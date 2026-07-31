from __future__ import annotations

from dataclasses import replace

from src.core.agent import ActParseError, Agent, Phase, PhaseKind
from src.core.config import GameCfg
from src.core.memory import MemoryEntry, both_agreed, render_turns
from src.games.base import PairingRecord
from src.games.prompts import notes_context, reflect_context, talk_context
from src.games.talk_rules import make_talk_rule
from src.providers.base import ProviderError
from src.strategy.base import PlayStrategy

# Outcome from A's perspective -> outcome from B's perspective.
_FLIP = {"CC": "CC", "DD": "DD", "DC": "CD", "CD": "DC"}


class ReputationPD:
    def __init__(self, cfg: GameCfg, strategy: PlayStrategy | None = None):
        self.cfg = cfg
        # The rules are no longer assembled here: the whole system (with the rules) is given as
        # a single string in AgentSpec.system_prompt, and the payoffs are substituted into
        # Agent.system_prompt from Phase.game_cfg. The strategy now lives on the agent
        # (agent.setup.play_strategy). The object is built lazily and cached by
        # (play_strategy, prediction_mapping). An explicitly passed `strategy` is a uniform
        # override (for tests and the simple case) on top of the per-agent one.
        self._override = strategy
        self._strategy_cache: dict[tuple[str, str], PlayStrategy] = {}

    def _strategy_for(self, agent: Agent) -> PlayStrategy:
        if self._override is not None:
            return self._override
        key = (agent.setup.play_strategy, agent.setup.prediction_mapping)
        st = self._strategy_cache.get(key)
        if st is None:
            from src.strategy.base import make_strategy  # lazy import: games<->strategy cycle
            st = make_strategy(key[0], key[1], self.cfg)
            self._strategy_cache[key] = st
        return st

    def resolve(self, x: int, y: int) -> tuple[str, float, float]:
        p = self.cfg.payoffs
        if x == y:
            return ("CC", p.R, p.R)
        if x == (y + 1) % 10:
            return ("DC", p.T, p.S)  # x betrayed y
        if y == (x + 1) % 10:
            return ("CD", p.S, p.T)  # y betrayed x
        return ("DD", p.P, p.P)

    async def play_pairing(self, a: Agent, b: Agent, round: int) -> PairingRecord:
        # No rng: the matcher fixes who opens cheap-talk via argument order (a opens).
        # The raw data of every LLM call is accumulated incrementally: on an LLM failure
        # (ProviderError) the pairing doesn't crash, but is returned as aborted
        # (finished=False) with the whole accumulated L2 log.
        transcript: list[dict] = []
        post_calls: list = []        # decide/predict/reflect calls (talk calls come from transcript)
        try:
            await self._cheap_talk(a, b, round, transcript)
            feed_a = self._feed(transcript, a.id)   # egocentric: its own turns are tagged <you>
            feed_b = self._feed(transcript, b.id)
            # The same closing reason the agents will see in this round's history (see memory).
            reason = (self.cfg.reason_agreed if both_agreed(transcript, a.id, b.id)
                      else self.cfg.reason_limit)
            da = await self._strategy_for(a).decide(a, b.id, round, feed_a, reason)
            post_calls += list(da.calls)
            db = await self._strategy_for(b).decide(b, a.id, round, feed_b, reason)
            post_calls += list(db.calls)
            x, y = da.number, db.number
            outcome, pa, pb = self.resolve(x, y)
            a.score += pa
            b.score += pb

            ra = rb = None
            usages = [t["usage"] for t in transcript] + [da.usage, db.usage]
            if self.cfg.reflection:
                # Reflection happens before the memory write: the round's facts arrive via
                # the context, while the agent's diary still shows only past rounds.
                ra, ua, ca = await self._reflect(a, b.id, round, feed_a, x, y, pa)
                post_calls += list(ca)
                rb, ub, cb = await self._reflect(b, a.id, round, feed_b, y, x, pb)
                post_calls += list(cb)
                usages += [ua, ub]

            public = _public(transcript)
            self._remember(a, b.id, round, public, da, y, outcome, pa, pb, ra)
            self._remember(b, a.id, round, public, db, x, _FLIP[outcome], pb, pa, rb)

            # Memory notes: every N rounds the agents consolidate their memory into notes
            # (after _remember — so the current round is already in memory). Note calls are
            # attached to the pairing. The consolidation trigger is the number of games
            # PLAYED BY THE AGENT (len(memory.entries) after _remember), not the round number:
            # idle rounds don't count, and both agents decide independently. Each note call
            # is accumulated right away (like reflect): if note(b) fails, note(a)'s already
            # captured L2 log isn't lost and makes it into the aborted pairing.
            na = nb = None
            if self._notes_due(a):
                na, una, cna = await self._make_notes(a, b.id, round)
                post_calls += list(cna); usages.append(una)
            if self._notes_due(b):
                nb, unb, cnb = await self._make_notes(b, a.id, round)
                post_calls += list(cnb); usages.append(unb)
            return PairingRecord(
                round=round, a_id=a.id, b_id=b.id, transcript=public,
                a_number=x, b_number=y,
                a_rationale=da.rationale, b_rationale=db.rationale,
                outcome=outcome, a_payoff=pa, b_payoff=pb, usage=_sum_usage(usages),
                a_predicted=da.predicted, b_predicted=db.predicted,
                a_reflection=ra, b_reflection=rb,
                a_notes=na, b_notes=nb,
                finished=True, llm_calls=_talk_calls(transcript) + post_calls,
            )
        except (ProviderError, ActParseError) as e:
            # Aborted pairing: no results (numbers/outcome/payoff = None), but the full raw
            # L2 log is kept — talk calls + completed decide/reflect + the failing ones (e.calls).
            calls = _talk_calls(transcript) + post_calls + list(e.calls)
            return PairingRecord(
                round=round, a_id=a.id, b_id=b.id, transcript=_public(transcript),
                usage=_usage_from_calls(calls), finished=False, llm_calls=calls,
            )

    async def _reflect(self, agent: Agent, partner_id: str, round: int, feed: str,
                       my_number: int, partner_number: int,
                       payoff: float) -> tuple[str, tuple[int, int], tuple]:
        """Ask the agent to reflect on the revealed outcome of the round.

        Args:
            agent: The agent making sense of the outcome.
            partner_id: Identifier of the partner in the current round.
            round: Round number.
            feed: Rendered negotiation history.
            my_number: The number chosen by the agent itself.
            partner_number: The number chosen by the partner.
            payoff: The agent's payoff for this round.

        Returns:
            A triple (reflection text, request usage, raw phase LLMCall's).
        """
        ctx = reflect_context(self.cfg, partner_id, round, feed, me_id=agent.id,
                              my_number=my_number, partner_number=partner_number, payoff=payoff,
                              score=agent.score)
        res = await agent.act(Phase(PhaseKind.REFLECT, ctx, game_cfg=self.cfg))
        return res.data["reflection"], res.usage, res.calls

    def _notes_due(self, agent: Agent) -> bool:
        """Is it time for the agent to consolidate its memory: every memory_notes_every games PLAYED.

        The game counter = len(agent.memory.entries) (one entry per round played; idle
        rounds don't add one). Called after _remember, so the current round is already counted."""
        every = self.cfg.memory_notes_every
        return bool(every) and len(agent.memory.entries) % every == 0

    async def _make_notes(self, agent: Agent, partner_id: str,
                          round: int) -> tuple[str, tuple[int, int], tuple]:
        """Consolidate the agent's memory into personal notes (NOTE phase) and store them in agent.memory.

        The agent sees its whole memory (old notes + buffer of new rounds) as history in
        Agent.act and rewrites it into new notes; from this point on render() sends the
        notes instead of the full history. A failure (ProviderError/ActParseError) propagates
        and aborts the pairing, like any LLM call.

        Args:
            agent: The agent consolidating its memory.
            partner_id: The co-player of the round that triggered the consolidation.
            round: Round number (for {round} in the template).

        Returns:
            A triple (notes text, request usage, raw NOTE-phase LLMCall's).
        """
        ctx = notes_context(self.cfg, round, score=agent.score, partner=partner_id)
        res = await agent.act(Phase(PhaseKind.NOTE, ctx, game_cfg=self.cfg))
        agent.memory.set_notes(res.data["notes"])
        return res.data["notes"], res.usage, res.calls

    async def _cheap_talk(self, a: Agent, b: Agent, round: int, transcript: list[dict]) -> None:
        # Fills the passed-in transcript in place (so that on an LLM failure the partial
        # cheap-talk and its L2 log aren't lost — each turn carries its own "calls").
        # The stop rule is a pluggable module (src/games/talk_rules.py): skip_turn decides
        # whether an already-ready speaker stays silent (latch), is_over decides whether it's
        # time to end the negotiation.
        rule = make_talk_rule(self.cfg.talk_stop_rule)
        ready = {a.id: False, b.id: False}
        order = [a, b]  # a opens; the matcher sets orientation via pairing order
        i = 0
        while len(transcript) < self.cfg.max_talk_turns:
            cur, oth = order[i % 2], order[(i + 1) % 2]
            i += 1
            if rule.skip_turn(cur.id, ready):
                if rule.is_over(ready):
                    break
                continue  # latched: stays silent while the other matures
            ctx = talk_context(self.cfg, oth.id, round, self._feed(transcript, cur.id),
                               cur.score)
            res = await cur.act(Phase(PhaseKind.TALK, ctx, game_cfg=self.cfg))
            transcript.append(
                {
                    "speaker": cur.id,
                    "text": res.public_text,
                    "ready": res.data["ready"],
                    "usage": res.usage,
                    "calls": res.calls,      # raw L2 log of the turn (turn_idx is added by the game)
                }
            )
            # next_ready decides whether to overwrite the flag with the signal (revocable) or latch it (sticky).
            ready[cur.id] = rule.next_ready(ready[cur.id], res.data["ready"])
            # Each agent necessarily speaks at least once: ending needs BOTH ready,
            # and an agent is marked ready only after it has spoken.
            if rule.is_over(ready):
                break

    def _feed(self, transcript: list[dict], me_id: str) -> str:
        # Live feed of the current round — the same <you>/<name> tags as the history (shared code).
        return render_turns(transcript, me_id, self.cfg.msg_self, self.cfg.msg_partner)

    def _remember(self, agent, partner_id, round, public_transcript, mine, partner_number,
                  outcome, payoff, partner_payoff, reflection=None):
        agent.memory.add(
            MemoryEntry(
                round=round,
                my_id=agent.id,
                partner_id=partner_id,
                transcript=public_transcript,
                my_number=mine.number,
                my_rationale=mine.rationale,
                partner_number=partner_number,
                outcome=outcome,
                payoff=payoff,
                partner_payoff=partner_payoff,
                score=agent.score - payoff,   # score BEFORE this round — as in the phase header
                my_predicted=mine.predicted,
                my_reflection=reflection,
            )
        )


def _public(transcript: list[dict]) -> list[dict]:
    return [{"speaker": t["speaker"], "text": t["text"], "ready": t["ready"]} for t in transcript]


def _talk_calls(transcript: list[dict]) -> list:
    """Collect the LLMCall's of cheap-talk turns, stamping each with turn_idx (index in transcript)."""
    out: list = []
    for ti, t in enumerate(transcript):
        out += [replace(c, turn_idx=ti) for c in t.get("calls", ())]
    return out


def _usage_from_calls(calls: list) -> dict:
    """Usage of an aborted pairing — total tokens and number of HTTP calls from its L2 log."""
    return {
        "prompt_tokens": sum(c.prompt_tokens for c in calls),
        "completion_tokens": sum(c.completion_tokens for c in calls),
        "calls": len(calls),
    }


def _sum_usage(usages: list) -> dict:
    pt = ct = calls = 0
    for u in usages:
        if u is None:
            continue
        pt += u[0]
        ct += u[1]
        calls += 1
    return {"prompt_tokens": pt, "completion_tokens": ct, "calls": calls}

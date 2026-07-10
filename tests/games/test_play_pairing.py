from __future__ import annotations

from conftest import ScriptedProvider

from src.core.agent import Agent, AgentSetup
from src.core.config import DEFAULT_RULES, GameCfg, ProviderCfg
from src.games.reputation_pd import ReputationPD
from src.providers.base import HttpAttempt, ProviderUnavailable


def _agent(id, replies):
    cfg = ProviderCfg(base_url="http://x/v1", model="m")
    # the rules are now part of system_prompt (no more concatenation) — we put them here so
    # that the DECIDE phase still gets the rules text in the system prompt
    return Agent(id, AgentSetup(f"You are {id}.\n\n" + DEFAULT_RULES, cfg), ScriptedProvider(replies))


class RaisingProvider:
    """Provider double that raises ProviderUnavailable with the given HttpAttempts."""

    def __init__(self, attempts):
        self._attempts = tuple(attempts)

    async def complete(self, *, system, messages, temperature, max_tokens):
        e = ProviderUnavailable("down")
        e.request = {"model": "m", "messages": []}
        e.attempts = self._attempts
        raise e

    async def aclose(self):
        pass


def _raising_agent(id, attempts):
    cfg = ProviderCfg(base_url="http://x/v1", model="m")
    return Agent(id, AgentSetup(f"You are {id}.", cfg), RaisingProvider(attempts))


def _decide(n, rationale="r"):
    return '{"number": %d, "rationale": "%s"}' % (n, rationale)


def _talk(msg, ready):
    # The agent-facing key for closing the chat is now "finish" (internally still stored as ready).
    return '{"message": "%s", "finish": %s}' % (msg, "true" if ready else "false")


# ---- Slice 2: decision only (talk off) ----

async def test_decision_only_cc():
    g = ReputationPD(GameCfg(max_talk_turns=0, rationale=True))
    a = _agent("A1", [_decide(4, "ra")])
    b = _agent("A2", [_decide(4, "rb")])
    rec = await g.play_pairing(a, b, round=1)
    assert rec.transcript == []
    assert (rec.a_number, rec.b_number, rec.outcome) == (4, 4, "CC")
    assert (rec.a_payoff, rec.b_payoff) == (3.0, 3.0)
    assert a.score == 3.0 and b.score == 3.0
    ea, eb = a.memory.entries[0], b.memory.entries[0]
    assert (ea.my_number, ea.partner_number, ea.my_rationale, ea.outcome) == (4, 4, "ra", "CC")
    assert (eb.my_number, eb.partner_number, eb.my_rationale, eb.outcome) == (4, 4, "rb", "CC")


async def test_decision_dc_mirrored_outcome():
    g = ReputationPD(GameCfg(max_talk_turns=0))
    a = _agent("A1", [_decide(5)])  # a == b+1 -> a betrayed b
    b = _agent("A2", [_decide(4)])
    rec = await g.play_pairing(a, b, 1)
    assert rec.outcome == "DC"
    assert (rec.a_payoff, rec.b_payoff) == (5.0, 0.0)
    assert a.memory.entries[0].outcome == "DC"   # I betrayed
    assert b.memory.entries[0].outcome == "CD"   # I was betrayed (mirrored)
    assert b.memory.entries[0].my_number == 4 and b.memory.entries[0].partner_number == 5


async def test_rationale_privacy():
    g = ReputationPD(GameCfg(max_talk_turns=0, rationale=True))
    a = _agent("A1", [_decide(1, "secret-a")])
    b = _agent("A2", [_decide(7, "secret-b")])
    rec = await g.play_pairing(a, b, 1)
    assert rec.a_rationale == "secret-a" and rec.b_rationale == "secret-b"
    assert a.memory.entries[0].my_rationale == "secret-a"
    # b's private rationale must not leak into a's memory entry
    assert "secret-b" not in str(a.memory.entries[0])


# ---- Slice 3: cheap-talk loop ----

async def test_talk_until_both_ready_latch():
    g = ReputationPD(GameCfg(max_talk_turns=6))
    a = _agent("A1", [_talk("hi", False), _talk("ok 4", True), _decide(4)])
    b = _agent("A2", [_talk("4?", True), _decide(4)])
    rec = await g.play_pairing(a, b, 1)  # a first
    assert [t["speaker"] for t in rec.transcript] == ["A1", "A2", "A1"]
    assert rec.transcript[-1]["ready"] is True
    assert rec.outcome == "CC"


async def test_talk_ceiling_caps_turns():
    g = ReputationPD(GameCfg(max_talk_turns=2))
    a = _agent("A1", [_talk("a1", False), _decide(3)])
    b = _agent("A2", [_talk("b1", False), _decide(5)])
    rec = await g.play_pairing(a, b, 1)
    assert [t["speaker"] for t in rec.transcript] == ["A1", "A2"]  # capped at 2


async def test_min_one_each_even_if_first_ready():
    g = ReputationPD(GameCfg(max_talk_turns=6))
    a = _agent("A1", [_talk("ready now", True), _decide(2)])
    b = _agent("A2", [_talk("ok", True), _decide(2)])
    rec = await g.play_pairing(a, b, 1)
    assert [t["speaker"] for t in rec.transcript] == ["A1", "A2"]  # b still gets a turn


async def test_latched_agent_stays_silent():
    g = ReputationPD(GameCfg(max_talk_turns=6))
    a = _agent("A1", [_talk("done", True), _decide(0)])
    b = _agent("A2", [_talk("hmm", False), _talk("ok", True), _decide(0)])
    rec = await g.play_pairing(a, b, 1)
    assert [t["speaker"] for t in rec.transcript] == ["A1", "A2", "A2"]


async def test_revocable_finisher_keeps_talking():
    # both_ready_revocable: the first one to set finish does NOT fall silent — it keeps
    # taking turns; the dialogue only breaks off once finish is set for BOTH (here — at t4).
    g = ReputationPD(GameCfg(max_talk_turns=6, talk_stop_rule="both_ready_revocable"))
    a = _agent("A1", [_talk("done", True), _talk("still here", True), _decide(0)])
    b = _agent("A2", [_talk("hmm", False), _talk("ok", True), _decide(0)])
    rec = await g.play_pairing(a, b, 1)
    assert [t["speaker"] for t in rec.transcript] == ["A1", "A2", "A1", "A2"]  # A1 speaks twice
    assert rec.transcript[-1]["ready"] is True
    assert rec.outcome == "CC"


async def test_revocable_stops_only_on_mutual_finish():
    # A single finish does not stop it: A finishes at t1, B does not; we keep going until B finishes.
    g = ReputationPD(GameCfg(max_talk_turns=6, talk_stop_rule="both_ready_revocable"))
    a = _agent("A1", [_talk("a1", True), _talk("a2", True), _decide(2)])
    b = _agent("A2", [_talk("b1", False), _talk("b2", True), _decide(2)])
    rec = await g.play_pairing(a, b, 1)
    assert len(rec.transcript) == 4                       # didn't break off on the first finish
    assert [t["speaker"] for t in rec.transcript] == ["A1", "A2", "A1", "A2"]


async def test_committed_finish_is_sticky_but_keeps_talking():
    # both_ready_committed: the agent keeps talking (like revocable), but finish is STICKY —
    # A sets finish at t1, tries to revoke it at t3 (finish=false), but ready[A] stays
    # true, so the second finish from B at t4 ends the chat.
    g = ReputationPD(GameCfg(max_talk_turns=6, talk_stop_rule="both_ready_committed"))
    a = _agent("A1", [_talk("done", True), _talk("nvm", False), _decide(0)])
    b = _agent("A2", [_talk("hmm", False), _talk("ok", True), _decide(0)])
    rec = await g.play_pairing(a, b, 1)
    assert [t["speaker"] for t in rec.transcript] == ["A1", "A2", "A1", "A2"]
    assert rec.transcript[2]["ready"] is False   # A publicly said finish=false on their 2nd turn…
    assert rec.outcome == "CC"                   # …but the chat closed anyway (finish was sticky)


async def test_first_speaker_is_first_arg():
    # The matcher fixes who opens via pairing order: the first positional agent speaks first.
    g = ReputationPD(GameCfg(max_talk_turns=6))
    a = _agent("A1", [_talk("a", True), _decide(0)])
    b = _agent("A2", [_talk("b", True), _decide(0)])
    rec = await g.play_pairing(a, b, 1)  # a passed first -> A1 opens
    assert rec.transcript[0]["speaker"] == "A1"

    # swap the arguments -> A2 opens (matcher returned (A2, A1))
    a = _agent("A1", [_talk("a", True), _decide(0)])
    b = _agent("A2", [_talk("b", True), _decide(0)])
    rec = await g.play_pairing(b, a, 1)  # b passed first -> A2 opens
    assert rec.transcript[0]["speaker"] == "A2"


async def test_usage_aggregated():
    g = ReputationPD(GameCfg(max_talk_turns=2))
    a = _agent("A1", [_talk("a", False), _decide(1)])
    b = _agent("A2", [_talk("b", False), _decide(2)])
    rec = await g.play_pairing(a, b, 1)
    assert rec.usage["calls"] == 4  # 2 talk + 2 decide acts
    assert rec.usage["prompt_tokens"] == 8 and rec.usage["completion_tokens"] == 12


async def test_decide_context_contains_feed_and_ids():
    g = ReputationPD(GameCfg(max_talk_turns=2))
    a = _agent("A1", [_talk("take 4 plz", True), _decide(4)])
    b = _agent("A2", [_talk("ok", True), _decide(4)])
    await g.play_pairing(a, b, 7)
    system, messages = a.provider.calls[-1]  # a's DECIDE call
    ctx = messages[-1].content
    assert "A2" in ctx and "Round 7" in ctx and "take 4 plz" in ctx
    assert "<you>take 4 plz</you>" in ctx   # the feed is egocentric: your own messages are <you>
    assert "<A2>ok</A2>" in ctx             # the opponent's message is tagged with their name
    assert "0 to 9" in system  # rules went into the system prompt


# ---- Rationale switched off in config ----

async def test_rationale_off_asks_bare_number():
    # rationale=false -> bare static template: only the number, no rationale is stored
    g = ReputationPD(GameCfg(max_talk_turns=0, rationale=False))
    a = _agent("A1", ['{"number": 4}'])
    b = _agent("A2", ['{"number": 4, "rationale": "volunteered"}'])
    rec = await g.play_pairing(a, b, 1)
    assert (rec.a_number, rec.b_number, rec.outcome) == (4, 4, "CC")
    assert rec.a_rationale == "" and rec.b_rationale == ""
    assert a.memory.entries[0].my_rationale == ""
    _, messages = a.provider.calls[-1]   # a's DECIDE call
    assert "rationale" not in messages[-1].content.lower()


# ---- Reflection after the outcome ----

def _reflect(text):
    return '{"reflection": "%s"}' % text


async def test_reflection_off_by_default():
    g = ReputationPD(GameCfg(max_talk_turns=0))
    a = _agent("A1", [_decide(4)])        # no reflect reply queued: an extra call would crash
    b = _agent("A2", [_decide(4)])
    rec = await g.play_pairing(a, b, 1)
    assert rec.a_reflection is None and rec.b_reflection is None
    assert a.memory.entries[0].my_reflection is None
    assert rec.usage["calls"] == 2


async def test_reflection_stored_in_memory_and_record():
    g = ReputationPD(GameCfg(max_talk_turns=0, reflection=True))
    a = _agent("A1", [_decide(5), _reflect("betrayal worked")])
    b = _agent("A2", [_decide(4), _reflect("A1 cannot be trusted")])
    rec = await g.play_pairing(a, b, 1)
    assert rec.a_reflection == "betrayal worked"
    assert rec.b_reflection == "A1 cannot be trusted"
    assert a.memory.entries[0].my_reflection == "betrayal worked"
    assert b.memory.entries[0].my_reflection == "A1 cannot be trusted"
    assert rec.usage["calls"] == 4  # 2 decide + 2 reflect


async def test_reflection_context_reveals_round_result():
    g = ReputationPD(GameCfg(max_talk_turns=0, reflection=True))
    a = _agent("A1", [_decide(5), _reflect("ra")])
    b = _agent("A2", [_decide(4), _reflect("rb")])
    await g.play_pairing(a, b, 7)
    _, messages = a.provider.calls[-1]   # a's REFLECT call
    ctx = messages[-1].content
    assert "A2" in ctx and "Round 7" in ctx
    assert "5" in ctx and "4" in ctx     # both revealed numbers
    assert "A1 (you) picked 5" in ctx    # the agent itself — "<name> (you)", the opponent — by name


async def test_reflection_privacy():
    g = ReputationPD(GameCfg(max_talk_turns=0, reflection=True))
    a = _agent("A1", [_decide(4), _reflect("secret-ref-a")])
    b = _agent("A2", [_decide(4), _reflect("secret-ref-b")])
    await g.play_pairing(a, b, 1)
    # b's reflection must not leak into a's memory entry
    assert "secret-ref-b" not in str(a.memory.entries[0])


# ---- Memory notes: periodic memory consolidation ----

def _note(text):
    return '{"notes": "%s"}' % text


async def test_memory_notes_off_by_default():
    g = ReputationPD(GameCfg(max_talk_turns=0))   # memory_notes_every=0
    a = _agent("A1", [_decide(4)])                # no extra note call is queued -> would crash
    b = _agent("A2", [_decide(4)])
    rec = await g.play_pairing(a, b, 3)
    assert rec.a_notes is None and rec.b_notes is None
    assert a.memory.notes is None and b.memory.notes is None
    assert rec.usage["calls"] == 2


async def test_memory_notes_keyed_on_played_rounds_not_round_number():
    # Consolidation is counted by pairings PLAYED by the agent, not by round number: the agent
    # has played only 1 pairing (even though round=5), 1 % 2 != 0 -> no consolidation.
    g = ReputationPD(GameCfg(max_talk_turns=0, memory_notes_every=2))
    a = _agent("A1", [_decide(4)])
    b = _agent("A2", [_decide(4)])
    rec = await g.play_pairing(a, b, 5)
    assert rec.a_notes is None and rec.b_notes is None
    assert a.memory.notes is None


async def test_memory_notes_taken_after_n_played_rounds():
    g = ReputationPD(GameCfg(max_talk_turns=0, memory_notes_every=2))
    a = _agent("A1", [_decide(4), _decide(4), _note("A2 cooperates")])
    b = _agent("A2", [_decide(4), _decide(4), _note("A1 cooperates")])
    rec1 = await g.play_pairing(a, b, 1)          # 1 pairing played -> no consolidation
    assert rec1.a_notes is None and a.memory.notes is None
    rec2 = await g.play_pairing(a, b, 2)          # 2 played -> consolidation for both
    assert rec2.a_notes == "A2 cooperates" and rec2.b_notes == "A1 cooperates"
    assert a.memory.notes == "A2 cooperates" and b.memory.notes == "A1 cooperates"
    assert a.memory.noted_upto == 2 and b.memory.noted_upto == 2  # both pairings consolidated
    note_calls = [c for c in rec2.llm_calls if c.phase == "note"]
    assert len(note_calls) == 2 and all(c.turn_idx is None for c in note_calls)


async def test_note_failure_aborts_pairing_as_unfinished():
    g = ReputationPD(GameCfg(max_talk_turns=0, memory_notes_every=1))   # consolidation every pairing
    a = _agent("A1", [_decide(4), _note("ok")])   # decide ok, note ok
    b = _agent("A2", [_decide(4), "nope", "nope", "nope"])  # note: invalid JSON -> ActParseError
    rec = await g.play_pairing(a, b, 1)
    assert rec.finished is False
    statuses = [c.status for c in rec.llm_calls]
    assert statuses.count("parse_error") == 3     # b's three failed note attempts
    assert any(c.phase == "note" and c.status == "ok" for c in rec.llm_calls)  # a's note succeeded


# ---- Task 6: strategy delegation + prediction persistence ----

async def test_direct_strategy_leaves_predicted_none():
    g = ReputationPD(GameCfg(max_talk_turns=0))   # default DirectStrategy
    a = _agent("A1", [_decide(4)])
    b = _agent("A2", [_decide(4)])
    rec = await g.play_pairing(a, b, 1)
    assert rec.a_predicted is None and rec.b_predicted is None
    assert a.memory.entries[0].my_predicted is None


async def test_prediction_strategy_records_and_remembers_predictions():
    from src.strategy.mappings import get_mapping
    from src.strategy.prediction import PredictionStrategy

    g = ReputationPD(GameCfg(max_talk_turns=0),
                     strategy=PredictionStrategy(get_mapping("one_above"), GameCfg(max_talk_turns=0)))
    a = _agent("A1", ['{"number": 4, "rationale": "pa"}'])   # predicts 4 -> chooses 5
    b = _agent("A2", ['{"number": 4, "rationale": "pb"}'])   # predicts 4 -> chooses 5
    rec = await g.play_pairing(a, b, 1)
    assert (rec.a_predicted, rec.b_predicted) == (4, 4)
    assert (rec.a_number, rec.b_number) == (5, 5)
    assert rec.outcome == "CC"
    # private scratchpad: the predicted value lives in the acting agent's memory
    assert a.memory.entries[0].my_predicted == 4
    assert "pa" in str(a.memory.entries[0])
    # the partner's prediction reasoning never leaks into a's memory entry
    assert "pb" not in str(a.memory.entries[0])


# ---- L2: finished flag + llm_calls capture ----

async def test_finished_pairing_collects_talk_and_decide_calls():
    g = ReputationPD(GameCfg(max_talk_turns=2))
    a = _agent("A1", [_talk("hi", True), _decide(4)])
    b = _agent("A2", [_talk("ok", True), _decide(4)])
    rec = await g.play_pairing(a, b, 1)
    assert rec.finished is True
    phases = [(c.phase, c.turn_idx) for c in rec.llm_calls]
    assert ("talk", 0) in phases and ("talk", 1) in phases   # talk calls are tagged with turn_idx
    assert ("decide", None) in phases                        # decide — no turn_idx
    assert all(c.status == "ok" for c in rec.llm_calls)


async def test_provider_error_aborts_pairing_as_unfinished():
    g = ReputationPD(GameCfg(max_talk_turns=0))
    a = _agent("A1", [_decide(4)])
    b = _raising_agent("A2", [HttpAttempt("network", None, {"m": 1}, None, None, "boom")])
    rec = await g.play_pairing(a, b, 1)
    assert rec.finished is False
    assert rec.a_number is None and rec.outcome is None      # no result
    assert a.score == 0.0 and b.score == 0.0                 # nothing scored
    statuses = [c.status for c in rec.llm_calls]
    assert "ok" in statuses          # a's decide succeeded
    assert "network" in statuses     # b's decide failed


async def test_parse_exhaustion_aborts_pairing_as_unfinished():
    g = ReputationPD(GameCfg(max_talk_turns=0))
    a = _agent("A1", [_decide(4)])
    b = _agent("A2", ["nope", "nope", "nope"])   # never valid JSON -> ActParseError, no substitution
    rec = await g.play_pairing(a, b, 1)
    assert rec.finished is False
    assert rec.a_number is None and rec.outcome is None       # no result
    assert a.score == 0.0 and b.score == 0.0                  # nothing scored
    statuses = [c.status for c in rec.llm_calls]
    assert "ok" in statuses                       # a's decide succeeded
    assert statuses.count("parse_error") == 3     # b's three failed attempts logged

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum

from src.core.config import GameCfg, PromptVariants, ProviderCfg
from src.core.jsonextract import extract_json_obj
from src.core.memory import Memory
from src.providers.base import LLMProvider, Message, ProviderError

_MAX_PARSE_RETRIES = 2

_log = logging.getLogger(__name__)

# The seam between two adjacent <game> blocks: a closing </game> followed, after a single
# whitespace gap, by an opening <game>. We merge such blocks into one (e.g. the round result
# line and the opening of the next round), keeping the separator itself.
_GAME_SEAM = re.compile(r"</game>(\s*)<game>")


def _merge_game_blocks(text: str) -> str:
    """Remove </game>…<game> seams between adjacent blocks — yields one transcript."""
    return _GAME_SEAM.sub(r"\1", text)


# Substitution order: the note text goes last so its content is never rescanned.
_FRAGMENT_KEYS = ("recent_rounds", "history_lines", "history", "notes_line", "notes")


def _fill_fragments(template: str, frags: dict[str, str]) -> str:
    """Substitute the memory fragments, wiping sections whose memory is empty.

    The template splits into paragraphs (blank-line-delimited groups). A paragraph that
    names fragments which ALL render empty is dropped whole — its headers included; that is
    how a with_history text degrades into the no_history message on the first round. In a
    kept paragraph an individual empty fragment vanishes with just its own line.
    """
    out = []
    for paragraph in template.split("\n\n"):
        keys = [k for k in _FRAGMENT_KEYS if "{%s}" % k in paragraph]
        if keys and all(not frags[k] for k in keys):
            continue
        for k in keys:
            if frags[k]:
                paragraph = paragraph.replace("{%s}" % k, frags[k])
            else:
                paragraph = (paragraph.replace("{%s}\n" % k, "")
                             .replace("{%s}" % k, ""))
        out.append(paragraph)
    return "\n\n".join(out)


class ActParseError(Exception):
    """The agent could not get a valid phase response after all parse-retries.

    No fallback / substitution: the pairing is aborted (finished=0) and the episode stops.
    Carries the failed call log (`calls`) so the pairing can persist it before aborting,
    exactly like a ProviderError.
    """

    calls: tuple = ()
    agent_id: str | None = None
    phase: str | None = None


class PhaseKind(Enum):
    TALK = "talk"
    DECIDE = "decide"
    PREDICT = "predict"
    REFLECT = "reflect"
    NOTE = "note"


@dataclass(frozen=True)
class Phase:
    kind: PhaseKind
    context: str | PromptVariants   # the step prompt with facts already filled
    game_cfg: GameCfg | None = None  # history transcript templates + payoffs for substitution into system; comes from the game


@dataclass(frozen=True)
class LLMCall:
    """Raw L2 log of one `provider.complete()` call (including parse retries).

    Self-contained record: everything except `round_idx`/`pair_idx`/`call_idx` (added by
    `Storage.observe`). `turn_idx` is set by the game for the TALK phase.

    Attributes:
        agent_id: Who made the call.
        phase: Phase (talk/decide/predict/reflect).
        attempt: Agent.act parse attempt (1..3).
        http_attempt: Network retry within one complete() call (1..5).
        status: ok | parse_error | bad_json | bad_shape | http_error | server_error | network.
        status_code: HTTP status code of the attempt (None on a network error).
        request: The exact payload sent (None if the provider didn't return it).
        response: Extracted response text (only on the final ok attempt).
        response_raw: Exact response body as a string (resp.text, incl. a 5xx body).
        error: Failure message.
        prompt_tokens: Prompt tokens (on the final ok attempt).
        completion_tokens: Completion tokens (on the final ok attempt).
        turn_idx: Message turn index for TALK; None for other phases.
    """

    agent_id: str
    phase: str
    attempt: int
    http_attempt: int
    status: str
    status_code: int | None
    request: dict | None
    response: str | None
    response_raw: str | None
    error: str | None
    prompt_tokens: int
    completion_tokens: int
    turn_idx: int | None = None


@dataclass(frozen=True)
class ActResult:
    public_text: str | None     # TALK -> message; DECIDE -> None
    data: dict                  # TALK -> {message, ready}; DECIDE -> {number, rationale}
    usage: tuple[int, int]      # (prompt_tokens, completion_tokens), summed over all attempts
    calls: tuple[LLMCall, ...] = ()   # L2 log of all attempts of this act()


@dataclass(frozen=True)
class AgentSetup:
    system_prompt: str        # the agent's full system prompt (ONE template string); {id} and payoffs are substituted by Agent.system_prompt
    provider_cfg: ProviderCfg
    # This agent's play strategy — plain strings (core does not import strategy; the strategy
    # object is assembled at the game level, see ReputationPD._strategy_for).
    play_strategy: str = "direct"        # "direct" | "prediction"
    prediction_mapping: str = "match"    # predict->choice mapping (prediction strategy only)
    deceptive: bool = False              # evolution: this agent counts as deceptive (see EvolutionCfg)
    invincible: bool = False             # evolution: this agent never dies (see EvolutionCfg)


# Correction text for retries lives in the config (GameCfg.*_correction), not hardcoded here.
# For phases assembled without game_cfg (unit tests), we take the default GameCfg(). One correction
# per phase now (DECIDE/PREDICT no longer have a _bare variant) — write it to match the phase prompt.
_DEFAULT_GAME_CFG = GameCfg()


def _correction(game_cfg: "GameCfg | None", kind: PhaseKind) -> str:
    cfg = game_cfg if game_cfg is not None else _DEFAULT_GAME_CFG
    if kind is PhaseKind.DECIDE:
        return cfg.decide_correction
    if kind is PhaseKind.PREDICT:
        return cfg.predict_correction
    if kind is PhaseKind.TALK:
        return cfg.talk_correction
    if kind is PhaseKind.REFLECT:
        return cfg.reflect_correction
    return cfg.note_correction  # PhaseKind.NOTE


class Agent:
    def __init__(
        self,
        id: str,
        setup: AgentSetup,
        provider: LLMProvider,
        *,
        context_window: int | None = None,
    ):
        self.id = id
        self.setup = setup
        self.provider = provider
        self.memory = Memory()
        self.score = 0.0
        self.parse_failures = 0
        self._window = context_window

    def system_prompt(self, game_cfg: "GameCfg | None" = None) -> str:
        """The agent's full system prompt: self.setup.system_prompt template with {id} and payoffs substituted.

        There is no longer the old assembly (identity + persona + rules) — system is given
        as a single string. {id} is always substituted; payoff parameters
        {R}/{T}/{P}/{S}/{max_talk_turns} are substituted when game_cfg is known (for all
        production phases it travels in Phase.game_cfg)."""
        system = self.setup.system_prompt.replace("{id}", self.id)
        if game_cfg is not None:
            p = game_cfg.payoffs
            system = (system
                      .replace("{R}", f"{p.R:g}").replace("{T}", f"{p.T:g}")
                      .replace("{P}", f"{p.P:g}").replace("{S}", f"{p.S:g}")
                      .replace("{max_talk_turns}", str(game_cfg.max_talk_turns)))
        return system

    async def act(self, phase: Phase) -> ActResult:
        system = self.system_prompt(phase.game_cfg)
        # NOTE folds memory into notes — for that it needs to see it in full (no window),
        # so nothing is lost during consolidation; the other phases use the window.
        window = None if phase.kind is PhaseKind.NOTE else self._window
        diary = self.memory.render(window, phase.game_cfg)  # [] or [user message with the history transcript]
        history = diary[0].content if diary else ""
        context = phase.context
        if isinstance(context, PromptVariants):
            # "Past" excludes the just-played round at NOTE time (already in memory).
            past = len(self.memory.entries) - (1 if phase.kind is PhaseKind.NOTE else 0)
            if past > 0 or context.no_history is None:
                context = context.with_history
            else:
                context = context.no_history
            owns = True
        else:
            owns = any("{%s}" % k in context for k in _FRAGMENT_KEYS)
        if owns:
            frags = self.memory.fragments(window, phase.game_cfg)
            frags["history"] = history
            base = _fill_fragments(context, frags)
        else:
            # Legacy assembly (stored configs): memory diary precedes the phase context.
            base = f"{history}\n\n{context}" if history else context
        cfg = self.setup.provider_cfg

        prompt_toks = 0
        comp_toks = 0
        calls: list[LLMCall] = []
        correction: str | None = None
        for attempt in range(1, _MAX_PARSE_RETRIES + 2):
            # one user message: the assembled step prompt (+ correction on parse retry)
            content = base
            if correction is not None:
                content = f"{content}\n\n{correction}"
            content = _merge_game_blocks(content)   # merge the </game>…<game> seams
            messages = [Message("user", content)]
            if phase.kind is not PhaseKind.TALK and _log.isEnabledFor(logging.DEBUG):
                _log.debug(_render_trace(self.id, phase.kind, attempt, system, messages))
            try:
                comp = await self.provider.complete(
                    system=system,
                    messages=messages,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                )
            except ProviderError as e:
                # a failed call is also L2-log rows; attach context to the exception, re-raise
                calls.extend(_calls_from_attempts(self.id, phase.kind, attempt, e.attempts))
                e.agent_id, e.phase, e.attempt = self.id, phase.kind.value, attempt
                e.calls = tuple(calls)
                raise
            prompt_toks += comp.prompt_tokens
            comp_toks += comp.completion_tokens
            data = _parse(phase.kind, comp.text)
            calls.extend(_calls_from_attempts(
                self.id, phase.kind, attempt, comp.attempts, parsed=data is not None))
            if data is not None:
                return _result(phase.kind, data, (prompt_toks, comp_toks), tuple(calls))
            correction = _correction(phase.game_cfg, phase.kind)

        # all parse-retries exhausted: no fallback — abort the pairing (finished=0)
        self.parse_failures += 1
        err = ActParseError(
            f"{self.id} {phase.kind.value}: no valid JSON after {_MAX_PARSE_RETRIES + 1} attempts")
        err.agent_id, err.phase = self.id, phase.kind.value
        err.calls = tuple(calls)
        raise err


def _calls_from_attempts(agent_id, kind, attempt, attempts, *, parsed: bool | None = None):
    """Unroll one complete() call's HTTP attempts into LLMCall rows (one per attempt).

    On the final HTTP-successful attempt, the `ok` status is changed to `parse_error` if the
    phase validator rejected the text (`parsed=False`). For a failed complete(), `parsed` is
    not set.
    """
    out = []
    n = len(attempts)
    for i, at in enumerate(attempts, start=1):
        status = at.status
        if parsed is not None and i == n and at.status == "ok" and not parsed:
            status = "parse_error"
        out.append(LLMCall(
            agent_id=agent_id, phase=kind.value, attempt=attempt, http_attempt=i,
            status=status, status_code=at.status_code, request=at.request,
            response=at.response, response_raw=at.response_raw, error=at.error,
            prompt_tokens=at.prompt_tokens, completion_tokens=at.completion_tokens,
        ))
    return out


def _result(kind: PhaseKind, data: dict, usage: tuple[int, int],
            calls: tuple[LLMCall, ...] = ()) -> ActResult:
    public = data["message"] if kind is PhaseKind.TALK else None
    return ActResult(public_text=public, data=data, usage=usage, calls=calls)


def _parse(kind: PhaseKind, text: str) -> dict | None:
    obj = extract_json_obj(text)
    if obj is None:
        return None
    if kind in (PhaseKind.DECIDE, PhaseKind.PREDICT):
        return _validate_decide(obj)
    if kind is PhaseKind.TALK:
        return _validate_talk(obj)
    if kind is PhaseKind.REFLECT:
        return _validate_reflect(obj)
    if kind is PhaseKind.NOTE:
        return _validate_notes(obj)
    return None


def _validate_decide(obj: dict) -> dict | None:
    n = obj.get("number")
    if isinstance(n, bool):  # bool is a subclass of int — reject explicitly
        return None
    if not isinstance(n, int):
        try:
            n = int(str(n).strip())
        except (TypeError, ValueError):
            return None
    if not (0 <= n <= 9):
        return None
    rationale = obj.get("rationale", "")
    if not isinstance(rationale, str):
        rationale = str(rationale)
    return {"number": n, "rationale": rationale}


def _validate_talk(obj: dict) -> dict | None:
    message = obj.get("message")
    if message is None:  # message is required; trigger a retry
        return None
    if not isinstance(message, str):
        message = str(message)
    # The agent-facing key is "finish" (close the chat); internally we still store it as "ready".
    return {"message": message, "ready": _coerce_bool(obj.get("finish"))}


def _validate_reflect(obj: dict) -> dict | None:
    reflection = obj.get("reflection")
    if reflection is None:  # the key is required; otherwise retry with a correction
        return None
    if not isinstance(reflection, str):
        reflection = str(reflection)
    return {"reflection": reflection}


def _validate_notes(obj: dict) -> dict | None:
    # "note" (current configs) or "notes" (stored ones); normalized to the internal "notes".
    notes = obj.get("note", obj.get("notes"))
    if notes is None:  # the key is required; otherwise retry with a correction
        return None
    if not isinstance(notes, str):
        notes = str(notes)
    return {"notes": notes}


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "yes", "y", "1"):
            return True
    return False  # missing / unknown -> default False (keep talking)


def _render_trace(agent_id: str, kind: PhaseKind, attempt: int,
                  system: str, messages: list[Message]) -> str:
    """Render the exact LLM input (system + all messages) for the log record.

    Args:
        agent_id: Identifier of the agent making the request.
        kind: Request phase (DECIDE, PREDICT or REFLECT).
        attempt: Request attempt number (1..3, retries due to parse errors).
        system: Full system prompt (persona + rules).
        messages: Request messages — one user message (memory diary + phase
            context + correction), assembled in Agent.act.

    Returns:
        Multi-line log record text.
    """
    parts = [
        f"LLM input: agent {agent_id}, phase {kind.value}, attempt {attempt}",
        f"--- system ---\n{system}",
    ]
    parts += [f"--- {m.role} ---\n{m.content}" for m in messages]
    return "\n".join(parts)

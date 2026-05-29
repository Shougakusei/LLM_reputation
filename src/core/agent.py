from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from enum import Enum

from src.core.config import ProviderCfg
from src.core.memory import Memory
from src.providers.base import LLMProvider, Message

_MAX_PARSE_RETRIES = 2


class PhaseKind(Enum):
    TALK = "talk"
    DECIDE = "decide"


@dataclass(frozen=True)
class Phase:
    kind: PhaseKind
    context: str          # rendered situation + output instruction (becomes a user message)
    rules: str = ""       # static game rules; the agent puts them in `system` after the persona


@dataclass(frozen=True)
class ActResult:
    public_text: str | None     # TALK -> message; DECIDE -> None
    data: dict                  # TALK -> {message, ready}; DECIDE -> {number, rationale}
    usage: tuple[int, int]      # (prompt_tokens, completion_tokens), summed over all attempts


@dataclass(frozen=True)
class AgentSetup:
    persona: str
    provider_cfg: ProviderCfg


_CORRECTION = {
    PhaseKind.DECIDE: (
        "Respond with ONLY valid JSON, nothing else: "
        '{"number": <integer 0-9>, "rationale": "<short reason>"}'
    ),
    PhaseKind.TALK: (
        "Respond with ONLY valid JSON, nothing else: "
        '{"message": "<your message>", "ready": <true|false>}'
    ),
}


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

    async def act(self, phase: Phase) -> ActResult:
        system = self.setup.persona
        if phase.rules:
            system = f"{system}\n\n{phase.rules}"
        base = self.memory.render(self._window) + [Message("user", phase.context)]
        cfg = self.setup.provider_cfg

        prompt_toks = 0
        comp_toks = 0
        correction: str | None = None
        for _ in range(_MAX_PARSE_RETRIES + 1):
            messages = base if correction is None else base + [Message("user", correction)]
            comp = await self.provider.complete(
                system=system,
                messages=messages,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )
            prompt_toks += comp.prompt_tokens
            comp_toks += comp.completion_tokens
            data = _parse(phase.kind, comp.text)
            if data is not None:
                return _result(phase.kind, data, (prompt_toks, comp_toks))
            correction = _CORRECTION[phase.kind]

        self.parse_failures += 1
        return _result(phase.kind, _fallback(phase.kind), (prompt_toks, comp_toks))


def _result(kind: PhaseKind, data: dict, usage: tuple[int, int]) -> ActResult:
    public = data["message"] if kind is PhaseKind.TALK else None
    return ActResult(public_text=public, data=data, usage=usage)


def _parse(kind: PhaseKind, text: str) -> dict | None:
    obj = _extract_json_obj(text)
    if obj is None:
        return None
    if kind is PhaseKind.DECIDE:
        return _validate_decide(obj)
    if kind is PhaseKind.TALK:
        return _validate_talk(obj)
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
    return {"message": message, "ready": _coerce_bool(obj.get("ready"))}


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


def _fallback(kind: PhaseKind) -> dict:
    if kind is PhaseKind.DECIDE:
        return {"number": random.randint(0, 9), "rationale": "(unparsed)"}
    return {"message": "", "ready": True}


def _extract_json_obj(text: str) -> dict | None:
    candidates = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1).strip())
    block = _first_brace_block(text)
    if block:
        candidates.append(block)
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _first_brace_block(text: str) -> str | None:
    # Naive balanced-brace scan: good enough for prose-wrapped JSON; does not account
    # for braces inside string values (rare in our outputs).
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None

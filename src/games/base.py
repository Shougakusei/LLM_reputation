from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from src.core.agent import Agent


@dataclass
class PairingRecord:
    round: int
    a_id: str
    b_id: str
    transcript: list[dict]          # public cheap-talk: [{speaker, text, ready}]
    # Pairing results. NULL/empty if the pairing was aborted (finished=False, LLM failure):
    a_number: int | None = None
    b_number: int | None = None
    a_k: int = 1                    # each side's winning gap this round (1 unless game.gap > 1)
    b_k: int = 1
    a_rationale: str | None = None  # private; never shown to the partner, kept for analysis
    b_rationale: str | None = None
    outcome: str | None = None      # from A's perspective: CC / DC / CD / DD
    a_payoff: float | None = None
    b_payoff: float | None = None
    usage: dict = field(default_factory=dict)   # {"prompt_tokens", "completion_tokens", "calls"}
    a_predicted: int | None = None  # prediction strategy; None for direct
    b_predicted: int | None = None
    a_reflection: str | None = None  # post-game reflection; None if disabled
    b_reflection: str | None = None
    a_notes: str | None = None      # memory notes after the round; None if not consolidated this round
    b_notes: str | None = None
    finished: bool = True           # False = pairing aborted by an LLM failure (no results)
    llm_calls: list = field(default_factory=list)   # raw LLMCall's of the pairing (L2 log)


class Game(Protocol):
    async def play_pairing(self, a: Agent, b: Agent, round: int) -> PairingRecord: ...

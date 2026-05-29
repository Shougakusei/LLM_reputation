from __future__ import annotations

from dataclasses import dataclass

from src.providers.base import Message


@dataclass
class MemoryEntry:
    round: int
    partner_id: str
    transcript: list[dict]  # [{speaker, text, ready}]
    my_number: int
    my_rationale: str
    partner_number: int
    outcome: str
    payoff: float


class Memory:
    def __init__(self) -> None:
        self.entries: list[MemoryEntry] = []

    def add(self, entry: MemoryEntry) -> None:
        self.entries.append(entry)

    def render(self, window: int | None) -> list[Message]:
        if window is None:
            entries = self.entries
        elif window <= 0:
            entries = []
        else:
            entries = self.entries[-window:]
        if not entries:
            return []
        diary = "Past rounds (oldest first):\n\n" + "\n\n".join(
            _render_entry(e) for e in entries
        )
        return [Message("user", diary)]


def _render_entry(e: MemoryEntry) -> str:
    lines = [f"[Round {e.round} · vs {e.partner_id}]"]
    if e.transcript:
        lines.append("Talk:")
        for turn in e.transcript:
            ready = str(bool(turn.get("ready"))).lower()
            lines.append(f"  {turn.get('speaker', '?')}: {turn.get('text', '')} (ready={ready})")
    reason = f" (reason: {e.my_rationale})" if e.my_rationale else ""
    lines.append(
        f"Decision: I chose {e.my_number}{reason}. "
        f"{e.partner_id} chose {e.partner_number}. "
        f"Outcome: {e.outcome}. Payoff: {e.payoff:g}."
    )
    return "\n".join(lines)

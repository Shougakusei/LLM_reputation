from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class RoundPlan:
    pairings: list[tuple[str, str]]   # disjoint pairs (a, b); a opens cheap-talk
    idle: list[str]                   # who sits out (odd N) — 0 or 1 in the MVP
    events: list[dict]                # seam for interactive matchmakers; random -> []


class Matchmaker(Protocol):
    def setup(self, agent_ids: list[str], cfg) -> None: ...
    async def plan_round(self, agent_ids: list[str], round: int, rng, actor=None) -> RoundPlan: ...
    #   rng: per-round stream, derived by the caller (Random(f"{seed}:matchmaker:{round}")), so
    #        round r's partition depends only on (ids, r) — resumable without replaying the stream.
    #   actor: callback to query an agent (interactive matchmakers); random ignores it


def make_matchmaker(kind: str) -> Matchmaker:
    if kind == "random":
        from src.matchmaking.random_mm import RandomMatchmaker

        return RandomMatchmaker()
    if kind == "sequence":
        from src.matchmaking.sequence import SequenceMatchmaker

        return SequenceMatchmaker()
    raise ValueError(f"unknown matchmaker kind: {kind!r}")

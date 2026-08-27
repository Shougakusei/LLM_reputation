from __future__ import annotations

from src.matchmaking.base import RoundPlan


class SequenceMatchmaker:
    """Manual schedule: the roster order is the play order.

    The first agent (the subject) meets each following agent in turn — round r pairs
    it with the r-th agent of the roster captured at `setup`; everyone else idles.
    The chat opener is drawn from the per-round rng, so orientation is reproducible
    by seed and resumable like the random matchmaker.
    """

    def setup(self, agent_ids, cfg=None) -> None:
        self._order = list(agent_ids)

    async def plan_round(self, agent_ids, round, rng, actor=None) -> RoundPlan:
        if round >= len(self._order):
            raise ValueError(f"sequence matchmaker: round {round} has no partner "
                             f"({len(self._order) - 1} partners after the subject)")
        subject, partner = self._order[0], self._order[round]
        live = set(agent_ids)
        for aid in (subject, partner):
            if aid not in live:
                raise ValueError(f"sequence matchmaker: agent {aid!r} is not alive in round {round}")
        pair = (subject, partner) if rng.random() < 0.5 else (partner, subject)
        idle = [aid for aid in agent_ids if aid not in pair]
        return RoundPlan(pairings=[pair], idle=idle, events=[])

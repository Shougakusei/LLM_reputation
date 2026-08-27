from __future__ import annotations

import random

import pytest

from src.matchmaking import make_matchmaker
from src.matchmaking.sequence import SequenceMatchmaker

IDS = ["S", "N1", "N2", "N3"]


def _mm():
    mm = SequenceMatchmaker()
    mm.setup(list(IDS), cfg=None)
    return mm


def test_factory_returns_sequence():
    assert isinstance(make_matchmaker("sequence"), SequenceMatchmaker)


async def test_round_r_pairs_first_agent_with_rth_agent_rest_idle():
    mm = _mm()
    for r, npc in ((1, "N1"), (2, "N2"), (3, "N3")):
        plan = await mm.plan_round(IDS, r, random.Random(r))
        assert len(plan.pairings) == 1
        assert set(plan.pairings[0]) == {"S", npc}
        assert sorted(plan.idle) == sorted(set(IDS) - {"S", npc})
        assert plan.events == []


async def test_opener_is_drawn_from_the_round_rng():
    mm = _mm()
    openers = {(await mm.plan_round(IDS, 1, random.Random(seed))).pairings[0][0]
               for seed in range(20)}
    assert openers == {"S", "N1"}                     # both orientations occur
    a = await mm.plan_round(IDS, 1, random.Random(7))
    b = await mm.plan_round(IDS, 1, random.Random(7))
    assert a == b                                     # deterministic per seed


async def test_round_beyond_the_sequence_raises():
    mm = _mm()
    with pytest.raises(ValueError, match="round 4"):
        await mm.plan_round(IDS, 4, random.Random(0))


async def test_missing_agent_raises():
    mm = _mm()
    with pytest.raises(ValueError, match="N2"):
        await mm.plan_round(["S", "N1", "N3"], 2, random.Random(0))


async def test_order_fixed_at_setup_not_by_live_list():
    mm = _mm()
    plan = await mm.plan_round(["N3", "N2", "N1", "S"], 1, random.Random(0))
    assert set(plan.pairings[0]) == {"S", "N1"}

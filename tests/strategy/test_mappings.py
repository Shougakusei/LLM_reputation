from __future__ import annotations

import pytest

from src.strategy.mappings import get_mapping


def test_match_returns_predicted_unchanged():
    f = get_mapping("match")
    assert [f(p, 3) for p in range(10)] == list(range(10))


def test_k_above_shifts_by_k_mod_10():
    f = get_mapping("k_above")
    assert [f(p, 1) for p in range(10)] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    assert [f(p, 3) for p in range(10)] == [3, 4, 5, 6, 7, 8, 9, 0, 1, 2]


def test_k_above_wraps_around_the_cycle():
    assert get_mapping("k_above")(9, 1) == 0
    assert get_mapping("k_above")(8, 5) == 3


def test_one_above_is_a_legacy_alias_of_k_above():
    assert get_mapping("one_above") is get_mapping("k_above")


def test_unknown_mapping_raises():
    with pytest.raises(ValueError):
        get_mapping("nope")

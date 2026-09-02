from __future__ import annotations

from src.core.config import GameCfg, Payoffs
from src.games.reputation_pd import ReputationPD


def _game(**kw):
    return ReputationPD(GameCfg(**kw))


def test_equal_is_cc():
    g = _game()
    for n in range(10):
        assert g.resolve(n, n) == ("CC", 3.0, 3.0)


def test_x_off_by_one_is_dc():
    g = _game()
    for y in range(10):
        x = (y + 1) % 10
        assert g.resolve(x, y) == ("DC", 5.0, 0.0)


def test_y_off_by_one_is_cd():
    g = _game()
    for x in range(10):
        y = (x + 1) % 10
        assert g.resolve(x, y) == ("CD", 0.0, 5.0)


def test_wraparound_zero_beats_nine():
    g = _game()
    assert g.resolve(0, 9) == ("DC", 5.0, 0.0)
    assert g.resolve(9, 0) == ("CD", 0.0, 5.0)


def test_exhaustive_classification():
    g = _game()
    R, T, P, S = 3.0, 5.0, 1.0, 0.0
    for x in range(10):
        for y in range(10):
            got = g.resolve(x, y)
            if x == y:
                assert got == ("CC", R, R)
            elif x == (y + 1) % 10:
                assert got == ("DC", T, S)
            elif y == (x + 1) % 10:
                assert got == ("CD", S, T)
            else:
                assert got == ("DD", P, P)


def test_no_mutual_betrayal():
    g = _game()
    for x in range(10):
        for y in range(10):
            _, px, py = g.resolve(x, y)
            assert not (px == 5.0 and py == 5.0)  # off-by-one is never mutual


def test_payoff_invariants():
    p = Payoffs()
    assert p.T > p.R > p.P > p.S
    assert 2 * p.R > p.T + p.S


def test_custom_payoffs():
    g = _game(payoffs=Payoffs(R=2, T=4, P=1, S=0))
    assert g.resolve(3, 3) == ("CC", 2, 2)
    assert g.resolve(4, 3) == ("DC", 4, 0)
    assert g.resolve(2, 5) == ("DD", 1, 1)


def test_resolve_with_per_player_gaps():
    # gap mode: each player wins only by exactly ITS OWN k over the other
    g = _game()
    assert g.resolve(7, 4, ka=3, kb=2) == ("DC", 5.0, 0.0)   # a is k_a=3 above b
    assert g.resolve(4, 6, ka=3, kb=2) == ("CD", 0.0, 5.0)   # b is k_b=2 above a
    assert g.resolve(5, 4, ka=3, kb=2) == ("DD", 1.0, 1.0)   # +1 is no longer a hit
    assert g.resolve(1, 8, ka=3, kb=2) == ("DC", 5.0, 0.0)   # wraps around the cycle
    assert g.resolve(4, 4, ka=3, kb=2) == ("CC", 3.0, 3.0)


def test_resolve_defaults_to_gap_one():
    g = _game()
    assert g.resolve(5, 4) == g.resolve(5, 4, ka=1, kb=1)

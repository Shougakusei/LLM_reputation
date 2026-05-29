from __future__ import annotations

from src.core.memory import Memory, MemoryEntry


def _entry(round=1, partner="A2", my=4, partner_num=4, outcome="CC", payoff=3.0):
    return MemoryEntry(
        round=round,
        partner_id=partner,
        transcript=[
            {"speaker": "A1", "text": "let us both take 4", "ready": False},
            {"speaker": partner, "text": "ok, 4", "ready": True},
        ],
        my_number=my,
        my_rationale="agreed on 4",
        partner_number=partner_num,
        outcome=outcome,
        payoff=payoff,
    )


def test_empty_renders_nothing():
    m = Memory()
    assert m.render(None) == []
    assert m.render(5) == []


def test_single_entry_content():
    m = Memory()
    m.add(_entry(round=3, partner="A5"))
    msgs = m.render(None)
    assert len(msgs) == 1
    msg = msgs[0]
    assert msg.role == "user"
    text = msg.content
    assert "Round 3" in text and "A5" in text
    assert "I chose 4" in text and "agreed on 4" in text
    assert "A5 chose 4" in text
    assert "Outcome: CC" in text and "Payoff: 3" in text
    assert "ready=false" in text and "ready=true" in text


def test_window_limits_to_last_k():
    m = Memory()
    for r in range(1, 6):
        m.add(_entry(round=r))
    text = m.render(2)[0].content
    assert "Round 4" in text and "Round 5" in text
    assert "Round 1" not in text and "Round 3" not in text


def test_window_none_returns_all():
    m = Memory()
    for r in range(1, 4):
        m.add(_entry(round=r))
    text = m.render(None)[0].content
    assert all(f"Round {r}" in text for r in (1, 2, 3))


def test_window_zero_returns_nothing():
    m = Memory()
    m.add(_entry())
    assert m.render(0) == []

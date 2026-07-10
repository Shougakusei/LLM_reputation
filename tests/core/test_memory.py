from __future__ import annotations

from src.core.memory import Memory, MemoryEntry


def _entry(round=1, partner="A2", my=4, partner_num=4, outcome="CC",
           payoff=3.0, partner_payoff=3.0, score=0.0):
    return MemoryEntry(
        round=round,
        my_id="A1",
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
        partner_payoff=partner_payoff,
        score=score,
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
    assert "<game>Round 3 · opponent A5" in text
    assert "<you>let us both take 4</you>" in text   # own line tagged <you>
    assert "<A5>ok, 4</A5>" in text                   # opponent line tagged with the id
    assert "The choice has been accepted. A5 chose 4" in text  # revealing result line
    assert "Payoffs: you = 3, A5 = 3" in text         # both payoffs on one line
    assert "Outcome" not in text and "CC" not in text  # the raw outcome code doesn't leak into the transcript


def test_result_line_shows_running_total():
    m = Memory()
    m.add(_entry(round=3, partner="A5", score=12.0))   # 12 before the round + payoff 3 = 15 after
    text = m.render(None)[0].content
    assert "Your total score after round 3 is 15 points" in text


def test_close_reason_reflects_who_ended_the_chat():
    m = Memory()
    m.add(_entry())                                    # A1 ready=False -> hit the limit
    assert "the messages number limit has been reached" in m.render(None)[0].content
    m2 = Memory()
    e = _entry()
    e.transcript[0]["ready"] = True                    # now both set finish
    m2.add(e)
    assert "both players agreed to stop" in m2.render(None)[0].content


def test_render_shows_both_payoffs_distinctly():
    m = Memory()
    m.add(_entry(partner="A5", payoff=5.0, partner_payoff=0.0))   # you outbid the opponent
    text = m.render(None)[0].content
    assert "Payoffs: you = 5, A5 = 0" in text


def test_reflection_rendered_after_outcome():
    # the takeaway line renders only when the experiment's history_prompt asks for it
    from src.core.config import DEFAULT_HISTORY_PROMPT, GameCfg

    cfg = GameCfg(history_prompt=DEFAULT_HISTORY_PROMPT + "\n<you>(my takeaway: {my_reflection})</you>")
    m = Memory()
    e = _entry(round=2, partner="A3")
    e.my_reflection = "A3 kept the agreement; cooperating with them pays off"
    m.add(e)
    text = m.render(None, cfg)[0].content
    assert "A3 kept the agreement" in text
    # reflection comes after the revealing result line
    assert text.index("Your total score") < text.index("A3 kept the agreement")


def test_entry_without_reflection_renders_no_reflection_line():
    m = Memory()
    m.add(_entry())
    text = m.render(None)[0].content
    assert "reflection" not in text.lower()


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


def test_set_notes_marks_buffer_boundary():
    m = Memory()
    m.add(_entry(round=1))
    m.add(_entry(round=2))
    m.set_notes("rounds 1-2 went fine")
    assert m.notes == "rounds 1-2 went fine"
    assert m.noted_upto == 2          # both entries collapsed -> buffer empty


def test_render_with_notes_replaces_history_but_keeps_recent_buffer():
    m = Memory()
    m.add(_entry(round=1))
    m.add(_entry(round=2))
    m.set_notes("R1-2: opponent A2 keeps agreements")
    m.add(_entry(round=3, partner="A7"))   # played after consolidation -> buffer
    text = m.render(None)[0].content
    # section header in <game>, the notes themselves — in <you>
    assert "<game>Your notes from earlier rounds:</game>\n<you>R1-2: opponent A2 keeps agreements</you>" in text
    assert "<game>Your rounds since those notes:</game>" in text  # label above the fresh buffer, in <game>
    assert "Round 1" not in text and "Round 2" not in text  # collapsed rounds are not rendered raw
    assert "Round 3" in text and "A7" in text               # fresh buffer — rendered raw


def test_render_with_notes_only_when_buffer_empty():
    m = Memory()
    m.add(_entry(round=1))
    m.set_notes("note text")
    msgs = m.render(None)                  # buffer empty -> notes only (not [])
    assert len(msgs) == 1
    assert msgs[0].content == "<game>Your notes from earlier rounds:</game>\n<you>note text</you>"  # <game> label + <you> notes
    assert "Your rounds since those notes:" not in msgs[0].content  # no buffer -> no label either
    assert "Round 1" not in msgs[0].content


def test_history_prompt_controls_what_is_shown():
    # what the agent sees of its own past (rationale block? takeaway?) is decided entirely by
    # the experiment's history_prompt — the default shows only the number line and the result.
    from src.core.config import GameCfg
    from src.core.memory import Memory, MemoryEntry

    e = MemoryEntry(round=1, my_id="A1", partner_id="A2", transcript=[], my_number=5,
                    my_rationale="risky", partner_number=5, outcome="CC", payoff=3.0,
                    partner_payoff=3.0, my_predicted=4)
    e.my_reflection = "trust holds"

    m = Memory()
    m.add(e)
    default = m.render(None, GameCfg())[0].content
    assert "<you>5</you>" in default                     # the number line
    assert "risky" not in default and "trust holds" not in default and "predicted" not in default

    rich = GameCfg(history_prompt=(
        "<game>Round {round} with {partner}</game>\n"
        "<you>rationale: {my_rationale}\nnumber: {my_number}</you>\n"
        "<game>{partner} chose {partner_number}.</game>\n"
        "<you>(my takeaway: {my_reflection})</you>"
    ))
    m2 = Memory()
    m2.add(e)
    text = m2.render(None, rich)[0].content
    assert "rationale: risky" in text and "(my takeaway: trust holds)" in text


# ── Collapsed rendering (opt-in via a single history_prompt / notes_view) ──

def _collapsed_cfg(history_prompt=None, **over):
    from src.core.config import (
        DEFAULT_HISTORY_PROMPT,
        DEFAULT_NOTES_BUFFER,
        DEFAULT_NOTES_VIEW,
        GameCfg,
    )
    base = dict(
        history_prompt=history_prompt or DEFAULT_HISTORY_PROMPT,
        notes_view=DEFAULT_NOTES_VIEW,
        notes_buffer=DEFAULT_NOTES_BUFFER,
    )
    base.update(over)
    return GameCfg(**base)


def test_collapsed_history_prompt_renders_turns_and_result():
    cfg = _collapsed_cfg()
    m = Memory()
    m.add(_entry(round=3, partner="A5", my=4, score=12.0))
    text = m.render(None, cfg)[0].content
    assert "opponent A5" in text                                   # DEFAULT_HISTORY_PROMPT header
    assert "<you>let us both take 4</you>" in text                 # {feed} rendered
    assert "<A5>ok, 4</A5>" in text
    assert "<you>4</you>" in text                                  # the number line
    assert "The choice has been accepted. A5 chose 4" in text
    assert "Your total score after round 3 is 15 points" in text   # 12 + 3


def test_collapsed_history_prompt_drops_turns_line_when_no_transcript():
    cfg = _collapsed_cfg()
    e = MemoryEntry(round=1, my_id="A1", partner_id="A2", transcript=[], my_number=5,
                    my_rationale="", partner_number=5, outcome="CC", payoff=3.0, partner_payoff=3.0)
    m = Memory()
    m.add(e)
    text = m.render(None, cfg)[0].content
    assert "{feed}" not in text
    # the header line is immediately followed by the close line — no empty line where turns were
    assert "The chat has been opened.</game>\n<game>The chat has been closed" in text


def test_collapsed_number_line_renders_through_msg_self():
    # {my_number_line} uses msg_self, so a custom self-tag applies to the number line too.
    cfg = _collapsed_cfg(msg_self="<self>{text}</self>")
    m = Memory()
    m.add(_entry(my=4))
    text = m.render(None, cfg)[0].content
    assert "<self>4</self>" in text                       # the number line via the custom msg_self
    assert "<self>let us both take 4</self>" in text      # cheap-talk lines too
    assert "<you>" not in text                             # nothing falls back to the hardcoded tag


def test_collapsed_notes_line_renders_through_msg_self():
    # {notes_line} uses msg_self too — the saved notes carry the same self-tag as the agent's lines.
    cfg = _collapsed_cfg(msg_self="<self>{text}</self>")
    m = Memory()
    m.add(_entry(round=1))
    m.set_notes("keep trusting A2")
    text = m.render(None, cfg)[0].content
    assert "<self>keep trusting A2</self>" in text
    assert "<you>" not in text


def test_collapsed_custom_history_prompt_is_used_verbatim():
    # An experiment supplies its own history_prompt (here: rationale block); the code fills
    # placeholders and does not branch on flags.
    tmpl = (
        "<game>Round {round} with {partner}:\nThe chat has been opened.</game>\n{feed}\n"
        "<game>The chat has been closed as {reason}. Give your rationale first, then choose the number.</game>\n"
        "<you>rationale: {my_rationale}\nnumber: {my_number}</you>\n"
        "<game>The choice has been accepted. {partner} chose {partner_number}. "
        "Payoffs: you = {payoff}, {partner} = {partner_payoff}.\n"
        "Your total score after round {round} is {total} points.</game>"
    )
    cfg = _collapsed_cfg(history_prompt=tmpl)
    m = Memory()
    m.add(_entry(partner="A5"))
    text = m.render(None, cfg)[0].content
    assert "<you>rationale: agreed on 4\nnumber: 4</you>" in text
    assert "Give your rationale first" in text


def test_collapsed_notes_view_and_full_buffer():
    cfg = _collapsed_cfg()
    m = Memory()
    m.add(_entry(round=1))
    m.add(_entry(round=2))
    m.set_notes("R1-2 fine")
    m.add(_entry(round=3, partner="A7"))                # played after consolidation -> buffer
    text = m.render(None, cfg)[0].content
    assert "<game>Your notes from earlier rounds:</game>\n<you>R1-2 fine</you>" in text
    assert "<game>Your rounds since those notes:</game>" in text
    assert "Round 3" in text and "A7" in text
    assert "Round 1" not in text and "Round 2" not in text


def test_collapsed_notes_view_only_when_buffer_empty():
    cfg = _collapsed_cfg()
    m = Memory()
    m.add(_entry(round=1))
    m.set_notes("note text")
    content = m.render(None, cfg)[0].content
    # nothing buffered -> the buffer section is not appended
    assert content == "<game>Your notes from earlier rounds:</game>\n<you>note text</you>"


def test_legacy_path_still_used_when_history_prompt_unset():
    # Default GameCfg leaves history_prompt None -> the piecewise legacy path is rendered.
    from src.core.config import GameCfg

    m = Memory()
    m.add(_entry(round=3, partner="A5"))
    text = m.render(None, GameCfg())[0].content
    assert "The choice has been accepted. A5 chose 4" in text

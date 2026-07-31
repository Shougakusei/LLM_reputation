from __future__ import annotations

import logging

import pytest
from conftest import ScriptedProvider

from src.core.agent import ActParseError, Agent, AgentSetup, Phase, PhaseKind
from src.core.config import GameCfg, ProviderCfg
from src.core.memory import MemoryEntry
from src.providers.base import HttpAttempt, ProviderUnavailable


class RaisingProvider:
    """Provider double: raises ProviderUnavailable with the given HttpAttempts."""

    def __init__(self, attempts):
        self._attempts = tuple(attempts)
        self.calls = 0

    async def complete(self, *, system, messages, temperature, max_tokens):
        self.calls += 1
        e = ProviderUnavailable("boom")
        e.request = {"model": "m", "messages": []}
        e.attempts = self._attempts
        raise e

    async def aclose(self):
        pass


def _agent(provider, system="You are A.", **kw):
    cfg = ProviderCfg(base_url="http://x/v1", model="m", temperature=0.0, max_tokens=64)
    return Agent("A1", AgentSetup(system, cfg), provider, **kw)


def _decide(context="Pick a number."):
    return Phase(PhaseKind.DECIDE, context)


def _talk(context="Negotiate."):
    return Phase(PhaseKind.TALK, context)


async def test_decide_clean_json():
    p = ScriptedProvider(['{"number": 4, "rationale": "ok"}'])
    r = await _agent(p).act(_decide())
    assert r.data == {"number": 4, "rationale": "ok"}
    assert r.public_text is None
    assert len(p.calls) == 1


async def test_decide_bare_correction_omits_rationale():
    # Bug: in bare mode (rationale=false) the retry correction must not require rationale —
    # previously the engine forced {"rationale","number"}, contradicting the prompt, which asked for {"number"}.
    p = ScriptedProvider(["nope", '{"number": 3}'])
    phase = Phase(PhaseKind.DECIDE, "Pick a number.", game_cfg=GameCfg(rationale=False))
    r = await _agent(p).act(phase)
    assert r.data["number"] == 3
    correction = p.calls[1][1][-1].content        # last user message of the second attempt
    assert '"number"' in correction
    assert "rationale" not in correction          # bare — without rationale


async def test_decide_correction_text_comes_from_config():
    # The correction text comes from GameCfg, not from a hardcoded string.
    p = ScriptedProvider(["nope", '{"number": 1}'])
    phase = Phase(PhaseKind.DECIDE, "Pick a number.",
                  game_cfg=GameCfg(decide_correction="JUST_NUMBER_PLEASE"))
    await _agent(p).act(phase)
    assert "JUST_NUMBER_PLEASE" in p.calls[1][1][-1].content


async def test_decide_json_in_fence():
    p = ScriptedProvider(['```json\n{"number": 7, "rationale": "x"}\n```'])
    r = await _agent(p).act(_decide())
    assert r.data["number"] == 7


async def test_decide_json_among_prose():
    p = ScriptedProvider(['Sure! {"number": 2, "rationale": "y"} done'])
    r = await _agent(p).act(_decide())
    assert r.data["number"] == 2


async def test_decide_string_number_coerced():
    p = ScriptedProvider(['{"number": "3", "rationale": "z"}'])
    r = await _agent(p).act(_decide())
    assert r.data["number"] == 3


async def test_decide_out_of_range_then_valid():
    p = ScriptedProvider(
        ['{"number": 15, "rationale": "bad"}', '{"number": 3, "rationale": "good"}']
    )
    r = await _agent(p).act(_decide())
    assert r.data["number"] == 3
    assert len(p.calls) == 2


async def test_decide_invalid_twice_then_valid():
    p = ScriptedProvider(["nope", "still nope", '{"number": 1, "rationale": "r"}'])
    r = await _agent(p).act(_decide())
    assert r.data["number"] == 1
    assert len(p.calls) == 3


async def test_decide_persistent_failure_raises():
    p = ScriptedProvider(["no", "no", "no"])
    a = _agent(p)
    with pytest.raises(ActParseError) as ei:        # no substitution: abort instead of a fallback number
        await a.act(_decide())
    assert a.parse_failures == 1
    assert len(p.calls) == 3                         # retried with _CORRECTION, then gave up
    assert [c.status for c in ei.value.calls] == ["parse_error"] * 3
    assert ei.value.agent_id == "A1" and ei.value.phase == "decide"


async def test_decide_rejects_bool_number():
    # JSON `true` must not be accepted as the integer 1.
    p = ScriptedProvider(['{"number": true, "rationale": "x"}', '{"number": 5, "rationale": "ok"}'])
    r = await _agent(p).act(_decide())
    assert r.data["number"] == 5
    assert len(p.calls) == 2


def _reflect(context="Reflect on the outcome."):
    return Phase(PhaseKind.REFLECT, context)


async def test_reflect_clean_json():
    p = ScriptedProvider(['{"reflection": "partner kept the deal"}'])
    r = await _agent(p).act(_reflect())
    assert r.data == {"reflection": "partner kept the deal"}
    assert r.public_text is None
    assert len(p.calls) == 1


async def test_reflect_invalid_then_valid_retries_with_correction():
    p = ScriptedProvider(["nope", '{"reflection": "ok"}'])
    r = await _agent(p).act(_reflect())
    assert r.data["reflection"] == "ok"
    assert len(p.calls) == 2
    _, messages = p.calls[1]
    assert len(messages) == 1                     # correction is merged into the same user message
    assert "reflection" in messages[-1].content  # correction names the expected key


async def test_reflect_persistent_failure_raises():
    p = ScriptedProvider(["no", "no", "no"])
    a = _agent(p)
    with pytest.raises(ActParseError):
        await a.act(_reflect())
    assert a.parse_failures == 1
    assert len(p.calls) == 3


def _note(context="Summarize your memory."):
    return Phase(PhaseKind.NOTE, context)


async def test_note_clean_json():
    p = ScriptedProvider(['{"notes": "A2 keeps deals; undercut A5"}'])
    r = await _agent(p).act(_note())
    assert r.data == {"notes": "A2 keeps deals; undercut A5"}
    assert r.public_text is None              # note — not a public message
    assert len(p.calls) == 1


async def test_note_accepts_singular_note_key():
    # Configs may ask for {"note": ...}; the reply normalizes to the internal "notes" key.
    p = ScriptedProvider(['{"note": "watch A5"}'])
    r = await _agent(p).act(_note())
    assert r.data == {"notes": "watch A5"}


async def test_note_invalid_then_valid_retries_with_correction():
    p = ScriptedProvider(["nope", '{"notes": "ok"}'])
    r = await _agent(p).act(_note())
    assert r.data["notes"] == "ok"
    assert len(p.calls) == 2
    assert "notes" in p.calls[1][1][-1].content   # correction names the expected key


async def test_note_renders_full_memory_ignoring_window():
    # window=1, but NOTE must see the whole memory (notes are built from the full history)
    p = ScriptedProvider(['{"notes": "done"}'])
    a = _agent(p, context_window=1)
    for r in (1, 2, 3):
        a.memory.add(MemoryEntry(round=r, my_id="A1", partner_id="A2", transcript=[],
                                 my_number=4, my_rationale="", partner_number=4,
                                 outcome="CC", payoff=3.0, partner_payoff=3.0))
    await a.act(_note())
    _, messages = p.calls[-1]
    content = messages[-1].content
    assert "Round 1" in content and "Round 2" in content and "Round 3" in content


async def test_game_blocks_merged_at_history_live_seam():
    # The junction "previous round's result line </game>" + "<game> opening of current" is merged
    # into a single <game> block, so the transcript isn't jittered by an extra tag close/open.
    p = ScriptedProvider(['{"number": 0, "rationale": ""}'])
    a = _agent(p)
    a.memory.add(MemoryEntry(round=1, my_id="A1", partner_id="A2", transcript=[],
                             my_number=4, my_rationale="", partner_number=4,
                             outcome="CC", payoff=3.0, partner_payoff=3.0))
    await a.act(Phase(PhaseKind.DECIDE,
                      "<game>Round 2 · opponent A2\nThe chat has been open.</game>"))
    _, messages = p.calls[-1]
    content = messages[-1].content
    assert "</game>" in content and "<game>" in content   # tags remain
    assert "</game>\n\n<game>" not in content             # but the history↔current junction is cleaned up
    assert "Your total score after round 1" in content    # content of both blocks survived
    assert "Round 2 · opponent A2" in content


async def test_game_blocks_merged_at_phase_junction_within_round():
    # A round without negotiation: the opening block (TALK phase) runs right into the closing
    # block (DECIDE phase) — the </game><game> junction between phases also collapses into one block.
    import re
    p = ScriptedProvider(['{"number": 0, "rationale": ""}'])
    a = _agent(p)
    a.memory.add(MemoryEntry(round=1, my_id="A1", partner_id="A2", transcript=[],   # no talk
                             my_number=5, my_rationale="", partner_number=5,
                             outcome="CC", payoff=3.0, partner_payoff=3.0))
    await a.act(Phase(PhaseKind.DECIDE, "<game>Round 2 · opponent A2\nThe chat has been open.</game>"))
    content = p.calls[-1][1][-1].content
    assert not re.search(r"</game>\s*<game>", content)   # no phase junction remains
    assert "The chat has been open." in content and "The chat has been closed" in content  # both phases intact


async def test_notes_buffer_labels_and_buffer_live_seam_merges():
    # Under the header labels come the notes (<game>…</game>) and the buffer; the buffer in turn
    # joins the live current round — this </game><game> junction gets merged.
    import re
    p = ScriptedProvider(['{"number": 0, "rationale": ""}'])
    a = _agent(p)
    a.memory.set_notes("R1: faced A2, matched on 5")
    a.memory.add(MemoryEntry(round=2, my_id="A1", partner_id="A2", transcript=[],
                             my_number=5, my_rationale="", partner_number=5,
                             outcome="CC", payoff=3.0, partner_payoff=3.0))
    await a.act(Phase(PhaseKind.DECIDE,
                      "<game>Round 3 · opponent A2\nThe chat has been open.</game>"))
    content = p.calls[-1][1][-1].content
    assert "Your notes from earlier rounds:" in content        # labels are present
    assert "Your rounds since those notes:" in content
    assert "R1: faced A2" in content                           # notes are in place
    assert not re.search(r"</game>\s*<game>", content)         # buffer↔live round junction merged
    assert "Round 2" in content and "Round 3 · opponent A2" in content


async def test_system_is_the_agent_system_prompt_verbatim():
    # No more concatenation: system = AgentSetup.system_prompt verbatim (only {id} is substituted).
    p = ScriptedProvider(['{"number": 0, "rationale": ""}'])
    await _agent(p, system="PERSONA\n\nGAME RULES").act(Phase(PhaseKind.DECIDE, "SITUATION"))
    system, messages = p.calls[0]
    assert system == "PERSONA\n\nGAME RULES"
    assert messages[-1].role == "user"
    assert messages[-1].content == "SITUATION"
    assert len(messages) == 1  # empty memory -> only the situation message


async def test_system_prompt_substitutes_id():
    p = ScriptedProvider(['{"number": 0, "rationale": ""}'])
    await _agent(p, system="Ты ИИ-игрок {id}. Play well.").act(Phase(PhaseKind.DECIDE, "S"))
    system, _ = p.calls[0]
    assert system == "Ты ИИ-игрок A1. Play well."     # {id} substituted by the agent, the rest verbatim


async def test_system_prompt_substitutes_payoffs_from_game_cfg():
    # Substitution of payoffs {R}/{T}/{P}/{S}/{max_talk_turns} moved from rules_text into system_prompt.
    from src.core.config import GameCfg
    p = ScriptedProvider(['{"number": 0, "rationale": ""}'])
    agent = _agent(p, system="R={R} T={T} P={P} S={S} budget={max_talk_turns}")
    await agent.act(Phase(PhaseKind.DECIDE, "S", game_cfg=GameCfg(max_talk_turns=4)))
    system, _ = p.calls[0]
    assert system == "R=3 T=5 P=1 S=0 budget=4"


async def test_usage_summed_over_retries():
    p = ScriptedProvider(["bad", '{"number": 5, "rationale": "r"}'])
    r = await _agent(p).act(_decide())
    assert r.usage == (4, 6)  # 2 calls * (2 prompt, 3 completion)


async def test_talk_clean_json():
    # Agent-facing key is "finish"; internally it's mapped to data["ready"].
    p = ScriptedProvider(['{"message": "let us both take 4", "finish": true}'])
    r = await _agent(p).act(_talk())
    assert r.data == {"message": "let us both take 4", "ready": True}
    assert r.public_text == "let us both take 4"
    assert len(p.calls) == 1


async def test_talk_finish_coercion():
    for raw, expected in [("yes", True), ("1", True), (1, True),
                          ("false", False), ("no", False), (0, False)]:
        p = ScriptedProvider(['{"message": "m", "finish": %s}'
                              % (raw if isinstance(raw, int) else '"%s"' % raw)])
        r = await _agent(p).act(_talk())
        assert r.data["ready"] is expected, (raw, expected)


async def test_talk_finish_missing_defaults_false():
    p = ScriptedProvider(['{"message": "still thinking"}'])
    r = await _agent(p).act(_talk())
    assert r.data == {"message": "still thinking", "ready": False}


async def test_talk_message_missing_then_valid():
    p = ScriptedProvider(['{"finish": true}', '{"message": "ok", "finish": true}'])
    r = await _agent(p).act(_talk())
    assert r.data["message"] == "ok"
    assert len(p.calls) == 2


async def test_talk_persistent_failure_raises():
    p = ScriptedProvider(["no", "no", "no"])
    a = _agent(p)
    with pytest.raises(ActParseError):               # talk too: abort, no empty-message substitution
        await a.act(_talk())
    assert a.parse_failures == 1
    assert len(p.calls) == 3


async def test_predict_phase_parses_number_and_rationale():
    p = ScriptedProvider(['{"number": 7, "rationale": "mid is safe"}'])
    r = await _agent(p).act(Phase(PhaseKind.PREDICT, "predict your partner"))
    assert r.data["number"] == 7
    assert r.data["rationale"] == "mid is safe"
    assert r.public_text is None  # PREDICT produces no public message


async def test_memory_diary_precedes_situation_in_one_message():
    p = ScriptedProvider(['{"number": 0, "rationale": ""}'])
    a = _agent(p)
    a.memory.add(
        MemoryEntry(
            round=2,
            my_id="A1",
            partner_id="A7",
            transcript=[{"speaker": "A7", "text": "take 5", "ready": True}],
            my_number=6,
            my_rationale="betray",
            partner_number=5,
            outcome="DC",
            payoff=5.0,
            partner_payoff=0.0,
        )
    )
    await a.act(Phase(PhaseKind.DECIDE, "SITUATION"))
    _system, messages = p.calls[0]
    assert len(messages) == 1 and messages[0].role == "user"   # diary and situation are merged
    content = messages[0].content
    assert "Round 2" in content and "A7" in content            # diary is present
    assert content.endswith("SITUATION")                       # situation is at the end
    assert content.index("Round 2") < content.index("SITUATION")   # diary precedes situation


async def test_history_placeholder_substitutes_memory_in_place():
    # A step prompt may place the memory itself via {history} — then nothing is prepended.
    p = ScriptedProvider(['{"number": 0, "rationale": ""}'])
    a = _agent(p)
    a.memory.add(
        MemoryEntry(
            round=2, my_id="A1", partner_id="A7",
            transcript=[{"speaker": "A7", "text": "take 5", "ready": True}],
            my_number=6, my_rationale="", partner_number=5, outcome="DC",
            payoff=5.0, partner_payoff=0.0,
        )
    )
    await a.act(Phase(PhaseKind.DECIDE, "HEAD\n\n{history}\n\nSITUATION"))
    content = p.calls[0][1][0].content
    assert content.startswith("HEAD")                        # memory was NOT prepended
    assert content.endswith("SITUATION")
    assert "Round 2" in content and "{history}" not in content
    assert content.index("HEAD") < content.index("Round 2") < content.index("SITUATION")


async def test_history_placeholder_dropped_with_blank_line_when_memory_empty():
    p = ScriptedProvider(['{"number": 0, "rationale": ""}'])
    await _agent(p).act(Phase(PhaseKind.DECIDE, "{history}\n\nSITUATION"))
    content = p.calls[0][1][0].content
    assert content == "SITUATION"                            # no leading blank lines left


async def test_history_placeholder_correction_still_appended_on_retry():
    p = ScriptedProvider(["nope", '{"number": 1, "rationale": ""}'])
    await _agent(p).act(Phase(PhaseKind.DECIDE, "{history}\n\nSITUATION"))
    second = p.calls[1][1][-1].content
    assert second.startswith("SITUATION")                    # placeholder handling unchanged
    assert "ONLY valid JSON" in second                       # correction at the end


def _dc_entry(round=2, partner="A7"):
    return MemoryEntry(
        round=round, my_id="A1", partner_id=partner,
        transcript=[{"speaker": partner, "text": "take 5", "ready": True}],
        my_number=6, my_rationale="", partner_number=5, outcome="DC",
        payoff=5.0, partner_payoff=0.0,
    )


async def test_variant_no_history_used_when_no_past_rounds():
    from src.core.config import PromptVariants
    p = ScriptedProvider(['{"number": 1, "rationale": ""}'])
    ctx = PromptVariants(no_history="FRESH", with_history="SEASONED {recent_rounds}")
    await _agent(p).act(Phase(PhaseKind.DECIDE, ctx))
    assert p.calls[0][1][0].content == "FRESH"


async def test_variant_with_history_used_when_past_rounds_exist():
    from src.core.config import PromptVariants
    p = ScriptedProvider(['{"number": 1, "rationale": ""}'])
    a = _agent(p)
    a.memory.add(_dc_entry())
    ctx = PromptVariants(no_history="FRESH", with_history="SEASONED\n\n{recent_rounds}\n\nEND")
    await a.act(Phase(PhaseKind.DECIDE, ctx))
    content = p.calls[0][1][0].content
    assert content.startswith("SEASONED")                    # variant chosen, nothing prepended
    assert "take 5" in content and content.endswith("END")


async def test_note_phase_excludes_current_round_from_variant_choice():
    # At NOTE time the just-played round is already in memory; it is not "past", so a lone
    # entry still selects no_history (the round itself arrives via {recent_rounds}).
    from src.core.config import PromptVariants
    p = ScriptedProvider(['{"notes": "n"}', '{"notes": "n"}'])
    a = _agent(p)
    a.memory.add(_dc_entry(round=1))
    ctx = PromptVariants(no_history="FIRST {recent_rounds}", with_history="LATER")
    await a.act(Phase(PhaseKind.NOTE, ctx))
    assert p.calls[0][1][0].content.startswith("FIRST")
    assert "take 5" in p.calls[0][1][0].content
    a.memory.add(_dc_entry(round=2))
    await a.act(Phase(PhaseKind.NOTE, ctx))
    assert p.calls[1][1][0].content == "LATER"               # a genuinely past round exists


async def test_fragment_placeholders_substituted_in_step_prompt():
    p = ScriptedProvider(['{"number": 1, "rationale": ""}'])
    a = _agent(p)
    cfg = GameCfg(history_line="R{round}:{partner}")
    a.memory.add(_dc_entry(round=1, partner="A7"))
    a.memory.set_notes("A7 betrays")
    a.memory.add(_dc_entry(round=2, partner="A9"))
    ctx = "L:{history_lines}|N:{notes_line}|R:{recent_rounds}|END"
    await a.act(Phase(PhaseKind.DECIDE, ctx, game_cfg=cfg))
    content = p.calls[0][1][0].content
    assert "L:R1:A7|" in content                             # folded round -> one line
    assert "N:<you>A7 betrays</you>|" in content             # the note through msg_self
    assert "A9 chose 5" in content and content.endswith("END")   # buffered round in full
    assert not content.startswith("<game>")                  # fragments used -> no prepend


async def test_empty_fragment_collapses_with_its_blank_line():
    p = ScriptedProvider(['{"number": 1, "rationale": ""}'])
    await _agent(p).act(Phase(PhaseKind.DECIDE, "A\n\n{recent_rounds}\n\nB"))
    assert p.calls[0][1][0].content == "A\n\nB"


async def test_paragraph_with_only_empty_fragments_vanishes_with_its_headers():
    # A paragraph (blank-line-delimited group) whose fragments ALL render empty is dropped
    # whole — headers inside it included. That is what derives no_history from with_history.
    p = ScriptedProvider(['{"number": 1, "rationale": ""}'])
    ctx = ("HEAD\n\n"
           "<game>The history of your interactions:\n{history_lines}</game>\n"
           "<game>Your memory note:</game>\n{notes_line}\n\n"
           "{recent_rounds}\n\n"
           "LIVE")
    await _agent(p).act(Phase(PhaseKind.DECIDE, ctx))
    assert p.calls[0][1][0].content == "HEAD\n\nLIVE"


async def test_paragraph_with_a_nonempty_fragment_stays_and_drops_empty_lines():
    p = ScriptedProvider(['{"number": 1, "rationale": ""}'])
    a = _agent(p)
    a.memory.add(_dc_entry())
    ctx = "T:\n{history_lines}\n{recent_rounds}\n\nEND"   # lines empty, recent non-empty
    await a.act(Phase(PhaseKind.DECIDE, ctx))
    content = p.calls[0][1][0].content
    assert content.startswith("T:\n")                    # header kept: paragraph not empty
    assert "take 5" in content and "{history_lines}" not in content


async def test_variant_without_no_history_falls_back_to_with_history():
    # no_history omitted -> with_history serves both states; empty-memory sections vanish
    # by the paragraph rule.
    from src.core.config import PromptVariants
    p = ScriptedProvider(['{"number": 1, "rationale": ""}', '{"number": 2, "rationale": ""}'])
    ctx = PromptVariants(with_history="T\n\nH:\n{recent_rounds}\n\nEND")
    a = _agent(p)
    await a.act(Phase(PhaseKind.DECIDE, ctx))
    assert p.calls[0][1][0].content == "T\n\nEND"        # round 1: section wiped
    a.memory.add(_dc_entry())
    await a.act(Phase(PhaseKind.DECIDE, ctx))
    assert "take 5" in p.calls[1][1][0].content          # later: section shown


def _trace_records(caplog):
    return [r for r in caplog.records if r.name == "src.core.agent"]


async def test_decide_logs_full_llm_input_at_debug(caplog):
    caplog.set_level(logging.DEBUG, logger="src.core.agent")
    p = ScriptedProvider(['{"number": 4, "rationale": "ok"}'])
    await _agent(p, system="PERSONA\n\nGAME RULES").act(
        Phase(PhaseKind.DECIDE, "SITUATION")
    )
    records = _trace_records(caplog)
    assert len(records) == 1
    msg = records[0].getMessage()
    assert "PERSONA" in msg and "GAME RULES" in msg  # system prompt
    assert "SITUATION" in msg                        # phase context message


async def test_predict_logs_llm_input_at_debug(caplog):
    caplog.set_level(logging.DEBUG, logger="src.core.agent")
    p = ScriptedProvider(['{"number": 7, "rationale": "guess"}'])
    await _agent(p).act(Phase(PhaseKind.PREDICT, "PREDICT-CTX"))
    records = _trace_records(caplog)
    assert len(records) == 1
    assert "PREDICT-CTX" in records[0].getMessage()


async def test_talk_logs_no_llm_input(caplog):
    caplog.set_level(logging.DEBUG, logger="src.core.agent")
    p = ScriptedProvider(['{"message": "hi", "finish": true}'])
    await _agent(p).act(_talk())
    assert _trace_records(caplog) == []


async def test_decide_retry_logs_attempt_with_correction(caplog):
    caplog.set_level(logging.DEBUG, logger="src.core.agent")
    p = ScriptedProvider(["bad", '{"number": 1, "rationale": "r"}'])
    await _agent(p).act(_decide())
    records = _trace_records(caplog)
    assert len(records) == 2  # one record per provider attempt
    second = records[1].getMessage()
    assert "ONLY valid JSON" in second  # the correction message is part of the input


# ---- L2: capturing raw calls (ActResult.calls) ----

async def test_clean_decide_records_one_ok_call():
    p = ScriptedProvider(['{"number": 4, "rationale": "ok"}'])
    r = await _agent(p).act(_decide())
    assert len(r.calls) == 1
    c = r.calls[0]
    assert (c.agent_id, c.phase, c.attempt, c.http_attempt) == ("A1", "decide", 1, 1)
    assert c.status == "ok"
    assert c.status_code == 200
    assert c.response == '{"number": 4, "rationale": "ok"}'
    assert c.request["messages"][0]["role"] == "system"   # verbatim payload in hand
    assert c.prompt_tokens == 2 and c.completion_tokens == 3


async def test_parse_retry_records_two_calls_with_statuses():
    p = ScriptedProvider(["garbage", '{"number": 1, "rationale": "r"}'])
    r = await _agent(p).act(_decide())
    assert [c.status for c in r.calls] == ["parse_error", "ok"]
    assert [c.attempt for c in r.calls] == [1, 2]          # separate complete() per attempt
    assert r.calls[0].response == "garbage"               # raw invalid response saved


async def test_all_parse_fail_raises_with_logged_calls():
    p = ScriptedProvider(["x", "y", "z"])
    with pytest.raises(ActParseError) as ei:
        await _agent(p).act(_decide())
    assert [c.status for c in ei.value.calls] == ["parse_error"] * 3   # all three attempts logged


async def test_provider_error_reraised_with_calls_and_context():
    attempts = [
        HttpAttempt("server_error", 503, {"m": 1}, None, "busy", "HTTP 503"),
        HttpAttempt("network", None, {"m": 1}, None, None, "boom"),
    ]
    p = RaisingProvider(attempts)
    with pytest.raises(ProviderUnavailable) as ei:
        await _agent(p).act(_decide())
    e = ei.value
    assert (e.agent_id, e.phase, e.attempt) == ("A1", "decide", 1)
    # each HTTP attempt is unwrapped into an LLMCall with the agent's context
    assert [c.status for c in e.calls] == ["server_error", "network"]
    assert [c.http_attempt for c in e.calls] == [1, 2]
    assert e.calls[0].response_raw == "busy"              # 5xx body saved

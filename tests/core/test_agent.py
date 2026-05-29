from __future__ import annotations

from conftest import ScriptedProvider

from src.core.agent import Agent, AgentSetup, Phase, PhaseKind
from src.core.config import ProviderCfg
from src.core.memory import MemoryEntry


def _agent(provider, persona="You are A.", **kw):
    cfg = ProviderCfg(base_url="http://x/v1", model="m", temperature=0.0, max_tokens=64)
    return Agent("A1", AgentSetup(persona, cfg), provider, **kw)


def _decide(context="Pick a number.", rules="RULES"):
    return Phase(PhaseKind.DECIDE, context, rules=rules)


def _talk(context="Negotiate.", rules="RULES"):
    return Phase(PhaseKind.TALK, context, rules=rules)


async def test_decide_clean_json():
    p = ScriptedProvider(['{"number": 4, "rationale": "ok"}'])
    r = await _agent(p).act(_decide())
    assert r.data == {"number": 4, "rationale": "ok"}
    assert r.public_text is None
    assert len(p.calls) == 1


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


async def test_decide_persistent_failure_fallback():
    p = ScriptedProvider(["no", "no", "no"])
    a = _agent(p)
    r = await a.act(_decide())
    assert 0 <= r.data["number"] <= 9
    assert r.data["rationale"] == "(unparsed)"
    assert a.parse_failures == 1
    assert len(p.calls) == 3


async def test_decide_rejects_bool_number():
    # JSON `true` must not be accepted as the integer 1.
    p = ScriptedProvider(['{"number": true, "rationale": "x"}', '{"number": 5, "rationale": "ok"}'])
    r = await _agent(p).act(_decide())
    assert r.data["number"] == 5
    assert len(p.calls) == 2


async def test_system_and_messages_assembly():
    p = ScriptedProvider(['{"number": 0, "rationale": ""}'])
    await _agent(p, persona="PERSONA").act(Phase(PhaseKind.DECIDE, "SITUATION", rules="GAME RULES"))
    system, messages = p.calls[0]
    assert "PERSONA" in system and "GAME RULES" in system
    assert messages[-1].role == "user"
    assert messages[-1].content == "SITUATION"
    assert len(messages) == 1  # empty memory -> only the situation message


async def test_usage_summed_over_retries():
    p = ScriptedProvider(["bad", '{"number": 5, "rationale": "r"}'])
    r = await _agent(p).act(_decide())
    assert r.usage == (4, 6)  # 2 calls * (2 prompt, 3 completion)


async def test_talk_clean_json():
    p = ScriptedProvider(['{"message": "let us both take 4", "ready": true}'])
    r = await _agent(p).act(_talk())
    assert r.data == {"message": "let us both take 4", "ready": True}
    assert r.public_text == "let us both take 4"
    assert len(p.calls) == 1


async def test_talk_ready_coercion():
    for raw, expected in [("yes", True), ("1", True), (1, True),
                          ("false", False), ("no", False), (0, False)]:
        p = ScriptedProvider(['{"message": "m", "ready": %s}'
                              % (raw if isinstance(raw, int) else '"%s"' % raw)])
        r = await _agent(p).act(_talk())
        assert r.data["ready"] is expected, (raw, expected)


async def test_talk_ready_missing_defaults_false():
    p = ScriptedProvider(['{"message": "still thinking"}'])
    r = await _agent(p).act(_talk())
    assert r.data == {"message": "still thinking", "ready": False}


async def test_talk_message_missing_then_valid():
    p = ScriptedProvider(['{"ready": true}', '{"message": "ok", "ready": true}'])
    r = await _agent(p).act(_talk())
    assert r.data["message"] == "ok"
    assert len(p.calls) == 2


async def test_talk_persistent_failure_fallback():
    p = ScriptedProvider(["no", "no", "no"])
    a = _agent(p)
    r = await a.act(_talk())
    assert r.data == {"message": "", "ready": True}  # fallback stops the negotiation
    assert r.public_text == ""
    assert a.parse_failures == 1


async def test_memory_diary_precedes_situation_in_prompt():
    p = ScriptedProvider(['{"number": 0, "rationale": ""}'])
    a = _agent(p)
    a.memory.add(
        MemoryEntry(
            round=2,
            partner_id="A7",
            transcript=[{"speaker": "A7", "text": "take 5", "ready": True}],
            my_number=6,
            my_rationale="betray",
            partner_number=5,
            outcome="DC",
            payoff=5.0,
        )
    )
    await a.act(Phase(PhaseKind.DECIDE, "SITUATION", rules="R"))
    _system, messages = p.calls[0]
    assert len(messages) == 2
    assert messages[0].role == "user" and "Round 2" in messages[0].content and "A7" in messages[0].content
    assert messages[-1].content == "SITUATION"

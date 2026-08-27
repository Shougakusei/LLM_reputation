from __future__ import annotations

from conftest import ScriptedProvider

from src.core.agent import Agent, AgentSetup
from src.core.config import GameCfg, ProviderCfg
from src.strategy.direct import DirectStrategy


def _agent(replies):
    cfg = ProviderCfg(base_url="http://x/v1", model="m")
    return Agent("A1", AgentSetup("You are A1.", cfg), ScriptedProvider(replies))


async def test_direct_returns_parsed_number_no_prediction():
    agent = _agent(['{"number": 6, "rationale": "because"}'])
    d = await DirectStrategy(GameCfg(rationale=True)).decide(agent, "A2", round=1, feed="")
    assert d.number == 6
    assert d.rationale == "because"
    assert d.predicted is None
    assert d.predicted_rationale is None


async def test_direct_threads_llm_calls():
    agent = _agent(['{"number": 6, "rationale": "because"}'])
    d = await DirectStrategy(GameCfg()).decide(agent, "A2", round=1, feed="")
    assert [c.phase for c in d.calls] == ["decide"]   # raw call threaded up
    assert d.calls[0].status == "ok"


async def test_direct_rationale_off_asks_bare_number_and_drops_text():
    agent = _agent(['{"number": 6, "rationale": "volunteered anyway"}'])
    d = await DirectStrategy(GameCfg(rationale=False)).decide(agent, "A2", round=1, feed="")
    assert d.number == 6
    assert d.rationale == ""                      # rationale=false -> bare template, rationale not stored
    _, messages = agent.provider.calls[0]
    assert "rationale" not in messages[-1].content.lower()


async def test_direct_choice_mapping_shifts_the_played_number():
    # choice_mapping one_above: the agent decides honestly, the engine plays +1 mod 10.
    from src.strategy.mappings import get_mapping
    agent = _agent(['{"number": 9}'])
    d = await DirectStrategy(GameCfg(), get_mapping("one_above")).decide(agent, "A2", round=1, feed="")
    assert d.number == 0                          # 9 -> 0 on the cycle


async def test_make_strategy_threads_choice_mapping():
    from src.strategy.base import make_strategy
    agent = _agent(['{"number": 4}'])
    st = make_strategy("direct", "match", GameCfg(), choice_mapping="one_above")
    assert (await st.decide(agent, "A2", round=1, feed="")).number == 5

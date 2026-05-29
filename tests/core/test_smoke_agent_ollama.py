from __future__ import annotations

import httpx
import pytest

from src.core.agent import Agent, AgentSetup, Phase, PhaseKind
from src.core.config import ProviderCfg
from src.providers import make_provider

OLLAMA_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "llama3:8b"  # non-reasoning -> fast, content comes directly


def _ollama_up() -> bool:
    try:
        httpx.get("http://localhost:11434/api/tags", timeout=1.0)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_up(), reason="local Ollama not reachable")
async def test_decide_against_ollama():
    cfg = ProviderCfg(base_url=OLLAMA_URL, model=OLLAMA_MODEL, temperature=0.0, max_tokens=256)
    provider = make_provider(cfg)
    agent = Agent("A1", AgentSetup("You are a player in a simple number game.", cfg), provider)
    phase = Phase(
        PhaseKind.DECIDE,
        'Choose a number from 0 to 9. Respond ONLY as JSON: '
        '{"number": <integer 0-9>, "rationale": "<short>"}',
        rules="This is a test round.",
    )
    try:
        r = await agent.act(phase)
    finally:
        await provider.aclose()
    assert 0 <= r.data["number"] <= 9
    assert isinstance(r.data["rationale"], str)


@pytest.mark.skipif(not _ollama_up(), reason="local Ollama not reachable")
async def test_talk_against_ollama():
    cfg = ProviderCfg(base_url=OLLAMA_URL, model=OLLAMA_MODEL, temperature=0.0, max_tokens=256)
    provider = make_provider(cfg)
    agent = Agent("A1", AgentSetup("You are a player negotiating in a number game.", cfg), provider)
    phase = Phase(
        PhaseKind.TALK,
        "Tell your partner which number you propose. Respond ONLY as JSON: "
        '{"message": "<your line>", "ready": <true|false>}',
        rules="This is a test round.",
    )
    try:
        r = await agent.act(phase)
    finally:
        await provider.aclose()
    assert isinstance(r.data["message"], str) and r.public_text == r.data["message"]
    assert isinstance(r.data["ready"], bool)

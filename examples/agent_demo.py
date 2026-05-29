"""Demo of the agent layer (layer 2) against a local Ollama.

Run from the repo root so `src` is importable:

    PYTHONPATH=. .venv/bin/python examples/agent_demo.py
"""

import asyncio

from src.core.agent import Agent, AgentSetup, Phase, PhaseKind
from src.core.config import ProviderCfg
from src.core.memory import MemoryEntry
from src.providers import make_provider

# The game rules live in `Phase.rules`; the agent puts them in `system` after its persona.
RULES = (
    "Game: you and a partner each secretly pick a number 0-9. "
    "Equal numbers => both cooperate (3 points each). "
    "If yours is exactly one higher (mod 10) you betrayed them (you 5, them 0); "
    "otherwise both get 1."
)


async def main():
    cfg = ProviderCfg(
        base_url="http://localhost:11434/v1",
        model="llama3:8b",  # non-reasoning: answers directly. For qwen3 raise max_tokens (>=512).
        temperature=0.7,
        max_tokens=256,
    )
    provider = make_provider(cfg)
    agent = Agent("A1", AgentSetup("You are a pragmatic, self-interested player.", cfg), provider)

    try:
        # 1. TALK: the agent produces one negotiation line (+ a `ready` flag).
        talk = await agent.act(
            Phase(
                PhaseKind.TALK,
                "Open the negotiation with A2. Respond ONLY as JSON: "
                '{"message": "<your line>", "ready": <true|false>}',
                rules=RULES,
            )
        )
        print("=== TALK ===")
        print("message:", repr(talk.public_text))
        print("ready  :", talk.data["ready"])
        print("tokens   :", talk.usage, " parse_failures:", agent.parse_failures)
        print()

        # 2. DECIDE: the agent secretly picks a number 0-9 (+ private rationale).
        decide = await agent.act(
            Phase(
                PhaseKind.DECIDE,
                "Now secretly choose your number. Respond ONLY as JSON: "
                '{"number": <0-9>, "rationale": "<short>"}',
                rules=RULES,
            )
        )
        print("=== DECIDE ===")
        print("number   :", decide.data["number"])
        print("rationale:", decide.data["rationale"])
        print("tokens   :", decide.usage, " parse_failures:", agent.parse_failures)
        print()

        # 3. MEMORY: record a finished round; it will be rendered into the next prompt.
        agent.memory.add(
            MemoryEntry(
                round=1,
                partner_id="A2",
                transcript=[{"speaker": "A2", "text": "let's both take 5", "ready": True}],
                my_number=6,
                my_rationale="they trusted me, so I grabbed +1",
                partner_number=5,
                outcome="DC",
                payoff=5.0,
            )
        )
        diary = agent.memory.render(None)  # None = full history
        print("=== MEMORY (diary block prepended to the next act) ===")
        print(diary[0].content if diary else "(empty)")
    finally:
        await provider.aclose()


if __name__ == "__main__":
    asyncio.run(main())

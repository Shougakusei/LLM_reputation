from __future__ import annotations

from typing import Iterator, Protocol

from src.core.agent import Agent, AgentSetup
from src.providers import LLMProvider, make_provider


class Population:
    """Mutable roster of live agents with stable, non-reused ids (A1, A2, …).

    Owns the provider cache: agents sharing a (base_url, model) share one client
    (connection pooling), so `aclose()` closes each unique provider exactly once.
    Evolution mutators (remove/replace) are a documented seam now implemented by
    `remove()` and `draw_name()` — the MVP only ever calls `add()`.
    """

    def __init__(self, *, context_window: int | None = None):
        self._agents: list[Agent] = []
        self._by_id: dict[str, Agent] = {}
        self._counter = 0
        self._providers: dict[tuple[str, str], LLMProvider] = {}
        self._window = context_window
        self.name_pool: list[str] = []   # unused replacement names (set by the generator; evolution draws from it)

    @property
    def agents(self) -> list[Agent]:
        return self._agents

    def ids(self) -> list[str]:
        return [a.id for a in self._agents]

    def get(self, agent_id: str) -> Agent:
        return self._by_id[agent_id]

    def __iter__(self) -> Iterator[Agent]:
        return iter(self._agents)

    def __len__(self) -> int:
        return len(self._agents)

    def next_id(self) -> str:
        self._counter += 1            # only grows -> ids are never reused
        return f"A{self._counter}"

    def add(self, setup: AgentSetup, *, agent_id: str | None = None) -> Agent:
        cfg = setup.provider_cfg
        key = (cfg.base_url, cfg.model)
        provider = self._providers.get(key)
        if provider is None:
            provider = make_provider(cfg)
            self._providers[key] = provider
        aid = agent_id if agent_id is not None else self.next_id()
        agent = Agent(aid, setup, provider, context_window=self._window)
        self._agents.append(agent)
        self._by_id[agent.id] = agent
        return agent

    def remove(self, agent_id: str) -> None:
        """Remove a live agent from the roster (its id is never reused)."""
        agent = self._by_id.pop(agent_id)
        self._agents.remove(agent)

    def draw_name(self, rng) -> str:
        """Pop an unused name from the leftover pool at an rng-chosen index.

        The caller must ensure the pool is non-empty (evolution raises its own
        error with round context before calling this on an empty pool)."""
        return self.name_pool.pop(rng.randrange(len(self.name_pool)))

    async def aclose(self) -> None:
        for provider in self._providers.values():   # one entry per unique (base_url, model)
            await provider.aclose()


class PopulationGenerator(Protocol):
    def build(self, rng) -> Population: ...


def make_population(pop_cfg, *, context_window: int | None = None) -> PopulationGenerator:
    if pop_cfg.kind == "roster":
        from src.population.roster import RosterGenerator

        return RosterGenerator(pop_cfg, context_window=context_window)
    raise ValueError(f"unknown population kind: {pop_cfg.kind!r}")

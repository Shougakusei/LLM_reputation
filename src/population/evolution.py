"""Population evolution step: random death and replacement between rounds.

Deterministic given (population state, rng): the orchestrator derives a dedicated
rng stream per round (Random(f"{seed}:evolution:{r}")), so resume can re-derive
every roster change without replaying LLM calls. Rng consumption order is a
compatibility contract: one random() per live agent in roster order (invincible
agents consume their draw and ignore it), then per replacement at most one
random() (type — roll mode only; inherit mode never draws it) plus one
randrange() (name).
"""

from __future__ import annotations

from dataclasses import asdict

from src.core.agent import AgentSetup
from src.population.base import Population


class NamePoolExhausted(RuntimeError):
    """The name pool has no unused names left for a replacement agent."""


def evolve(pop: Population, pop_cfg, rng, round: int) -> list[dict]:
    """Kill each agent with probability death_prob (unless invincible) and spawn a replacement per death.

    In **roll mode** (replacement="roll"), the newborn is deceptive with probability
    d/N (d = current live deceptive count, N = fixed population size), forced
    deceptive while d < decept_min and forced normal while d >= decept_max, then
    clones the first matching spec (system prompt, strategy). In **inherit mode**
    (replacement="inherit"), the newborn clones the dying agent's full setup verbatim
    (system_prompt, play_strategy, prediction_mapping, deceptive). Both modes: fresh
    memory, score 0, born mortal, and unused pool name. Invincible agents consume
    their death draw but ignore it, preserving the rng stream for deterministic resume.

    Args:
        pop: Live population to mutate (remove the dead, add the newborn).
        pop_cfg: PopulationCfg with a non-None `evolution` block.
        rng: Dedicated per-round stream (Random(f"{seed}:evolution:{round}")).
        round: Round number, used in events and error messages.

    Returns:
        Event dicts, deaths first: {"type": "death", "agent", "score"} and
        {"type": "birth", "agent", "deceptive", "system_prompt", "provider"}.

    Raises:
        NamePoolExhausted: A replacement needs a name but the pool is empty; the
            run cannot be resumed past this point (enlarge the pool and start a
            new run).
    """
    ev = pop_cfg.evolution
    n_total = len(pop)
    deaths = [a for a in list(pop)
              if rng.random() < ev.death_prob and not a.setup.invincible]
    events: list[dict] = []
    for agent in deaths:
        pop.remove(agent.id)
        events.append({"type": "death", "agent": agent.id, "score": agent.score})
    for dead in deaths:
        if ev.replacement == "inherit":
            # the newborn takes the dying agent's role verbatim; no type draw
            deceptive = dead.setup.deceptive
            system_prompt = dead.setup.system_prompt
            play_strategy = dead.setup.play_strategy
            prediction_mapping = dead.setup.prediction_mapping
            choice_mapping = dead.setup.choice_mapping
            provider = pop_cfg.provider or dead.setup.provider_cfg
        else:
            d = sum(1 for a in pop if a.setup.deceptive)
            if d < ev.decept_min:
                deceptive = True
            elif d >= ev.decept_max:
                deceptive = False
            else:
                deceptive = rng.random() < d / n_total
            spec = next(s for s in pop_cfg.agents if s.deceptive == deceptive)
            system_prompt = spec.system_prompt
            play_strategy = spec.play_strategy
            prediction_mapping = spec.prediction_mapping
            choice_mapping = spec.choice_mapping
            provider = pop_cfg.provider or spec.provider
        if not pop.name_pool:
            raise NamePoolExhausted(
                f"round {round}: name pool exhausted — enlarge the name pools; "
                f"this run cannot be resumed past this point")
        name = pop.draw_name(rng)
        # newborns are always mortal: invincible is deliberately not passed (defaults False)
        pop.add(AgentSetup(system_prompt, provider,
                           play_strategy, prediction_mapping,
                           choice_mapping=choice_mapping, deceptive=deceptive),
                agent_id=name)
        events.append({"type": "birth", "agent": name, "deceptive": deceptive,
                       "system_prompt": system_prompt,
                       "provider": asdict(provider)})
    return events

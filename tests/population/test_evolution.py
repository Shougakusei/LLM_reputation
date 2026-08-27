from __future__ import annotations

import random

import pytest

from dataclasses import replace

from src.core.config import AgentSpec, EvolutionCfg, PopulationCfg, ProviderCfg
from src.population import base as popbase
from src.population import make_population
from src.population.evolution import NamePoolExhausted, evolve


class _StubProvider:
    def __init__(self, cfg):
        self.cfg = cfg

    async def complete(self, **kw):
        raise AssertionError("evolution tests must not call the LLM")

    async def aclose(self):
        pass


@pytest.fixture(autouse=True)
def _stub_providers(monkeypatch):
    monkeypatch.setattr(popbase, "make_provider", lambda cfg: _StubProvider(cfg))


class FakeRng:
    """Scripted rng: returns queued values for random() and randrange()."""

    def __init__(self, randoms=(), ranges=()):
        self._randoms = list(randoms)
        self._ranges = list(ranges)

    def random(self):
        return self._randoms.pop(0)

    def randrange(self, n):
        return self._ranges.pop(0)


def _pop_cfg(pool_size=20, death_prob=0.5, decept_min=0, decept_max=4):
    return PopulationCfg(
        kind="roster",
        agents=[AgentSpec(count=3, system_prompt="normal {id}"),
                AgentSpec(count=1, system_prompt="defect {id}", deceptive=True)],
        provider=ProviderCfg(base_url="http://x/v1", model="m"),
        first_name_pool=[f"Player {i}" for i in range(pool_size)],
        evolution=EvolutionCfg(death_prob=death_prob, decept_min=decept_min,
                               decept_max=decept_max),
    )


def _build(cfg, seed=0):
    return make_population(cfg).build(random.Random(seed))


def test_no_deaths_when_all_survival_draws_high():
    cfg = _pop_cfg()
    pop = _build(cfg)
    before = list(pop.ids())
    events = evolve(pop, cfg, FakeRng(randoms=[0.9, 0.9, 0.9, 0.9]), round=2)
    assert events == [] and pop.ids() == before


def test_dead_agent_replaced_with_fresh_normal_agent():
    cfg = _pop_cfg()
    pop = _build(cfg)
    dead_id = pop.agents[1].id
    pop.agents[1].score = 7.0
    # draws: 4 deaths (only #2 dies: 0.1 < 0.5), then type 0.9 >= d/N=1/4 -> normal, then name index 0
    rng = FakeRng(randoms=[0.9, 0.1, 0.9, 0.9, 0.9], ranges=[0])
    events = evolve(pop, cfg, rng, round=3)
    assert events[0] == {"type": "death", "agent": dead_id, "score": 7.0}
    birth = events[1]
    assert birth["type"] == "birth" and birth["deceptive"] is False
    assert birth["system_prompt"] == "normal {id}"
    newborn = pop.get(birth["agent"])
    assert newborn.score == 0.0 and newborn.memory.entries == []
    assert newborn.setup.deceptive is False
    assert birth["agent"] != dead_id and len(pop) == 4


def test_deceptive_type_draw_below_threshold_spawns_deceptive():
    cfg = _pop_cfg()
    pop = _build(cfg)
    # only the FIRST (normal) agent dies; survivors hold d=1, N=4 -> threshold 0.25
    rng = FakeRng(randoms=[0.1, 0.9, 0.9, 0.9, 0.2], ranges=[0])
    events = evolve(pop, cfg, rng, round=2)
    birth = next(e for e in events if e["type"] == "birth")
    assert birth["deceptive"] is True
    assert birth["system_prompt"] == "defect {id}"
    assert pop.get(birth["agent"]).setup.deceptive is True


def test_bounds_force_types_until_satisfied():
    # full turnover with decept_min=decept_max=2: d<2 forces deceptive twice, then d>=2 forces normal
    cfg = _pop_cfg(death_prob=1.0, decept_min=2, decept_max=2)
    pop = _build(cfg)
    evolve(pop, cfg, random.Random("evo"), round=2)
    decept = [a for a in pop if a.setup.deceptive]
    assert len(pop) == 4 and len(decept) == 2
    assert all(a.setup.system_prompt == "defect {id}" for a in decept)


def test_forced_branches_consume_no_type_random():
    """Exact-count oracle: forced type branches must not consume rng.random().

    With decept_min=decept_max=2, full turnover (death_prob=1.0), all 4 replacements
    are forced (d<2 -> True, d>=2 -> False), so zero rng.random() calls for type.
    FakeRng with precisely-sized randoms list will raise IndexError if any forced
    branch accidentally calls rng.random(), protecting against regression that would
    silently desync the RNG stream and break resume determinism (Tasks 4–7).
    """
    cfg = _pop_cfg(death_prob=1.0, decept_min=2, decept_max=2)
    pop = _build(cfg)
    # Exact draw count: 4 randoms (deaths, all < 1.0) + 0 (no type draws, all forced) + 4 ranges (names)
    # If any forced branch consumes rng.random(), pop(0) raises IndexError
    rng = FakeRng(randoms=[0.5, 0.5, 0.5, 0.5], ranges=[0, 1, 2, 3])
    events = evolve(pop, cfg, rng, round=2)

    deaths = [e for e in events if e["type"] == "death"]
    births = [e for e in events if e["type"] == "birth"]
    assert len(deaths) == 4 and len(births) == 4
    decept_births = [b for b in births if b["deceptive"]]
    assert len(decept_births) == 2  # d<2 twice, then d>=2 twice


def test_newborn_names_are_unused_pool_names():
    cfg = _pop_cfg(death_prob=1.0)
    pop = _build(cfg)
    initial = set(pop.ids())
    evolve(pop, cfg, random.Random(0), round=2)
    pool = set(cfg.first_name_pool)
    assert all(a.id in pool and a.id not in initial for a in pop)


def test_name_pool_exhaustion_raises_with_round():
    # pool of 5 for 4 agents -> 1 leftover name; full turnover needs 4
    cfg = _pop_cfg(pool_size=5, death_prob=1.0)
    pop = _build(cfg)
    with pytest.raises(NamePoolExhausted, match="round 2"):
        evolve(pop, cfg, random.Random(0), round=2)


def test_evolution_is_deterministic_across_rebuilds():
    cfg = _pop_cfg(death_prob=0.7)
    rosters = []
    for _ in range(2):
        pop = _build(cfg, seed=1)
        for r in (2, 3):
            evolve(pop, cfg, random.Random(f"5:evolution:{r}"), r)
        rosters.append([(a.id, a.setup.deceptive) for a in pop])
    assert rosters[0] == rosters[1]


def _inv_cfg(pool_size=20, death_prob=1.0, decept_min=0, decept_max=4):
    """Like _pop_cfg but with an invincible normal and an invincible deceptive spec.

    Roster order: 2 mortal normals, 1 invincible normal, 1 invincible deceptive."""
    return PopulationCfg(
        kind="roster",
        agents=[AgentSpec(count=2, system_prompt="normal {id}"),
                AgentSpec(count=1, system_prompt="normal inv {id}", invincible=True),
                AgentSpec(count=1, system_prompt="defect inv {id}", deceptive=True,
                          invincible=True)],
        provider=ProviderCfg(base_url="http://x/v1", model="m"),
        first_name_pool=[f"Player {i}" for i in range(pool_size)],
        evolution=EvolutionCfg(death_prob=death_prob, decept_min=decept_min,
                               decept_max=decept_max),
    )


def test_invincible_agents_survive_full_turnover():
    cfg = _inv_cfg(death_prob=1.0)
    pop = _build(cfg)
    inv_ids = [a.id for a in pop if a.setup.invincible]
    events = evolve(pop, cfg, random.Random(0), round=2)
    deaths = [e for e in events if e["type"] == "death"]
    assert len(deaths) == 2                                  # only the two mortals died
    assert all(e["agent"] not in inv_ids for e in deaths)
    assert all(i in pop.ids() for i in inv_ids)
    assert len(pop) == 4


def test_invincible_agents_still_consume_death_draws():
    # rng contract: one random() per live agent in roster order, invincible or not
    cfg = _inv_cfg(death_prob=0.5)
    pop = _build(cfg)
    rng = FakeRng(randoms=[0.9, 0.9, 0.9, 0.9])              # exactly one per live agent
    events = evolve(pop, cfg, rng, round=2)
    assert events == []
    assert rng._randoms == []                                # all four consumed, none skipped


def test_lethal_draw_ignored_for_invincible():
    cfg = _inv_cfg(death_prob=0.5)
    pop = _build(cfg)
    mortal_id = pop.agents[0].id
    inv_id = pop.agents[2].id                                # the invincible normal
    # draws: mortal #1 dies (0.1 < 0.5), mortal #2 survives, invincible #3 draws a
    # lethal 0.1 but lives, invincible #4 survives; then type 0.9 >= d/N=1/4 -> normal,
    # then name index 0
    rng = FakeRng(randoms=[0.1, 0.9, 0.1, 0.9, 0.9], ranges=[0])
    events = evolve(pop, cfg, rng, round=2)
    dead = [e["agent"] for e in events if e["type"] == "death"]
    assert dead == [mortal_id]
    assert inv_id in pop.ids()


def test_newborn_cloned_from_invincible_spec_is_mortal():
    # decept_min=2 forces the first replacement deceptive; the only deceptive spec is
    # invincible, so the newborn clones its prompt but must be born mortal
    cfg = _inv_cfg(death_prob=1.0, decept_min=2, decept_max=2)
    pop = _build(cfg)
    events = evolve(pop, cfg, random.Random(0), round=2)
    births = [e for e in events if e["type"] == "birth"]
    assert births[0]["deceptive"] is True
    assert births[0]["system_prompt"] == "defect inv {id}"
    for b in births:
        assert pop.get(b["agent"]).setup.invincible is False


def test_evolution_with_invincible_is_deterministic_across_rebuilds():
    cfg = _inv_cfg(death_prob=0.7)
    rosters = []
    for _ in range(2):
        pop = _build(cfg, seed=1)
        for r in (2, 3):
            evolve(pop, cfg, random.Random(f"5:evolution:{r}"), r)
        rosters.append([(a.id, a.setup.deceptive, a.setup.invincible) for a in pop])
    assert rosters[0] == rosters[1]


def _inherit_cfg(pool_size=20, death_prob=0.5):
    """Like _pop_cfg but with replacement="inherit" and no decept bounds.

    Roster order: 2 direct normals, 1 prediction normal, 1 deceptive."""
    return PopulationCfg(
        kind="roster",
        agents=[AgentSpec(count=2, system_prompt="normal {id}"),
                AgentSpec(count=1, system_prompt="predict {id}",
                          play_strategy="prediction", prediction_mapping="one_above"),
                AgentSpec(count=1, system_prompt="defect {id}", deceptive=True)],
        provider=ProviderCfg(base_url="http://x/v1", model="m"),
        first_name_pool=[f"Player {i}" for i in range(pool_size)],
        evolution=EvolutionCfg(death_prob=death_prob, replacement="inherit"),
    )


def test_inherit_newborn_clones_dying_agents_full_setup():
    cfg = _inherit_cfg()
    pop = _build(cfg)
    dead = pop.agents[3]                                     # the deceptive agent
    # draws: 4 deaths (only #4 dies), then ONE name index — no type draw in inherit mode
    rng = FakeRng(randoms=[0.9, 0.9, 0.9, 0.1], ranges=[0])
    events = evolve(pop, cfg, rng, round=2)
    assert events[0] == {"type": "death", "agent": dead.id, "score": dead.score}
    birth = events[1]
    assert birth["type"] == "birth" and birth["deceptive"] is True
    assert birth["system_prompt"] == "defect {id}"
    newborn = pop.get(birth["agent"])
    assert newborn.setup.deceptive is True
    assert newborn.setup.system_prompt == "defect {id}"
    assert newborn.score == 0.0 and newborn.memory.entries == []
    assert birth["agent"] != dead.id and len(pop) == 4


def test_inherit_preserves_strategy_and_mapping():
    cfg = _inherit_cfg()
    pop = _build(cfg)
    # only the prediction agent (roster position 3) dies
    rng = FakeRng(randoms=[0.9, 0.9, 0.1, 0.9], ranges=[0])
    events = evolve(pop, cfg, rng, round=2)
    birth = next(e for e in events if e["type"] == "birth")
    newborn = pop.get(birth["agent"])
    assert newborn.setup.play_strategy == "prediction"
    assert newborn.setup.prediction_mapping == "one_above"


def test_inherit_consumes_no_type_randoms():
    """Exact-count oracle: inherit mode must draw only deaths + names.

    Full turnover of 4 agents: exactly 4 random() (deaths) and 4 randrange() (names).
    FakeRng raises IndexError on any extra draw, catching a regression that would
    desync the rng stream and corrupt resumed runs."""
    cfg = _inherit_cfg(death_prob=1.0)
    pop = _build(cfg)
    rng = FakeRng(randoms=[0.5, 0.5, 0.5, 0.5], ranges=[0, 1, 2, 3])
    events = evolve(pop, cfg, rng, round=2)
    assert len([e for e in events if e["type"] == "birth"]) == 4
    assert rng._randoms == [] and rng._ranges == []


def test_inherit_full_turnover_pairs_each_newborn_with_its_predecessor():
    cfg = _inherit_cfg(death_prob=1.0)
    pop = _build(cfg)
    dying = list(pop)                                        # full turnover: everyone dies
    setup_by_id = {a.id: (a.setup.system_prompt, a.setup.deceptive) for a in dying}
    before = sorted(setup_by_id.values())
    events = evolve(pop, cfg, random.Random(0), round=2)
    after = sorted((a.setup.system_prompt, a.setup.deceptive) for a in pop)
    assert after == before                                   # roles frozen, names changed

    deaths = [e for e in events if e["type"] == "death"]
    births = [e for e in events if e["type"] == "birth"]
    # event order is all deaths first, then births, both in roster order, so
    # birth i replaces death i: pair them positionally and check each newborn
    # inherited its own predecessor's setup, not merely some setup from the batch
    # (a crossed-pairing bug would still pass the multiset check above).
    assert [d["agent"] for d in deaths] == [a.id for a in dying]
    for death, birth in zip(deaths, births):
        expected = setup_by_id[death["agent"]]
        assert (birth["system_prompt"], birth["deceptive"]) == expected
        newborn = pop.get(birth["agent"])
        assert (newborn.setup.system_prompt, newborn.setup.deceptive) == expected


def test_inherit_newborns_are_mortal_with_fresh_names():
    cfg = _inherit_cfg(death_prob=1.0)
    pop = _build(cfg)
    initial = set(pop.ids())
    evolve(pop, cfg, random.Random(0), round=2)
    assert all(a.id not in initial and a.setup.invincible is False for a in pop)


def test_inherit_is_deterministic_across_rebuilds():
    cfg = _inherit_cfg(death_prob=0.7)
    rosters = []
    for _ in range(2):
        pop = _build(cfg, seed=1)
        for r in (2, 3):
            evolve(pop, cfg, random.Random(f"5:evolution:{r}"), r)
        rosters.append([(a.id, a.setup.system_prompt, a.setup.deceptive) for a in pop])
    assert rosters[0] == rosters[1]


def test_inherit_newborn_keeps_dying_agents_provider_when_population_has_none():
    own = ProviderCfg(base_url="http://own/v1", model="own")
    other = ProviderCfg(base_url="http://other/v1", model="other")
    cfg = _inherit_cfg()
    cfg = replace(cfg, provider=None,
                  agents=[replace(cfg.agents[0], provider=own)]
                  + [replace(a, provider=other) for a in cfg.agents[1:]])
    pop = _build(cfg)
    rng = FakeRng(randoms=[0.1, 0.9, 0.9, 0.9], ranges=[0])   # the first agent dies
    birth = evolve(pop, cfg, rng, round=2)[1]
    assert birth["provider"]["model"] == "own"
    assert pop.get(birth["agent"]).setup.provider_cfg == own


def test_roll_newborn_takes_chosen_specs_provider_when_population_has_none():
    own = ProviderCfg(base_url="http://own/v1", model="own")
    other = ProviderCfg(base_url="http://other/v1", model="other")
    cfg = _pop_cfg(decept_min=1, decept_max=4)
    cfg = replace(cfg, provider=None,
                  agents=[replace(cfg.agents[0], provider=other), replace(cfg.agents[1], provider=own)])
    pop = _build(cfg)
    rng = FakeRng(randoms=[0.9, 0.9, 0.9, 0.1], ranges=[0])   # the deceptive one dies -> forced deceptive
    birth = evolve(pop, cfg, rng, round=2)[1]
    assert birth["deceptive"] is True and birth["provider"]["model"] == "own"


def test_population_provider_wins_for_newborns_too():
    own = ProviderCfg(base_url="http://own/v1", model="own")
    cfg = _inherit_cfg()
    cfg = replace(cfg, agents=[replace(cfg.agents[0], provider=own)] + list(cfg.agents[1:]))
    pop = _build(cfg)
    rng = FakeRng(randoms=[0.1, 0.9, 0.9, 0.9], ranges=[0])
    birth = evolve(pop, cfg, rng, round=2)[1]
    assert birth["provider"]["model"] == "m"                  # population-wide provider

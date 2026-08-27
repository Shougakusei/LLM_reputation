from __future__ import annotations

import random

import pytest

from src.core.agent import AgentSetup
from src.core.config import AgentSpec, PopulationCfg, ProviderCfg
from src.population import Population, make_population
from src.population import base as popbase
from src.population.roster import RosterGenerator


class FakeProvider:
    def __init__(self, cfg):
        self.cfg = cfg
        self.closed = 0

    async def complete(self, **kw):
        raise NotImplementedError

    async def aclose(self):
        self.closed += 1


@pytest.fixture
def created(monkeypatch):
    """Patch the provider factory so building a population creates FakeProviders;
    return the list of created providers for caching / aclose assertions."""
    made = []

    def factory(cfg):
        p = FakeProvider(cfg)
        made.append(p)
        return p

    monkeypatch.setattr(popbase, "make_provider", factory)
    return made


_PROVIDER = ProviderCfg(base_url="http://x/v1", model="m")


def _spec(system_prompt, count=1):
    return AgentSpec(system_prompt=system_prompt, count=count)


def _pop_cfg(specs, provider=_PROVIDER):
    return PopulationCfg(kind="roster", agents=specs, provider=provider)


def test_roster_expands_by_count_and_ids(created):
    specs = [_spec("p0", count=3), _spec("p1", count=2)]
    pop = make_population(_pop_cfg(specs)).build(random.Random(0))
    assert isinstance(pop, Population)
    assert pop.ids() == ["A1", "A2", "A3", "A4", "A5"]
    assert [a.setup.system_prompt for a in pop] == ["p0", "p0", "p0", "p1", "p1"]   # grouped by type
    assert len(pop) == 5


def test_provider_shared_across_all_agents(created):
    # provider is population-wide -> every agent dedups to one cached client
    specs = [_spec("p0", count=2), _spec("p1", count=1)]
    pop = make_population(_pop_cfg(specs)).build(random.Random(0))
    a1, a2, a3 = pop.get("A1"), pop.get("A2"), pop.get("A3")
    assert a1.provider is a2.provider is a3.provider   # same cfg -> shared client
    assert len(created) == 1                             # only one provider ever created


def test_context_window_threaded_to_agents(created):
    pop = make_population(_pop_cfg([_spec("p", count=2)]), context_window=3).build(random.Random(0))
    assert pop.get("A1")._window == 3 and pop.get("A2")._window == 3


async def test_aclose_closes_the_shared_provider_once(created):
    specs = [_spec("p0", count=2), _spec("p1", count=1)]
    pop = make_population(_pop_cfg(specs)).build(random.Random(0))
    await pop.aclose()
    assert len(created) == 1
    assert all(p.closed == 1 for p in created)   # the shared client closed exactly once


def test_make_population_unknown_kind_raises():
    cfg = PopulationCfg(kind="nope", agents=[_spec("p")], provider=_PROVIDER)
    with pytest.raises(ValueError):
        make_population(cfg)


def _pop_cfg_named(specs, firsts, lasts):
    return PopulationCfg(kind="roster", agents=specs, provider=_PROVIDER,
                         first_name_pool=firsts, last_name_pool=lasts)


def test_names_replace_ids_unique_first_and_last(created):
    firsts = ["Kurisu", "Mayuri", "Itaru", "Moeka"]
    lasts = ["Makise", "Shiina", "Hashida", "Kiryuu"]
    pop = make_population(_pop_cfg_named([_spec("p", count=3)], firsts, lasts)).build(random.Random(0))
    ids = pop.ids()
    assert len(ids) == 3
    first_parts = [i.split(" ")[0] for i in ids]
    last_parts = [i.split(" ")[1] for i in ids]
    assert len(set(first_parts)) == 3   # all first names unique
    assert len(set(last_parts)) == 3    # all last names unique
    assert all(f in firsts and l in lasts for f, l in zip(first_parts, last_parts))


def test_name_assignment_is_deterministic_per_seed(created):
    firsts = ["Kurisu", "Mayuri", "Itaru", "Moeka"]
    lasts = ["Makise", "Shiina", "Hashida", "Kiryuu"]
    cfg = _pop_cfg_named([_spec("p", count=3)], firsts, lasts)
    ids1 = make_population(cfg).build(random.Random(7)).ids()
    ids2 = make_population(cfg).build(random.Random(7)).ids()
    ids3 = make_population(cfg).build(random.Random(99)).ids()
    assert ids1 == ids2          # same seed -> same assignment
    assert ids1 != ids3          # different seed -> different assignment


def test_empty_pools_fall_back_to_a_ids(created):
    # No pools provided -> A1.. ids (keeps programmatic construction working).
    pop = make_population(_pop_cfg([_spec("p", count=2)])).build(random.Random(0))
    assert pop.ids() == ["A1", "A2"]


def test_only_first_pool_uses_names_without_surname(created):
    # single pool (no surnames) -> id = the pool entry itself (e.g. "Player 348")
    firsts = ["Player 348", "Player 712", "Player 905"]
    pop = make_population(_pop_cfg_named([_spec("p", count=3)], firsts, [])).build(random.Random(0))
    ids = pop.ids()
    assert len(ids) == 3 and len(set(ids)) == 3
    assert all(i in firsts for i in ids)


def test_only_last_pool_uses_names_without_first(created):
    # symmetric case: only last_name_pool given -> id = its entries
    lasts = ["348", "712", "905", "246"]
    pop = make_population(_pop_cfg_named([_spec("p", count=2)], [], lasts)).build(random.Random(0))
    assert all(i in lasts for i in pop.ids())


def test_single_pool_numeric_entries_become_str_ids(created):
    # numeric pool entries (YAML numbers) become string ids
    pop = make_population(_pop_cfg_named([_spec("p", count=2)], [348, 712, 905], [])).build(random.Random(0))
    assert all(isinstance(i, str) for i in pop.ids())


class _StubProvider:
    def __init__(self, cfg):
        self.cfg = cfg

    async def complete(self, **kw):
        raise AssertionError("roster tests must not call the LLM")

    async def aclose(self):
        pass


@pytest.fixture
def _stub_providers(monkeypatch):
    monkeypatch.setattr(popbase, "make_provider", lambda cfg: _StubProvider(cfg))


def _pool_cfg(pool_size=8):
    return PopulationCfg(
        kind="roster",
        agents=[AgentSpec(count=2, system_prompt="normal {id}"),
                AgentSpec(count=1, system_prompt="defect {id}", deceptive=True)],
        provider=ProviderCfg(base_url="http://x/v1", model="m"),
        first_name_pool=[f"P{i}" for i in range(pool_size)],
    )


def test_build_marks_deceptive_and_keeps_leftover_pool(_stub_providers):
    cfg = _pool_cfg()
    pop = make_population(cfg).build(random.Random(0))
    assert [a.setup.deceptive for a in pop] == [False, False, True]
    # leftover pool = the unused names, in original pool order
    used = set(pop.ids())
    assert pop.name_pool == [n for n in cfg.first_name_pool if n not in used]
    assert len(pop.name_pool) == 5


def test_agent_setup_deceptive_defaults_false():
    setup = AgentSetup("prompt", ProviderCfg(base_url="http://x/v1", model="m"))
    assert setup.deceptive is False


def test_remove_drops_agent_from_roster(_stub_providers):
    pop = make_population(_pool_cfg()).build(random.Random(0))
    victim = pop.ids()[1]
    pop.remove(victim)
    assert victim not in pop.ids() and len(pop) == 2
    with pytest.raises(KeyError):
        pop.get(victim)


def test_both_pools_leftover_is_deterministic_per_seed(_stub_providers):
    # Two-pool mode ("First Last"): building twice with the same seed must give the
    # same roster ids AND the same leftover name_pool (evolution draws replacements
    # from it, so its order/content must be reproducible on resume).
    firsts = [f"F{i}" for i in range(10)]
    lasts = [f"L{i}" for i in range(10)]
    cfg = _pop_cfg_named([_spec("p", count=4)], firsts, lasts)

    pop1 = make_population(cfg).build(random.Random(7))
    pop2 = make_population(cfg).build(random.Random(7))

    assert pop1.ids() == pop2.ids()
    assert pop1.name_pool == pop2.name_pool
    assert len(pop1.name_pool) == 6                    # 10 - 4 used, per pool
    for name in pop1.name_pool:
        first, last = name.split(" ")
        assert first in firsts and last in lasts        # leftovers are "First Last" from unused names
        assert name not in pop1.ids()


def test_draw_name_pops_from_leftover_pool(_stub_providers):
    pop = make_population(_pool_cfg()).build(random.Random(0))
    before = list(pop.name_pool)
    name = pop.draw_name(random.Random(7))
    assert name in before and name not in pop.name_pool
    assert len(pop.name_pool) == len(before) - 1


def test_agent_setup_invincible_defaults_false():
    setup = AgentSetup("prompt", ProviderCfg(base_url="http://x/v1", model="m"))
    assert setup.invincible is False


def test_build_marks_invincible_per_spec(_stub_providers):
    cfg = PopulationCfg(
        kind="roster",
        agents=[AgentSpec(count=1, system_prompt="normal {id}"),
                AgentSpec(count=1, system_prompt="normal inv {id}", invincible=True),
                AgentSpec(count=1, system_prompt="defect {id}", deceptive=True,
                          invincible=True)],
        provider=ProviderCfg(base_url="http://x/v1", model="m"),
        first_name_pool=[f"P{i}" for i in range(8)],
    )
    pop = RosterGenerator(cfg).build(random.Random(0))
    assert [a.setup.invincible for a in pop] == [False, True, True]


def test_population_provider_wins_over_agent_providers(created):
    own = ProviderCfg(base_url="http://subj/v1", model="subject")
    specs = [AgentSpec(system_prompt="s", provider=own), _spec("p1", count=2)]
    pop = make_population(_pop_cfg(specs)).build(random.Random(0))
    assert all(a.setup.provider_cfg == _PROVIDER for a in pop)   # the agent-level block is ignored
    assert len(created) == 1


def test_agent_providers_used_when_population_has_none(created):
    subj = ProviderCfg(base_url="http://subj/v1", model="subject")
    npc = ProviderCfg(base_url="http://npc/v1", model="npc")
    specs = [AgentSpec(system_prompt="s", provider=subj),
             AgentSpec(system_prompt="n", count=2, provider=npc)]
    pop = make_population(_pop_cfg(specs, provider=None)).build(random.Random(0))
    a1, a2, a3 = pop.get("A1"), pop.get("A2"), pop.get("A3")
    assert a1.setup.provider_cfg == subj and a2.setup.provider_cfg == npc
    assert a1.provider is not a2.provider and a2.provider is a3.provider
    assert len(created) == 2                             # one client per distinct provider


def test_choice_mapping_threaded_to_agent_setup(created):
    specs = [AgentSpec(system_prompt="s", choice_mapping="one_above"), _spec("p1")]
    pop = make_population(_pop_cfg(specs)).build(random.Random(0))
    assert pop.get("A1").setup.choice_mapping == "one_above"
    assert pop.get("A2").setup.choice_mapping == "match"

# Population Evolution (Agent Death & Replacement) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** At the start of each round r ≥ 2, every agent may die with configurable probability and is replaced by a fresh agent whose deceptive/normal type is sampled from the current deceptive fraction, clamped to `[decept_min, decept_max]`.

**Architecture:** A deterministic evolution step (`src/population/evolution.py`) driven by a dedicated per-round rng stream `Random(f"{seed}:evolution:{r}")`, mirroring the existing matchmaker-rng pattern so resume/extend stay reproducible. Deaths/births flow out only through the existing `RoundPlan.events` → observer channel; new `agents` columns (`born_round`/`died_round`/`deceptive`) are analysis-only.

**Tech Stack:** Python 3.12, frozen dataclasses config, pytest + pytest-asyncio (auto mode), SQLite, `uv` for all commands.

**Spec:** `docs/superpowers/specs/2026-08-05-population-evolution-design.md` — read it first.

## Global Constraints

- Everything in English: docstrings (Google-style), comments, prints, log/exception messages, docs — this project overrides the user-global Russian rule.
- TDD: write the failing test FIRST, run it to see it fail, then minimal implementation.
- `from __future__ import annotations` at the top of every module; rng objects are passed in, never created inside library code (the orchestrator/runner derive the per-round streams — that is the existing pattern).
- No printing or persistence inside `src/` — output leaves the engine only via the orchestrator `observer` callback.
- Dependencies only via `uv`; run tests with `uv run pytest ...` (pytest-asyncio auto mode — no `@pytest.mark.asyncio` needed; `pythonpath=["."]` — import `src.*` directly).
- Unit tests use stub providers (monkeypatch `src.population.base.make_provider`) — no network.
- Config objects are frozen dataclasses; validation happens once at load in `_validate` (`src/core/config.py`), fail fast with a clear message.
- Evolution rng consumption order is a compatibility contract (resume re-derives it): one `rng.random()` per live agent in roster order for deaths, then per replacement at most one `rng.random()` (type) plus one `rng.randrange()` (name). Never reorder these draws.

---

### Task 1: Config — `EvolutionCfg`, `deceptive` spec flag, validation

**Files:**
- Modify: `src/core/config.py` (AgentSpec ~line 290, PopulationCfg ~line 300, `_population_cfg` ~line 370, `_validate` ~line 387)
- Test: `tests/core/test_config_load.py` (append)

**Interfaces:**
- Consumes: existing `AgentSpec`, `PopulationCfg`, `_validate`, `episode_from_dict`.
- Produces: `EvolutionCfg` frozen dataclass with fields `death_prob: float`, `decept_min: int`, `decept_max: int`; `AgentSpec.deceptive: bool = False`; `PopulationCfg.evolution: EvolutionCfg | None = None`. Later tasks import `EvolutionCfg` from `src.core.config` and read `spec.deceptive` / `pop_cfg.evolution`.

- [ ] **Step 1: Write the failing tests** — append to `tests/core/test_config_load.py` (reuse its existing imports if it already imports `episode_from_dict`; otherwise add `from src.core.config import EvolutionCfg, episode_from_dict` and `import pytest`):

```python
def _evo_dict(evolution=None, agents=None, first_pool=None):
    """Minimal valid episode dict; evolution/agents/pool are override points."""
    d = {
        "seed": 1, "rounds": 3, "matchmaker": "random",
        "population": {
            "kind": "roster",
            "agents": agents if agents is not None else [
                {"count": 3, "system_prompt": "normal {id}"},
                {"count": 1, "system_prompt": "defect {id}", "deceptive": True},
            ],
            "provider": {"base_url": "http://x/v1", "model": "m"},
        },
    }
    if first_pool is not None:
        d["population"]["first_name_pool"] = first_pool
    if evolution is not None:
        d["population"]["evolution"] = evolution
    return d


def test_evolution_block_parses_into_cfg():
    d = _evo_dict(evolution={"death_prob": 0.1, "decept_min": 1, "decept_max": 3},
                  first_pool=[f"P{i}" for i in range(10)])
    cfg = episode_from_dict(d)
    assert cfg.population.evolution == EvolutionCfg(death_prob=0.1, decept_min=1, decept_max=3)
    assert [s.deceptive for s in cfg.population.agents] == [False, True]


def test_evolution_absent_means_none_and_deceptive_defaults_false():
    cfg = episode_from_dict(_evo_dict())
    assert cfg.population.evolution is None
    assert all(s.deceptive is False for s in cfg.population.agents)


def test_evolution_survives_asdict_roundtrip():
    from dataclasses import asdict
    d = _evo_dict(evolution={"death_prob": 0.5, "decept_min": 0, "decept_max": 4},
                  first_pool=[f"P{i}" for i in range(10)])
    cfg = episode_from_dict(d)
    again = episode_from_dict(asdict(cfg))
    assert again.population.evolution == cfg.population.evolution
    assert [s.deceptive for s in again.population.agents] == [False, True]


@pytest.mark.parametrize("evolution, message", [
    ({"death_prob": 1.5, "decept_min": 0, "decept_max": 4}, "death_prob"),
    ({"death_prob": -0.1, "decept_min": 0, "decept_max": 4}, "death_prob"),
    ({"death_prob": 0.1, "decept_min": 3, "decept_max": 2}, "decept_min <= decept_max"),
    ({"death_prob": 0.1, "decept_min": 0, "decept_max": 9}, "decept_min <= decept_max"),
    ({"death_prob": 0.1, "decept_min": 2, "decept_max": 4}, "initial deceptive count"),
])
def test_evolution_validation_rejects_bad_values(evolution, message):
    d = _evo_dict(evolution=evolution, first_pool=[f"P{i}" for i in range(10)])
    with pytest.raises(ValueError, match=message):
        episode_from_dict(d)


def test_evolution_requires_both_spec_kinds():
    d = _evo_dict(evolution={"death_prob": 0.1, "decept_min": 0, "decept_max": 4},
                  agents=[{"count": 4, "system_prompt": "normal {id}"}],
                  first_pool=[f"P{i}" for i in range(10)])
    with pytest.raises(ValueError, match="deceptive"):
        episode_from_dict(d)


def test_evolution_requires_oversized_name_pool():
    # pool of exactly 4 names for 4 agents -> no headroom for replacements
    d = _evo_dict(evolution={"death_prob": 0.1, "decept_min": 0, "decept_max": 4},
                  first_pool=[f"P{i}" for i in range(4)])
    with pytest.raises(ValueError, match="strictly larger"):
        episode_from_dict(d)


def test_evolution_requires_a_name_pool():
    d = _evo_dict(evolution={"death_prob": 0.1, "decept_min": 0, "decept_max": 4})
    with pytest.raises(ValueError, match="name pool"):
        episode_from_dict(d)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_config_load.py -k evolution -v`
Expected: FAIL — `ImportError: cannot import name 'EvolutionCfg'` (or NameError).

- [ ] **Step 3: Implement in `src/core/config.py`**

Add after `AgentSpec` (keep its docstring style):

```python
@dataclass(frozen=True)
class EvolutionCfg:
    """Population turnover: per-round death and replacement (see docs/configuration.md).

    At the start of each round r >= 2 every agent dies with probability `death_prob` and is
    replaced by a fresh agent; the replacement is deceptive with probability d/N (d = current
    live deceptive count, N = population size), forced deceptive while d < decept_min and
    forced normal while d >= decept_max."""

    death_prob: float
    decept_min: int
    decept_max: int
```

Extend `AgentSpec` with a new last field:

```python
    deceptive: bool = False          # marks the spec whose agents count as deceptive for evolution
```

Extend `PopulationCfg` with a new last field:

```python
    evolution: EvolutionCfg | None = None   # None = no death/replacement (default)
```

In `_population_cfg`, add `deceptive=a.get("deceptive", False)` to the `AgentSpec(...)` call and pass `evolution=EvolutionCfg(**d["evolution"]) if d.get("evolution") else None` to `PopulationCfg(...)`.

In `_validate`, after the existing name-pool loop (which already computed `pop` and `total`), add:

```python
    evolution = pop.get("evolution")
    if evolution is not None:
        p = evolution.get("death_prob")
        if not isinstance(p, (int, float)) or isinstance(p, bool) or not (0 <= p <= 1):
            raise ValueError(f"evolution.death_prob must be a number in [0, 1], got: {p!r}")
        lo, hi = evolution.get("decept_min"), evolution.get("decept_max")
        for key, v in (("decept_min", lo), ("decept_max", hi)):
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise ValueError(f"evolution.{key} must be an integer >= 0, got: {v!r}")
        if not (lo <= hi <= total):
            raise ValueError(
                f"evolution requires decept_min <= decept_max <= agent count, "
                f"got: {lo} <= {hi} <= {total}")
        specs = pop["agents"]
        if not any(a.get("deceptive") for a in specs) or all(a.get("deceptive") for a in specs):
            raise ValueError(
                "evolution requires at least one deceptive and one non-deceptive agent spec")
        n_decept = sum(a.get("count", 1) for a in specs if a.get("deceptive"))
        if not (lo <= n_decept <= hi):
            raise ValueError(
                f"initial deceptive count ({n_decept}) is outside "
                f"[decept_min, decept_max] = [{lo}, {hi}]")
        pools = [pop.get(k, []) for k in ("first_name_pool", "last_name_pool")]
        if not any(pools):
            raise ValueError(
                "evolution requires a name pool: replacements need fresh names "
                "(the A1..An fallback is not supported)")
        for key, pool_names in zip(("first_name_pool", "last_name_pool"), pools):
            if pool_names and len(pool_names) <= total:
                raise ValueError(
                    f"evolution requires {key} strictly larger than the agent count "
                    f"({total}) — replacements draw unused names from it")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/core/test_config_load.py -v` (whole file — the old tests must still pass)
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/config.py tests/core/test_config_load.py
git commit -m "feat(config): EvolutionCfg block and deceptive agent-spec flag"
```

---

### Task 2: Population seam — `remove`, leftover name pool, `AgentSetup.deceptive`

**Files:**
- Modify: `src/core/agent.py` (`AgentSetup`, ~line 103)
- Modify: `src/population/base.py` (`Population`)
- Modify: `src/population/roster.py` (`RosterGenerator.build`, `_sample_names`)
- Test: `tests/population/test_roster.py` (append)

**Interfaces:**
- Consumes: `AgentSpec.deceptive` from Task 1.
- Produces: `AgentSetup.deceptive: bool = False` (new last field); `Population.remove(agent_id: str) -> None`; `Population.name_pool: list[str]` (ordered unused names, set by the generator); `Population.draw_name(rng) -> str` (pops `name_pool[rng.randrange(len(name_pool))]`; caller must check non-empty first). Task 3 relies on all four.

- [ ] **Step 1: Write the failing tests** — append to `tests/population/test_roster.py`:

```python
import random

import pytest

from src.core.agent import AgentSetup
from src.core.config import AgentSpec, PopulationCfg, ProviderCfg
from src.population import base as popbase
from src.population.roster import RosterGenerator


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
    pop = RosterGenerator(cfg).build(random.Random(0))
    assert [a.setup.deceptive for a in pop] == [False, False, True]
    # leftover pool = the unused names, in original pool order
    used = set(pop.ids())
    assert pop.name_pool == [n for n in cfg.first_name_pool if n not in used]
    assert len(pop.name_pool) == 5


def test_agent_setup_deceptive_defaults_false():
    setup = AgentSetup("prompt", ProviderCfg(base_url="http://x/v1", model="m"))
    assert setup.deceptive is False


def test_remove_drops_agent_from_roster(_stub_providers):
    pop = RosterGenerator(_pool_cfg()).build(random.Random(0))
    victim = pop.ids()[1]
    pop.remove(victim)
    assert victim not in pop.ids() and len(pop) == 2
    with pytest.raises(KeyError):
        pop.get(victim)


def test_draw_name_pops_from_leftover_pool(_stub_providers):
    pop = RosterGenerator(_pool_cfg()).build(random.Random(0))
    before = list(pop.name_pool)
    name = pop.draw_name(random.Random(7))
    assert name in before and name not in pop.name_pool
    assert len(pop.name_pool) == len(before) - 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/population/test_roster.py -v`
Expected: new tests FAIL (`deceptive` unexpected keyword / missing attribute `name_pool` / `remove`); old tests PASS.

- [ ] **Step 3: Implement**

`src/core/agent.py` — add a last field to `AgentSetup`:

```python
    deceptive: bool = False              # evolution: this agent counts as deceptive (see EvolutionCfg)
```

`src/population/base.py` — in `Population.__init__` add:

```python
        self.name_pool: list[str] = []   # unused replacement names (set by the generator; evolution draws from it)
```

and add methods (this fulfils the "evolution mutators are a documented seam" note in the class docstring — update that sentence to say the seam is now implemented by `remove`):

```python
    def remove(self, agent_id: str) -> None:
        """Remove a live agent from the roster (its id is never reused)."""
        agent = self._by_id.pop(agent_id)
        self._agents.remove(agent)

    def draw_name(self, rng) -> str:
        """Pop an unused name from the leftover pool at an rng-chosen index.

        The caller must ensure the pool is non-empty (evolution raises its own
        error with round context before calling this on an empty pool)."""
        return self.name_pool.pop(rng.randrange(len(self.name_pool)))
```

`src/population/roster.py` — `_sample_names` now also returns the ordered leftovers, and `build` stores them and passes `deceptive`:

```python
def build(self, rng) -> Population:
    ...
    pop = Population(context_window=self._window)
    names, leftover = _sample_names(self._cfg, rng)
    pop.name_pool = leftover
    i = 0
    for spec in self._cfg.agents:
        for _ in range(spec.count):
            pop.add(AgentSetup(spec.system_prompt, self._cfg.provider,
                               spec.play_strategy, spec.prediction_mapping,
                               deceptive=spec.deceptive),
                    agent_id=names[i])
            i += 1
    return pop
```

```python
def _sample_names(cfg, rng) -> tuple[list[str | None], list[str]]:
    """Sample unique ids from the name pools; also return the unused leftovers.

    (Keep the existing three-mode docstring and add:) The second element is the
    leftover pool in original pool order — evolution draws replacement names from
    it. Empty pools give an empty leftover (evolution is rejected at validation)."""
    total = sum(spec.count for spec in cfg.agents)
    firsts, lasts = cfg.first_name_pool, cfg.last_name_pool
    if firsts and lasts:
        f = rng.sample(firsts, total)
        l = rng.sample(lasts, total)
        rem_f = [x for x in firsts if x not in set(f)]
        rem_l = [x for x in lasts if x not in set(l)]
        return ([f"{a} {b}" for a, b in zip(f, l)],
                [f"{a} {b}" for a, b in zip(rem_f, rem_l)])
    if firsts or lasts:
        pool = firsts or lasts
        used = rng.sample(pool, total)
        taken = set(used)
        return [str(x) for x in used], [str(x) for x in pool if x not in taken]
    return [None] * total, []
```

Update the two existing call sites of `_sample_names`'s return value if any test helpers unpack it (only `build` calls it in `src/`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/population/ tests/core/test_agent.py -v`
Expected: PASS (including pre-existing roster/agent tests).

- [ ] **Step 5: Commit**

```bash
git add src/core/agent.py src/population/base.py src/population/roster.py tests/population/test_roster.py
git commit -m "feat(population): remove/draw_name seam, leftover name pool, deceptive setup flag"
```

---

### Task 3: Evolution step — `src/population/evolution.py`

**Files:**
- Create: `src/population/evolution.py`
- Test: `tests/population/test_evolution.py` (new)

**Interfaces:**
- Consumes: `Population.remove/draw_name/name_pool/add` (Task 2), `AgentSetup(..., deceptive=)` (Task 2), `PopulationCfg.evolution` + `AgentSpec.deceptive` (Task 1).
- Produces: `evolve(pop: Population, pop_cfg: PopulationCfg, rng: random.Random, round: int) -> list[dict]` and `class NamePoolExhausted(RuntimeError)`. Event shapes (relied on by Tasks 4–7):
  - `{"type": "death", "agent": <id>, "score": <float>}`
  - `{"type": "birth", "agent": <id>, "deceptive": <bool>, "system_prompt": <str>, "provider": <dict>}`

- [ ] **Step 1: Write the failing tests** — create `tests/population/test_evolution.py`:

```python
from __future__ import annotations

import random

import pytest

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/population/test_evolution.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.population.evolution'`.

- [ ] **Step 3: Implement** — create `src/population/evolution.py`:

```python
"""Population evolution step: random death and replacement between rounds.

Deterministic given (population state, rng): the orchestrator derives a dedicated
rng stream per round (Random(f"{seed}:evolution:{r}")), so resume can re-derive
every roster change without replaying LLM calls. Rng consumption order is a
compatibility contract: one random() per live agent in roster order (deaths),
then per replacement at most one random() (type) plus one randrange() (name).
"""

from __future__ import annotations

from dataclasses import asdict

from src.core.agent import AgentSetup
from src.population.base import Population


class NamePoolExhausted(RuntimeError):
    """The name pool has no unused names left for a replacement agent."""


def evolve(pop: Population, pop_cfg, rng, round: int) -> list[dict]:
    """Kill each agent with probability death_prob and spawn a replacement per death.

    The replacement is deceptive with probability d/N (d = current live deceptive
    count, N = fixed population size), forced deceptive while d < decept_min and
    forced normal while d >= decept_max. It clones the first matching spec
    (system prompt, strategy), starts with a fresh memory and score 0, and takes
    an unused name from the population's leftover pool.

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
    deaths = [a for a in list(pop) if rng.random() < ev.death_prob]
    events: list[dict] = []
    for agent in deaths:
        pop.remove(agent.id)
        events.append({"type": "death", "agent": agent.id, "score": agent.score})
    for _ in deaths:
        d = sum(1 for a in pop if a.setup.deceptive)
        if d < ev.decept_min:
            deceptive = True
        elif d >= ev.decept_max:
            deceptive = False
        else:
            deceptive = rng.random() < d / n_total
        spec = next(s for s in pop_cfg.agents if s.deceptive == deceptive)
        if not pop.name_pool:
            raise NamePoolExhausted(
                f"round {round}: name pool exhausted — enlarge the name pools; "
                f"this run cannot be resumed past this point")
        name = pop.draw_name(rng)
        pop.add(AgentSetup(spec.system_prompt, pop_cfg.provider,
                           spec.play_strategy, spec.prediction_mapping,
                           deceptive=spec.deceptive),
                agent_id=name)
        events.append({"type": "birth", "agent": name, "deceptive": deceptive,
                       "system_prompt": spec.system_prompt,
                       "provider": asdict(pop_cfg.provider)})
    return events
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/population/ -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/population/evolution.py tests/population/test_evolution.py
git commit -m "feat(population): deterministic death/replacement evolution step"
```

---

### Task 4: Orchestrator integration

**Files:**
- Modify: `src/core/orchestrator.py` (`run_episode` loop, ~lines 55–63)
- Test: `tests/core/test_orchestrator.py` (append)

**Interfaces:**
- Consumes: `evolve` + `NamePoolExhausted` (Task 3), `cfg.population.evolution` (Task 1).
- Produces: `run_episode` applies evolution at the start of each round r ≥ 2 with `random.Random(f"{cfg.seed}:evolution:{r}")` and prepends its events to `RoundPlan.events` before calling the observer. Tasks 5–7 rely on those events reaching the observer.

- [ ] **Step 1: Write the failing tests** — append to `tests/core/test_orchestrator.py` (add `EvolutionCfg` to the existing `src.core.config` import):

```python
def _evo_cfg(rounds=3, seed=0):
    return EpisodeCfg(
        seed=seed, rounds=rounds, matchmaker="random",
        population=PopulationCfg(
            kind="roster",
            agents=[AgentSpec(count=3, system_prompt="normal {id}"),
                    AgentSpec(count=1, system_prompt="defect {id}", deceptive=True)],
            provider=ProviderCfg(base_url="http://x/v1", model="m"),
            first_name_pool=[f"Player {i}" for i in range(40)],
            evolution=EvolutionCfg(death_prob=1.0, decept_min=0, decept_max=4),
        ),
        game=GameCfg(max_talk_turns=0),
    )


async def test_evolution_replaces_agents_between_rounds(providers):
    plans = {}
    await _run(_evo_cfg(), observer=lambda r, p, recs: plans.__setitem__(r, p))
    assert plans[1].events == []                        # never before round 1
    deaths = [e for e in plans[2].events if e["type"] == "death"]
    births = [e for e in plans[2].events if e["type"] == "birth"]
    assert len(deaths) == 4 and len(births) == 4        # death_prob=1.0 -> full turnover
    round1 = {x for pair in plans[1].pairings for x in pair} | set(plans[1].idle)
    round2 = {x for pair in plans[2].pairings for x in pair} | set(plans[2].idle)
    assert round1.isdisjoint(round2)                    # everyone was replaced


async def test_no_evolution_events_without_config(providers):
    plans = {}
    await _run(_cfg(n=4, rounds=2), observer=lambda r, p, recs: plans.__setitem__(r, p))
    assert plans[1].events == [] and plans[2].events == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_orchestrator.py -k evolution -v`
Expected: `test_evolution_replaces_agents_between_rounds` FAILS (events empty, rosters identical); the no-config test passes already.

- [ ] **Step 3: Implement** — in `src/core/orchestrator.py`:

Add imports: `from dataclasses import replace` and `from src.population.evolution import evolve`.

In the round loop, between `game = ReputationPD(cfg_r.game)` and the matchmaker rng line, insert:

```python
        ev_events: list[dict] = []
        if r >= 2 and cfg.population.evolution is not None:
            # dedicated per-round stream, like the matchmaker: resume re-derives it (see evolve)
            ev_events = evolve(pop, cfg.population,
                               random.Random(f"{cfg.seed}:evolution:{r}"), r)
```

and after `plan = await mm.plan_round(...)`:

```python
        if ev_events:
            plan = replace(plan, events=ev_events + plan.events)
```

Also extend the `run_episode` docstring with one sentence: with `population.evolution` set, a death/replacement step runs before pairing each round r ≥ 2 and its events are prepended to `RoundPlan.events`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/core/ -v`
Expected: PASS (all pre-existing orchestrator/schedule tests included).

- [ ] **Step 5: Commit**

```bash
git add src/core/orchestrator.py tests/core/test_orchestrator.py
git commit -m "feat(orchestrator): evolution step at round start, events via RoundPlan"
```

---

### Task 5: Storage — schema migration, deceptive on insert, event persistence

**Files:**
- Modify: `src/storage/schema.py` (agents CREATE TABLE + `init_schema`)
- Modify: `src/storage/store.py` (`begin` ~line 96, `observe` ~line 247)
- Test: `tests/storage/test_storage.py` (append)

**Interfaces:**
- Consumes: event dict shapes from Task 3; `a.setup.deceptive` from Task 2.
- Produces: `agents` columns `born_round INTEGER NOT NULL DEFAULT 1`, `died_round INTEGER` (NULL = alive), `deceptive INTEGER NOT NULL DEFAULT 0`; `Storage.observe` inserts newborn rows and stamps the dead. Task 7 (replay) queries these columns.

- [ ] **Step 1: Write the failing tests** — append to `tests/storage/test_storage.py` (add to its imports if missing: `import sqlite3`, `import random`, `from src.core.config import AgentSpec, EpisodeCfg, GameCfg, PopulationCfg, ProviderCfg`, `from src.matchmaking import RoundPlan`, `from src.population import base as popbase, make_population`, `from src.storage import Storage`, `import pytest`):

```python
class _EvoStubProvider:
    def __init__(self, cfg):
        self.cfg = cfg

    async def complete(self, **kw):
        raise AssertionError("storage tests must not call the LLM")

    async def aclose(self):
        pass


@pytest.fixture
def _evo_stub_providers(monkeypatch):
    monkeypatch.setattr(popbase, "make_provider", lambda cfg: _EvoStubProvider(cfg))


def _evo_storage_cfg():
    return EpisodeCfg(
        seed=0, rounds=2, matchmaker="random",
        population=PopulationCfg(
            kind="roster",
            agents=[AgentSpec(count=1, system_prompt="normal {id}"),
                    AgentSpec(count=1, system_prompt="defect {id}", deceptive=True)],
            provider=ProviderCfg(base_url="http://x/v1", model="m")),
        game=GameCfg(max_talk_turns=0))


def test_begin_records_deceptive_flag(tmp_path, _evo_stub_providers):
    st = Storage(str(tmp_path / "t.db"))
    cfg = _evo_storage_cfg()
    pop = make_population(cfg.population).build(random.Random(0))
    rid = st.begin(cfg, pop)
    rows = dict(st.conn.execute(
        "SELECT agent_id, deceptive FROM agents WHERE run_id=?", (rid,)))
    assert rows == {"A1": 0, "A2": 1}
    st.close()


def test_observe_persists_evolution_events(tmp_path, _evo_stub_providers):
    st = Storage(str(tmp_path / "t.db"))
    cfg = _evo_storage_cfg()
    pop = make_population(cfg.population).build(random.Random(0))
    rid = st.begin(cfg, pop)
    plan = RoundPlan(pairings=[], idle=[], events=[
        {"type": "death", "agent": "A1", "score": 5.0},
        {"type": "birth", "agent": "Player 9", "deceptive": True,
         "system_prompt": "defect {id}",
         "provider": {"base_url": "http://x/v1", "model": "m"}},
    ])
    st.observe(2, plan, [])
    assert st.conn.execute(
        "SELECT died_round, final_score FROM agents WHERE run_id=? AND agent_id='A1'",
        (rid,)).fetchone() == (2, 5.0)
    assert st.conn.execute(
        "SELECT born_round, deceptive, system_prompt FROM agents "
        "WHERE run_id=? AND agent_id='Player 9'", (rid,)).fetchone() == (2, 1, "defect {id}")
    st.close()


def test_migration_adds_evolution_columns_to_old_db(tmp_path):
    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE agents (
               run_id INTEGER NOT NULL, agent_id TEXT NOT NULL,
               system_prompt TEXT, provider TEXT NOT NULL, final_score REAL,
               PRIMARY KEY (run_id, agent_id))""")
    conn.commit()
    conn.close()
    st = Storage(path)
    cols = {row[1] for row in st.conn.execute("PRAGMA table_info(agents)")}
    assert {"born_round", "died_round", "deceptive"} <= cols
    st.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/storage/test_storage.py -v`
Expected: new tests FAIL (`no such column: deceptive` / `died_round`); old tests PASS.

- [ ] **Step 3: Implement**

`src/storage/schema.py` — extend the agents CREATE TABLE (new DBs get the columns directly):

```sql
CREATE TABLE IF NOT EXISTS agents (
    run_id        INTEGER NOT NULL,
    agent_id      TEXT NOT NULL,
    system_prompt TEXT,
    provider      TEXT NOT NULL,
    final_score   REAL,
    born_round    INTEGER NOT NULL DEFAULT 1,  -- evolution: round whose start spawned this agent
    died_round    INTEGER,                     -- evolution: round whose start killed it; NULL = alive
    deceptive     INTEGER NOT NULL DEFAULT 0,  -- evolution: spec's deceptive flag (analysis only)
    PRIMARY KEY (run_id, agent_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);
```

and migrate existing DBs in `init_schema` (executescript skips the existing table):

```python
def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    # additive migration for DBs created before the evolution columns existed
    cols = {row[1] for row in conn.execute("PRAGMA table_info(agents)")}
    for name, ddl in (("born_round", "INTEGER NOT NULL DEFAULT 1"),
                      ("died_round", "INTEGER"),
                      ("deceptive", "INTEGER NOT NULL DEFAULT 0")):
        if name not in cols:
            conn.execute(f"ALTER TABLE agents ADD COLUMN {name} {ddl}")
```

`src/storage/store.py` — in `begin`, change the agents insert to include the flag:

```python
            self._conn.executemany(
                "INSERT INTO agents(run_id, agent_id, system_prompt, provider, deceptive) "
                "VALUES (?,?,?,?,?)",
                [
                    (run_id, a.id, a.setup.system_prompt,
                     json.dumps(asdict(a.setup.provider_cfg)), int(a.setup.deceptive))
                    for a in pop
                ],
            )
```

In `observe`, right after the rounds-row insert inside the transaction, add:

```python
            for e in plan.events:                       # evolution: deaths/births at round start
                if e.get("type") == "death":
                    self._conn.execute(
                        "UPDATE agents SET died_round=?, final_score=? "
                        "WHERE run_id=? AND agent_id=?",
                        (round, e.get("score"), rid, e["agent"]))
                elif e.get("type") == "birth":
                    self._conn.execute(
                        "INSERT INTO agents(run_id, agent_id, system_prompt, provider, "
                        "born_round, deceptive) VALUES (?,?,?,?,?,?)",
                        (rid, e["agent"], e.get("system_prompt"),
                         json.dumps(e.get("provider")), round,
                         int(bool(e.get("deceptive")))))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/storage/ -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/storage/schema.py src/storage/store.py tests/storage/test_storage.py
git commit -m "feat(storage): persist evolution events; born/died/deceptive agent columns"
```

---

### Task 6: Runner — narration, resume replay, tolerant state apply

**Files:**
- Modify: `src/runner.py` (`narrate_round` ~line 26, `_apply_run_state` ~line 129, `resume_run` ~lines 175–186)
- Test: `tests/test_runner.py` (append)

**Interfaces:**
- Consumes: `evolve` (Task 3), events in `plan.events` (Task 4), stored rows (Task 5).
- Produces: live narration lines `"† <id> died (score <s>)"` / `"+ <id> joined the game"`; `resume_run` re-derives the evolved roster before applying state. Task 7's replay output mirrors the same wording.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_runner.py` (extend its `src.core.config` import with `EvolutionCfg`; add `import random`, `import sqlite3`, `from src.population import make_population`, `from src.population.evolution import evolve`):

```python
def _evo_runner_cfg(rounds=2, seed=0):
    return EpisodeCfg(
        seed=seed, rounds=rounds, matchmaker="random",
        population=PopulationCfg(
            kind="roster",
            agents=[AgentSpec(count=3, system_prompt="normal {id}"),
                    AgentSpec(count=1, system_prompt="defect {id}", deceptive=True)],
            provider=ProviderCfg(base_url="http://x/v1", model="m"),
            first_name_pool=[f"Player {i}" for i in range(40)],
            evolution=EvolutionCfg(death_prob=1.0, decept_min=0, decept_max=4)),
        game=GameCfg(max_talk_turns=0))


async def test_narration_prints_deaths_and_births(tmp_path, capsys):
    db = str(tmp_path / "t.db")
    await runner.run_experiment(_evo_runner_cfg(), db)
    out = capsys.readouterr().out
    assert "died" in out and "joined the game" in out


async def test_resume_replays_evolution_deterministically(tmp_path):
    db = str(tmp_path / "t.db")
    cfg = _evo_runner_cfg(rounds=2)
    rid = await runner.run_experiment(cfg, db, quiet=True)
    await runner.resume_run(rid, db, rounds=4, quiet=True)
    # expected live roster after round 4 = fresh build + evolution replay for rounds 2..4
    pop = make_population(cfg.population).build(random.Random(cfg.seed))
    try:
        for r in range(2, 5):
            evolve(pop, cfg.population, random.Random(f"{cfg.seed}:evolution:{r}"), r)
        expected = set(pop.ids())
    finally:
        await pop.aclose()
    conn = sqlite3.connect(db)
    try:
        ids = set()
        for a, b in conn.execute(
                "SELECT a_id, b_id FROM pairings WHERE run_id=? AND round_idx=4", (rid,)):
            ids |= {a, b}
        for (aid,) in conn.execute(
                "SELECT agent_id FROM idle WHERE run_id=? AND round_idx=4", (rid,)):
            ids.add(aid)
        assert ids == expected
    finally:
        conn.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_runner.py -k "evolution or deaths" -v`
Expected: narration test FAILS (no death lines); resume test FAILS (round-4 ids don't match — resume did not replay evolution, or KeyError in `_apply_run_state`).

- [ ] **Step 3: Implement** — in `src/runner.py`:

Add import: `from src.population.evolution import evolve`.

`narrate_round` — right after the `ROUND {r}` header print, before the idle line:

```python
    for e in plan.events:                              # evolution events happened at round start
        if e.get("type") == "death":
            print(f"  † {e['agent']} died (score {e['score']:g})")
        elif e.get("type") == "birth":
            mark = " (deceptive)" if e.get("deceptive") else ""
            print(f"  + {e['agent']} joined the game{mark}")
```

`_apply_run_state` — tolerate ids without stored state (dead agents' state is simply not applied because they are not in `pop`; a just-born agent may have no rows yet):

```python
    for agent in pop:
        agent.score = state.scores.get(agent.id, 0.0)
        if agent.id in state.memories:
            agent.memory = state.memories[agent.id]
```

Update its docstring: ids matched by deterministic rebuild + evolution replay; missing ids keep fresh state.

`resume_run` — compute `start` BEFORE applying state, replay evolution, then apply (replace the current two lines `_apply_run_state(pop, state)` / `start = state.last_round + 1`):

```python
    start = state.last_round + 1
    if cfg.population.evolution is not None:
        # re-derive roster changes for already-played rounds; round `start` itself
        # is evolved by run_episode (it evolves every round it plays)
        for r in range(2, start):
            evolve(pop, cfg.population, random.Random(f"{cfg.seed}:evolution:{r}"), r)
    _apply_run_state(pop, state)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_runner.py -v`
Expected: PASS (including all pre-existing runner tests).

- [ ] **Step 5: Commit**

```bash
git add src/runner.py tests/test_runner.py
git commit -m "feat(runner): narrate evolution events; replay them on resume"
```

---

### Task 7: Replay — show deaths/births from stored runs

**Files:**
- Modify: `replay.py` (round loop, ~line 340)
- Test: `tests/test_replay.py` (append)

**Interfaces:**
- Consumes: `agents.born_round/died_round/deceptive` columns (Task 5).
- Produces: replay prints the same `"† ... died"` / `"+ ... joined the game"` lines as live narration. No API for later tasks.

- [ ] **Step 1: Write the failing test** — append to `tests/test_replay.py` (self-contained: its own stub-provider fixture and config helper; add imports `import random`, `import sqlite3`, `import pytest`, `import replay as replay_mod`, `from src import runner`, `from src.core.config import AgentSpec, EpisodeCfg, EvolutionCfg, GameCfg, PopulationCfg, ProviderCfg`, `from src.population import base as popbase`, `from src.providers.base import Completion` — skip any already present):

```python
class _EvoFixedProvider:
    def __init__(self, cfg):
        self.cfg = cfg

    async def complete(self, **kw):
        return Completion(text='{"number": 4, "rationale": "r"}',
                          prompt_tokens=2, completion_tokens=3, raw={})

    async def aclose(self):
        pass


@pytest.fixture
def _evo_providers(monkeypatch):
    monkeypatch.setattr(popbase, "make_provider", lambda cfg: _EvoFixedProvider(cfg))


def _evo_replay_cfg():
    return EpisodeCfg(
        seed=0, rounds=2, matchmaker="random",
        population=PopulationCfg(
            kind="roster",
            agents=[AgentSpec(count=3, system_prompt="normal {id}"),
                    AgentSpec(count=1, system_prompt="defect {id}", deceptive=True)],
            provider=ProviderCfg(base_url="http://x/v1", model="m"),
            first_name_pool=[f"Player {i}" for i in range(40)],
            evolution=EvolutionCfg(death_prob=1.0, decept_min=0, decept_max=4)),
        game=GameCfg(max_talk_turns=0))


async def test_replay_shows_evolution_events(tmp_path, capsys, _evo_providers):
    db = str(tmp_path / "t.db")
    rid = await runner.run_experiment(_evo_replay_cfg(), db, quiet=True)
    capsys.readouterr()                                 # drop the run's own output
    conn = sqlite3.connect(db)
    try:
        replay_mod.replay(conn, rid)
    finally:
        conn.close()
    out = capsys.readouterr().out
    assert "died" in out and "joined the game" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_replay.py -k evolution -v`
Expected: FAIL — replay output has no death/birth lines.

- [ ] **Step 3: Implement** — in `replay.py`, inside `replay(...)`: before the `for r in rounds:` loop compute (replay opens raw sqlite3 connections, so old DBs may lack the columns — guard):

```python
    has_evolution = any(
        row[1] == "died_round" for row in conn.execute("PRAGMA table_info(agents)")
    )
```

and immediately after the `ROUND {r}` header print inside the loop:

```python
        if has_evolution:
            for aid, score in conn.execute(
                    "SELECT agent_id, final_score FROM agents "
                    "WHERE run_id=? AND died_round=? ORDER BY agent_id", (run_id, r)):
                print(f"  † {aid} died (score {(score or 0):g})")
            for aid, dec in conn.execute(
                    "SELECT agent_id, deceptive FROM agents "
                    "WHERE run_id=? AND born_round=? ORDER BY agent_id", (run_id, r)):
                print(f"  + {aid} joined the game{' (deceptive)' if dec else ''}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_replay.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add replay.py tests/test_replay.py
git commit -m "feat(replay): show evolution deaths/births per round"
```

---

### Task 8: Documentation

**Files:**
- Modify: `docs/configuration.md`, `docs/architecture.md`, `docs/database.md`, `CLAUDE.md` (project map)

No test cycle — verification is the full suite plus a docs read-through.

- [ ] **Step 1: `docs/configuration.md`** — document the new fields in the population section:
  - `population.agents[*].deceptive: bool` (default false) — marks the deceptive spec.
  - `population.evolution: {death_prob, decept_min, decept_max}` block with the exact sampling rule from the spec (round start r ≥ 2, forced types outside `[decept_min, decept_max]`, probability `d/N` otherwise) and a YAML example mirroring `config/research.yaml`'s two-spec layout.
  - Requirements: at least one deceptive and one normal spec; initial deceptive count within bounds; each provided name pool strictly larger than the agent count (replacements draw unused names; choose a generously oversized pool — a run whose pool runs dry aborts and cannot be resumed past that point, because resume replays the config stored in the DB).

- [ ] **Step 2: `docs/architecture.md`** — in the orchestrator/round-loop section, add the evolution step: dedicated rng stream `Random(f"{seed}:evolution:{r}")` before pairing, events prepended to `RoundPlan.events` (still the single output channel), resume re-derives roster changes deterministically (runner replays rounds `2..start_round-1`, `run_episode` evolves each round it plays). Note the rng consumption-order contract from `src/population/evolution.py`.

- [ ] **Step 3: `docs/database.md`** — document the three new `agents` columns (`born_round` default 1, `died_round` NULL = alive, `deceptive`), the additive `ALTER TABLE` migration on `Storage` init, and that they are analysis-only (resume never reads them). Add one example query: deceptive share over time / who died when.

- [ ] **Step 4: `CLAUDE.md`** — in the project map, extend the `population/` line: `Population (live roster, provider cache, remove/draw_name) + RosterGenerator + evolution.py (death/replacement step)`.

- [ ] **Step 5: Run the full suite and commit**

Run: `uv run pytest`
Expected: all tests PASS.

```bash
git add docs/configuration.md docs/architecture.md docs/database.md CLAUDE.md
git commit -m "docs: population evolution (death/replacement) reference"
```

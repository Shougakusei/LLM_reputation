# `evolution.replacement: inherit` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second population-evolution replacement policy, `inherit`, where a newborn agent clones the dying agent's full role instead of rolling a type via the d/N rule.

**Architecture:** A new `replacement: "roll" | "inherit"` field on the frozen `EvolutionCfg` dataclass (default `"roll"` = today's behavior). `evolve()` branches on it: `inherit` copies the dying agent's setup (system prompt, strategy, mapping, deceptive flag) and consumes no type `random()` draw. `_hash_config_dict` strips the new default-valued keys so every existing config keeps its `config_hash`. Spec: `docs/superpowers/specs/2026-08-07-evolution-inherit-replacement-design.md`.

**Tech Stack:** Python 3.12, frozen dataclasses, pytest (asyncio auto mode, `pythonpath=["."]`).

## Global Constraints

- All code text — docstrings (Google-style), comments, error messages, test names — in **English** (project override of the user's global Russian rule).
- TDD: write the failing test first, run it to see it fail, then minimal implementation.
- Dependencies only via `uv`; run tests with `uv run pytest ...`. No new dependencies are needed.
- No printing/persistence inside `src/`.
- `from __future__ import annotations` is already at the top of every touched module — keep it.
- RNG compatibility contract (`src/population/evolution.py` module docstring): one `random()` per live agent in roster order, then per replacement **at most** one `random()` (type, roll mode only) plus one `randrange()` (name). Never reorder draws.
- Commit after each task with a conventional-commit message.

---

### Task 1: Config — `replacement` field, optional `decept_min`/`decept_max`, validation

**Files:**
- Modify: `src/core/config.py` (EvolutionCfg ~line 327, `_validate` evolution block ~lines 506–540)
- Test: `tests/core/test_config_load.py` (evolution tests live after `_evo_dict`, ~line 786)

**Interfaces:**
- Consumes: nothing new.
- Produces: `EvolutionCfg(death_prob: float, decept_min: int | None = None, decept_max: int | None = None, replacement: str = "roll")`. Tasks 2–3 rely on `EvolutionCfg.replacement` and on `decept_min`/`decept_max` accepting `None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/core/test_config_load.py` (after `test_evolution_requires_a_name_pool`, before `test_invincible_flag_parses_per_spec`; reuse the existing `_evo_dict` helper at line 786):

```python
def test_replacement_defaults_to_roll():
    d = _evo_dict(evolution={"death_prob": 0.1, "decept_min": 1, "decept_max": 3},
                  first_pool=[f"P{i}" for i in range(10)])
    cfg = episode_from_dict(d)
    assert cfg.population.evolution.replacement == "roll"


def test_replacement_inherit_parses_without_decept_bounds():
    d = _evo_dict(evolution={"death_prob": 0.1, "replacement": "inherit"},
                  first_pool=[f"P{i}" for i in range(10)])
    cfg = episode_from_dict(d)
    assert cfg.population.evolution == EvolutionCfg(
        death_prob=0.1, decept_min=None, decept_max=None, replacement="inherit")


def test_replacement_rejects_unknown_value():
    d = _evo_dict(evolution={"death_prob": 0.1, "replacement": "mutate"},
                  first_pool=[f"P{i}" for i in range(10)])
    with pytest.raises(ValueError, match="replacement"):
        episode_from_dict(d)


def test_roll_mode_still_requires_decept_bounds():
    d = _evo_dict(evolution={"death_prob": 0.1},
                  first_pool=[f"P{i}" for i in range(10)])
    with pytest.raises(ValueError, match="decept_min"):
        episode_from_dict(d)


def test_inherit_mode_allows_single_flag_population():
    # No deceptive spec at all: fine under inherit (pure turnover), rejected under roll.
    d = _evo_dict(evolution={"death_prob": 0.1, "replacement": "inherit"},
                  agents=[{"count": 4, "system_prompt": "normal {id}"}],
                  first_pool=[f"P{i}" for i in range(10)])
    cfg = episode_from_dict(d)
    assert cfg.population.evolution.replacement == "inherit"


def test_inherit_mode_ignores_decept_bounds_when_present():
    # decept_min/decept_max may be present but are ignored: values that roll mode
    # would reject (initial deceptive count 1 outside [3, 3]) load fine.
    d = _evo_dict(evolution={"death_prob": 0.1, "replacement": "inherit",
                             "decept_min": 3, "decept_max": 3},
                  first_pool=[f"P{i}" for i in range(10)])
    cfg = episode_from_dict(d)
    assert cfg.population.evolution.decept_min == 3


def test_inherit_survives_asdict_roundtrip():
    from dataclasses import asdict
    d = _evo_dict(evolution={"death_prob": 0.5, "replacement": "inherit"},
                  first_pool=[f"P{i}" for i in range(10)])
    cfg = episode_from_dict(d)
    again = episode_from_dict(asdict(cfg))
    assert again.population.evolution == cfg.population.evolution
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/core/test_config_load.py -k "replacement or inherit or roll_mode" -v`

Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'replacement'` for the parse tests; `test_roll_mode_still_requires_decept_bounds` fails because omitting the bounds currently raises with message "must be an integer" (match on "decept_min" may already pass — that is acceptable; the test locks the behavior in) and `test_replacement_rejects_unknown_value` fails with TypeError instead of ValueError.

- [ ] **Step 3: Implement the config change**

In `src/core/config.py`, replace the `EvolutionCfg` dataclass:

```python
@dataclass(frozen=True)
class EvolutionCfg:
    """Population turnover: per-round death and replacement (see docs/configuration.md).

    At the start of each round r >= 2 every agent dies with probability `death_prob` and is
    replaced by a fresh agent. How the replacement's role is chosen depends on `replacement`:
      "roll"    — deceptive with probability d/N (d = current live deceptive count,
                  N = population size), forced deceptive while d < decept_min and forced
                  normal while d >= decept_max; clones the first spec matching the flag.
      "inherit" — clones the dying agent's full setup (system prompt, strategy, mapping,
                  deceptive flag); decept_min/decept_max are ignored and may be omitted.
    """

    death_prob: float
    decept_min: int | None = None    # required (int) in roll mode; ignored in inherit mode
    decept_max: int | None = None    # required (int) in roll mode; ignored in inherit mode
    replacement: str = "roll"        # "roll" | "inherit"
```

In `_validate`, restructure the evolution block (currently lines 506–540). The `death_prob`, mortal-agent, and name-pool checks apply in both modes; everything decept-related becomes roll-only:

```python
    evolution = pop.get("evolution")
    if evolution is not None:
        p = evolution.get("death_prob")
        if not isinstance(p, (int, float)) or isinstance(p, bool) or not (0 <= p <= 1):
            raise ValueError(f"evolution.death_prob must be a number in [0, 1], got: {p!r}")
        mode = evolution.get("replacement", "roll")
        if mode not in ("roll", "inherit"):
            raise ValueError(
                f"evolution.replacement must be 'roll' or 'inherit', got: {mode!r}")
        specs = pop["agents"]
        if mode == "roll":
            lo, hi = evolution.get("decept_min"), evolution.get("decept_max")
            for key, v in (("decept_min", lo), ("decept_max", hi)):
                if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                    raise ValueError(f"evolution.{key} must be an integer >= 0, got: {v!r}")
            if not (lo <= hi <= total):
                raise ValueError(
                    f"evolution requires decept_min <= decept_max <= agent count, "
                    f"got: {lo} <= {hi} <= {total}")
            if not any(a.get("deceptive") for a in specs) or all(a.get("deceptive") for a in specs):
                raise ValueError(
                    "evolution requires at least one deceptive and one non-deceptive agent spec")
            n_decept = sum(a.get("count", 1) for a in specs if a.get("deceptive"))
            if not (lo <= n_decept <= hi):
                raise ValueError(
                    f"initial deceptive count ({n_decept}) is outside "
                    f"[decept_min, decept_max] = [{lo}, {hi}]")
        if all(a.get("invincible") for a in specs):
            raise ValueError(
                "evolution requires at least one mortal (non-invincible) agent")
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

Note the only structural changes vs. today: the `mode` check is new, `specs` is hoisted above the roll-only block, and the four decept checks are indented under `if mode == "roll":`. No check text changes, so existing parametrized message-match tests keep passing.

`_population_cfg` needs no change: `EvolutionCfg(**d["evolution"])` picks up the new keyword with its default.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/core/test_config_load.py -v`
Expected: all PASS (the new tests plus every pre-existing evolution test).

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest`
Expected: PASS. (`tests/population/test_evolution.py` still passes — it constructs `EvolutionCfg` with explicit `decept_min`/`decept_max` keywords.)

- [ ] **Step 6: Commit**

```bash
git add src/core/config.py tests/core/test_config_load.py
git commit -m "feat(config): evolution.replacement roll|inherit, decept bounds optional in inherit mode"
```

---

### Task 2: `evolve()` — inherit branch

**Files:**
- Modify: `src/population/evolution.py` (module docstring, `evolve` docstring, replacement loop lines 57–78)
- Test: `tests/population/test_evolution.py`

**Interfaces:**
- Consumes: `EvolutionCfg.replacement` from Task 1.
- Produces: no new public API — `evolve(pop, pop_cfg, rng, round) -> list[dict]` keeps its signature and event shapes (`{"type": "death", "agent", "score"}` / `{"type": "birth", "agent", "deceptive", "system_prompt", "provider"}`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/population/test_evolution.py` (reuse `_StubProvider`, `FakeRng`, `_build`; note `_pop_cfg` builds 3 normal + 1 deceptive agents):

```python
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


def test_inherit_full_turnover_preserves_type_composition():
    cfg = _inherit_cfg(death_prob=1.0)
    pop = _build(cfg)
    before = sorted((a.setup.system_prompt, a.setup.deceptive) for a in pop)
    evolve(pop, cfg, random.Random(0), round=2)
    after = sorted((a.setup.system_prompt, a.setup.deceptive) for a in pop)
    assert after == before                                   # roles frozen, names changed


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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/population/test_evolution.py -k inherit -v`
Expected: FAIL — the current roll logic draws a type `random()` (IndexError from `FakeRng` in the exact-count tests) and clones the first matching spec, so `test_inherit_preserves_strategy_and_mapping` gets `play_strategy == "direct"`. (`test_inherit_...` failures may also surface as `TypeError` on `ev.decept_min` being `None` — comparing `d < None`.)

- [ ] **Step 3: Implement the inherit branch**

In `src/population/evolution.py`, replace the replacement loop (`for _ in deaths:` ... end of function) so each birth is paired with its dead agent:

```python
    for dead in deaths:
        if ev.replacement == "inherit":
            # the newborn takes the dying agent's role verbatim; no type draw
            deceptive = dead.setup.deceptive
            system_prompt = dead.setup.system_prompt
            play_strategy = dead.setup.play_strategy
            prediction_mapping = dead.setup.prediction_mapping
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
        if not pop.name_pool:
            raise NamePoolExhausted(
                f"round {round}: name pool exhausted — enlarge the name pools; "
                f"this run cannot be resumed past this point")
        name = pop.draw_name(rng)
        # newborns are always mortal: invincible is deliberately not passed (defaults False)
        pop.add(AgentSetup(system_prompt, pop_cfg.provider,
                           play_strategy, prediction_mapping,
                           deceptive=deceptive),
                agent_id=name)
        events.append({"type": "birth", "agent": name, "deceptive": deceptive,
                       "system_prompt": system_prompt,
                       "provider": asdict(pop_cfg.provider)})
    return events
```

Update the module docstring's rng-contract sentence (lines 5–8) to:

```
compatibility contract: one random() per live agent in roster order (invincible
agents consume their draw and ignore it), then per replacement at most one
random() (type — roll mode only; inherit mode never draws it) plus one
randrange() (name).
```

Update the `evolve` docstring's first paragraph to describe both modes (mirror the `EvolutionCfg` docstring from Task 1: roll = d/N with bounds and first-matching-spec clone; inherit = clone of the dying agent's setup; both: fresh memory, score 0, born mortal, unused pool name).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/population/test_evolution.py -v`
Expected: all PASS — the 6 new inherit tests and all pre-existing roll/invincible tests (the roll branch is byte-for-byte the same draws in the same order).

- [ ] **Step 5: Run the full suite and commit**

Run: `uv run pytest`
Expected: PASS.

```bash
git add src/population/evolution.py tests/population/test_evolution.py
git commit -m "feat(evolution): inherit replacement mode clones the dying agent's role"
```

---

### Task 3: Config hash — strip the new default-valued keys

**Files:**
- Modify: `src/storage/store.py` (`_hash_config_dict`, lines 44–69)
- Test: `tests/storage/test_storage.py` (hash tests around line 458)

**Interfaces:**
- Consumes: `EvolutionCfg.replacement` / nullable `decept_min`/`decept_max` from Task 1 (via `asdict`).
- Produces: nothing new — `_hash_config_dict(d: dict) -> str` behavior contract: every pre-feature config dict hashes identically to its post-feature `asdict` twin.

- [ ] **Step 1: Write the failing tests**

Append to `tests/storage/test_storage.py` next to `test_hash_config_dict_keeps_evolution_when_enabled` (reuse its imports: `_cfg`, `replace`, `asdict`, `json`, `AgentSpec`, `EvolutionCfg`, `_hash_config_dict`):

```python
def _evo_variant(base, **evo_kwargs):
    pop = replace(
        base.population,
        agents=[AgentSpec(count=2), AgentSpec(count=1, deceptive=True)],
        first_name_pool=["Alice", "Bob", "Carol", "Dave"],
        evolution=EvolutionCfg(**evo_kwargs),
    )
    return replace(base, population=pop)


def test_hash_config_dict_strips_default_replacement_and_null_bounds():
    # A stored pre-feature evolution config has no `replacement` key and int bounds.
    # Its post-feature asdict twin carries replacement="roll" — same hash required.
    cfg = _evo_variant(_cfg(seed=1), death_prob=0.1, decept_min=0, decept_max=1)
    d = asdict(cfg)
    assert d["population"]["evolution"]["replacement"] == "roll"

    legacy = json.loads(json.dumps(d))         # deep copy via round-trip
    legacy["population"]["evolution"].pop("replacement")
    assert _hash_config_dict(d) == _hash_config_dict(legacy)


def test_hash_config_dict_strips_none_decept_bounds():
    # An inherit config without bounds must hash the same whether the None keys
    # are present (asdict) or absent (hand-written dict).
    cfg = _evo_variant(_cfg(seed=1), death_prob=0.1, replacement="inherit")
    d = asdict(cfg)
    assert d["population"]["evolution"]["decept_min"] is None

    bare = json.loads(json.dumps(d))
    bare["population"]["evolution"].pop("decept_min")
    bare["population"]["evolution"].pop("decept_max")
    assert _hash_config_dict(d) == _hash_config_dict(bare)


def test_hash_config_dict_inherit_is_a_new_design():
    base = _cfg(seed=1)
    roll = _evo_variant(base, death_prob=0.1, decept_min=0, decept_max=1)
    inherit = _evo_variant(base, death_prob=0.1, decept_min=0, decept_max=1,
                           replacement="inherit")
    assert _hash_config_dict(asdict(roll)) != _hash_config_dict(asdict(inherit))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/storage/test_storage.py -k "strips_default_replacement or strips_none_decept or inherit_is_a_new_design" -v`
Expected: the two `strips_*` tests FAIL (hashes differ — no stripping yet); `inherit_is_a_new_design` PASSES already (any key difference changes the hash) — it is a regression guard for the stripping logic about to be added.

- [ ] **Step 3: Implement the stripping**

In `src/storage/store.py` `_hash_config_dict`, after the existing `population.pop("evolution")` handling (line 56) and before the agents loop, add:

```python
        evolution = population.get("evolution")
        if isinstance(evolution, dict):
            evolution = dict(evolution)
            if evolution.get("replacement", "roll") == "roll":
                evolution.pop("replacement", None)
            for key in ("decept_min", "decept_max"):
                if evolution.get(key) is None:
                    evolution.pop(key, None)
            population["evolution"] = evolution
```

Extend the docstring sentence about normalization (lines 44–48) to mention the two new strips: `replacement: "roll"` and `None`-valued `decept_min`/`decept_max` inside `population.evolution`, kept so pre-feature evolution configs keep their hash.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/storage/ -v`
Expected: all PASS, including the pre-existing `test_hash_config_dict_*` pair.

- [ ] **Step 5: Run the full suite and commit**

Run: `uv run pytest`
Expected: PASS.

```bash
git add src/storage/store.py tests/storage/test_storage.py
git commit -m "feat(storage): keep config_hash stable across the evolution.replacement default"
```

---

### Task 4: Documentation

**Files:**
- Modify: `docs/configuration.md` (*Population evolution*, lines 207–276)
- Modify: `docs/architecture.md` (*Population evolution*, lines 198–232)
- Modify: `docs/database.md` (the config_hash normalization note — search for "evolution")
- Modify: `CLAUDE.md` (the evolution `<important>` block)
- Modify: `config/reference.yaml` (population section — add the missing commented `evolution` block)

**Interfaces:**
- Consumes: the final behavior from Tasks 1–3.
- Produces: nothing — docs only.

- [ ] **Step 1: Update `docs/configuration.md`**

In the *Population evolution* section:
- Add `replacement: roll` to the YAML example with a comment: `# roll (default) | inherit — how a replacement's role is chosen`.
- Add a bullet after the `decept_min`/`decept_max` one:
  - `evolution.replacement` — `"roll"` (default): the type is rolled by the d/N rule below; `"inherit"`: the newborn clones the dying agent's full setup (system prompt, play strategy, prediction mapping, deceptive flag). In inherit mode `decept_min`/`decept_max` are ignored and may be omitted, and the type composition of the population can never change.
- In the **Sampling rule**, mark steps 2–3 as roll-mode only and add the inherit alternative: for each dead agent the replacement clones that agent's setup, with **no** type `random()` draw; step 4 (name `randrange()`) applies in both modes.
- In **Validation requirements**, scope the first two bullets (mixed specs, initial count in bounds) plus the `decept_min <= decept_max <= count` rule to roll mode, and note inherit mode skips them (any spec mix allowed); mortal-agent and name-pool rules apply in both modes; `replacement` must be `roll` or `inherit`.

- [ ] **Step 2: Update `docs/architecture.md`**

In its *Population evolution* section, add one short paragraph: `evolution.replacement: inherit` replaces the type roll with a clone of the dying agent's setup (no type draw; the per-replacement rng cost drops to the single name `randrange()`), so the population's role composition is invariant under turnover. Update the rng-contract sentence (around line 222) to the same "at most one random() (type — roll mode only)" wording used in the module docstring.

- [ ] **Step 3: Update `docs/database.md`**

Where the config_hash normalization is described, extend the list of stripped defaults with `population.evolution.replacement == "roll"` and `None`-valued `decept_min`/`decept_max`.

- [ ] **Step 4: Update `CLAUDE.md`**

In the `<important if="you are modifying population evolution...">` block, change "then per replacement at most one `random()` (type) + one `randrange()` (name)" to note the type draw happens in roll mode only, and mention the `replacement: roll | inherit` knob in one clause. Keep the block terse — details stay in `docs/configuration.md`.

- [ ] **Step 5: Add the evolution block to `config/reference.yaml`**

The reference catalogue currently omits evolution entirely (pre-existing gap). Add after the name pools (line 44), matching the file's comment style:

```yaml
  # ── Population evolution (EvolutionCfg) — optional; absent -> no turnover ────
  # Per-round death & replacement from round 2 on. Needs deceptive/invincible flags
  # on agent specs and a name pool strictly larger than the agent count.
  # evolution:
  #   death_prob: 0.1     # per-agent, per-round probability of dying, in [0, 1]
  #   replacement: roll   # roll (default): type rolled d/N within the bounds below;
  #                       # inherit: newborn clones the dying agent's role (bounds ignored)
  #   decept_min: 1       # roll mode only: deceptive count never pushed below this
  #   decept_max: 4       # roll mode only: replacements never push deceptive count above this
```

- [ ] **Step 6: Verify and commit**

Run: `uv run pytest` (docs-only change — a green run confirms nothing was accidentally touched).
Skim the four edited docs once for stale "roll-only described as the only behavior" sentences.

```bash
git add docs/configuration.md docs/architecture.md docs/database.md CLAUDE.md config/reference.yaml
git commit -m "docs: document evolution.replacement inherit mode"
```

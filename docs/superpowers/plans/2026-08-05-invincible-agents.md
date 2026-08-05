# Invincible Agents (Evolution-Exempt) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A per-spec `invincible: true` flag that exempts agents from evolution deaths — a fixed backbone of survivors (normal and/or deceptive) inside a churning population.

**Architecture:** The flag rides the existing spec→setup→evolve pipeline (`AgentSpec.invincible` → `AgentSetup.invincible` → `evolve()` ignores the death draw). The rng consumption contract stays verbatim: one `random()` per live agent in roster order — an invincible agent's draw is consumed and ignored. Newborns never inherit invincibility (they may clone an invincible spec's prompt but are born mortal). One analysis-only `agents.invincible` column with the standard additive migration and config-hash normalization.

**Tech Stack:** Python 3.12, frozen dataclasses config, pytest + pytest-asyncio (auto mode), SQLite, `uv` for all commands.

**Spec:** `docs/superpowers/specs/2026-08-05-invincible-agents-design.md` — read it first.

## Global Constraints

- Everything in English: docstrings (Google-style), comments, prints, log/exception messages, docs — this project overrides the user-global Russian rule.
- TDD: write the failing test FIRST, run it to see it fail, then minimal implementation.
- `from __future__ import annotations` at the top of every module; rng objects are passed in, never created inside library code.
- No printing or persistence inside `src/` — output leaves the engine only via the orchestrator `observer` callback.
- Dependencies only via `uv`; run tests with `uv run pytest ...` (pytest-asyncio auto mode; `pythonpath=["."]` — import `src.*` directly).
- Unit tests use stub providers (monkeypatch `src.population.base.make_provider`) — no network.
- Evolution rng consumption order is a compatibility contract (resume re-derives it): one `rng.random()` per live agent in roster order for deaths — **regardless of invincibility, the draw is consumed; an invincible agent ignores its result** — then per replacement at most one `rng.random()` (type) plus one `rng.randrange()` (name). Never reorder or skip these draws.
- Like `deceptive`, the `invincible` YAML flag is honored only when `population.evolution` is present; otherwise it is normalized to `False` at load.

---

### Task 1: Config — `invincible` spec flag, normalization, mortal-agent validation

**Files:**
- Modify: `src/core/config.py` (`AgentSpec` ~line 291, `_population_cfg` ~line 393, `_validate` evolution block ~lines 453–485)
- Test: `tests/core/test_config_load.py` (append; reuse the existing `_evo_dict` helper at ~line 645 and the file's existing `episode_from_dict` / `pytest` imports)

**Interfaces:**
- Consumes: existing `AgentSpec`, `_population_cfg`, `_validate`, `episode_from_dict`, test helper `_evo_dict(evolution=None, agents=None, first_pool=None)`.
- Produces: `AgentSpec.invincible: bool = False` (new LAST field, after `deceptive`). Later tasks read `spec.invincible`. New validation error containing the word `"mortal"` when every spec is invincible and evolution is on.

- [ ] **Step 1: Write the failing tests** — append to `tests/core/test_config_load.py`:

```python
def test_invincible_flag_parses_per_spec():
    d = _evo_dict(evolution={"death_prob": 0.1, "decept_min": 1, "decept_max": 3},
                  agents=[
                      {"count": 2, "system_prompt": "normal {id}"},
                      {"count": 1, "system_prompt": "normal inv {id}", "invincible": True},
                      {"count": 1, "system_prompt": "defect {id}", "deceptive": True,
                       "invincible": True},
                  ],
                  first_pool=[f"P{i}" for i in range(10)])
    cfg = episode_from_dict(d)
    assert [s.invincible for s in cfg.population.agents] == [False, True, True]


def test_invincible_normalized_false_without_evolution():
    d = _evo_dict(agents=[
        {"count": 3, "system_prompt": "normal {id}"},
        {"count": 1, "system_prompt": "defect {id}", "deceptive": True, "invincible": True},
    ])
    cfg = episode_from_dict(d)
    assert all(s.invincible is False for s in cfg.population.agents)


def test_invincible_survives_asdict_roundtrip():
    from dataclasses import asdict
    d = _evo_dict(evolution={"death_prob": 0.5, "decept_min": 0, "decept_max": 4},
                  agents=[
                      {"count": 3, "system_prompt": "normal {id}"},
                      {"count": 1, "system_prompt": "defect {id}", "deceptive": True,
                       "invincible": True},
                  ],
                  first_pool=[f"P{i}" for i in range(10)])
    cfg = episode_from_dict(d)
    again = episode_from_dict(asdict(cfg))
    assert [s.invincible for s in again.population.agents] == [False, True]


def test_evolution_rejects_all_invincible_population():
    d = _evo_dict(evolution={"death_prob": 0.1, "decept_min": 1, "decept_max": 3},
                  agents=[
                      {"count": 3, "system_prompt": "normal {id}", "invincible": True},
                      {"count": 1, "system_prompt": "defect {id}", "deceptive": True,
                       "invincible": True},
                  ],
                  first_pool=[f"P{i}" for i in range(10)])
    with pytest.raises(ValueError, match="mortal"):
        episode_from_dict(d)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_config_load.py -k invincible -v`
Expected: FAIL — `AgentSpec` has no attribute `invincible` (parses test), and the reject test fails because no error is raised.

- [ ] **Step 3: Implement in `src/core/config.py`**

Extend `AgentSpec` with a new LAST field (after `deceptive`, keep the comment style):

```python
    invincible: bool = False         # evolution: this agent never dies (draw consumed, ignored)
```

In `_population_cfg` (~line 393), extend the `AgentSpec(...)` call — mirror the existing `deceptive` gating exactly:

```python
                  deceptive=a.get("deceptive", False) if evolution is not None else False,
                  invincible=a.get("invincible", False) if evolution is not None else False)
```

In `_validate`, inside the `if evolution is not None:` block, right after the initial-deceptive-count check (`raise ValueError(f"initial deceptive count ...")`, ~line 475), add:

```python
        if all(a.get("invincible") for a in specs):
            raise ValueError(
                "evolution requires at least one mortal (non-invincible) agent")
```

(`specs` is already in scope from the deceptive-spec checks above.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/core/test_config_load.py -v` (whole file — the old tests must still pass)
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/config.py tests/core/test_config_load.py
git commit -m "feat(config): invincible agent-spec flag with mortal-agent validation"
```

---

### Task 2: Plumbing — `AgentSetup.invincible`, roster passes the flag

**Files:**
- Modify: `src/core/agent.py` (`AgentSetup`, ~line 103)
- Modify: `src/population/roster.py` (`RosterGenerator.build`, ~line 31)
- Test: `tests/population/test_roster.py` (append; the file already has `_stub_providers`/`_StubProvider` fixtures at ~line 144 and `_pool_cfg` at ~line 160 — reuse the fixture; ensure `AgentSetup`, `AgentSpec`, `PopulationCfg`, `ProviderCfg`, `RosterGenerator`, `random` are imported at MODULE level, adding any that are missing or currently imported locally inside a test)

**Interfaces:**
- Consumes: `AgentSpec.invincible` (Task 1).
- Produces: `AgentSetup.invincible: bool = False` (new LAST field, after `deceptive`); `RosterGenerator.build` sets it from `spec.invincible`. Task 3 reads `agent.setup.invincible`.

- [ ] **Step 1: Write the failing tests** — append to `tests/population/test_roster.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/population/test_roster.py -k invincible -v`
Expected: FAIL — `AgentSetup` has no field `invincible` / `TypeError: unexpected keyword argument`.

- [ ] **Step 3: Implement**

`src/core/agent.py` — add a new LAST field to `AgentSetup` (after `deceptive`):

```python
    invincible: bool = False             # evolution: this agent never dies (see EvolutionCfg)
```

`src/population/roster.py` — in `build`, extend the `AgentSetup(...)` call:

```python
                pop.add(AgentSetup(spec.system_prompt, self._cfg.provider,
                                   spec.play_strategy, spec.prediction_mapping,
                                   deceptive=spec.deceptive,
                                   invincible=spec.invincible),
                        agent_id=names[i])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/population/ tests/core/test_agent.py -v`
Expected: PASS (including all pre-existing roster/agent tests).

- [ ] **Step 5: Commit**

```bash
git add src/core/agent.py src/population/roster.py tests/population/test_roster.py
git commit -m "feat(population): carry invincible flag from spec to agent setup"
```

---

### Task 3: Evolution — ignore death draws for invincible agents; newborns always mortal

**Files:**
- Modify: `src/population/evolution.py` (death phase ~line 48; docstrings)
- Test: `tests/population/test_evolution.py` (append; reuse the file's `FakeRng` (~line 29), `_pop_cfg` (~line 43), `_build` (~line 55) and its existing imports of `random`, `pytest`, `AgentSpec`, `EvolutionCfg`, `PopulationCfg`, `ProviderCfg`, `evolve`)

**Interfaces:**
- Consumes: `AgentSetup.invincible` via `agent.setup.invincible` (Task 2).
- Produces: `evolve()` behavior change only — no signature or event-shape change. Contract for all later tasks and docs: every live agent still consumes exactly one death `rng.random()` in roster order; invincible agents ignore the result; newborn `AgentSetup`s always have `invincible=False` (the `AgentSetup(...)` call in `evolve` does not pass `invincible`, so the dataclass default applies — keep it that way).

- [ ] **Step 1: Write the failing tests** — append to `tests/population/test_evolution.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/population/test_evolution.py -v`
Expected: the new invincible tests FAIL (invincible agents die: survive-turnover, lethal-draw, newborn-mortal tests break); the draw-count and determinism tests may pass already — that is fine, the behavioral tests are the RED gate. Old tests PASS.

- [ ] **Step 3: Implement** — in `src/population/evolution.py`:

Change the death-selection line (~line 48) so the draw is always consumed but ignored for invincible agents (evaluation order matters — `rng.random()` must run for every agent, so it stays first in the condition):

```python
    deaths = [a for a in list(pop)
              if rng.random() < ev.death_prob and not a.setup.invincible]
```

Do NOT change the `AgentSetup(...)` call in the replacement loop — it does not pass `invincible`, so newborns get the dataclass default `False`. Add one comment above it:

```python
        # newborns are always mortal: invincible is deliberately not passed (defaults False)
```

Update the module docstring's contract sentence and the `evolve` docstring: one `random()` per live agent in roster order — invincible agents consume their draw and ignore it (so toggling flags never reshuffles other agents' draws); replacements clone the first type-matching spec's prompt/strategy but are always born mortal.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/population/ -v`
Expected: PASS (including the pre-existing forced-branch draw-count oracle).

- [ ] **Step 5: Commit**

```bash
git add src/population/evolution.py tests/population/test_evolution.py
git commit -m "feat(evolution): invincible agents ignore death draws; newborns always mortal"
```

---

### Task 4: Storage — `agents.invincible` column, migration, config-hash normalization

**Files:**
- Modify: `src/storage/schema.py` (agents CREATE TABLE ~line 24, `init_schema` migration tuple ~line 152)
- Modify: `src/storage/store.py` (`_hash_config_dict` ~lines 35–65, `begin` agents insert ~line 116)
- Test: `tests/storage/test_storage.py` (append; reuse `_evo_stub_providers` (~line 541), `_cfg`, `RoundPlan`, `Storage`, `make_population`, `random`, `sqlite3`, `json`, `asdict`, `_hash_config_dict` — all already imported in that file)

**Interfaces:**
- Consumes: `a.setup.invincible` (Task 2); `AgentSpec.invincible` in `asdict` output (Task 1).
- Produces: `agents.invincible INTEGER NOT NULL DEFAULT 0` (analysis-only; resume never reads it); `begin` stamps it; birth-event inserts leave the default 0; `_hash_config_dict` drops falsy `invincible` keys so pre-feature configs keep their hash. No API for later tasks.

- [ ] **Step 1: Write the failing tests** — append to `tests/storage/test_storage.py`:

```python
def _inv_storage_cfg():
    return EpisodeCfg(
        seed=0, rounds=2, matchmaker="random",
        population=PopulationCfg(
            kind="roster",
            agents=[AgentSpec(count=1, system_prompt="normal {id}"),
                    AgentSpec(count=1, system_prompt="defect {id}", deceptive=True,
                              invincible=True)],
            provider=ProviderCfg(base_url="http://x/v1", model="m")),
        game=GameCfg(max_talk_turns=0))


def test_begin_records_invincible_flag(tmp_path, _evo_stub_providers):
    st = Storage(str(tmp_path / "t.db"))
    cfg = _inv_storage_cfg()
    pop = make_population(cfg.population).build(random.Random(0))
    rid = st.begin(cfg, pop)
    rows = dict(st.conn.execute(
        "SELECT agent_id, invincible FROM agents WHERE run_id=?", (rid,)))
    assert rows == {"A1": 0, "A2": 1}
    st.close()


def test_birth_rows_default_mortal(tmp_path, _evo_stub_providers):
    st = Storage(str(tmp_path / "t.db"))
    cfg = _inv_storage_cfg()
    pop = make_population(cfg.population).build(random.Random(0))
    rid = st.begin(cfg, pop)
    plan = RoundPlan(pairings=[], idle=[], events=[
        {"type": "birth", "agent": "Player 9", "deceptive": True,
         "system_prompt": "defect {id}",
         "provider": {"base_url": "http://x/v1", "model": "m"}}])
    st.observe(2, plan, [])
    assert st.conn.execute(
        "SELECT invincible FROM agents WHERE run_id=? AND agent_id='Player 9'",
        (rid,)).fetchone() == (0,)
    st.close()


def test_migration_adds_invincible_column_to_old_db(tmp_path):
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
    assert "invincible" in cols
    st.close()


def test_hash_config_dict_strips_default_invincible_noise():
    # `invincible: False` on every spec must not perturb pre-feature hashes
    cfg = _cfg(seed=1)
    d = asdict(cfg)
    assert all(a["invincible"] is False for a in d["population"]["agents"])
    stripped = json.loads(json.dumps(d))       # deep copy via round-trip
    stripped["population"].pop("evolution")
    for a in stripped["population"]["agents"]:
        a.pop("deceptive")
        a.pop("invincible")
    assert _hash_config_dict(d) == _hash_config_dict(stripped)
```

(If `EpisodeCfg`/`GameCfg` are not yet imported at module level in this test file, add them to the existing `src.core.config` import.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/storage/test_storage.py -v`
Expected: new tests FAIL (`no such column: invincible`; the hash test fails because `invincible` is stripped from `stripped` but survives normalization in `d`); old tests PASS.

- [ ] **Step 3: Implement**

`src/storage/schema.py` — in the agents CREATE TABLE, after the `deceptive` line:

```sql
    invincible    INTEGER NOT NULL DEFAULT 0,
```

and extend the migration tuple in `init_schema`:

```python
    for name, ddl in (("born_round", "INTEGER NOT NULL DEFAULT 1"),
                      ("died_round", "INTEGER"),
                      ("deceptive", "INTEGER NOT NULL DEFAULT 0"),
                      ("invincible", "INTEGER NOT NULL DEFAULT 0")):
```

`src/storage/store.py` — in `begin`, extend the agents insert:

```python
                "INSERT INTO agents(run_id, agent_id, system_prompt, provider, deceptive, "
                "invincible) VALUES (?,?,?,?,?,?)",
                [
                    (run_id, a.id, a.setup.system_prompt,
                     json.dumps(asdict(a.setup.provider_cfg)), int(a.setup.deceptive),
                     int(a.setup.invincible))
                    for a in pop
                ],
```

(Leave the birth-event insert in `observe` untouched — the column default 0 is the contract.)

In `_hash_config_dict`, replace the per-spec `deceptive` strip with a two-key strip and extend the docstring sentence to mention `invincible`:

```python
            new_agents = []
            for a in agents:
                if isinstance(a, dict):
                    drop = {k for k in ("deceptive", "invincible") if not a.get(k)}
                    if drop:
                        a = {k: v for k, v in a.items() if k not in drop}
                new_agents.append(a)
            population["agents"] = new_agents
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/storage/ -v`
Expected: PASS (including the pre-existing hash-normalization tests).

- [ ] **Step 5: Commit**

```bash
git add src/storage/schema.py src/storage/store.py tests/storage/test_storage.py
git commit -m "feat(storage): invincible agent column; keep pre-feature config_hash"
```

---

### Task 5: Documentation

**Files:**
- Modify: `docs/configuration.md`, `docs/architecture.md`, `docs/database.md`

No test cycle — verification is the full suite plus a docs read-through. Read each doc's evolution-related section first so additions blend in.

- [ ] **Step 1: `docs/configuration.md`** — in the population-evolution section:
  - document `population.agents[*].invincible: bool` (default false): the agent never dies during evolution; honored only when `evolution` is present (normalized to false otherwise, like `deceptive`); any spec may combine `invincible` with `deceptive`.
  - the requirement: at least one agent must be mortal when evolution is on.
  - replacement rule note: replacements clone the first type-matching spec's prompt/strategy — possibly an invincible spec's — but are always born mortal; invincible agents count in `d`/`N` for the replacement sampling.
  - extend the section's YAML example with an invincible spec line (keep the existing anchor style).

- [ ] **Step 2: `docs/architecture.md`** — in the evolution-step paragraph, add: invincible agents consume their death draw and ignore it (the rng consumption-order contract is unchanged, so toggling `invincible` flags never reshuffles which other agents die), and newborns are always mortal.

- [ ] **Step 3: `docs/database.md`** — add the `agents.invincible` column (analysis-only, default 0, additive migration like the other evolution columns; birth rows are always 0).

- [ ] **Step 4: Run the full suite and commit**

Run: `uv run pytest`
Expected: all tests PASS.

```bash
git add docs/configuration.md docs/architecture.md docs/database.md
git commit -m "docs: invincible agents reference"
```

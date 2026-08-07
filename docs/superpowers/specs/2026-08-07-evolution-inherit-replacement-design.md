# Design: `evolution.replacement: inherit` — replacement inherits the dying agent's role

Date: 2026-08-07
Status: approved

## Problem

Population evolution currently has one replacement policy: when an agent dies, the
newborn's type is **rolled** — deceptive with probability d/N (d = live deceptive
count, N = population size), clamped by `decept_min`/`decept_max` — and the first
agent spec matching that flag is cloned. We want a second policy where the newborn
simply **takes the role of the agent it replaces**, so the population's type
composition is frozen while names, memories, and scores still turn over.

## Config (`src/core/config.py`)

- `EvolutionCfg` gains `replacement: str = "roll"`:
  - `"roll"` — today's behavior, unchanged (default).
  - `"inherit"` — the newborn clones the dying agent's role.
- `decept_min` / `decept_max` become optional: `int | None = None`.
- `_validate`:
  - `replacement` must be `"roll"` or `"inherit"`, else fail fast.
  - **roll mode** keeps every current check: `decept_min`/`decept_max` present,
    integers `>= 0`, `decept_min <= decept_max <= agent count`; at least one
    deceptive and one non-deceptive spec; initial deceptive count within
    `[decept_min, decept_max]`.
  - **inherit mode** skips all of those: `decept_min`/`decept_max` may be omitted
    and are ignored if present; any spec mix is allowed (including all-honest or
    all-deceptive populations — pure turnover with no deception axis).
  - Checks that apply in **both** modes: `death_prob` in [0, 1], at least one
    mortal (non-invincible) agent, a name pool for replacements.

## Evolution step (`src/population/evolution.py`)

- `evolve()` branches on `ev.replacement`:
  - **roll**: unchanged.
  - **inherit**: for each dying agent, the newborn clones that agent's full setup —
    `system_prompt`, `play_strategy`, `prediction_mapping`, `deceptive` — with a
    fresh name drawn from the pool, fresh memory, score 0, and always mortal
    (`invincible` never inherited). Death→birth pairing follows the `deaths` list
    order (roster order), same as today's event order.
- RNG consumption:
  - inherit mode consumes **no** type `random()` — only the one name `randrange()`
    per replacement. This stays within the documented contract ("per replacement
    at most one `random()` plus one `randrange()`"); the mode is fixed by config,
    so `Random(f"{seed}:evolution:{r}")` re-derivation on resume is unaffected.
  - The one-`random()`-per-live-agent death sweep is identical in both modes.
  - Module docstring updated to state the per-mode consumption explicitly.
- Birth events keep the same shape: `{"type": "birth", "agent", "deceptive",
  "system_prompt", "provider"}`.

## Config hash (`src/storage/store.py`)

`_hash_config_dict` additionally strips, inside `population.evolution`:

- `replacement` when equal to `"roll"` (the default), and
- `decept_min` / `decept_max` when `None`,

so every stored/existing config — evolution-free or evolution-enabled — keeps its
current `config_hash`. Inherit-mode configs hash as a genuinely new design.

## Tests (TDD — failing tests first)

`tests/population/test_evolution.py` (or a sibling module, following existing layout):

- inherit: newborn clones the dying agent's exact setup (prompt, strategy,
  mapping, deceptive), has a fresh pool name, fresh memory, score 0, mortal.
- inherit: no type draw consumed — assert rng stream consumption (e.g. via a
  scripted/counting rng or by comparing streams across modes).
- inherit with multiple simultaneous deaths of different types: each newborn
  matches its own predecessor.
- validation: inherit without `decept_min`/`decept_max` loads; roll without them
  fails fast; unknown `replacement` value fails fast; inherit accepts a
  single-flag (e.g. all non-deceptive) population; roll still rejects it.

`tests/storage/` hash tests:

- an evolution config's hash is unchanged by the new default field
  (`replacement: "roll"`, `decept_min`/`decept_max` set);
- an evolution-free config's hash is unchanged;
- an inherit config hashes differently from its roll counterpart.

## Docs to update

- `docs/configuration.md` — *Population evolution* section: the `replacement`
  knob, optionality of `decept_min`/`decept_max`, validation differences.
- `config/reference.yaml` — commented `replacement` knob in the evolution block.
- `docs/architecture.md` — evolution description, if it states the roll policy.
- `CLAUDE.md` — the evolution `<important>` block (rng contract wording).

## Out of scope

- No per-spec replacement policies, no heredity of memory/score, no mutation of
  inherited roles. YAGNI.
- No strategy-pattern/factory for replacement policies — it's a two-value config
  switch inside `evolve()`; extract a policy seam only if a third mode ever lands.

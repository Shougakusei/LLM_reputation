# Invincible Agents (Evolution-Exempt) — Design

**Date:** 2026-08-05
**Status:** Approved
**Builds on:** `2026-08-05-population-evolution-design.md` (merged)

## Goal

Let a config declare agents that never die during population evolution: a fixed
backbone of survivors inside an otherwise churning population. Two kinds are
expressible — normal invincible agents and deceptive ("defect") invincible
agents — each with its own prompt, count, and strategy.

## Non-goals

- No invincible replacements: newborns are always mortal, even when they clone
  an invincible spec's prompt (see Sampling below).
- No per-round scheduling of invincibility (consistent with the evolution
  non-goal: no change-points for evolution parameters).
- No mid-run granting or revoking of invincibility.

## Config surface

```yaml
population:
  agents:
    - {count: 6, system_prompt: *normal}
    - {count: 2, system_prompt: *normal_inv, invincible: true}
    - {count: 1, system_prompt: *defect, deceptive: true}
    - {count: 1, system_prompt: *defect_inv, deceptive: true, invincible: true}
  evolution:
    death_prob: 0.1
    decept_min: 1
    decept_max: 4
```

- `AgentSpec` (frozen dataclass, `src/core/config.py`) gains
  `invincible: bool = False` as a new last field.
- Like `deceptive`, the flag is honored only when `population.evolution` is
  present; without an evolution block it is normalized to `False` at load
  (invincibility is meaningless when nothing dies).
- `_validate` additions when `evolution` is present:
  - at least one agent must be mortal (`invincible` false on at least one spec
    instance): an all-invincible population makes evolution a silent no-op and
    is rejected as a config mistake;
  - existing rules are unchanged — in particular "at least one deceptive and
    one non-deceptive spec" still counts specs regardless of invincibility,
    because replacements clone prompts from any type-matching spec.

## Sampling semantics (deltas to the evolution rule)

1. **Death phase — rng contract unchanged.** `evolve()` still consumes exactly
   one `rng.random()` per live agent in roster order. An invincible agent's
   draw is consumed but its result is ignored — the agent never dies.
   Consequence (deliberate): with the same seed and roster, toggling
   `invincible` flags does not change which *other* agents die.
2. **Replacement counts.** Invincible agents count in `d` (live deceptive
   count) and `N` (fixed population size) for the `d / N` type sampling and in
   the initial-count bounds check. A persistent deceptive backbone therefore
   keeps the deceptive fraction — and the chance of deceptive replacements —
   elevated; that is the point of the feature.
3. **Replacement spec selection unchanged; newborns always mortal.** The
   replacement still clones the first spec whose `deceptive` matches the
   sampled type (system prompt + play strategy + prediction mapping) — that
   spec may be an invincible one — but the newborn's own `invincible` is
   always `False`. The invincible population can only shrink to its configured
   set, never grow. A config with only-invincible deceptive agents plus mortal
   normals is legal: deceptive replacements clone the invincible defect
   spec's prompt and are born mortal.

Per-replacement rng consumption is unchanged: at most one `random()` (type,
only in the probabilistic branch) plus one `randrange()` (name).

## Plumbing & persistence

- `AgentSetup` (`src/core/agent.py`) gains `invincible: bool = False` as a new
  last field; `RosterGenerator.build` passes `invincible=spec.invincible`.
- `evolve()` reads `agent.setup.invincible` in the death phase and forces
  `invincible=False` on newborn `AgentSetup`s.
- `agents` table gains `invincible INTEGER NOT NULL DEFAULT 0` — analysis-only
  (resume never reads it), added via the same additive `ALTER TABLE` pattern in
  `init_schema`. `Storage.begin` stamps it from `a.setup.invincible`; birth
  inserts leave the default 0 (newborns are mortal by construction).
- `_hash_config_dict` normalization: drop `invincible` from agent-spec dicts
  when falsy, exactly like `deceptive` — evolution-free and pre-feature
  configs keep their `config_hash`.
- No changes to the orchestrator, narration, replay, or resume: `evolve()`
  stays deterministic, so resume's evolution replay reproduces invincible
  survivors automatically from the rebuilt roster.

## Error handling

- New failure message from `_validate` (English, fail-fast): evolution with all
  agents invincible → "evolution requires at least one mortal (non-invincible)
  agent".

## Testing (TDD, stub providers, no network)

- **config:** `invincible` parses per spec; defaults `False`; survives the
  `asdict` roundtrip; normalized to `False` when `evolution` absent;
  all-invincible + evolution rejected with the message above.
- **evolution unit:** with `death_prob=1.0`, invincible agents survive while
  every mortal dies; the exact-draw-count oracle extended to prove invincible
  agents still consume one death draw each (scripted rng sized so a skipped
  draw fails); a newborn cloned from an invincible spec has
  `setup.invincible is False`; determinism across rebuilds with mixed flags.
- **roster:** `build` marks `setup.invincible` per spec.
- **storage:** `begin` stamps the column; migration adds it to a pre-existing
  DB; birth-event rows keep `invincible=0`.
- **config-hash:** flag-free configs hash identically to pre-feature configs.

## Docs to update after implementation

- `docs/configuration.md` — `invincible` flag in the population/evolution
  section, mortal-agent requirement.
- `docs/architecture.md` — one sentence in the evolution step (draw consumed
  and ignored; newborns always mortal).
- `docs/database.md` — the new `agents.invincible` column.

# Population Evolution: Agent Death & Replacement — Design

**Date:** 2026-08-05
**Status:** Approved

## Goal

Add a configurable evolution mechanic to the episode loop: at each round boundary
every agent may die with a fixed probability and is replaced by a fresh agent. The
replacement is *deceptive* (uses the defect system prompt) with probability equal to
the current deceptive fraction of the population, clamped so the deceptive count
stays within configured bounds. This lets the research study reputation dynamics
under population turnover.

## Non-goals

- No population state carried across runs/episodes (each run still builds its roster
  fresh from config + seed).
- No fitness-based selection (death is uniform-random, not score-driven).
- No per-round scheduling of evolution parameters (no change-points).
- Nothing kills surplus deceptive agents: bounds are enforced only through
  replacement choices.

## Config surface

```yaml
population:
  agents:
    - {count: 8, play_strategy: direct, system_prompt: *system_default}
    - {count: 2, play_strategy: direct, system_prompt: *system_defect, deceptive: true}
  evolution:            # optional block; absent -> feature off, zero behavior change
    death_prob: 0.1     # per-agent, per-round probability of dying
    decept_min: 1       # deceptive count never pushed below this by replacements
    decept_max: 4       # replacements never push deceptive count above this
  first_name_pool: [ ... strictly more names than agents ... ]
```

- `AgentSpec` (frozen dataclass, `src/core/config.py`) gains optional
  `deceptive: bool = False`.
- `PopulationCfg` gains optional `evolution: EvolutionCfg | None = None` with frozen
  `EvolutionCfg(death_prob: float, decept_min: int, decept_max: int)`.
- `_validate` fails fast when `evolution` is present and any of:
  - `death_prob` outside `[0, 1]`;
  - not `0 <= decept_min <= decept_max <= total agent count`;
  - no spec with `deceptive: true`, or no spec without it;
  - initial deceptive count outside `[decept_min, decept_max]`;
  - the effective name pool is not strictly larger than the initial population
    (replacements need headroom; empty pools / A{n} fallback mode is rejected when
    evolution is on).

## Sampling semantics (exact rule)

At the **start of each round r >= 2** (never before round 1). Applying at round
start makes the roster for round r a pure function of `(seed, r)` — extend/resume
stay deterministic regardless of where a run previously stopped.

1. Derive `rng = random.Random(f"{seed}:evolution:{r}")` — a dedicated stream per
   round, mirroring the matchmaker pattern.
2. Each live agent independently dies with probability `death_prob` (one
   `rng.random() < death_prob` draw per agent, in current roster order).
3. Remove all dead agents, then create replacements **one at a time**. For each,
   with `d` = current live deceptive count and `N` = fixed population size:
   - if `d < decept_min` -> replacement is deceptive (forced);
   - elif `d >= decept_max` -> replacement is normal (forced);
   - else -> deceptive with probability `d / N` (one rng draw).
4. A deceptive replacement clones the first `deceptive: true` spec (system prompt +
   play strategy + prediction mapping); a normal replacement clones the first
   non-deceptive spec. The newborn starts with a fresh empty memory and score 0.
5. The newborn's name is drawn by `rng` from the unused remainder of the name pool
   (dead agents' names are never reused — a reused name would inherit the dead
   agent's reputation). If the pool is exhausted mid-run, the episode aborts with a
   clear error naming the round and instructing to enlarge the pool.

## Architecture

**Chosen: deterministic rng step in the orchestrator** (over an event-sourced
alternative that would persist deaths/births and read them back on resume — more
storage code, DB becomes correctness-critical, no benefit given the existing
per-round-rng determinism discipline).

- New module `src/population/evolution.py`: an `evolve(pop, pop_cfg, rng) ->
  list[dict]` step that mutates the population and returns events. Pure with
  respect to everything except `pop` and `rng` — no I/O, no printing.
- `Population` gains the documented seam mutator `remove(agent_id)` and keeps a set
  of all ids ever used, so names are never reused. `RosterGenerator` stores the
  ordered leftover name pool on the `Population` after the initial build, so later
  draws are deterministic.
- `run_episode` (`src/core/orchestrator.py`) calls the evolution step at the top of
  each round r >= 2 (before `plan_round`) when `cfg.population.evolution` is set,
  and appends the returned events to `RoundPlan.events` — deaths/births reach the
  outside world only through the existing observer channel.
- Event shapes:
  - `{"type": "death", "agent": <id>, "score": <final score>}`
  - `{"type": "birth", "agent": <id>, "deceptive": <bool>, "system_prompt": <str>,
    "provider": <dict>}` — the prompt/provider ride along so Storage can insert the
    newborn's `agents` row without reaching back into the population.

## Persistence & presentation

- `agents` table gains `born_round INTEGER NOT NULL DEFAULT 1`,
  `died_round INTEGER` (NULL = alive), `deceptive INTEGER NOT NULL DEFAULT 0`.
  Additive `ALTER TABLE` migration for existing DBs at `Storage` init.
  These columns are **analysis-only** — resume never reads them.
- `Storage.observe` processes `plan.events`: inserts newborn agent rows
  (`born_round = r`, prompt, provider, deceptive flag) and stamps `died_round = r`
  plus `final_score` for the dead.
- `narrate_round` (runner) prints deaths/births at the top of the round;
  `replay.py` shows the same events when replaying stored runs.

## Resume

`resume_run` rebuilds the initial roster from config + seed as today, then re-runs
the evolution steps for the already-played rounds `2..start_round - 1` with the same
per-round rng streams (`run_episode` then applies round `start_round`'s own
evolution step itself, as it does for every round it plays)
(pure rng + counts — no LLM calls, no DB reads), producing the exact live roster at
the resume point. Stored scores/memories are then applied to live agents;
`_apply_run_state` switches from `state.scores[agent.id]` to `.get(agent.id, ...)`
because newborns may have no stored state yet (dead agents' stored state is simply
not applied).

## Error handling

- Name pool exhaustion mid-run: raise a dedicated error at the round boundary with
  the round number and a hint to enlarge the pool. Such a run **cannot be resumed
  past that point**: resume replays the stored config (not the YAML file), and
  `rng.sample` over a different-sized pool would change every name draw anyway.
  Document in configuration.md: choose a generously oversized pool when evolution
  is enabled.
- All new failure messages in English, raised from `_validate` or the evolution
  step; no printing inside `src/`.

## Testing (TDD, ScriptedProvider, no network)

- **config:** each validation failure above; `evolution` absent -> `None` and no
  behavior change; `deceptive` defaults to False.
- **evolution unit:** forced-deceptive below `decept_min`; forced-normal at
  `decept_max`; probabilistic path with a seeded rng (both outcomes exercised);
  newborn gets fresh memory/score/name from the pool; name-pool exhaustion raises;
  determinism: same rng seed -> identical events.
- **orchestrator:** events appear in the observer's `RoundPlan`; roster changes
  between rounds; newborn participates in later rounds with an empty diary; no
  evolution before round 1's play.
- **resume:** play k rounds with evolution, resume -> identical derived roster
  (ids and deceptive types), scores/memories applied to survivors only.
- **storage:** newborn row inserted with `born_round`; `died_round` and
  `final_score` stamped; migration adds columns to a pre-existing DB.

## Docs to update after implementation

- `docs/configuration.md` — `evolution` block, `deceptive` spec flag, pool-size
  requirement, append-only pool rule.
- `docs/architecture.md` — evolution step in the round loop, event channel.
- `docs/database.md` — new `agents` columns.
- `CLAUDE.md` project map if a new module warrants a line.

# Architecture

A research engine that studies whether an **institution of reputation** emerges on
its own in a population of LLM agents that repeatedly play a coordination game with
cheap talk.

One **episode** = one YAML config. The orchestrator builds a population of agents; a
matchmaker pairs them each round; within each pair the agents exchange a few short
messages (cheap talk) and then secretly pick a number 0–9. Payoffs reward matching
and punish being undercut. Everything the run produces leaves the engine through a
single `observer` callback; the runner persists it to SQLite.

## Layered design

Dependencies flow one way, bottom → top. A lower layer never imports a higher one.

```
providers/     OpenAI-compatible HTTP client + retries          (no deps on us)
   ↑
core/          Agent, Memory, config dataclasses, orchestrator   (depends on providers)
   ↑
games/         ReputationPD game + prompt builders
strategy/      PlayStrategy: how an agent turns a round into a number
   ↑
population/    Population (live roster + provider cache)
matchmaking/   Matchmaker: who plays whom each round
   ↑
core/orchestrator.py   run_episode: glues all of the above
   ↑
storage/ · stats/ · judge/ · runner.py   persistence & analysis (callers, not the engine)
```

| layer | code |
|-------|------|
| provider | `src/providers/` — `LLMProvider` Protocol, httpx client, retries, error types |
| agent + memory | `src/core/agent.py`, `src/core/memory.py` |
| config | `src/core/config.py` — frozen dataclasses, `load_episode`, `_validate` |
| game | `src/games/` — `ReputationPD`, prompt builders, talk-stop rules |
| strategy | `src/strategy/` — `direct` / `prediction` |
| population | `src/population/` — `Population`, `RosterGenerator` |
| matchmaking | `src/matchmaking/` — `RandomMatchmaker` |
| orchestrator | `src/core/orchestrator.py` — `run_episode` |
| persistence | `src/storage/` — SQLite schema, `Storage`, record types (see [database.md](./database.md)) |
| analysis | `src/stats/`, `src/judge/`, `src/runner.py` |

`games/` and `strategy/` are mutually referential by design; the cycle is broken with
**lazy imports** (see `src/games/reputation_pd.py`). Prompt builders live in
`src/games/prompts.py`, which imports neither game nor strategy, so both share it
without a cycle.

## The game: ReputationPD

`src/games/reputation_pd.py`. A repeated, PD-flavoured coordination game.

- Each round two agents secretly pick an integer 0–9.
- **Equal** numbers → both score `R` (mutual cooperation).
- One number **exactly one higher** than the other, **mod 10** (0 follows 9) → the
  higher scores `T` (betrayal), the lower scores `S`.
- **Anything else** → both score `P` (miscoordination).
- Outcome strings are from A's perspective (`CC`/`DC`/`CD`/`DD`); `_FLIP`
  mirrors them for B's memory.

Payoff invariants (`Payoffs`, `src/core/config.py`): `T > R > P > S` and `2R > T + S`
(mutual cooperation beats alternating exploitation). The reference values are
`R,T,P,S = 3,5,1,0`.

### Pairing flow (`ReputationPD.play_pairing`)

1. **Cheap talk** (`_cheap_talk`). Agents alternate short messages; `a` always opens
   (the matcher fixes orientation via pairing order — no rng in the game). Every agent
   speaks at least once, up to a hard ceiling of `cfg.max_talk_turns`. Chat closes only
   when **both** agents set `finish: true` (the agent-facing JSON key; stored internally
   as `ready`). The exact stopping behaviour is a pluggable seam, `TalkStopRule`
   (`src/games/talk_rules.py`), with three variants along two axes — does a finished
   agent keep talking, and is `finish` revocable:
   - **`both_ready_latch`** (default) — the first finisher latches silent and waits;
   - **`both_ready_revocable`** — it keeps talking, and a later `finish: false` takes
     its readiness back;
   - **`both_ready_committed`** — it keeps talking, but once set `finish` is sticky.

   `make_talk_rule(name)` is the factory; the name is validated at load.
2. **Decide.** Each agent's own `PlayStrategy.decide` runs on the public talk feed and
   returns a `Decision` (final number + optional rationale, plus an optional prediction).
3. **Resolve.** Scores mutate on the agents.
4. **Reflect** (optional, `game.reflection: true`). Each agent makes one extra `REFLECT`
   LLM call over the revealed result (both numbers + own payoff) and returns a short
   reflection, private to its author like the rationale.
5. **Record.** A `PairingRecord` (`src/games/base.py`) is returned, and each agent's
   `Memory` gets an entry (including any reflection). See *Memory rendering* below.
6. **Memory notes** (optional, `game.memory_notes_every: N`). After every N rounds an
   agent has actually played (counted per-agent as `len(memory.entries)`; idle rounds
   don't count), it makes a `NOTE` call that rewrites its whole memory into private
   notes. From then on `Memory.render` sends those notes **instead of** the raw round
   history, plus a raw buffer of rounds played since the last consolidation. With
   `game.history_line` set, the folded rounds additionally survive as compact one-liners
   (facts only: round, partner, payoff) rendered into `{lines}` of `notes_view`.

A failed LLM call in any phase aborts the pairing (see *Failure handling*).

## Strategies (`src/strategy/`)

`PlayStrategy` (Protocol, `strategy/base.py`) maps round state → `Decision`.

- **direct** (`strategy/direct.py`): one `DECIDE` phase; the agent picks the number itself.
- **prediction** (`strategy/prediction.py`): one `PREDICT` phase (the agent predicts the
  partner's number), then a pure `PredictionMapping` (`strategy/mappings.py`) turns that
  prediction into its own choice — `match` (copy → cooperate) or `one_above` (rational
  best-response, off-by-one for `T`).

**Strategy is per-agent.** It lives on `AgentSpec`/`AgentSetup` as plain strings (`core`
does not import `strategy`), so one population can mix direct and prediction agents. The
game builds each agent's strategy from `agent.setup` and caches it by
`(play_strategy, prediction_mapping)`; `make_strategy(...)` is the factory. Add one:
implement the Protocol, register it in `make_strategy`, and extend `_validate` if it
needs validation.

## Agent & phases (`src/core/agent.py`)

An `Agent` owns its `Memory`, a running `score`, and an `LLMProvider`. `Agent.act` builds a
**single** user message per call. A step prompt arrives as a string or a `PromptVariants`
pair — the one conditional in assembly picks `no_history`/`with_history` by whether any
round was played *before the current one* (at NOTE time the just-played round is already in
memory and does not count; a missing `no_history` falls back to `with_history`). The chosen
text then gets the **memory fragments** substituted in place (`Memory.fragments` +
`_fill_fragments`): `{recent_rounds}`, `{history_lines}`, `{notes_line}`/`{notes}`, and
`{history}` (the engine-composed block). Substitution follows the **paragraph rule**: a
blank-line-delimited paragraph whose fragments all render empty is dropped whole (headers
included), which is how a `with_history` text degrades into the first-round message; inside
a kept paragraph an empty fragment vanishes with just its own line. A plain-string prompt
naming no fragment gets the memory **prepended** instead (legacy assembly, kept so stored
configs replay unchanged). Agent.act then calls the provider and parses the reply as JSON
with up to `_MAX_PARSE_RETRIES` correction retries; on total failure it raises
`ActParseError` (no substitution/fallback).

Five `PhaseKind`s: `TALK`, `DECIDE`, `PREDICT`, `REFLECT`, `NOTE`. **All phase prompts are
static templates** — only named placeholders are substituted, never assembled from text
chunks. `PREDICT` mirrors `DECIDE` byte-for-byte except the directive. DECIDE/PREDICT each
use one `decide_prompt` template each (no `_bare` variant); the `game.rationale` flag does not
pick a template — it only gates whether the returned rationale is read/stored, so each experiment
writes its single prompt to match (`{"rationale","number"}` when on, `{"number"}` when off). JSON
extraction is lenient — raw, fenced, and balanced-brace candidates are all tried
(`src/core/jsonextract.py`).

The **system prompt** is the agent's entire system message, taken verbatim; only `{id}`
and the payoff placeholders `{R}/{T}/{P}/{S}/{max_talk_turns}` are substituted.

### Memory rendering

The whole LLM input is one continuous **game transcript**. `Memory.render`
(`src/core/memory.py`) replays each past round with **one full-round template**,
`GameCfg.history_prompt` — header, the round's cheap-talk at `{feed}` (rendered with
`msg_self`/`msg_partner`, the same tags the live prompts use), the close line, the agent's
response, and the result. What the agent sees of its own past (a rationale block, a
takeaway, …) is decided entirely by the experiment's `history_prompt` — the code does not
branch on flags. Templates ride into `render` via `Phase.game_cfg` (which also carries the
payoffs). After gluing history to the live prompt, `Agent.act` collapses adjacent `<game>`
blocks (`_merge_game_blocks`) so the input is one running transcript rather than a series
of closed/reopened blocks. With memory notes on, `notes_view` renders the saved notes
(via `msg_self`, as the agent's own memo) and `notes_buffer` appends the raw rounds played
since the last consolidation; an optional `history_line` keeps a compact one-liner per
folded round at `{lines}` inside `notes_view` (see [configuration.md](./configuration.md)).

### LLM input trace

For DECIDE/PREDICT/REFLECT calls, `Agent.act` logs the exact LLM input (system prompt,
diary, phase context, retry corrections) at DEBUG via the `src.core.agent` logger
(`_render_trace`) — one record per provider attempt. Silent unless the caller configures
logging: set `LLM_TRACE=1` (env var or `.env`) and the runner attaches a handler. TALK
calls are not traced.

## Population & matchmaking

- `Population` (`src/population/base.py`) is a mutable roster that owns a **provider cache**
  keyed by `(base_url, model)`, so agents sharing a model share one httpx client
  (connection pooling); `aclose()` closes each unique provider once.
- `RosterGenerator` (`src/population/roster.py`) builds from an explicit spec list, cycled
  up to `n_agents`, sampling unique ids from the config name pools.
- `src/population/evolution.py` (`evolve`) implements the optional death/replacement step
  (see *Population evolution* below), using `Population.remove`/`draw_name` as its
  mutators over the live roster.
- `RandomMatchmaker` (`src/matchmaking/random_mm.py`) shuffles ids into disjoint pairs; an
  odd one out sits idle and earns `idle_payoff`. It uses its **own** rng stream
  (`Random(f"{seed}:matchmaker")`), so partitions are reproducible by seed regardless of
  what the games do.

## The orchestrator (`src/core/orchestrator.py`)

`run_episode(cfg, pop, *, observer)` has **side effects only** — it mutates agents
(score/memory) and emits each round to `observer`. It returns nothing: the **caller owns
the population**, builds it, reads final scores from it, and `aclose()`s it.

- The `observer` callback `(round, RoundPlan, list[PairingRecord])` is the **only output
  channel**. There is deliberately no `db_path` in the config — persistence plugs in here.
- Pairings within a round run concurrently under an `asyncio.Semaphore(cfg.max_concurrency)`.
- With a `schedule` (per-round change-points), the game is rebuilt from `cfg_for_round`
  each round, so payoffs, talk turns, prompts, etc. can phase across a run.

### Population evolution (death & replacement)

With `population.evolution` set ([configuration.md](./configuration.md)), `run_episode`
runs a death/replacement step (`src.population.evolution.evolve`) at the **start of every
round `r >= 2`**, before that round's `plan_round` — so a newborn can be paired the same
round it's born, and the roster for round `r` is a pure function of `(seed, r)` regardless
of where a run previously stopped (extend/resume stay deterministic). It reads the
**top-level** `cfg.population` — evolution parameters are not per-round schedulable (a
deliberate non-goal; `schedule` patches do not reach them).

It uses its own dedicated rng stream, `Random(f"{cfg.seed}:evolution:{r}")` — mirroring
the per-round matchmaker stream — and mutates `pop` in place (`Population.remove`/`add`).
The returned event dicts (`{"type": "death", "agent", "score"}` /
`{"type": "birth", "agent", "deceptive", "system_prompt", "provider"}`) are **prepended**
to `RoundPlan.events`, so deaths/births still reach the outside world only through the
existing `observer` channel, ahead of that round's idle/pairing events.

Resume re-derives roster changes instead of reading them back from the DB: `resume_run`
(`src/runner.py`) replays `evolve` for every already-played round `2..start_round - 1`
with the same per-round rng streams (pure — no LLM calls, no DB reads), and `run_episode`
then evolves round `start_round` itself, exactly as it does for every round it plays.
Stored scores/memories are applied afterward with `.get(...)` tolerance, so newborns (ids
the stored run never saw) simply keep their fresh state (score 0, empty memory).

`evolution.py`'s rng-consumption order is a documented compatibility contract: one
`random()` per live agent in roster order (deaths), then per replacement at most one
`random()` (type — skipped when forced by `decept_min`/`decept_max`) plus one
`randrange()` (name draw).

### Failure handling

LLM failures do **not** fail-fast mid-round. `play_pairing` catches a `ProviderError`
(or `ActParseError`) and returns the pairing as **unfinished** (`PairingRecord.finished =
False`, results NULL, full raw L2 log kept). The round finishes and is emitted to
`observer` (so the failure is persisted), then `run_episode` raises `EpisodeAborted` —
stopping the episode at a round boundary with `runs.finished_at` left NULL as a crash
marker. Such a run can be resumed later (see [configuration.md](./configuration.md),
*Resuming or extending*).

## Persistence, runner & analysis (callers)

The engine writes nothing itself. The **runner** (`src/runner.py`) is the caller that wires
the observer to storage:

- `run(cfg, db, name)` builds the population, drives `run_episode`, persists every round
  and every raw LLM call to SQLite, scores the run, optionally invokes the judge, and
  returns the new `run_id`.
- `resume_run(run_id, db, rounds)` reloads a stored run's config, rebuilds the population
  from the same seed, rehydrates score + memory from the DB, and plays only the missing
  rounds — to finish an aborted run or extend a finished one.

`src/storage/` holds the SQLite layer: `schema.py` (the `CREATE TABLE`s), `Storage`
(`store.py`), and record dataclasses (`records.py`). Run identity is an incremental integer
`run_id`; `runs.config_hash` groups runs of one **design** (config minus `judge` and
`rounds`). Full schema and query patterns: [database.md](./database.md).

`src/stats/` aggregates verdicts by design with Wilson confidence intervals
(`aggregate.py`, `wilson.py`, `selection.py`).

## LLM judge & keyword judge (`src/judge/`)

Optional post-episode analysis, invoked by the runner/callers after `run_episode` returns —
the orchestrator is untouched.

- **LLM judge** (`judge.py`): one call over the **public cheap-talk transcript only**
  (`transcript.py` builds it; private rationales, reflections, payoffs are hidden). Returns
  a `JudgeVerdict` (emerged: bool, explanation, evidence — validated message references),
  stored in `judge_verdicts`. Configured by its **own** `ProviderCfg`, fully independent of
  the agents (agents on Together.ai, judge on Ollama, or vice versa). Excluded from
  `config_hash`, so toggling it never changes a run's design family.
- **Keyword judge** (`keyword.py`): an LLM-free alternative. For a given term it counts how
  many distinct speakers mentioned it (case-sensitive substring match over public message
  text), stored in `keyword_counts`.

## Intentional seams (not all built out)

- **Persistence via the observer** — the DB never leaks into the orchestrator or `src/`.
- **Selection / evolution** — implemented (`src/population/evolution.py`, see *Population
  evolution* above) as uniform-random death + replacement. Not built out: fitness-based
  selection (death is uniform-random, not score-driven) and per-round scheduling of
  evolution parameters.
- **Interactive matchmakers** — `plan_round(..., actor=...)` and `RoundPlan.events` exist for
  matchmakers that query agents; `random` ignores them.
- **New provider / strategy / matchmaker / population / talk-rule** — each is a Protocol with
  a `make_*` factory; register there and extend `_validate` if needed (see
  [development.md](./development.md)).

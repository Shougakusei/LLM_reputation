# CLAUDE.md

Research simulation engine studying whether **reputation** emerges on its own in
populations of LLM agents that repeatedly play a coordination game with cheap talk.
One **episode** = one YAML config: the orchestrator builds a population, a matchmaker
pairs agents each round, each pair exchanges short messages (cheap-talk) and then
secretly picks a number 0–9; payoffs reward matching and punish being undercut.

The shared, checked-in documentation lives in `docs/` — keep it in sync after any change
(see Doc index below).

## Stack

Python 3.12, asyncio, httpx (OpenAI-compatible LLM calls), PyYAML, python-dotenv.
Tooling: `uv`, `pytest` + `pytest-asyncio` (auto mode — no `@pytest.mark.asyncio`
needed). No web framework — runs are driven from root scripts (`experiment.py`, `research.py`).

## Project map

```
src/
├── providers/      OpenAI-compatible HTTP client + retries (Together.ai, Ollama)
│   ├── base.py         LLMProvider Protocol, Message, Completion, error types
│   └── openai_compat.py  the real httpx client + make_provider
├── core/
│   ├── agent.py        Agent.act: memory + phase context -> LLM -> parsed JSON;
│   │                   phases TALK/DECIDE/PREDICT/REFLECT/NOTE; DEBUG trace of LLM input
│   ├── memory.py       per-agent diary of past rounds + optional consolidated notes
│   │                   (memory_notes_every), rendered into the prompt
│   ├── jsonextract.py  lenient JSON extraction (raw / fenced / balanced-brace)
│   ├── config.py       frozen dataclasses + load_episode + _validate (fail fast)
│   └── orchestrator.py run_episode: rounds loop, semaphore, observer callback
├── games/          reputation_pd.py (ReputationPD), prompts.py (English builders),
│                   talk_rules.py (TalkStopRule + make_talk_rule), base.py (PairingRecord)
├── strategy/       PlayStrategy: direct | prediction (map via match/one_above)
├── judge/          judge.py (LLM judge), keyword.py (deterministic term judge),
│                   transcript.py (public cheap-talk builder), base.py
├── population/     Population (live roster, provider cache, remove/draw_name) + RosterGenerator
│                   + evolution.py (death/replacement; deceptive & invincible agent flags)
├── matchmaking/    Matchmaker Protocol + RandomMatchmaker (disjoint pairs, idle)
├── storage/        SQLite persistence: schema.py, store.py (Storage), records.py
├── stats/          verdict aggregation: aggregate.py, wilson.py, selection.py
└── runner.py       run() / resume_run(): wires the observer to Storage, scores, judges
config/             one YAML = one episode; reference.yaml is a commented catalogue of
                    every knob; research*.yaml are research conditions
tests/              mirrors src/; unit tests stub the LLM, smoke tests hit Ollama
docs/               checked-in English docs (architecture / configuration / database / development)
```

Dependency flow is one-directional bottom→top: providers → core → games/strategy →
population/matchmaking → orchestrator → storage/stats/judge/runner. `games` and
`strategy` reference each other via lazy imports (`src/games/reputation_pd.py`).

## Commands

```bash
uv sync --extra dev                                   # install deps (incl. pytest)
uv run pytest                                          # all tests
uv run pytest tests/strategy/test_prediction.py       # single test file
uv run pytest -k resolve                              # by name
uv run python experiment.py [config.yaml] ["run name"]              # one run -> experiment.db
uv run python experiment.py --resume 87 [--rounds 20]              # finish/extend run #87
LLM_TRACE=1 uv run python experiment.py                            # + print exact LLM input per DECIDE/PREDICT/REFLECT
uv run python research.py                             # model×game sweep -> its own DB (idempotent/resumable)
uv run python replay.py <run_id|config_hash> [-c]    # replay a stored run (no LLM calls)
uv run python judge_runs.py                           # backfill LLM-judge verdicts
uv run python keyword_judge.py <term>                 # deterministic term-mention judge
uv run python find_gossip.py                          # third-player mentions in cheap talk
uv run python collect_stats.py                        # aggregate verdicts -> stats.json + stats.csv
uv run python plot_stats.py                           # stats.json -> stats.png (Wilson CI bars)
uv run python export_runs.py                          # split a DB into per-run .db files
```

`experiment.py` is the main entry point (there is no `examples/` demo). `research.py`
runs a grid (MODELS × `GAMES_PER_MODEL`) named `"<model> <n>"`: it first resumes every
unfinished run, then fills missing games by name, so re-running just continues.

Running an episode needs a reachable provider; the API key is read from `.env`
(`TOGETHER_API_KEY`). `.env` is gitignored — never commit it.

## Critical conventions (apply to nearly every task)

- **Everything is in English** — docstrings (Google-style), inline `#` comments, `print`s,
  logging (`_log.*` / `logging.*`), raised exception messages, LLM-facing prompt text
  (`src/games/prompts.py`), and `docs/` — across `src/`, root scripts, and `tests/`. This
  overrides the user's global default of Russian for these categories.
- TDD: write the failing test first (AAA, behavioural name), then minimal code.
- Manage dependencies only via `uv` (`uv add` / `uv remove` / `uv sync`) — never
  `pip` / `poetry`.
- No printing or persistence inside `src/` — output leaves the engine only through
  the orchestrator's `observer` callback (silent DEBUG logging is the one
  exception: handlers are configured by the caller, never in `src/`).
- `from __future__ import annotations` at the top of every module; rng objects are
  passed in, never created inside library code.

<important if="you are writing or modifying tests">
- Unit tests use the `ScriptedProvider` stub (per-package `tests/*/conftest.py`) —
  no network; queue exact JSON replies, assert on outcomes AND on the recorded
  `(system, messages)` calls.
- Smoke tests (`test_smoke_*_ollama.py`) hit a local Ollama and auto-skip if absent.
- `pythonpath = ["."]` is set — tests import `src.*` directly.
- Details and conventions: `docs/development.md`.
</important>

<important if="you are adding a new provider, strategy, matchmaker, population generator, or talk-stop rule">
Implement the Protocol and register it through its `make_*` factory (provider →
`make_provider`, strategy → `make_strategy`, matchmaker → `make_matchmaker`, population
→ `make_population`, talk-stop rule → `make_talk_rule`). If the new kind needs config
validation, extend `_validate` in `src/core/config.py`. Table: `docs/development.md`.
</important>

<important if="you are modifying the orchestrator or the episode run loop">
- The orchestrator's **only output channel is the `observer` callback**
  `(round, RoundPlan, list[PairingRecord])`; persistence lives in the runner, not `src/`.
- The **caller owns the `Population`**: it builds it and must `await pop.aclose()`
  (in a `finally`).
- Pairings in a round run concurrently under `asyncio.Semaphore(cfg.max_concurrency)`.
  LLM failures do NOT fail-fast mid-round: the pairing is returned unfinished
  (`finished=False`), the round is still emitted, then `EpisodeAborted` is raised at the
  round boundary (`finished_at` left NULL). Deep dive: `docs/architecture.md`.
</important>

<important if="you are adding or changing configuration fields or episode YAML">
- Config objects are frozen dataclasses in `src/core/config.py`; wire new fields
  through the `_*_cfg` builders / `load_episode`.
- Validate input once at load (`_validate`), fail fast with a clear message.
- Provider blocks are shared across agents via YAML anchors; `ProviderCfg` points
  at any OpenAI-compatible `/chat/completions` endpoint.
- Field reference, anchors, population pools, schedule change-points: `docs/configuration.md`.
</important>

<important if="you are changing Agent.act, phases, prompts, or memory rendering">
- The LLM input is assembled in `Agent.act` (`src/core/agent.py`): system =
  the agent's single `system_prompt` template taken **verbatim**, with only `{id}` and
  the payoff placeholders `{R}/{T}/{P}/{S}/{max_talk_turns}` substituted (the latter from
  `Phase.game_cfg`). A single user message: a step prompt is a string or a `PromptVariants`
  pair (`no_history`/`with_history`, picked by rounds played before the current one; NOTE
  excludes the just-played round; `no_history` optional). Memory fragments are substituted
  in place — `{recent_rounds}` `{history_lines}` `{notes_line}` `{notes}` `{history}` — via
  the paragraph rule: a blank-line-delimited paragraph whose fragments are all empty is
  dropped whole, headers included. A plain string naming none gets the memory prepended
  (legacy, for stored configs). (+ correction appended on JSON parse retry, then
  `ActParseError` — the pairing is aborted.)
- All phase prompts are static templates; only named placeholders are substituted.
  `PREDICT` mirrors `DECIDE`; there is one `decide_prompt` (no `_bare`) — the `rationale` flag
  only gates reading/storing the rationale (default off), not which template is sent.
- JSON extraction is lenient (`src/core/jsonextract.py`); validators per phase.
- DECIDE/PREDICT/REFLECT inputs are traced at DEBUG via the `src.core.agent` logger
  (`_render_trace`); keep the trace in sync if you change prompt assembly.
</important>

<important if="you are debugging what agents actually see or why they chose a number">
Run `LLM_TRACE=1 uv run python experiment.py` — it prints the full system prompt, memory
diary, and decide/predict context per provider attempt. In tests, use
`caplog.set_level(logging.DEBUG, logger="src.core.agent")` (examples at the bottom of
`tests/core/test_agent.py`).
</important>

<important if="you are modifying population evolution (death/replacement) or the deceptive/invincible flags">
- Rng consumption order in `src/population/evolution.py` is a compatibility contract:
  one `random()` per live agent in roster order (invincible agents consume and ignore
  their draw), then per replacement at most one `random()` (type) + one `randrange()`
  (name). Resume re-derives every roster change from `Random(f"{seed}:evolution:{r}")`,
  so reordering draws silently corrupts resumed runs.
- `deceptive` / `invincible` spec flags are honored only when `population.evolution` is
  set (normalized to false otherwise). Validation requires at least one deceptive and one
  non-deceptive spec, at least one mortal agent, and a name pool for replacements.
  Reference: `docs/configuration.md` (*Population evolution*).
</important>

<important if="you need to understand persistence or query stored runs">
SQLite: `src/storage/schema.py` is the source of truth. Run identity is an incremental
integer `run_id`; `config_hash` (config minus `judge` and `rounds`, with default-valued
evolution keys stripped so pre-feature configs keep their hash) groups a design family.
Tables, columns, and query patterns: `docs/database.md`.
</important>

## Doc index

Read the relevant doc before starting work on a related area, and update it after:

- `docs/architecture.md` — layered design, ReputationPD rules, pairing flow, strategies,
  agent phases, memory rendering, persistence, judge, intentional seams
- `docs/configuration.md` — episode YAML reference, game/population/judge/schedule blocks,
  provider anchors, resume/extend, how to add a config knob
- `docs/database.md` — SQLite schema, run_id vs config_hash, how to query runs
- `docs/development.md` — language rules, code patterns, testing, how to extend the engine

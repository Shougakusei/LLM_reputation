# Development

Working conventions for the codebase — language rules, patterns, testing, and how to
extend the engine. These are binding for any change.

## Language

**Everything is in English** — code (docstrings, inline comments), `print`s, logging,
error messages, LLM-facing prompt text, and docs. Docstrings are Google-style
(Args / Returns / Raises); see `src/games/prompts.py` for house style.

Known exception: the test fixture string `"Ты ИИ-игрок {id}. Play well."` in
`tests/core/test_agent.py` is intentionally kept as-is — it verifies that a system
prompt template passes through `Agent.act` verbatim (only `{id}` substituted),
regardless of language. Do not "fix" it to English.

## Code patterns

- **`from __future__ import annotations`** at the top of every module.
- **Frozen dataclasses for config** (`src/core/config.py`). Build them through the `_*_cfg`
  helpers and `load_episode`, never by hand, so validation runs.
- **Validate once at load** in `_validate` (fail fast, clear message) — not deep in the run.
- **rng is passed in, never created ad hoc** inside library code, so seeds control everything
  (`build(rng)`, `Matchmaker.setup(..., rng, ...)`).
- **Lazy imports to break cycles** between `games` and `strategy` — keep new cross-layer
  imports inside functions if they would otherwise cycle.

## Output & ownership

- The orchestrator's **only output channel is the `observer` callback**. Do not add printing
  or persistence inside `src/` — emit a `PairingRecord` and let the caller (runner) handle it.
  The one exception is silent DEBUG logging: handlers are configured by the caller, never in
  `src/`.
- **The caller owns the `Population`**: it builds it, reads final scores from it, and must
  `await pop.aclose()` in a `finally`.

## Dependency direction

`src → tests` only; never import from `tests/` into `src/`. Within `src/`, respect the
layering in [architecture.md](./architecture.md) — lower layers never import higher ones.

## Extending the engine (Protocols + factories)

Each seam is a `Protocol` implemented and registered through a `make_*` factory. To add a
new kind, implement the Protocol, register it in the factory, and — if it needs config
validation — extend `_validate` in `src/core/config.py`.

| seam | Protocol | factory |
|------|----------|---------|
| provider | `LLMProvider` (`src/providers/base.py`) | `make_provider` (`src/providers/openai_compat.py`) |
| strategy | `PlayStrategy` (`src/strategy/base.py`) | `make_strategy` (`src/strategy/base.py`) |
| matchmaker | `Matchmaker` (`src/matchmaking/base.py`) | `make_matchmaker` (`src/matchmaking/base.py`) |
| population | `PopulationGenerator` (`src/population/base.py`) | `make_population` (`src/population/base.py`) |
| talk-stop rule | `TalkStopRule` (`src/games/talk_rules.py`) | `make_talk_rule` (`src/games/talk_rules.py`) |

## Testing

`pytest` with `pytest-asyncio` in **auto mode** (`asyncio_mode = "auto"`) — `async def
test_*` needs no `@pytest.mark.asyncio`. `pythonpath = ["."]` is set, so tests import
`src.*` directly. Test files mirror the `src/` package layout under `tests/`.

```bash
uv run pytest                          # all tests
uv run pytest tests/strategy           # one package
uv run pytest tests/core/test_agent.py # one file
uv run pytest -k resolve               # by name
```

**Two kinds of tests:**

- **Unit tests** (the default) — no network. `ScriptedProvider` (defined per-package in
  `tests/*/conftest.py`) is an `LLMProvider` double that returns a queued list of reply
  strings in order and records every `(system, messages)` call. Queue the exact JSON replies
  the LLM would produce, then assert on outcomes **and** on what the agent was prompted with.
- **Smoke tests** — `test_smoke_*_ollama.py` hit a local Ollama at
  `http://localhost:11434/v1`, guarded by a `_ollama_up()` reachability check, so they
  silently skip when Ollama isn't running. They assert only on shape (valid outcome, number
  in range), not exact values.

**TDD:** write the failing test first (AAA — Arrange/Act/Assert, one assertion where
practical, behavioural name `test_<behaviour>`), then the minimal implementation.

## Determinism

`seed` makes runs reproducible: the population build and a derived matchmaker rng stream
(`Random(f"{seed}:matchmaker")`) are both seeded. The LLM is the only nondeterministic part
— hence unit tests stub it with `ScriptedProvider`.

## Dependencies & tooling

Manage dependencies **only** via `uv` (`uv add` / `uv remove` / `uv sync`) — never `pip` or
`poetry`. Install with `uv sync --extra dev`. Running an episode needs a reachable provider;
the API key is read from `.env` (`TOGETHER_API_KEY`), which is gitignored — never commit it.

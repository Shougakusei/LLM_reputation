# LLM_reputation

Does an institution of **reputation** emerge on its own in a group of AI agents?

A population of LLM agents plays a coordination game round after round, with pre-play
negotiation (cheap talk). The question is whether reputation forms between agents without
any external rules — purely from repeated interaction.

## How it works

One **episode** = one YAML config. The orchestrator builds a population of agents; a
matchmaker splits them into pairs each round; within each pair the agents exchange a few
short messages and then secretly pick a number 0–9. The payoff rewards matching and
punishes being undercut (a modified prisoner's dilemma on a 0–9 cycle). Every round is
persisted to SQLite for later replay and analysis.

The play strategy is set per agent in the config:

- `direct` — the agent picks a number directly;
- `prediction` — the agent predicts its partner's number, and a mapping (`match` /
  `one_above`) turns that prediction into its own choice.

Two optional, config-driven extensions:

- **population evolution** — each round agents may die and be replaced by newcomers;
  agent specs can be marked `deceptive` (tracked through replacements) or `invincible`
  (never dies). This represents the real world dynamics of a reputation in an open system;
- **memory notes** — every N rounds an agent compresses its diary into private notes
  that replace the raw history in its prompt.

Full design: [`docs/architecture.md`](docs/architecture.md).

## Install

```bash
uv sync --extra dev          # dependencies (including pytest)
```

Create a `.env` with your provider key (Together.ai, an OpenAI-compatible endpoint, is the
default). `.env` is gitignored — never commit it.

```
TOGETHER_API_KEY=your_key
```

## Run

```bash
uv run python experiment.py                       # config/experiment.yaml -> experiment.db
uv run python experiment.py config/other.yaml     # a different episode
uv run python experiment.py config/other.yaml "run name"   # + a human label

uv run python experiment.py --resume 87           # finish an aborted run #87
uv run python experiment.py --resume 87 --rounds 20   # extend #87 to 20 rounds
```

Each invocation appends a new run to `experiment.db`. Set `LLM_TRACE=1` to print the exact
LLM input of every decision. Config reference: [`docs/configuration.md`](docs/configuration.md);
[`config/reference.yaml`](config/reference.yaml) is a commented catalogue of every parameter
that also loads as a valid episode.

### Model sweep

`research.py` runs a grid (models × games) into its own DB, splitting each run into a
per-run file. It is idempotent/resumable — it first finishes any unfinished runs, then fills
missing games by name, so re-running just continues where it left off:

```bash
uv run python research.py                          # config/research.yaml
```

## Analysis scripts

All read a stored DB; run from the repo root. Data model:
[`docs/database.md`](docs/database.md).

```bash
uv run python replay.py                 # list runs in the DB
uv run python replay.py <run_id>        # replay a run round by round (no LLM calls)
uv run python replay.py <run_id> -c     # + prompts, roster and config
                                        # <run_id> is a number or a config_hash

uv run python judge_runs.py             # backfill LLM-judge verdicts over stored runs
uv run python keyword_judge.py <term>   # count distinct speakers of a term (LLM-free judge)
uv run python find_gossip.py            # find "gossip" — mentions of a third player in cheap talk
uv run python collect_stats.py          # aggregate verdicts by design -> stats.json + stats.csv
uv run python infection_validation.py   # infection study: does the cooperator prompt hold? (100 one-round games)
uv run python infection_research.py     # infection study: one subject vs a fixed sequence of fresh NPCs
uv run python infection_stats.py        # infection study: subject action per round position -> JSON
uv run python plot_stats.py             # stats.json -> stats.png (Wilson CI bars)
uv run python export_runs.py            # split a DB into per-run .db files
modal deploy modal_vllm.py              # self-host an open model on Modal GPUs (see docs/configuration.md)
```

### The LLM judge

An optional model that decides, after an episode, whether an institution of reputation
emerged — it sees only the public cheap-talk transcript. Enable it by adding a `judge:`
block to the config (see [`docs/configuration.md`](docs/configuration.md)). The judge is
configured fully independently of the agents (its own `provider`), so agents and judge can
run on different endpoints — e.g. agents on Together.ai, judge on a local Ollama.

## Tests

```bash
uv run pytest
```

Unit tests never hit the network (the LLM is replaced with a stub). Smoke tests talk to a
local Ollama and skip automatically if it is unavailable. Conventions:
[`docs/development.md`](docs/development.md).

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — layered design, the game, pairing flow,
  strategies, agent phases, persistence, judge, seams
- [`docs/configuration.md`](docs/configuration.md) — episode YAML reference
- [`docs/database.md`](docs/database.md) — SQLite schema and how to query it
- [`docs/development.md`](docs/development.md) — language rules, code patterns, testing,
  how to extend the engine

## Team

[Andrey Seryakov](https://github.com/AndreySeryakov) ·
[Ekaterina Krupkina](https://github.com/ktchka) ·
[Andrey Bystrov](https://github.com/Shougakusei) ·
[Dmitrii Salnikov](https://github.com/DmitrySalnikov)

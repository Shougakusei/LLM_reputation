# Data model (SQLite)

Every run is persisted to one SQLite file — "one DB, many runs". `experiment.py` writes
`experiment.db`; `research.py` writes its own DB and also splits each run into a
per-run file. The schema is defined in `src/storage/schema.py` and applied by
`init_schema`; `Storage` (`src/storage/store.py`) is the write/read API and record types
live in `src/storage/records.py`.

The engine itself never touches the DB — persistence happens in the runner via the
orchestrator's `observer` callback (see [architecture.md](./architecture.md)).

## Run identity: `run_id` vs `config_hash`

- **`run_id`** — an incremental integer (`INTEGER PRIMARY KEY AUTOINCREMENT`). This is a
  run's identity. Every invocation is a **new** run; re-running the same config does not
  de-dup, it just gets the next number (handy for repeated runs under a noisy LLM).
- **`config_hash`** — SHA-256 of the config **minus `judge` and `rounds`**. It groups runs
  of one **design** into a "family". `rounds` is excluded because, with per-round matchmaker
  rng, round `r` is identical regardless of total length — a longer run is just a shorter one
  continued. `judge` is excluded so toggling the judge never changes the family.
- `replay.py` accepts **either** an integer `run_id` or a `config_hash` (the latter resolves
  to the earliest run of that family).
- The hash also strips the population-evolution feature's default-valued keys
  (`population.evolution: None`, and `deceptive: False` / `invincible: False` on each
  agent spec) before hashing, so evolution-free configs keep the exact `config_hash` they
  had before the feature existed. Evolution-enabled configs keep both keys — that's a
  genuinely new design, and gets its own hash.
- `runs.finished_at` NULL = the run crashed/aborted mid-episode (a resume marker).

## Tables

| table | grain | notes |
|-------|-------|-------|
| `runs` | one run | `name`, `config` (JSON), `config_hash`, `seed`, `created_at`, `finished_at` |
| `agents` | one agent in a run | `system_prompt`, `provider` (JSON — model lives here), `final_score`, `born_round`, `died_round`, `deceptive`, `invincible` (see below) |
| `rounds` | one round | just `(run_id, round_idx)` |
| `idle` | agent idle in a round | the odd-one-out that sat a round out |
| `pairings` | one pair in a round | the heart of the results (see below) |
| `messages` | one cheap-talk line | `speaker`, `text`, `ready`; `turn_idx` orders them |
| `llm_calls` | one HTTP attempt | verbatim request/response of **every** call incl. retries & failures |
| `judge_verdicts` | one run | LLM-judge output: `emerged`, `explanation`, `evidence`, `model` |
| `keyword_counts` | one (run, term) | distinct-speaker count for a keyword-judge term |

### `pairings` (results)

Columns are from A's / B's perspective:

- `a_id`, `b_id` — the two agents; `a` opened the cheap talk.
- `finished` — `1` = played to completion, `0` = aborted by an LLM failure. Two CHECK
  constraints enforce: finished ⇒ results present; aborted ⇒ results NULL.
- `a_number`, `b_number` — the chosen integers 0–9.
- `a_outcome` — `CC` / `DC` / `CD` / `DD` from A's perspective; `a_payoff`, `b_payoff`.
- `a_rationale`, `b_rationale` — private reasoning (NULL when `game.rationale=false`).
- `a_predicted`, `b_predicted` — prediction-strategy guess (NULL for direct agents).
- `a_reflection`, `b_reflection` — post-game reflection (NULL when `game.reflection=false`).
- `a_notes`, `b_notes` — memory notes written after this round (NULL when not consolidated).
- `usage_prompt_tokens`, `usage_completion_tokens`, `usage_calls` — token accounting.

### `agents` (population evolution columns)

Four columns support the optional population-evolution feature
([configuration.md](./configuration.md), *Population evolution*):

- `born_round` (`INTEGER NOT NULL DEFAULT 1`) — the round the agent entered the
  population: `1` for every agent in the initial roster (the schema default), the
  evolution round number for a replacement.
- `died_round` (`INTEGER`, nullable) — the round the agent was removed; `NULL` = still
  alive as of `finished_at` (or the run's last played round, if unfinished).
- `deceptive` (`INTEGER NOT NULL DEFAULT 0`) — `1` if the agent was cloned from (or is an
  initial agent of) a `deceptive: true` spec, `0` otherwise.
- `invincible` (`INTEGER NOT NULL DEFAULT 0`) — `1` if the agent is an initial agent of an
  `invincible: true` spec, `0` otherwise. Replacements are always born mortal, so this is
  `0` on every birth row regardless of the spec they were cloned from.

Added by an **additive `ALTER TABLE`** migration that runs on every `Storage.__init__`
(`init_schema`, `src/storage/schema.py`), so opening a DB created before evolution
existed backfills the four columns onto existing rows with their defaults
(`born_round=1`, `died_round=NULL`, `deceptive=0`, `invincible=0`) — such a run simply
shows a full initial roster with no deaths.

These columns are **analysis-only**: `resume_run` never reads them back. The live roster
at a resume point is rebuilt by deterministically re-deriving evolution from
`(seed, round)`, not from stored agent rows (see [architecture.md](./architecture.md)).

### `llm_calls` (raw L2 audit log)

Every HTTP call to the provider, including network retries and parse-retries, is logged
verbatim: `phase` (`talk|decide|predict|reflect|note`), `attempt` (parse attempt 1..3),
`http_attempt` (network retry 1..5), `status` (`ok|parse_error|bad_json|bad_shape|
http_error|server_error|network`), `status_code`, the exact `request` payload, the extracted
`response` (final ok attempt only), the verbatim `response_raw` (incl. 5xx bodies), `error`,
and token counts. This is what lets you reconstruct exactly what an agent saw and said.

Indexes: `llm_calls(run_id, agent_id)`, `llm_calls(run_id, status)`, `runs(config_hash)`.

## Reading runs back

CLI scripts (run from the repo root; see the README for the full list):

- `replay.py <run_id|config_hash>` — replay a run round by round, no LLM calls; `-c` adds
  prompts, roster and config. Judge-cited messages are highlighted.
- `judge_runs.py` — backfill LLM-judge verdicts over stored runs (`--force`, `--config`,
  `--design`, `--name` filters).
- `keyword_judge.py <term>` — count distinct speakers of a term across runs → `keyword_counts`.
- `collect_stats.py` — aggregate judge verdicts by `config_hash` with 95% Wilson CIs →
  `stats.json` + `stats.csv`.
- `export_runs.py` — split a DB into per-run `.db` files.

The run-selection filters (`--design` / `--exclude-design` / `--name` / `--exclude-name`)
are shared by `judge_runs.py`, `keyword_judge.py` and `collect_stats.py`.

## Quick queries

Model per run lives in `agents.provider` (JSON). Count runs per model:

```python
import sqlite3, json
from collections import Counter

conn = sqlite3.connect("research.db")
model_of = {}                                  # run_id -> model (one agent is enough)
for run_id, provider in conn.execute("SELECT run_id, provider FROM agents"):
    model_of.setdefault(run_id, json.loads(provider).get("model"))

total, finished = Counter(), Counter()
for run_id, fin in conn.execute("SELECT run_id, finished_at FROM runs"):
    m = model_of.get(run_id)
    total[m] += 1
    if fin:
        finished[m] += 1
```

Finished pairings for a run, with numbers and payoffs:

```sql
SELECT round_idx, a_id, b_id, a_number, b_number, a_outcome, a_payoff, b_payoff
FROM pairings WHERE run_id = ? AND finished = 1
ORDER BY round_idx, pair_idx;
```

Who died when (population evolution), with their final score:

```sql
SELECT agent_id, died_round, final_score, deceptive
FROM agents
WHERE run_id = ? AND died_round IS NOT NULL
ORDER BY died_round;
```

Deceptive share of the live roster at the start of round `r` (agent already born, not
yet dead as of that round) — bind `(run_id, r, r)`, matching the `?` placeholder order:

```sql
SELECT
    SUM(deceptive) AS deceptive_alive,
    COUNT(*)       AS total_alive
FROM agents
WHERE run_id = ?
  AND born_round <= ?
  AND (died_round IS NULL OR died_round > ?);
```

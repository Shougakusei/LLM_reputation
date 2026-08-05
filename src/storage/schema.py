from __future__ import annotations

import sqlite3

# Normalized L1 schema (one DB, many runs). Configs are JSON (runs.config,
# agents.provider); everything operational is normalized. See agent-games-logger-plan §4.
#
# run_id — an integer autoincrement (human-readable 1, 2, 3 …), NOT a config hash.
# A run's identity is its number; the "design" instead is hashed into runs.config_hash (a hash
# of the config without `judge` and without `rounds`) — to group runs of the same design into
# a family (repeats and continuations of different length share config_hash). replay accepts
# both a number and a hash.
SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT,
    config      TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    seed        INTEGER NOT NULL,
    created_at  TEXT NOT NULL,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS agents (
    run_id        INTEGER NOT NULL,
    agent_id      TEXT NOT NULL,
    system_prompt TEXT,
    provider      TEXT NOT NULL,
    final_score   REAL,
    born_round    INTEGER NOT NULL DEFAULT 1,
    died_round    INTEGER,
    deceptive     INTEGER NOT NULL DEFAULT 0,
    invincible    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (run_id, agent_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS rounds (
    run_id    INTEGER NOT NULL,
    round_idx INTEGER NOT NULL,
    PRIMARY KEY (run_id, round_idx),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS idle (
    run_id    INTEGER NOT NULL,
    round_idx INTEGER NOT NULL,
    agent_id  TEXT NOT NULL,
    PRIMARY KEY (run_id, round_idx, agent_id),
    FOREIGN KEY (run_id, round_idx) REFERENCES rounds(run_id, round_idx) ON DELETE CASCADE,
    FOREIGN KEY (run_id, agent_id)  REFERENCES agents(run_id, agent_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS pairings (
    run_id      INTEGER NOT NULL,
    round_idx   INTEGER NOT NULL,
    pair_idx    INTEGER NOT NULL,
    a_id        TEXT NOT NULL,
    b_id        TEXT NOT NULL,
    finished    INTEGER NOT NULL DEFAULT 1,  -- 1 = played to completion; 0 = aborted by an LLM failure (no results)
    a_number    INTEGER,                     -- results are NULL if finished=0 (see CHECK)
    b_number    INTEGER,
    a_rationale TEXT,
    b_rationale TEXT,
    a_outcome   TEXT,
    a_payoff    REAL,
    b_payoff    REAL,
    a_predicted INTEGER,                    -- prediction strategy: a's guess of b's number (NULL for direct)
    b_predicted INTEGER,
    a_reflection TEXT,                      -- post-game reflection (NULL when game.reflection=false)
    b_reflection TEXT,
    a_notes     TEXT,                       -- memory notes after the round (NULL if not consolidated)
    b_notes     TEXT,
    usage_prompt_tokens     INTEGER,
    usage_completion_tokens INTEGER,
    usage_calls             INTEGER,
    PRIMARY KEY (run_id, round_idx, pair_idx),
    FOREIGN KEY (run_id, round_idx) REFERENCES rounds(run_id, round_idx) ON DELETE CASCADE,
    FOREIGN KEY (run_id, a_id) REFERENCES agents(run_id, agent_id) ON DELETE CASCADE,
    FOREIGN KEY (run_id, b_id) REFERENCES agents(run_id, agent_id) ON DELETE CASCADE,
    CHECK (finished = 0 OR a_number IS NOT NULL),   -- played to completion ⇒ result exists
    CHECK (finished = 1 OR a_number IS NULL)        -- aborted               ⇒ result is empty
);

CREATE TABLE IF NOT EXISTS messages (
    run_id    INTEGER NOT NULL,
    round_idx INTEGER NOT NULL,
    pair_idx  INTEGER NOT NULL,
    turn_idx  INTEGER NOT NULL,
    speaker   TEXT NOT NULL,
    text      TEXT NOT NULL,
    ready     INTEGER NOT NULL,
    PRIMARY KEY (run_id, round_idx, pair_idx, turn_idx),
    FOREIGN KEY (run_id, round_idx, pair_idx) REFERENCES pairings(run_id, round_idx, pair_idx) ON DELETE CASCADE,
    FOREIGN KEY (run_id, speaker) REFERENCES agents(run_id, agent_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS llm_calls (
    run_id        INTEGER NOT NULL,
    round_idx     INTEGER NOT NULL,
    pair_idx      INTEGER NOT NULL,
    call_idx      INTEGER NOT NULL,   -- call order within the pair (execution order)
    agent_id      TEXT    NOT NULL,   -- who made the call
    phase         TEXT    NOT NULL,   -- talk | decide | predict | reflect | note
    turn_idx      INTEGER,            -- NULL except for TALK; FK to the specific messages turn
    attempt       INTEGER NOT NULL,   -- Agent.act parse attempt (1..3)
    http_attempt  INTEGER NOT NULL,   -- network retry within complete() (1..5)
    status        TEXT    NOT NULL,   -- ok | parse_error | bad_json | bad_shape | http_error | server_error | network
    status_code   INTEGER,            -- HTTP status code of the attempt (NULL on a network error)
    request       TEXT    NOT NULL,   -- VERBATIM payload (JSON)
    response      TEXT,               -- extracted text (only on the final ok attempt)
    response_raw  TEXT,               -- VERBATIM resp.text body (incl. a 5xx body); NULL on a network error
    error         TEXT,               -- failure message
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (run_id, round_idx, pair_idx, call_idx),
    FOREIGN KEY (run_id, round_idx, pair_idx) REFERENCES pairings(run_id, round_idx, pair_idx) ON DELETE CASCADE,
    FOREIGN KEY (run_id, agent_id) REFERENCES agents(run_id, agent_id) ON DELETE CASCADE,
    FOREIGN KEY (run_id, round_idx, pair_idx, turn_idx)
        REFERENCES messages(run_id, round_idx, pair_idx, turn_idx) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS judge_verdicts (
    run_id      INTEGER PRIMARY KEY,
    emerged     INTEGER NOT NULL,
    explanation TEXT NOT NULL,
    evidence    TEXT NOT NULL,      -- JSON: [{"round":0,"pair":1,"turn":2}, ...]
    model       TEXT NOT NULL,      -- judge model, for the record
    created_at  TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS keyword_counts (
    run_id     INTEGER NOT NULL,
    term       TEXT NOT NULL,
    count      INTEGER NOT NULL,   -- number of distinct speakers who mentioned the term
    speakers   TEXT NOT NULL,      -- JSON list of speaker ids
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, term),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS ix_llm_calls_agent  ON llm_calls(run_id, agent_id);
CREATE INDEX IF NOT EXISTS ix_llm_calls_status ON llm_calls(run_id, status);
CREATE INDEX IF NOT EXISTS ix_runs_config_hash ON runs(config_hash);
"""


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    # Additive migration for databases created before the evolution columns existed.
    cols = {row[1] for row in conn.execute("PRAGMA table_info(agents)")}
    for name, ddl in (("born_round", "INTEGER NOT NULL DEFAULT 1"),
                      ("died_round", "INTEGER"),
                      ("deceptive", "INTEGER NOT NULL DEFAULT 0"),
                      ("invincible", "INTEGER NOT NULL DEFAULT 0")):
        if name not in cols:
            conn.execute(f"ALTER TABLE agents ADD COLUMN {name} {ddl}")
    conn.commit()

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from src.core.config import EpisodeCfg
from src.core.memory import Memory, MemoryEntry
from src.games.base import PairingRecord
from src.judge import JudgeVerdict, KeywordCount
from src.matchmaking import RoundPlan
from src.population import Population
from src.storage.schema import init_schema

# Outcome from A's perspective -> from B's perspective (same as _FLIP in reputation_pd).
_FLIP = {"CC": "CC", "DD": "DD", "DC": "CD", "CD": "DC"}


@dataclass
class RunState:
    """A run snapshot reconstructed from the DB for resuming.

    last_round — the index of the last recorded round (0 if none). scores — the accumulated
    score per agent. memories — the reconstructed Memory per agent (diary + notes), ready to
    be overlaid onto a freshly built population (A4)."""

    last_round: int
    scores: dict[str, float]
    memories: dict[str, Memory]


def _hash_config_dict(d: dict) -> str:
    """Hash of the "experiment design": the config without `judge` and without `rounds`.

    `judge` is analytics, not gameplay. `rounds` is excluded deliberately: with per-round
    rng, round r is identical regardless of the total length, so "20 rounds" is just "10
    rounds" played further; the round count is "how far the simulation got", not part of the
    design's identity. Hence a run, its repeats, and its extensions of different lengths
    share one config_hash (one "family")."""
    d = dict(d)
    d.pop("judge", None)
    d.pop("rounds", None)
    canon = json.dumps(d, sort_keys=True)               # stable across processes
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def _config_hash(cfg: EpisodeCfg) -> str:
    """A run's config_hash — the design hash (see _hash_config_dict). Not the run's identity
    (that's the integer runs.run_id), but a tag for grouping runs of the same design."""
    return _hash_config_dict(asdict(cfg))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Storage:
    """Persists one episode to SQLite (L1). Subscribes to the orchestrator's observer
    seam — the engine is unchanged. See agent-games-logger-plan.md."""

    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        init_schema(self._conn)
        self._run_id: int | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        """Read-only access to the connection (reconstructing records, listing runs)."""
        return self._conn

    def has_verdict(self, run_id: int) -> bool:
        """True if the run already has a judge verdict."""
        row = self._conn.execute(
            "SELECT 1 FROM judge_verdicts WHERE run_id=?", (run_id,)
        ).fetchone()
        return row is not None

    def begin(self, cfg: EpisodeCfg, pop: Population, name: str | None = None) -> int:
        """Step 0: write the run + agents; returns run_id — an integer autoincrement.

        Every call creates a NEW run (there is no longer config-based dedup: re-running the
        same config gets a new number). The config's tag goes into config_hash. `name` is an
        optional human-readable label (metadata)."""
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO runs(name, config, config_hash, seed, created_at) VALUES (?,?,?,?,?)",
                (name, json.dumps(asdict(cfg)), _config_hash(cfg), cfg.seed, _now()),
            )
            run_id = cur.lastrowid
            self._run_id = run_id
            self._conn.executemany(
                "INSERT INTO agents(run_id, agent_id, system_prompt, provider, deceptive) VALUES (?,?,?,?,?)",
                [
                    (run_id, a.id, a.setup.system_prompt,
                     json.dumps(asdict(a.setup.provider_cfg)), int(a.setup.deceptive))
                    for a in pop
                ],
            )
        return run_id

    def is_finished(self, run_id: int) -> bool:
        """True if the run has been played to completion (finished_at is set); False for an
        interrupted or missing run."""
        row = self._conn.execute(
            "SELECT finished_at FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        return bool(row and row[0])

    def unfinished_runs(self) -> list[tuple[int, str | None]]:
        """All unfinished runs (finished_at IS NULL) as (run_id, name), in ascending id order.

        Needed by aggregate scripts (research.py), which first finish off interrupted runs
        and then fill in what's missing."""
        return [
            (row[0], row[1])
            for row in self._conn.execute(
                "SELECT run_id, name FROM runs WHERE finished_at IS NULL ORDER BY run_id"
            )
        ]

    def run_id_by_name(self, name: str) -> int | None:
        """run_id of the first run with this name (or None). The name is a human-readable run
        label; the aggregate script uses it to look up what has already been computed or
        started (resume by run_id)."""
        row = self._conn.execute(
            "SELECT run_id FROM runs WHERE name=? ORDER BY run_id LIMIT 1", (name,)
        ).fetchone()
        return row[0] if row else None

    def run_config(self, run_id: int) -> str | None:
        """Return the run's stored config (JSON string), or None if it has none.
        Used when resuming: EpisodeCfg is reconstructed from it."""
        row = self._conn.execute(
            "SELECT config FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        return row[0] if row else None

    def resume(self, run_id: int, cfg: EpisodeCfg) -> None:
        """Prepare Storage to append to an existing run (resume/extend).

        Remembers run_id (for observe/finish), clears finished_at (marking it "in progress" —
        if the extension is interrupted, the run correctly stays unfinished), and updates the
        config (on extend the round count grew; config_hash is NOT touched — rounds isn't
        part of it, so the design family is preserved)."""
        self._run_id = run_id
        with self._conn:
            self._conn.execute(
                "UPDATE runs SET finished_at=NULL, config=? WHERE run_id=?",
                (json.dumps(asdict(cfg)), run_id),
            )

    def delete_run(self, run_id: int) -> None:
        """Delete the run and all its rows — child tables cascade
        (FK with ON DELETE CASCADE; PRAGMA foreign_keys=ON is set in __init__)."""
        with self._conn:
            self._conn.execute("DELETE FROM runs WHERE run_id=?", (run_id,))

    def load_state(self, run_id: int, idle_payoff: float) -> RunState:
        """Reconstruct a run's state from the DB (pure read) for resuming.

        For each agent, gathers Memory (a diary from messages/pairings, notes and their
        boundary noted_upto from a_notes/b_notes) and the accumulated score (sum of payoffs
        from finished pairs + idle_payoff for skipped rounds), plus the index of the last
        recorded round. Entries are built in round order so that the score field ("score
        BEFORE the round") in the diary matches a live run.

        Aborted pairs (finished=0) are excluded from memory and score — just as in a live
        game their agents didn't finish that round. idle_payoff is passed in from the config
        (it isn't in the DB)."""
        c = self._conn
        agent_ids = [r[0] for r in c.execute(
            "SELECT agent_id FROM agents WHERE run_id=? ORDER BY agent_id", (run_id,))]
        memories: dict[str, Memory] = {aid: Memory() for aid in agent_ids}
        running: dict[str, float] = {aid: 0.0 for aid in agent_ids}

        rounds = [r[0] for r in c.execute(
            "SELECT round_idx FROM rounds WHERE run_id=? ORDER BY round_idx", (run_id,))]

        idle_by_round: dict[int, list[str]] = defaultdict(list)
        for ri, aid in c.execute(
                "SELECT round_idx, agent_id FROM idle WHERE run_id=?", (run_id,)):
            idle_by_round[ri].append(aid)

        pair_by_round: dict[int, list] = defaultdict(list)
        for row in c.execute(
                """SELECT round_idx, pair_idx, a_id, b_id, a_number, b_number,
                          a_rationale, b_rationale, a_outcome, a_payoff, b_payoff,
                          a_predicted, b_predicted, a_reflection, b_reflection, a_notes, b_notes
                   FROM pairings WHERE run_id=? AND finished=1
                   ORDER BY round_idx, pair_idx""", (run_id,)):
            pair_by_round[row[0]].append(row)

        for ri in rounds:
            for aid in idle_by_round.get(ri, ()):          # idle: score only, no diary entry
                if aid in running:
                    running[aid] += idle_payoff
            for row in pair_by_round.get(ri, ()):
                (_, pair_idx, a_id, b_id, a_number, b_number, a_rationale, b_rationale,
                 a_outcome, a_payoff, b_payoff, a_predicted, b_predicted,
                 a_reflection, b_reflection, a_notes, b_notes) = row
                transcript = self._load_transcript(run_id, ri, pair_idx)
                # side A — as-is; side B — mirror the perspective (partner, numbers, outcome)
                self._restore_entry(memories[a_id], running, a_id, ri, b_id, transcript,
                                    a_number, a_rationale, b_number, a_outcome,
                                    a_payoff, b_payoff, a_predicted, a_reflection, a_notes)
                self._restore_entry(memories[b_id], running, b_id, ri, a_id, transcript,
                                    b_number, b_rationale, a_number, _FLIP.get(a_outcome, a_outcome),
                                    b_payoff, a_payoff, b_predicted, b_reflection, b_notes)

        last_round = rounds[-1] if rounds else 0
        return RunState(last_round=last_round, scores=dict(running), memories=memories)

    def _load_transcript(self, run_id: int, round_idx: int, pair_idx: int) -> list[dict]:
        return [
            {"speaker": s, "text": t, "ready": bool(r)}
            for s, t, r in self._conn.execute(
                "SELECT speaker, text, ready FROM messages "
                "WHERE run_id=? AND round_idx=? AND pair_idx=? ORDER BY turn_idx",
                (run_id, round_idx, pair_idx))
        ]

    @staticmethod
    def _restore_entry(memory: Memory, running: dict[str, float], aid: str, round: int,
                       partner: str, transcript: list[dict], my_number, my_rationale,
                       partner_number, outcome, payoff, partner_payoff, my_predicted,
                       my_reflection, notes) -> None:
        # score — the score BEFORE this round (same as in the live _remember: agent.score - payoff)
        memory.add(MemoryEntry(
            round=round, my_id=aid, partner_id=partner, transcript=transcript,
            my_number=my_number, my_rationale=my_rationale or "",
            partner_number=partner_number, outcome=outcome,
            payoff=payoff, partner_payoff=partner_payoff, score=running[aid],
            my_predicted=my_predicted, my_reflection=my_reflection,
        ))
        running[aid] += payoff
        if notes is not None:                              # the round on which the agent folded notes
            memory.set_notes(notes)

    def observe(self, round: int, plan: RoundPlan, recs: list[PairingRecord]) -> None:
        """Step R: one transaction per round — rounds + idle + pairings + messages.
        This is the orchestrator observer (sync; sqlite3 is synchronous)."""
        rid = self._run_id
        with self._conn:
            self._conn.execute(
                "INSERT INTO rounds(run_id, round_idx) VALUES (?,?)", (rid, round)
            )
            # Persist evolution events (deaths and births) at round start.
            for e in plan.events:
                if e.get("type") == "death":
                    self._conn.execute(
                        "UPDATE agents SET died_round=?, final_score=? "
                        "WHERE run_id=? AND agent_id=?",
                        (round, e.get("score"), rid, e["agent"]))
                elif e.get("type") == "birth":
                    self._conn.execute(
                        "INSERT INTO agents(run_id, agent_id, system_prompt, provider, "
                        "born_round, deceptive) VALUES (?,?,?,?,?,?)",
                        (rid, e["agent"], e.get("system_prompt"),
                         json.dumps(e.get("provider")), round,
                         int(bool(e.get("deceptive")))))
            self._conn.executemany(
                "INSERT INTO idle(run_id, round_idx, agent_id) VALUES (?,?,?)",
                [(rid, round, aid) for aid in plan.idle],
            )
            for pair_idx, rec in enumerate(recs):
                u = rec.usage or {}
                self._conn.execute(
                    """INSERT INTO pairings(
                           run_id, round_idx, pair_idx, a_id, b_id, finished,
                           a_number, b_number, a_rationale, b_rationale,
                           a_outcome, a_payoff, b_payoff, a_predicted, b_predicted,
                           a_reflection, b_reflection, a_notes, b_notes,
                           usage_prompt_tokens, usage_completion_tokens, usage_calls)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        rid, round, pair_idx, rec.a_id, rec.b_id, int(rec.finished),
                        rec.a_number, rec.b_number, rec.a_rationale, rec.b_rationale,
                        rec.outcome, rec.a_payoff, rec.b_payoff, rec.a_predicted, rec.b_predicted,
                        rec.a_reflection, rec.b_reflection, rec.a_notes, rec.b_notes,
                        u.get("prompt_tokens"), u.get("completion_tokens"), u.get("calls"),
                    ),
                )
                self._conn.executemany(
                    """INSERT INTO messages(run_id, round_idx, pair_idx, turn_idx, speaker, text, ready)
                       VALUES (?,?,?,?,?,?,?)""",
                    [
                        (rid, round, pair_idx, ti, t["speaker"], t["text"], int(bool(t["ready"])))
                        for ti, t in enumerate(rec.transcript)
                    ],
                )
                # L2: raw LLM calls (one row per HTTP attempt), call_idx — the order
                self._conn.executemany(
                    """INSERT INTO llm_calls(
                           run_id, round_idx, pair_idx, call_idx, agent_id, phase, turn_idx,
                           attempt, http_attempt, status, status_code,
                           request, response, response_raw, error,
                           prompt_tokens, completion_tokens)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    [
                        (rid, round, pair_idx, call_idx, c.agent_id, c.phase, c.turn_idx,
                         c.attempt, c.http_attempt, c.status, c.status_code,
                         json.dumps(c.request), c.response, c.response_raw, c.error,
                         c.prompt_tokens, c.completion_tokens)
                        for call_idx, c in enumerate(rec.llm_calls)
                    ],
                )

    def finish(self, pop: Population) -> None:
        """Step F: stamp finished_at and write each agent's final score."""
        rid = self._run_id
        with self._conn:
            self._conn.execute(
                "UPDATE runs SET finished_at=? WHERE run_id=?", (_now(), rid)
            )
            self._conn.executemany(
                "UPDATE agents SET final_score=? WHERE run_id=? AND agent_id=?",
                [(a.score, rid, a.id) for a in pop],
            )

    def save_verdict(self, verdict: JudgeVerdict, *, model: str, run_id: int | None = None) -> None:
        """Step J: save the LLM judge's verdict (one row per run).

        run_id=None means the current run (live path); an explicit run_id is for backfill,
        where one Storage instance evaluates many stored runs."""
        rid = run_id if run_id is not None else self._run_id
        with self._conn:
            self._conn.execute(
                """INSERT INTO judge_verdicts(run_id, emerged, explanation, evidence, model, created_at)
                   VALUES (?,?,?,?,?,?)""",
                (
                    rid,
                    int(verdict.emerged),
                    verdict.explanation,
                    json.dumps([asdict(e) for e in verdict.evidence]),
                    model,
                    _now(),
                ),
            )

    def save_keyword_count(self, count: KeywordCount, *, run_id: int) -> None:
        """Save a term-mention count for the run (upsert on (run_id, term)).

        Re-running the same term for the run replaces the previous row."""
        with self._conn:
            self._conn.execute(
                """INSERT OR REPLACE INTO
                       keyword_counts(run_id, term, count, speakers, created_at)
                   VALUES (?,?,?,?,?)""",
                (run_id, count.term, count.count,
                 json.dumps(list(count.speakers)), _now()),
            )

    def close(self) -> None:
        self._conn.close()

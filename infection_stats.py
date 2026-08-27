# Infection study, analysis: what does the subject do in each round position, per
# sequence and subject model?
#   uv run python infection_stats.py [--db db/infection.db] [--json infection_stats.json]
#
# The subject is the first agent of every run (roster order). Its action in a round is
# read off the two numbers: cooperate = same number, undercut = subject is one above the
# NPC, betrayed = NPC is one above the subject, miscoord = anything else. Per
# (model, sequence, round) the share of "cooperate" comes with a Wilson 95% interval.

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict

from src.stats.wilson import wilson_interval

ACTIONS = ("cooperate", "undercut", "betrayed", "miscoord")


def action(subject: int, npc: int) -> str:
    if subject == npc:
        return "cooperate"
    if subject == (npc + 1) % 10:
        return "undercut"
    if npc == (subject + 1) % 10:
        return "betrayed"
    return "miscoord"


def collect(db: str) -> dict:
    """{(label, sequence): {round: Counter(action)}} over finished pairings."""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        subjects = {run_id: aid for run_id, aid in conn.execute(
            "SELECT run_id, agent_id FROM agents WHERE rowid IN "
            "(SELECT MIN(rowid) FROM agents GROUP BY run_id)")}
        names = dict(conn.execute("SELECT run_id, name FROM runs"))
        rows = conn.execute(
            "SELECT run_id, round_idx, a_id, a_number, b_number FROM pairings "
            "WHERE finished = 1 AND a_number IS NOT NULL").fetchall()
    finally:
        conn.close()
    table: dict = defaultdict(lambda: defaultdict(Counter))
    for run_id, rnd, a_id, a_num, b_num in rows:
        label, sequence, _ = names[run_id].rsplit(" ", 2)
        subj, npc = (a_num, b_num) if a_id == subjects[run_id] else (b_num, a_num)
        table[(label, sequence)][rnd][action(subj, npc)] += 1
    return table


def summarize(table: dict) -> list[dict]:
    out = []
    for (label, sequence), rounds in sorted(table.items()):
        for rnd, counts in sorted(rounds.items()):
            n = sum(counts.values())
            k = counts["cooperate"]
            lo, hi = wilson_interval(k, n)
            out.append({"model": label, "sequence": sequence, "round": rnd,
                        "npc": "defector" if sequence[rnd - 1] == "P" else "cooperator",
                        "n": n, "cooperate": k / n, "ci": [lo, hi],
                        **{a: counts[a] for a in ACTIONS}})
    return out


def print_table(rows: list[dict]) -> None:
    last = None
    for r in rows:
        key = (r["model"], r["sequence"])
        if key != last:
            print(f"\n{r['model']}  {r['sequence']}")
            print("  round npc         n  cooperate [95% CI]     undercut betrayed miscoord")
            last = key
        print(f"  {r['round']:>5} {r['npc']:<10} {r['n']:>2}  {r['cooperate']:.2f} "
              f"[{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]     {r['undercut']:>8} {r['betrayed']:>8} {r['miscoord']:>8}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-round subject actions of the infection study.")
    ap.add_argument("--db", default="db/infection.db")
    ap.add_argument("--json", default="infection_stats.json")
    args = ap.parse_args()
    rows = summarize(collect(args.db))
    print_table(rows)
    with open(args.json, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\nwritten {args.json}")


if __name__ == "__main__":
    main()

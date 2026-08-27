# Infection study, analysis: what does the subject do in each round position, per
# sequence and subject model?
#   uv run python infection_stats.py [--db db/infection.db] [--json infection_stats.json]
#
# Every row is about the SUBJECT (the first agent of the run, roster order). Its number
# in a round is classified relative to the NPC's: same (matched), +1 (subject one above:
# an undercut), -1 (subject one below: it deviated downward — against an honest
# cooperator that is the subject breaking the deal, not the NPC), other (miscoordination).
# The "vs" column says which NPC type it faced (a defector is one above by construction).
# Per (model, sequence, round) the "same" share (`share`) comes with a Wilson 95% CI.

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict

from src.stats.wilson import wilson_interval

ACTIONS = ("same", "+1", "-1", "other")


def action(subject: int, npc: int) -> str:
    """The subject's number relative to the NPC's on the 0-9 cycle."""
    d = (subject - npc) % 10
    return {0: "same", 1: "+1", 9: "-1"}.get(d, "other")


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
            k = counts["same"]
            lo, hi = wilson_interval(k, n)
            out.append({"model": label, "sequence": sequence, "round": rnd,
                        "vs": "defector" if sequence[rnd - 1] == "P" else "cooperator",
                        "n": n, "share": k / n, "ci": [lo, hi],
                        **{a: counts[a] for a in ACTIONS}})
    return out


def print_table(rows: list[dict]) -> None:
    last = None
    for r in rows:
        key = (r["model"], r["sequence"])
        if key != last:
            print(f"\n{r['model']}  {r['sequence']}   (subject's number vs the NPC's)")
            print("  round vs          n  same [95% CI]         +1   -1  other")
            last = key
        print(f"  {r['round']:>5} {r['vs']:<10} {r['n']:>2}  {r['share']:.2f} "
              f"[{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]   {r['+1']:>4} {r['-1']:>4} {r['other']:>6}")


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

# Infection study, analysis: what does the subject do in each round position, per
# sequence and subject model?
#   uv run python infection_stats.py [--db db/infection.db] [--json infection_stats.json]
#
# Every row is about the SUBJECT (the first agent of the run, roster order). Its number in a
# round is read against the NPC's, knowing the NPC type:
#   vs a cooperator (plays the agreed number a): a -> cooperate, a+1 -> cheat, a-1 -> below
#     (deviated downward), anything else -> other;
#   vs a defector (plays a+1 by construction): a -> cooperate (the subject kept the deal and
#     got cheated), a+1 or a+2 -> cheat (the subject tried to undercut too), else -> other.
# Per (model, sequence, round) the cooperate share comes with a Wilson 95% CI; per run the
# rounds AFTER the first defector (all rounds for the baseline) are summarized.

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict

from src.stats.wilson import wilson_interval

ACTIONS = ("cooperate", "cheat", "below", "other")


def action(subject: int, npc: int, vs: str = "cooperator") -> str:
    """The subject's intent from the two numbers and the NPC type (see the header)."""
    d = (subject - npc) % 10
    if vs == "defector":
        return {9: "cooperate", 0: "cheat", 1: "cheat"}.get(d, "other")
    return {0: "cooperate", 1: "cheat", 9: "below"}.get(d, "other")


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
        vs = "defector" if sequence[rnd - 1] == "P" else "cooperator"
        table[(label, sequence)][rnd][action(subj, npc, vs)] += 1
    return table


def collect_runs(db: str) -> dict:
    """{(label, sequence): [per-run dict]} — the subject's intent in the rounds AFTER the first
    defector (all rounds for the baseline). Behaviour is stable within a run, so the run is
    the honest unit of observation."""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        subjects = {run_id: aid for run_id, aid in conn.execute(
            "SELECT run_id, agent_id FROM agents WHERE rowid IN "
            "(SELECT MIN(rowid) FROM agents GROUP BY run_id)")}
        names = dict(conn.execute("SELECT run_id, name FROM runs"))
        rows = conn.execute(
            "SELECT run_id, round_idx, a_id, a_number, b_number FROM pairings "
            "WHERE finished = 1 AND a_number IS NOT NULL ORDER BY run_id, round_idx").fetchall()
    finally:
        conn.close()
    per_run: dict = defaultdict(list)
    for run_id, rnd, a_id, a_num, b_num in rows:
        label, sequence, _ = names[run_id].rsplit(" ", 2)
        subj, npc = (a_num, b_num) if a_id == subjects[run_id] else (b_num, a_num)
        kind = "P" if sequence[rnd - 1] == "P" else "X"
        per_run[run_id].append((rnd, action(subj, npc, "defector" if kind == "P" else "cooperator"), kind))
    table: dict = defaultdict(list)
    for run_id, rounds in per_run.items():
        label, sequence, _ = names[run_id].rsplit(" ", 2)
        first_p = sequence.find("P")            # 0-based; rounds strictly after it count
        after = [act for rnd, act, _ in rounds if first_p < 0 or rnd - 1 > first_p]
        if not after:
            continue
        table[(label, sequence)].append({
            "rounds": rounds,
            "same_share": sum(1 for a in after if a == "cooperate") / len(after),
            "clean": all(a == "cooperate" for a in after),
            "undercut": any(a == "cheat" for a in after),
            "below": any(a == "below" for a in after),
        })
    return table


def summarize_runs(table: dict) -> list[dict]:
    out = []
    for (label, sequence), runs in sorted(table.items()):
        n = len(runs)
        k = sum(r["clean"] for r in runs)
        lo, hi = wilson_interval(k, n)
        out.append({"model": label, "sequence": sequence, "runs": n,
                    "same_share": sum(r["same_share"] for r in runs) / n,
                    "clean": k, "clean_share": k / n, "clean_ci": [lo, hi],
                    "undercut": sum(r["undercut"] for r in runs),
                    "below": sum(r["below"] for r in runs)})
    return out


def print_runs(rows: list[dict]) -> None:
    print("\nPer run — the subject in the rounds after the first defector (baseline: all rounds)")
    print(f"  {'subject / npc':46s} {'seq':8s} coop%  clean [95% CI]     cheat below")
    for r in rows:
        print(f"  {r['model']:46s} {r['sequence']:8s} {100 * r['same_share']:4.0f}%  "
              f"{r['clean']:>2}/{r['runs']:<2} [{r['clean_ci'][0]:.2f}, {r['clean_ci'][1]:.2f}]  "
              f"{r['undercut']:>4} {r['below']:>4}")


def summarize(table: dict) -> list[dict]:
    out = []
    for (label, sequence), rounds in sorted(table.items()):
        for rnd, counts in sorted(rounds.items()):
            n = sum(counts.values())
            k = counts["cooperate"]
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
            print(f"\n{r['model']}  {r['sequence']}   (subject's intent per round)")
            print("  round vs          n  cooperate [95% CI]   cheat below other")
            last = key
        print(f"  {r['round']:>5} {r['vs']:<10} {r['n']:>2}  {r['share']:.2f} "
              f"[{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]   {r['cheat']:>5} {r['below']:>5} {r['other']:>5}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-round subject actions of the infection study.")
    ap.add_argument("--db", default="db/infection.db")
    ap.add_argument("--json", default="infection_stats.json")
    args = ap.parse_args()
    rows = summarize(collect(args.db))
    print_table(rows)
    runs = summarize_runs(collect_runs(args.db))
    print_runs(runs)
    with open(args.json, "w") as f:
        json.dump({"rounds": rows, "runs": runs}, f, indent=1)
    print(f"\nwritten {args.json}")


if __name__ == "__main__":
    main()

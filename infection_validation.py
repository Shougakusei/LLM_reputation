# Infection study, phase 1: does the cooperator prompt hold on the NPC model?
#   uv run python infection_validation.py                 # run the missing games, then report
#   uv run python infection_validation.py --report        # report only (no LLM calls)
#   uv run python infection_validation.py --parallel 8    # concurrent runs (default PARALLEL)
#
# Runs config/infection_validation.yaml (two cooperators, one round) N times as independent
# runs named "cooperator <i>", resuming aborted ones first — idempotent, re-run to continue.
# Runs are played PARALLEL at a time (each is its own episode + DB writer; WAL + busy timeout
# make that safe) so a rented GPU is not left idle. A run "holds" when both agents chose the
# same number (outcome CC).

from __future__ import annotations

import argparse
import asyncio
import os
import sqlite3
import time

from dotenv import load_dotenv

from src.core.config import load_episode
from src.runner import resume_run, run
from src.stats.wilson import wilson_interval
from src.storage import Storage

load_dotenv()

CONFIG = "config/infection_validation.yaml"
DB = "db/infection_validation.db"
GAMES = 100
PARALLEL = 8


async def _play_missing(games: int, parallel: int) -> None:
    st = Storage(DB)
    try:
        unfinished = st.unfinished_runs()
        missing = [f"cooperator {i}" for i in range(1, games + 1)
                   if st.run_id_by_name(f"cooperator {i}") is None]
    finally:
        st.close()
    sem = asyncio.Semaphore(parallel)

    async def resume(run_id, name):
        async with sem:
            print(f"resume {name}")
            await resume_run(run_id, DB, quiet=True)

    async def play(name):
        async with sem:
            print(f"calculating {name}")
            t0 = time.monotonic()
            await run(load_episode(CONFIG), DB, name, quiet=True)   # fresh random seed per load
            print(f"done {name} {time.monotonic() - t0:.1f}s")

    await asyncio.gather(*[resume(r, n) for r, n in unfinished])
    await asyncio.gather(*[play(n) for n in missing])


def report() -> None:
    if not os.path.exists(DB):
        print("no runs yet")
        return
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT p.run_id, r.name, p.finished, p.a_number, p.b_number, p.a_outcome "
            "FROM pairings p JOIN runs r USING (run_id) WHERE p.round_idx = 1 ORDER BY p.run_id"
        ).fetchall()
        chats = {}
        for run_id, speaker, text in conn.execute(
                "SELECT run_id, speaker, text FROM messages ORDER BY run_id, turn_idx"):
            chats.setdefault(run_id, []).append(f"{speaker}: {text}")
    finally:
        conn.close()
    finished = [r for r in rows if r[2]]
    held = [r for r in finished if r[5] == "CC"]
    n, k = len(finished), len(held)
    print(f"runs: {len(rows)} total, {n} finished, {len(rows) - n} aborted")
    if n:
        lo, hi = wilson_interval(k, n)
        print(f"both kept the deal (CC): {k}/{n} = {k / n:.2f}  [Wilson 95%: {lo:.2f}, {hi:.2f}]")
    for run_id, name, _, a, b, outcome in finished:
        if outcome == "CC":
            continue
        kind = ("undercut" if (a - b) % 10 == 1 or (b - a) % 10 == 1 else "miscoordination")
        print(f"\n--- {name} (run {run_id}): {a} vs {b} -> {outcome} ({kind})")
        for line in chats.get(run_id, []):
            print("   ", line)


def main() -> None:
    ap = argparse.ArgumentParser(description="Cooperator-prompt validation sweep + report.")
    ap.add_argument("--report", action="store_true", help="report only, play nothing")
    ap.add_argument("--games", type=int, default=GAMES, help=f"target number of runs (default {GAMES})")
    ap.add_argument("--parallel", type=int, default=PARALLEL, help=f"concurrent runs (default {PARALLEL})")
    args = ap.parse_args()
    if not args.report:
        asyncio.run(_play_missing(args.games, args.parallel))
    report()


if __name__ == "__main__":
    main()

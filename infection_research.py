# Infection study, phase 2: subject models × NPC sequences × repeats.
#   uv run python infection_research.py
#
# Grid: for every subject model in SUBJECTS, every sequence in SEQUENCES and repeat
# 1..REPEATS, one run of config/infection.yaml named "<label> <sequence> <i>". The agent
# list is rebuilt per run from the sequence string: X = cooperator, P = defector (the
# same cooperator prompt + choice_mapping: one_above); the subject keeps the config's
# prompt and gets the swept model. Idempotent: unfinished runs are resumed first, then
# missing names are filled — re-run to continue. PARALLEL runs play at a time (each its
# own episode + DB writer; WAL + busy timeout make that safe) so a rented GPU stays busy.

from __future__ import annotations

import asyncio
import sqlite3
import time
from dataclasses import replace

from dotenv import load_dotenv

from export_runs import _out_dir_for, export_run
from src.core.config import EpisodeCfg, load_episode
from src.runner import resume_run, run
from src.storage import Storage

load_dotenv()

CONFIG = "config/infection.yaml"
DB = "db/infection.db"
SPLIT_DIR = _out_dir_for(DB)
REPEATS = 10
PARALLEL = 8
SUBJECTS = [                     # (label for run names, Ollama/Together model id)
    ("gpt-oss-120b", "gpt-oss:120b"),
    ("qwen3-8b", "qwen3:8b"),
]
SEQUENCES = ["XXXXXXX", "PXXXXXX", "XPXXXXX", "PPXXXXX"]   # X = cooperator, P = defector


def cfg_for(model_id: str, sequence: str) -> EpisodeCfg:
    """Fresh config (new random seed) with the subject model and NPC sequence swapped in."""
    cfg = load_episode(CONFIG)
    specs = cfg.population.agents
    subject = replace(specs[0], provider=replace(specs[0].provider, model=model_id))
    cooperator = replace(next(s for s in specs[1:] if s.choice_mapping == "match"), count=1)
    defector = replace(cooperator, choice_mapping="one_above")
    npcs = [defector if ch == "P" else cooperator for ch in sequence]
    return replace(cfg, rounds=len(sequence),
                   population=replace(cfg.population, agents=[subject] + npcs))


def _split_off(run_id: int) -> None:
    conn = sqlite3.connect(DB)
    try:
        export_run(conn, run_id, SPLIT_DIR, overwrite=True)
    finally:
        conn.close()


async def _main() -> None:
    st = Storage(DB)
    try:
        unfinished = st.unfinished_runs()
        missing = [(label, model_id, sequence, f"{label} {sequence} {i}")
                   for label, model_id in SUBJECTS for sequence in SEQUENCES
                   for i in range(1, REPEATS + 1)
                   if st.run_id_by_name(f"{label} {sequence} {i}") is None]
    finally:
        st.close()
    sem = asyncio.Semaphore(PARALLEL)

    async def resume(run_id, name):
        async with sem:
            print(f"resume {name}")
            await resume_run(run_id, DB, quiet=True)
            _split_off(run_id)

    async def play(model_id, sequence, name):
        async with sem:
            print(f"calculating {name}")
            t0 = time.monotonic()
            run_id = await run(cfg_for(model_id, sequence), DB, name, quiet=True)
            _split_off(run_id)
            print(f"done {name} {time.monotonic() - t0:.1f}s")

    await asyncio.gather(*[resume(r, n) for r, n in unfinished])
    await asyncio.gather(*[play(m, s, n) for _, m, s, n in missing])


if __name__ == "__main__":
    asyncio.run(_main())

# uv run python research.py

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

load_dotenv()                       # API keys from .env (TOGETHER_API_KEY)

CONFIG = "config/research_notes.yaml"
DB = "db/ternary_bonsai_notes_100_rounds.db"
SPLIT_DIR = _out_dir_for(DB)        # folder with per-run files = DB name without extension (qwen3.db -> qwen3/)
TARGET_ROUNDS = load_episode(CONFIG).rounds   # target number of rounds = rounds from the config (currently 10)
GAMES_PER_MODEL = 1

def _split_off(run_id: int) -> None:
    """Export a run into a separate file SPLIT_DIR/<number>.db, overwriting an existing one."""
    conn = sqlite3.connect(DB)
    try:
        export_run(conn, run_id, SPLIT_DIR, overwrite=True)
    finally:
        conn.close()
MODELS = [
    # ("llama-3-8b",      "meta-llama/Meta-Llama-3-8B-Instruct-Lite"),
    # ("qwen2.5-7b",      "Qwen/Qwen2.5-7B-Instruct-Turbo"),
    # ("deepseek-v4-pro", "deepseek-ai/DeepSeek-V4-Pro"),
    # ("gpt-oss-20b",     "openai/gpt-oss-20b"),
    # ("qwen3-FP8",       "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"),
    ("ternary-bonsai", "Prism-ML/Ternary-Bonsai-27B"),
]


def _cfg_for_model(model_id: str) -> EpisodeCfg:
    """Fresh research config with the model swapped in.

    load_episode rereads research.yaml in full, so `seed: random` gives a NEW seed
    on every call (each game is its own). We only change provider.model — the rest of the
    design (population, payoffs, prompts) is fixed by the config."""
    cfg = load_episode(CONFIG)
    provider = replace(cfg.population.provider, model=model_id)
    return replace(cfg, population=replace(cfg.population, provider=provider))


async def _extend_existing() -> None:
    """Phase 1: bring EVERY existing run up to TARGET_ROUNDS.

    resume_run with rounds=TARGET_ROUNDS finishes off interrupted runs AND extends already
    completed ones (rounds is excluded from config_hash — the design doesn't change). Runs
    that already reached the target are skipped by resume_run ("nothing to do"), so this
    phase is idempotent."""
    conn = sqlite3.connect(DB)
    try:
        # fresh/empty DB: the table doesn't exist yet (Storage will create it in phase 2) — nothing to extend
        has_runs = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='runs'"
        ).fetchone()
        runs = conn.execute(
            "SELECT run_id, name FROM runs ORDER BY run_id"
        ).fetchall() if has_runs else []
    finally:
        conn.close()
    for run_id, name in runs:
        print(f"extend {name} -> {TARGET_ROUNDS} rounds")
        await resume_run(run_id, DB, rounds=TARGET_ROUNDS, quiet=True)
        _split_off(run_id)                        # extended — overwrite the file


async def _fill_missing() -> None:
    """Phase 2: fill in the missing games per the plan (model × number). Run name = '<model> <i>',
    with the iteration number i starting at 1 (per model).

    We look up by name: if a run with that name already exists (finished in phase 1 or
    earlier, or still open) — skip it; otherwise compute a new one. This way we continue
    from the first not-yet-run entry. Prints `calculating <name>` before starting and
    `done <wall-time>` after."""
    for label, model_id in MODELS:
        for i in range(1, GAMES_PER_MODEL + 1):      # iteration number for the model — starting at 1
            name = f"{label} {i}"
            st = Storage(DB)
            try:
                exists = st.run_id_by_name(name) is not None
            finally:
                st.close()
            if exists:
                continue
            print(f"calculating {name}")
            t0 = time.monotonic()
            run_id = await run(_cfg_for_model(model_id), DB, name, quiet=True)
            _split_off(run_id)                       # computed — export to file
            print(f"done {time.monotonic() - t0:.1f}s")


async def _main() -> None:
    await _extend_existing()
    await _fill_missing()


if __name__ == "__main__":
    asyncio.run(_main())

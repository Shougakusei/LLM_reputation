"""Run one reputation experiment from a YAML config.

The experiment definition lives in a YAML file — `config/experiment.yaml` by default.
This script only loads that file and hands it to the runner; all the run logic
(build population → drive episode → persist + narrate → score) lives in src/runner.py.
Point it at another config to run a different episode (e.g. config/example.yaml).

Run from the repo root. Optional positional args: a config path, then a human label for
the run (stored in runs.name):

    uv run python experiment.py [config.yaml] ["run name"]

Every run is appended to one DB ("one DB, many runs") under a fresh incremental run_id
(1, 2, 3 …). Each invocation is a NEW run — re-running the same config no longer de-dups,
it just gets the next number (handy for repeated runs of one config under a noisy LLM).
The config's hash is still stored (runs.config_hash) to group runs of the same design.

Resume an unfinished run or extend a finished one to more rounds (by run number):

    uv run python experiment.py --resume 87              # finish #87 to its configured rounds
    uv run python experiment.py --resume 87 --rounds 20  # grow #87 to 20 rounds

Read a stored run back, round by round, with:

    uv run python replay.py <run_id>        # run_id — a number or config_hash
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import replace

from dotenv import load_dotenv

from src.core.config import JudgeCfg, ProviderCfg, load_episode  # noqa: F401 (JudgeCfg/ProviderCfg used in commented example below)
from src.runner import resume_run, run

load_dotenv()                       # pick up API keys from .env (e.g. TOGETHER_API_KEY)

DB = "db/experiment.db"                 # one DB, many runs; appended to on every run
DEFAULT_CONFIG = "config/research_notes.yaml"

# --- LLM judge (optional) ------------------------------------------------------
# A separate model that, at the end of the episode, decides whether the reputation
# institution emerged. The judge only sees the public cheap-talk; the verdict is
# printed, saved to the DB and highlighted in replay.py. None = judge disabled.
# To enable: uncomment JUDGE below — it overrides the judge block from the YAML
# (or set the `judge:` block directly in the YAML config — that's equivalent).
JUDGE = None
# JUDGE = JudgeCfg(provider=ProviderCfg(
#     base_url="https://api.together.xyz/v1",
#     api_key_env="TOGETHER_API_KEY",
#     model="Qwen/Qwen2.5-72B-Instruct-Turbo",
# ))


def configure_llm_trace() -> None:
    """LLM_TRACE=1 (env or .env) -> DEBUG output of the LLM input for the DECIDE/PREDICT/REFLECT phases.

    The handler is configured by this caller script, not the engine: src/core/agent.py
    only writes to the `src.core.agent` logger (CLAUDE.md: handlers configured by the
    caller, never in src/).
    """
    if os.environ.get("LLM_TRACE", "0") in ("", "0"):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("\n%(message)s"))
    logger = logging.getLogger("src.core.agent")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)


def _flag(args, name):
    """Value of the flag `name X`, or None if the flag is absent."""
    return args[args.index(name) + 1] if name in args else None


if __name__ == "__main__":
    configure_llm_trace()           # opt-in LLM input tracing (env/.env), before launch
    args = sys.argv[1:]
    if "--resume" in args:
        # Resume/extend an existing run by number:
        #   experiment.py --resume 87            # finish interrupted #87 up to its rounds
        #   experiment.py --resume 87 --rounds 20 # extend #87 to 20 rounds
        run_id = int(_flag(args, "--resume"))
        rounds = _flag(args, "--rounds")
        asyncio.run(resume_run(run_id, DB, int(rounds) if rounds is not None else None))
    else:
        config_path = args[0] if args and not args[0].startswith("-") else DEFAULT_CONFIG
        name = args[1] if len(args) > 1 and not args[1].startswith("-") else None
        cfg = load_episode(config_path)
        if JUDGE is not None:       # Python override of the judge on top of the YAML (see JUDGE above)
            cfg = replace(cfg, judge=JUDGE)
        asyncio.run(run(cfg, DB, name))

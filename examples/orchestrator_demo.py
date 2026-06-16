"""Demo / entry point for the orchestrator (layer 5): build a population from a config,
run one episode over it (population -> matchmaker -> game, all rounds), narrating each
round's negotiations via the observer seam, then printing the scoreboard.

run_episode returns nothing: the caller owns the population (reads scores from it,
closes it) and collects per-round data via the observer — the same seam the Logger
layer will use to persist rounds.

Run from the repo root (needs API-доступ к провайдеру из конфигурации; ключ берётся из
.env). Путь к конфигурации можно передать первым аргументом, иначе берётся example.yaml:

    uv run python examples/orchestrator_demo.py [config/episode.yaml]

Трассировка LLM-входа перед выбором числа: LLM_TRACE=1 (флаг можно задать в .env).
"""

import asyncio
import logging
import os
import random
import sys

from dotenv import load_dotenv

from src.core.config import load_episode
from src.core.orchestrator import EpisodeAborted, run_episode
from src.judge import JudgeError, judge_episode
from src.population import make_population
from src.providers.base import ProviderError
from src.runner import print_verdict

load_dotenv()                       # подхватить ключи API из .env (например TOGETHER_API_KEY)

if os.environ.get("LLM_TRACE", "0") not in ("", "0"):
    # Включить трассировку LLM-входа фаз DECIDE/PREDICT/REFLECT (см. src/core/agent.py)
    _trace_handler = logging.StreamHandler()
    _trace_handler.setFormatter(logging.Formatter("\n%(message)s"))
    _trace_logger = logging.getLogger("src.core.agent")
    _trace_logger.setLevel(logging.DEBUG)
    _trace_logger.addHandler(_trace_handler)

CONFIG = sys.argv[1] if len(sys.argv) > 1 else "config/example.yaml"


def narrate_round(r, plan, recs):
    print(f"\n{'─' * 60}\n  ROUND {r}")
    if plan.idle:
        print(f"  idle (sat out): {', '.join(plan.idle)}")
    for rec in recs:
        print(f"\n  {rec.a_id} vs {rec.b_id}  ({rec.a_id} opens):")
        if rec.transcript:
            for i, t in enumerate(rec.transcript, 1):
                print(f"    {i}. {t['speaker']}: {t['text']}   [ready={t['ready']}]")
        else:
            print("    (no messages exchanged)")
        # reasoning is shown before the choices it led to (absent when game.rationale=false)
        if rec.a_rationale:
            print(f"    {rec.a_id} reason: {rec.a_rationale}")
        if rec.b_rationale:
            print(f"    {rec.b_id} reason: {rec.b_rationale}")
        if rec.a_predicted is not None or rec.b_predicted is not None:
            print(
                f"    predicted: {rec.a_id} guessed {rec.b_id}={rec.a_predicted}, "
                f"{rec.b_id} guessed {rec.a_id}={rec.b_predicted}"
            )
        if not rec.finished:
            # Aborted pairing (LLM error): no numbers/outcome/payoffs were produced.
            print("    (pairing aborted — LLM call failed, no choices made)")
            continue
        print(
            f"    choices: {rec.a_id}={rec.a_number}, {rec.b_id}={rec.b_number}"
            f"  ->  {rec.outcome}   (payoffs {rec.a_id}={rec.a_payoff:g}, {rec.b_id}={rec.b_payoff:g})"
        )
        if rec.a_reflection is not None:
            print(f"      {rec.a_id} reflects: {rec.a_reflection}")
        if rec.b_reflection is not None:
            print(f"      {rec.b_id} reflects: {rec.b_reflection}")


async def main():
    cfg = load_episode(CONFIG)
    n_agents = sum(a.count for a in cfg.population.agents)
    print(
        f"Running {cfg.rounds} rounds, {n_agents} agents, "
        f"matchmaker={cfg.matchmaker}, seed={cfg.seed}"
    )
    pop = make_population(cfg.population, context_window=cfg.context_window).build(
        random.Random(cfg.seed)
    )
    records = []

    def observer(r, plan, recs):
        narrate_round(r, plan, recs)
        records.extend(recs)        # collect per-round facts via the observer channel

    try:
        await run_episode(cfg, pop, observer=observer)
    except EpisodeAborted as e:
        # An aborted pairing (LLM failure) stops the episode. The usual cause is the
        # provider: a wrong key in .env, an exhausted balance/credits, or an unavailable model.
        print(
            f"\nepisode aborted at round {e.round}: an LLM call failed "
            f"({e.rec.a_id} vs {e.rec.b_id}). Check the provider from the config — "
            f"the key in .env, the credit balance, and model availability."
        )
        return
    finally:
        await pop.aclose()

    bar = "=" * 60
    print(f"\n{bar}\n  SCOREBOARD ({len(records)} games over {cfg.rounds} rounds)\n{bar}")
    scores = {a.id: a.score for a in pop}      # final state read from the caller-owned pop
    for agent_id, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        games = sum(1 for rec in records if agent_id in (rec.a_id, rec.b_id))
        print(f"  {agent_id}: {score:g}   ({games} games)")

    outcomes = {}
    for rec in records:
        outcomes[rec.outcome] = outcomes.get(rec.outcome, 0) + 1
    print("\noutcomes:", outcomes)

    if cfg.judge is not None:       # опциональный LLM-судья: только печать (БД здесь нет)
        try:
            verdict = await judge_episode(cfg.judge, records)
        except (JudgeError, ProviderError) as e:
            print(f"\nсудья не смог вынести вердикт: {e}")
        else:
            print_verdict(verdict)


if __name__ == "__main__":
    asyncio.run(main())

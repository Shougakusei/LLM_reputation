from __future__ import annotations

import asyncio
import inspect
import random
from dataclasses import replace
from typing import Awaitable, Callable, Optional

from src.core.config import EpisodeCfg, cfg_for_round
from src.games.base import PairingRecord
from src.games.reputation_pd import ReputationPD
from src.matchmaking import RoundPlan, make_matchmaker
from src.population import Population
from src.population.evolution import evolve

# Called at the end of each round; sync or async. This is the orchestrator's ONLY
# output channel — the Logger layer will implement it to persist each round.
Observer = Callable[[int, RoundPlan, list[PairingRecord]], Optional[Awaitable[None]]]


class EpisodeAborted(RuntimeError):
    """Episode stopped because a pairing was aborted by an LLM failure.

    Raised AFTER the round is written to the observer — so the aborted pairing (finished=0)
    and its L2 log are already in the DB, while `runs.finished_at` stays NULL as a crash marker.
    """

    def __init__(self, round: int, rec: PairingRecord):
        super().__init__(f"round {round}: pairing {rec.a_id} vs {rec.b_id} aborted by LLM failure")
        self.round = round
        self.rec = rec


async def _guarded(coro, sem: asyncio.Semaphore):
    async with sem:
        return await coro


async def run_episode(cfg: EpisodeCfg, pop: Population, *, observer: Observer | None = None,
                      start_round: int = 1) -> None:
    """Drive one episode over a caller-owned population.

    Side effects only: mutates agents (score / memory) and emits each round to
    `observer`. Returns nothing — final state lives on `pop`, which the caller builds,
    inspects (scores / memory) and closes (`aclose`). `pop` must come from
    `cfg.population`, e.g. make_population(cfg.population, ...).build(Random(cfg.seed)).

    `start_round` (>=1) — which round to start playing from: 1 for a new episode, last+1
    for resuming. The matchmaker is seeded PER ROUND (Random(f"{seed}:matchmaker:{r}")),
    so round r's pairing is the same regardless of whether we play from the start or
    resume — there is no need to rewind the rng stream (see resume in runner).

    With `population.evolution` set, a death/replacement step runs before pairing each
    round r ≥ 2 and its events are prepended to `RoundPlan.events`.
    """
    mm = make_matchmaker(cfg.matchmaker)
    mm.setup(pop.ids(), cfg)
    # seed / max_concurrency / matchmaker — the frame for the whole run (not patched per round).
    sem = asyncio.Semaphore(cfg.max_concurrency)
    for r in range(start_round, cfg.rounds + 1):                       # rounds are numbered from 1
        cfg_r = cfg_for_round(cfg, r)            # per-round config: payoffs/talk/prompts/idle
        game = ReputationPD(cfg_r.game)          # rebuilt each time — the game is stateless between pairs; strategy is per-agent (agent.setup)
        ev_events: list[dict] = []
        if r >= 2 and cfg.population.evolution is not None:
            # dedicated per-round stream, like the matchmaker: resume re-derives it (see evolve)
            ev_events = evolve(pop, cfg.population,
                               random.Random(f"{cfg.seed}:evolution:{r}"), r)
        rng = random.Random(f"{cfg.seed}:matchmaker:{r}")             # M1: a separate rng per round
        plan = await mm.plan_round(pop.ids(), r, rng, actor=None)
        if ev_events:
            plan = replace(plan, events=ev_events + plan.events)
        recs = await asyncio.gather(*[                                  # fail-fast (C2)
            _guarded(game.play_pairing(pop.get(a), pop.get(b), r), sem)
            for a, b in plan.pairings
        ])
        for c in plan.idle:
            pop.get(c).score += cfg_r.idle_payoff                      # C3
        if observer is not None:                                       # the only output channel
            res = observer(r, plan, recs)
            if inspect.isawaitable(res):
                await res
        aborted = next((rec for rec in recs if not rec.finished), None)
        if aborted is not None:                # aborted pairing: round already written -> stop
            raise EpisodeAborted(r, aborted)

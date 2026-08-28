"""Experiment runner — drive one episode, persist it, narrate it live.

This is the reusable engine behind `experiment.py`: it owns no config of its own.
`experiment.py` defines the config and calls `run(cfg, db_path, name)`; tests or other
scripts can do the same. Persistence (Storage) and the live console narration are wired
together into one observer so the DB and the printed transcript never diverge.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import replace

from src.core.config import EpisodeCfg, episode_from_dict
from src.core.orchestrator import EpisodeAborted, run_episode
from src.judge import JudgeError, JudgeVerdict, judge_episode
from src.population import make_population
from src.population.evolution import NamePoolExhausted, evolve
from src.providers.base import ProviderError
from src.storage import Storage


def narrate_round(r, plan, recs) -> None:
    """Print one round to console as soon as it's played (the live observer)."""
    print(f"\n{'─' * 60}\n  ROUND {r}")
    for e in plan.events:                              # evolution events happened at round start
        if e.get("type") == "death":
            print(f"  † {e['agent']} died (score {e['score']:g})")
        elif e.get("type") == "birth":
            mark = " (deceptive)" if e.get("deceptive") else ""
            print(f"  + {e['agent']} joined the game{mark}")
    if plan.idle:
        print(f"  idle (sat out): {', '.join(plan.idle)}")
    for rec in recs:
        print(f"\n  {rec.a_id} vs {rec.b_id}  ({rec.a_id} opens):")
        if rec.transcript:
            for i, t in enumerate(rec.transcript, 1):
                mark = "   [finish=true]" if t["ready"] else ""   # finish=false is not printed
                print(f"    {i}. {t['speaker']}: {t['text']}{mark}")
        else:
            print("    (no messages exchanged)")
        if not rec.finished:                       # aborted pairing: no result
            print("    (pairing aborted by LLM failure — no result)")
            continue
        # reasoning is shown before the choices it led to (empty when the prompt asked only for a number)
        if rec.a_rationale:
            print(f"    {rec.a_id} reason: {rec.a_rationale}")
        if rec.b_rationale:
            print(f"    {rec.b_id} reason: {rec.b_rationale}")
        if rec.a_predicted is not None or rec.b_predicted is not None:
            print(
                f"    predicted: {rec.a_id} guessed {rec.b_id}={rec.a_predicted}, "
                f"{rec.b_id} guessed {rec.a_id}={rec.b_predicted}"
            )
        print(
            f"    choices: {rec.a_id}={rec.a_number}, {rec.b_id}={rec.b_number}"
            f"  ->  {rec.outcome}   (payoffs {rec.a_id}={rec.a_payoff:g}, {rec.b_id}={rec.b_payoff:g})"
        )
        if rec.a_reflection is not None:
            print(f"      {rec.a_id} reflects: {rec.a_reflection}")
        if rec.b_reflection is not None:
            print(f"      {rec.b_id} reflects: {rec.b_reflection}")


def print_verdict(verdict: JudgeVerdict) -> None:
    """Print the LLM judge's verdict (without citations — highlighting lives in replay)."""
    bar = "=" * 60
    print(f"\n{bar}\n  JUDGE VERDICT\n{bar}")
    print(f"  reputation institute emerged: {'YES' if verdict.emerged else 'NO'}")
    print(f"  {verdict.explanation}")
    print(f"  evidence: {len(verdict.evidence)} message(s) — replay will highlight them in color")


async def _judge_and_store(cfg, records, st) -> None:
    """Call the judge after the episode; its failure must not lose the run's results."""
    try:
        verdict = await judge_episode(cfg, records)
    except (JudgeError, ProviderError) as e:
        print(f"\njudge failed to reach a verdict: {e} — run saved without a verdict")
        return
    print_verdict(verdict)
    st.save_verdict(verdict, model=cfg.provider.model)


async def run_experiment(cfg: EpisodeCfg, db_path: str, name: str | None = None,
                         quiet: bool = False) -> int:
    """Build the population, run the episode, persist + narrate each round, score it.

    Returns the run_id (an auto-incrementing integer). Every run is a new run: there is
    no longer dedup by config (re-running the same config creates a new number).
    Resuming/extending aborted or finished runs is a separate, explicit path by run
    number (see resume, A5).

    `quiet=True` mutes the per-round narration and the final scoreboard (for sweeps of
    hundreds of runs, see research.py) — the DB persistence stays complete; abort
    messages remain."""
    pop = make_population(cfg.population, context_window=cfg.context_window).build(
        random.Random(cfg.seed)
    )
    st = Storage(db_path)
    try:
        run_id = st.begin(cfg, pop, name)        # INSERT runs+agents; a new number

        records: list = []                       # accumulate records for the LLM judge

        def observer(r, plan, recs):             # persist AND (if not quiet) narrate each round live
            st.observe(r, plan, recs)
            if not quiet:
                narrate_round(r, plan, recs)
            records.extend(rec for rec in recs if rec.finished)   # judge sees only finished pairings

        try:
            await run_episode(cfg, pop, observer=observer)
        except EpisodeAborted as e:
            # the aborted pairing is already persisted; run stays without finished_at (crash marker)
            print(f"\nepisode aborted: {e} — run saved without finished_at")
            return run_id
        except NamePoolExhausted as e:
            # the evolution step ran out of replacement names before this round was recorded;
            # run stays without finished_at — enlarge the name pool and start a new run
            print(f"\n{e} — run saved without finished_at; "
                  f"this run cannot be resumed past this point (enlarge the name pool and start a new run)")
            return run_id
        st.finish(pop)

        if not quiet:
            bar = "=" * 60
            print(f"\n{bar}\n  FINAL SCOREBOARD\n{bar}")
            for a in sorted(pop, key=lambda a: a.score, reverse=True):
                print(f"  {a.id}: {a.score:g}")

        if cfg.judge is not None:                # optional LLM judge (see JudgeCfg)
            await _judge_and_store(cfg.judge, records, st)
    finally:
        st.close()
        await pop.aclose()
    return run_id


def _apply_run_state(pop, state) -> None:
    """Apply the restored state (score + memory) onto a freshly built population.

    Agent ids are matched by deterministic rebuild (same config + seed) and evolution replay.
    Missing ids (newborns after evolution replay) keep fresh state (score 0, empty memory).
    The memory window lives on the agent, not on the Memory object, so replacing memory
    does not lose it."""
    for agent in pop:
        agent.score = state.scores.get(agent.id, 0.0)
        if agent.id in state.memories:
            agent.memory = state.memories[agent.id]


async def resume_run(run_id: int, db_path: str, rounds: int | None = None,
                     quiet: bool = False) -> int:
    """Resume or extend an existing run by its number.

    Without `rounds` — finish playing an aborted run up to `rounds` from its stored
    config. With `rounds` — grow it to that number (extend); if that many or more
    rounds are already played, do nothing. A run holding an aborted pairing is first
    rewound to that round and replayed from there, whatever its finished state. Past rounds are read from the DB (the
    actual pairings), new ones are played with per-round rng, so runs recorded before
    the switch to per-round rng are also resumable.

    `quiet=True` (for sweeps, see research.py) mutes narration/headers/scoreboard —
    abort messages remain; the DB is written in full."""
    st = Storage(db_path)
    config_json = st.run_config(run_id)
    if config_json is None:
        print(f"run {run_id} not found in {db_path}")
        st.close()
        return run_id

    cfg = episode_from_dict(json.loads(config_json))
    if rounds is not None:
        cfg = replace(cfg, rounds=rounds)               # extend: grow the target number of rounds
    # An aborted pairing leaves a hole the later rounds were played over; rewind to it and
    # replay from there (the memory of every agent is rebuilt from the surviving rounds).
    hole = st.first_aborted_round(run_id)
    if hole is not None:
        st.rewind(run_id, hole)
        if not quiet:
            print(f"run {run_id}: round {hole} has an aborted pairing — rewinding and replaying from it")
    state = st.load_state(run_id, cfg.idle_payoff)
    # "Nothing to finish playing" holds only if the run is ALREADY closed (finished_at set). If all
    # rounds are recorded but finished_at is empty — this is an episode aborted on the LAST round
    # (the aborted pairing is written to the DB before EpisodeAborted, so the last round's
    # round-row exists). There are no new rounds, but the run still needs to be closed — we go
    # through the common path below: run_episode plays 0 rounds, st.finish sets finished_at.
    # Otherwise such a run would never finish (resume would keep returning "nothing to do").
    if state.last_round >= cfg.rounds and st.is_finished(run_id):
        if not quiet:
            print(f"run {run_id}: already played {state.last_round} rounds (>= {cfg.rounds}) — nothing to do")
        st.close()
        return run_id

    pop = make_population(cfg.population, context_window=cfg.context_window).build(
        random.Random(cfg.seed)
    )
    start = state.last_round + 1
    if cfg.population.evolution is not None:
        # re-derive roster changes for already-played rounds; round `start` itself
        # is evolved by run_episode (it evolves every round it plays)
        for r in range(2, start):
            evolve(pop, cfg.population, random.Random(f"{cfg.seed}:evolution:{r}"), r)
    _apply_run_state(pop, state)
    if not quiet:
        if start > cfg.rounds:                          # all rounds exist — just closing the aborted run
            print(f"Finalizing run {run_id} into {db_path}: all {cfg.rounds} rounds recorded, "
                  f"setting finished_at ({len(pop)} agents)")
        else:
            print(f"Resuming run {run_id} into {db_path}: rounds {start}..{cfg.rounds}, "
                  f"{len(pop)} agents, seed={cfg.seed}")
    try:
        st.resume(run_id, cfg)                          # _run_id, clear finished_at, update config
        def observer(r, plan, recs):                    # persist AND (if not quiet) narrate each new round
            st.observe(r, plan, recs)
            if not quiet:
                narrate_round(r, plan, recs)

        try:
            await run_episode(cfg, pop, observer=observer, start_round=start)
        except EpisodeAborted as e:
            print(f"\nepisode aborted: {e} — run saved without finished_at")
            return run_id
        except NamePoolExhausted as e:
            print(f"\n{e} — run saved without finished_at; "
                  f"this run cannot be resumed past this point (enlarge the name pool and start a new run)")
            return run_id
        st.finish(pop)

        if not quiet:
            bar = "=" * 60
            print(f"\n{bar}\n  FINAL SCOREBOARD\n{bar}")
            for a in sorted(pop, key=lambda a: a.score, reverse=True):
                print(f"  {a.id}: {a.score:g}")
            if cfg.judge is not None:                    # the judge is separate analysis over the full episode
                print("\n(the LLM judge does not run on resume/extend — evaluate the run separately)")
    finally:
        st.close()
        await pop.aclose()
    if not quiet:
        print(f"\nrun_id={run_id}   (replay: uv run python replay.py {run_id})")
    return run_id


async def run(cfg: EpisodeCfg, db_path: str, name: str | None = None,
              quiet: bool = False) -> int:
    """Top-level entry: print a header, run the experiment, print the replay hint.

    `quiet=True` (for sweeps, see research.py) suppresses the header, narration and hint —
    nothing goes out except abort messages; the DB is written in full."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    if not quiet:
        n_agents = sum(a.count for a in cfg.population.agents)
        print(f"Running experiment{f' {name!r}' if name else ''} into {db_path}: "
              f"{n_agents} agents, {cfg.rounds} rounds, seed={cfg.seed}")

    run_id = await run_experiment(cfg, db_path, name, quiet=quiet)
    if not quiet:
        print(f"\nrun_id={run_id}   "
              f"(replay: uv run python replay.py {run_id})")
    return run_id

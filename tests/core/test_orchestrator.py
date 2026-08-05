from __future__ import annotations

import random

import pytest

from src.core import orchestrator as orch
from src.core.config import (
    AgentSpec, ChangePoint, EpisodeCfg, EvolutionCfg, GameCfg, Payoffs, PopulationCfg, ProviderCfg,
)
from src.population import base as popbase
from src.population import make_population
from src.providers.base import Completion, ProviderUnavailable


class FixedProvider:
    """Returns a constant DECIDE reply for every call — order-independent, so it is
    safe under gather even when shared by all agents via the provider cache."""

    def __init__(self, cfg, number: int = 4):
        self.cfg = cfg
        self.closed = 0
        self._reply = '{"number": %d, "rationale": "r"}' % number

    async def complete(self, **kw):
        return Completion(text=self._reply, prompt_tokens=2, completion_tokens=3, raw={})

    async def aclose(self):
        self.closed += 1


class FailAfterProvider:
    """Returns ok-DECIDE the first `ok_calls` times, then raises ProviderUnavailable —
    to verify the episode stops on the broken pair after the recorded rounds."""

    def __init__(self, cfg, ok_calls: int):
        self.cfg = cfg
        self.closed = 0
        self._left = ok_calls
        self._reply = '{"number": 4, "rationale": "r"}'

    async def complete(self, **kw):
        if self._left <= 0:
            e = ProviderUnavailable("down")
            e.request = {"model": "m", "messages": []}
            e.attempts = ()
            raise e
        self._left -= 1
        return Completion(text=self._reply, prompt_tokens=2, completion_tokens=3, raw={})

    async def aclose(self):
        self.closed += 1


@pytest.fixture
def providers(monkeypatch):
    made = []
    monkeypatch.setattr(popbase, "make_provider", lambda cfg: made.append(FixedProvider(cfg)) or made[-1])
    return made


def _cfg(n=3, rounds=2, seed=0):
    spec = AgentSpec(count=n)
    return EpisodeCfg(
        seed=seed,
        rounds=rounds,
        matchmaker="random",
        population=PopulationCfg(kind="roster", agents=[spec],
                                 provider=ProviderCfg(base_url="http://x/v1", model="m")),
        game=GameCfg(max_talk_turns=0),     # decision-only -> deterministic CC for all
    )


async def _run(cfg, observer=None, start_round=1):
    """Caller owns the population: build it, run, close it; return it for inspection."""
    pop = make_population(cfg.population, context_window=cfg.context_window).build(
        random.Random(cfg.seed)
    )
    try:
        await orch.run_episode(cfg, pop, observer=observer, start_round=start_round)
    finally:
        await pop.aclose()
    return pop


async def test_run_episode_drives_rounds(providers):
    records = []
    pop = await _run(_cfg(n=3, rounds=2), observer=lambda r, p, recs: records.extend(recs))
    # 3 agents -> 1 pair + 1 idle per round; 2 rounds -> 2 pairings via the observer
    assert len(records) == 2
    assert all(rec.outcome == "CC" for rec in records)
    # final scores live on the (caller-owned) population
    scores = {a.id: a.score for a in pop}
    assert set(scores) == {"A1", "A2", "A3"}
    # per round: 2 paired +3 (CC) + 1 idle +1 = 7; 2 rounds -> 14
    assert sum(scores.values()) == pytest.approx(14.0)
    # one shared provider (same base_url/model), closed exactly once by the caller
    assert len(providers) == 1 and providers[0].closed == 1


async def test_resume_from_start_round_skips_earlier_and_reproduces_pairings(providers):
    # per-round rng -> round r's pairing is the same whether we run from round 1 or "resume" from round 3
    full = {}
    await _run(_cfg(n=4, rounds=4), observer=lambda r, p, recs: full.__setitem__(r, p.pairings))
    resumed = {}
    await _run(_cfg(n=4, rounds=4),
               observer=lambda r, p, recs: resumed.__setitem__(r, p.pairings),
               start_round=3)
    assert set(resumed) == {3, 4}                                   # rounds 1-2 skipped
    assert resumed[3] == full[3] and resumed[4] == full[4]          # the same pairings as in the full run


async def test_observer_gets_each_round(providers):
    seen = []

    async def observer(r, plan, recs):
        seen.append((r, plan, recs))

    await _run(_cfg(n=4, rounds=3), observer=observer)
    assert [r for r, _, _ in seen] == [1, 2, 3]
    for r, plan, recs in seen:
        assert len(recs) == len(plan.pairings)      # recs align with the round's pairings
        assert len(plan.pairings) == 2 and plan.idle == []   # N=4 -> 2 pairs, no idle


async def test_sync_observer_supported(providers):
    seen = []
    await _run(_cfg(n=2, rounds=2), observer=lambda r, plan, recs: seen.append(r))
    assert seen == [1, 2]


async def test_aborts_after_recording_round_on_llm_failure(monkeypatch):
    # n=2, max_talk_turns=0 -> 1 pair/round = 2 decide calls; ok_calls=2 -> round 1 intact, round 2 breaks
    monkeypatch.setattr(popbase, "make_provider", lambda cfg: FailAfterProvider(cfg, ok_calls=2))
    seen = []
    with pytest.raises(orch.EpisodeAborted) as ei:
        await _run(_cfg(n=2, rounds=3), observer=lambda r, plan, recs: seen.append((r, recs)))
    assert ei.value.round == 2                         # failed on round 2, not round 3
    assert [r for r, _ in seen] == [1, 2]              # round 1 AND the broken round 2 are recorded
    assert seen[0][1][0].finished is True              # round 1 played to completion
    assert seen[1][1][0].finished is False             # round 2 broken


async def test_llm_failure_aborts_episode_and_closes_providers(monkeypatch):
    # provider raises -> the pair catches the failure (finished=0) -> run_episode raises EpisodeAborted
    # -> the caller still closes the providers
    class BoomProvider:
        def __init__(self, cfg):
            self.closed = 0

        async def complete(self, **kw):
            raise ProviderUnavailable("boom")

        async def aclose(self):
            self.closed += 1

    made = []
    monkeypatch.setattr(popbase, "make_provider", lambda cfg: made.append(BoomProvider(cfg)) or made[-1])
    with pytest.raises(orch.EpisodeAborted):
        await _run(_cfg(n=2, rounds=1))
    assert made and all(p.closed == 1 for p in made)


async def test_per_round_game_params_change_via_schedule(providers):
    # n=2 -> 1 pair/round, max_talk_turns=0 -> deterministic CC (both take 4).
    # The patch from round 2 changes payoff R (CC) from 3 to 7. Round 1 runs on the base config.
    spec = AgentSpec(count=2)
    cfg = EpisodeCfg(
        seed=0, rounds=3, matchmaker="random",
        population=PopulationCfg(kind="roster", agents=[spec],
                                 provider=ProviderCfg(base_url="http://x/v1", model="m")),
        game=GameCfg(max_talk_turns=0, payoffs=Payoffs(R=3)),
        schedule=(ChangePoint(from_round=2, patch={"game": {"payoffs": {"R": 7}}}),),
    )
    pop = await _run(cfg)
    # CC every round: R1 → +3, R2 → +7, R3 → +7 (sticky) for each of the two agents
    assert sum(a.score for a in pop) == pytest.approx(2 * (3 + 7 + 7))


async def test_schedule_patch_honored_on_resume(providers):
    # resuming from round 2 must see round 2's patch (same cfg_for_round materialization)
    spec = AgentSpec(count=2)
    cfg = EpisodeCfg(
        seed=0, rounds=2, matchmaker="random",
        population=PopulationCfg(kind="roster", agents=[spec],
                                 provider=ProviderCfg(base_url="http://x/v1", model="m")),
        game=GameCfg(max_talk_turns=0, payoffs=Payoffs(R=3)),
        schedule=(ChangePoint(from_round=2, patch={"game": {"payoffs": {"R": 7}}}),),
    )
    pop = await _run(cfg, start_round=2)             # playing only round 2 (R=7)
    assert sum(a.score for a in pop) == pytest.approx(2 * 7)


def _pred_cfg(n=2, rounds=1, seed=0):
    # strategy is now per-agent: the whole population is prediction/one_above
    spec = AgentSpec(count=n, play_strategy="prediction", prediction_mapping="one_above")
    return EpisodeCfg(
        seed=seed,
        rounds=rounds,
        matchmaker="random",
        population=PopulationCfg(kind="roster", agents=[spec],
                                 provider=ProviderCfg(base_url="http://x/v1", model="m")),
        game=GameCfg(max_talk_turns=0),
    )


async def test_prediction_strategy_threaded_through_orchestrator(providers):
    # FixedProvider replies number=4 for every call -> predict 4 -> one_above -> choose 5.
    records = []
    await _run(_pred_cfg(n=2, rounds=1), observer=lambda r, p, recs: records.extend(recs))
    assert len(records) == 1
    rec = records[0]
    assert rec.a_predicted == 4 and rec.b_predicted == 4
    assert rec.a_number == 5 and rec.b_number == 5
    assert rec.outcome == "CC"


def _evo_cfg(rounds=3, seed=0):
    return EpisodeCfg(
        seed=seed, rounds=rounds, matchmaker="random",
        population=PopulationCfg(
            kind="roster",
            agents=[AgentSpec(count=3, system_prompt="normal {id}"),
                    AgentSpec(count=1, system_prompt="defect {id}", deceptive=True)],
            provider=ProviderCfg(base_url="http://x/v1", model="m"),
            first_name_pool=[f"Player {i}" for i in range(40)],
            evolution=EvolutionCfg(death_prob=1.0, decept_min=0, decept_max=4),
        ),
        game=GameCfg(max_talk_turns=0),
    )


async def test_evolution_replaces_agents_between_rounds(providers):
    plans = {}
    await _run(_evo_cfg(), observer=lambda r, p, recs: plans.__setitem__(r, p))
    assert plans[1].events == []                        # never before round 1
    deaths = [e for e in plans[2].events if e["type"] == "death"]
    births = [e for e in plans[2].events if e["type"] == "birth"]
    assert len(deaths) == 4 and len(births) == 4        # death_prob=1.0 -> full turnover
    round1 = {x for pair in plans[1].pairings for x in pair} | set(plans[1].idle)
    round2 = {x for pair in plans[2].pairings for x in pair} | set(plans[2].idle)
    assert round1.isdisjoint(round2)                    # everyone was replaced


async def test_no_evolution_events_without_config(providers):
    plans = {}
    await _run(_cfg(n=4, rounds=2), observer=lambda r, p, recs: plans.__setitem__(r, p))
    assert plans[1].events == [] and plans[2].events == []

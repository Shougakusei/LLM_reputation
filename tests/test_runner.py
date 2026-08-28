from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import replace

import pytest

from src import runner
from src.core.config import (
    AgentSpec, EpisodeCfg, EvolutionCfg, GameCfg, JudgeCfg, PopulationCfg, ProviderCfg,
)
from src.judge import JudgeError, JudgeVerdict, MessageRef
from src.population import base as popbase
from src.population import make_population
from src.population.evolution import evolve
from src.providers.base import Completion


class FixedProvider:
    def __init__(self, cfg):
        self.cfg = cfg

    async def complete(self, **kw):
        return Completion(text='{"number": 4, "rationale": "r"}', prompt_tokens=2, completion_tokens=3, raw={})

    async def aclose(self):
        pass


@pytest.fixture(autouse=True)
def _fake_providers(monkeypatch):
    monkeypatch.setattr(popbase, "make_provider", lambda cfg: FixedProvider(cfg))


def _cfg(judge=None, rounds=1, n=2):
    spec = AgentSpec(count=n)
    return EpisodeCfg(
        seed=0, rounds=rounds, matchmaker="random",
        population=PopulationCfg(kind="roster", agents=[spec],
                                 provider=ProviderCfg(base_url="http://x/v1", model="m")),
        game=GameCfg(max_talk_turns=0),
        judge=judge,
    )


def _final_scores(db, rid):
    import sqlite3

    conn = sqlite3.connect(db)
    try:
        return dict(conn.execute(
            "SELECT agent_id, final_score FROM agents WHERE run_id=?", (rid,)).fetchall())
    finally:
        conn.close()


def _judge():
    return JudgeCfg(provider=ProviderCfg(base_url="http://j/v1", model="judge-m"))


# ---- quiet mode (for sweeps: research.py runs hundreds of runs) ----

async def test_quiet_run_suppresses_narration(tmp_path, capsys):
    db = str(tmp_path / "t.db")
    await runner.run_experiment(_cfg(rounds=1), db, quiet=True)
    out = capsys.readouterr().out
    assert "ROUND" not in out and "FINAL SCOREBOARD" not in out
    assert "Running experiment" not in out


async def test_loud_run_narrates_by_default(tmp_path, capsys):
    db = str(tmp_path / "t.db")
    await runner.run_experiment(_cfg(rounds=1), db)        # quiet=False by default
    out = capsys.readouterr().out
    assert "ROUND" in out and "FINAL SCOREBOARD" in out


async def test_quiet_resume_suppresses_narration(tmp_path, capsys):
    db = str(tmp_path / "t.db")
    rid = await runner.run_experiment(_cfg(rounds=2), db, quiet=True)
    capsys.readouterr()
    await runner.resume_run(rid, db, rounds=4, quiet=True)  # extend 2 -> 4 quietly
    out = capsys.readouterr().out
    assert "ROUND" not in out and "Resuming run" not in out


async def test_quiet_run_still_persists(tmp_path):
    db = str(tmp_path / "t.db")
    rid = await runner.run_experiment(_cfg(rounds=1, n=2), db, quiet=True)
    assert _final_scores(db, rid)                          # data written despite the silence


async def test_rerunning_config_creates_new_run_and_leaves_existing(tmp_path):
    import random
    import sqlite3

    from src.population import make_population
    from src.storage import Storage

    db = str(tmp_path / "t.db")
    cfg = _cfg()
    # an aborted run of the same config already exists (begin without finish)
    st = Storage(db)
    pop = make_population(cfg.population, context_window=cfg.context_window).build(random.Random(cfg.seed))
    rid = st.begin(cfg, pop)
    st.close()
    await pop.aclose()

    run_id = await runner.run_experiment(cfg, db)      # no dedup -> new run, old one untouched
    assert run_id != rid

    conn = sqlite3.connect(db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 2                            # both saved
        assert conn.execute("SELECT finished_at FROM runs WHERE run_id=?", (rid,)).fetchone()[0] is None       # the aborted one was never finished
        assert conn.execute("SELECT finished_at FROM runs WHERE run_id=?", (run_id,)).fetchone()[0] is not None  # the new one finished
    finally:
        conn.close()


async def test_rerunning_finished_config_creates_second_run(tmp_path):
    import sqlite3

    db = str(tmp_path / "t.db")
    cfg = _cfg()
    first = await runner.run_experiment(cfg, db)       # finished to completion
    second = await runner.run_experiment(cfg, db)      # same config -> new number, not "nothing to do"
    assert first != second

    conn = sqlite3.connect(db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 2
        hs = [h for (h,) in conn.execute("SELECT config_hash FROM runs")]
        assert hs[0] == hs[1]                          # same config -> same config_hash
    finally:
        conn.close()


# ---- resume / extend (by explicit run number) ----

async def test_resume_unfinished_run_completes_to_configured_rounds(tmp_path):
    # Reference — a straight 4-round run. Then we build an ABORTED run of the same config
    # (2 of 4 played, without finish) and resume it — the result must match the reference.
    import random

    from src.core.orchestrator import run_episode
    from src.population import make_population
    from src.storage import Storage

    cfg = _cfg(n=3, rounds=4)
    ref_db = str(tmp_path / "ref.db")
    ref_id = await runner.run_experiment(cfg, ref_db)
    ref_scores = _final_scores(ref_db, ref_id)

    db = str(tmp_path / "t.db")
    pop = make_population(cfg.population, context_window=cfg.context_window).build(random.Random(cfg.seed))
    st = Storage(db)
    rid = st.begin(cfg, pop)                                  # config holds rounds=4
    await run_episode(replace(cfg, rounds=2), pop, observer=st.observe)   # but only 2 played, without finish
    st.close()
    await pop.aclose()

    done = await runner.resume_run(rid, db)                  # without rounds -> finish up to the saved 4
    assert done == rid
    assert _final_scores(db, rid) == ref_scores              # resumed == straight run


async def test_extend_finished_run_matches_straight_run(tmp_path):
    # A finished 2-round run, extended to 4, must match a straight 4-round run.
    cfg2 = _cfg(n=3, rounds=2)
    cfg4 = _cfg(n=3, rounds=4)
    ref_db = str(tmp_path / "ref.db")
    ref_id = await runner.run_experiment(cfg4, ref_db)
    ref_scores = _final_scores(ref_db, ref_id)

    db = str(tmp_path / "t.db")
    rid = await runner.run_experiment(cfg2, db)              # finished to 2
    same = await runner.resume_run(rid, db, rounds=4)        # extend to 4
    assert same == rid
    assert _final_scores(db, rid) == ref_scores
    import sqlite3
    conn = sqlite3.connect(db)
    try:
        # rounds in the saved config grew to 4, finished_at set again
        assert json.loads(conn.execute("SELECT config FROM runs WHERE run_id=?", (rid,)).fetchone()[0])["rounds"] == 4
        assert conn.execute("SELECT finished_at FROM runs WHERE run_id=?", (rid,)).fetchone()[0] is not None
        assert conn.execute("SELECT COUNT(*) FROM rounds WHERE run_id=?", (rid,)).fetchone()[0] == 4
    finally:
        conn.close()


async def test_extend_honors_schedule_patch_for_new_rounds(tmp_path):
    # A run with a patch taking effect at round 3; the short run (2 rounds) doesn't reach it,
    # extend to 4 must play rounds 3-4 already under the patch and match a straight run of the
    # same schedule for 4 rounds.
    from src.core.config import ChangePoint

    sched = (ChangePoint(from_round=3, patch={"game": {"payoffs": {"R": 7}}}),)
    cfg4 = replace(_cfg(n=3, rounds=4), schedule=sched)
    cfg2 = replace(cfg4, rounds=2)

    ref_db = str(tmp_path / "ref.db")
    ref_id = await runner.run_experiment(cfg4, ref_db)
    ref_scores = _final_scores(ref_db, ref_id)

    db = str(tmp_path / "t.db")
    rid = await runner.run_experiment(cfg2, db)         # 2 played (patch not yet active)
    same = await runner.resume_run(rid, db, rounds=4)   # extend to 4 -> rounds 3-4 under the patch
    assert same == rid
    assert _final_scores(db, rid) == ref_scores


async def test_resume_already_complete_is_noop(tmp_path, capsys):
    db = str(tmp_path / "t.db")
    rid = await runner.run_experiment(_cfg(rounds=2), db)
    capsys.readouterr()
    res = await runner.resume_run(rid, db)                   # already played 2 of 2
    out = capsys.readouterr().out.lower()
    assert res == rid
    assert "nothing to do" in out


async def test_resume_finalizes_run_aborted_on_final_round(tmp_path):
    # An episode aborted on the LAST round: all round rows are already written (the aborted pair
    # is written to the DB before EpisodeAborted), but finished_at is not set — last_round == rounds.
    # The old guard used to return "nothing to do" here, and the run stayed forever unfinished.
    # resume must CLOSE it (set finished_at), without playing any new rounds.
    import random
    import sqlite3

    from src.core.orchestrator import run_episode
    from src.population import make_population
    from src.storage import Storage

    cfg = _cfg(n=3, rounds=2)
    db = str(tmp_path / "t.db")
    pop = make_population(cfg.population, context_window=cfg.context_window).build(random.Random(cfg.seed))
    st = Storage(db)
    rid = st.begin(cfg, pop)
    await run_episode(cfg, pop, observer=st.observe)        # both rounds played, but WITHOUT finish
    assert not st.is_finished(rid)
    st.close()
    await pop.aclose()

    done = await runner.resume_run(rid, db)                 # last_round(2) >= rounds(2), but not finished
    assert done == rid
    conn = sqlite3.connect(db)
    try:
        assert conn.execute(
            "SELECT finished_at FROM runs WHERE run_id=?", (rid,)).fetchone()[0] is not None
        assert conn.execute(
            "SELECT COUNT(*) FROM rounds WHERE run_id=?", (rid,)).fetchone()[0] == 2  # no new rounds
    finally:
        conn.close()
    import sqlite3
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM rounds WHERE run_id=?", (rid,)).fetchone()[0] == 2  # no new rounds
    finally:
        conn.close()


async def test_no_judge_block_means_no_judge_call(tmp_path, monkeypatch):
    called = []
    async def fake_judge(cfg, records):
        called.append(1)
    monkeypatch.setattr(runner, "judge_episode", fake_judge)
    await runner.run_experiment(_cfg(), str(tmp_path / "t.db"))
    assert called == []


async def test_judge_verdict_printed_and_persisted(tmp_path, monkeypatch, capsys):
    seen = {}
    async def fake_judge(cfg, records):
        seen["model"] = cfg.provider.model
        seen["n_records"] = len(records)
        return JudgeVerdict(emerged=True, explanation="history-based trust",
                            evidence=[MessageRef(round=0, pair=0, turn=0)])
    monkeypatch.setattr(runner, "judge_episode", fake_judge)

    run_id = await runner.run_experiment(_cfg(judge=_judge()), str(tmp_path / "t.db"))

    assert seen == {"model": "judge-m", "n_records": 1}   # the judge received the collected records
    out = capsys.readouterr().out
    assert "JUDGE VERDICT" in out and "YES" in out and "history-based trust" in out

    import sqlite3
    conn = sqlite3.connect(str(tmp_path / "t.db"))
    try:
        row = conn.execute("SELECT emerged, model FROM judge_verdicts WHERE run_id=?", (run_id,)).fetchone()
        assert row == (1, "judge-m")
    finally:
        conn.close()


async def test_judge_failure_does_not_lose_the_run(tmp_path, monkeypatch, capsys):
    async def failing_judge(cfg, records):
        raise JudgeError("judge returned an unparseable response after a retry")
    monkeypatch.setattr(runner, "judge_episode", failing_judge)

    run_id = await runner.run_experiment(_cfg(judge=_judge()), str(tmp_path / "t.db"))

    assert run_id is not None                         # episode was saved
    assert "judge failed to reach a verdict" in capsys.readouterr().out

    import sqlite3
    conn = sqlite3.connect(str(tmp_path / "t.db"))
    try:
        assert conn.execute("SELECT finished_at FROM runs").fetchone()[0] is not None
        assert conn.execute("SELECT COUNT(*) FROM judge_verdicts").fetchone()[0] == 0
    finally:
        conn.close()


# ---- evolution: narration and resume replay ----

def _evo_runner_cfg(rounds=2, seed=0):
    return EpisodeCfg(
        seed=seed, rounds=rounds, matchmaker="random",
        population=PopulationCfg(
            kind="roster",
            agents=[AgentSpec(count=3, system_prompt="normal {id}"),
                    AgentSpec(count=1, system_prompt="defect {id}", deceptive=True)],
            provider=ProviderCfg(base_url="http://x/v1", model="m"),
            first_name_pool=[f"Player {i}" for i in range(40)],
            evolution=EvolutionCfg(death_prob=1.0, decept_min=0, decept_max=4)),
        game=GameCfg(max_talk_turns=0))


async def test_narration_prints_deaths_and_births(tmp_path, capsys):
    db = str(tmp_path / "t.db")
    await runner.run_experiment(_evo_runner_cfg(), db)
    out = capsys.readouterr().out
    assert "died" in out and "joined the game" in out


def _exhausting_cfg(rounds=3):
    # name pool sized to leave exactly one leftover name -> death_prob=1.0 kills all 4
    # agents on the very first evolution round (round 2), the second replacement in that
    # same round finds the pool empty and exhausts it deterministically. Pool must still be
    # strictly larger than the agent count to pass config validation on resume's reload.
    return EpisodeCfg(
        seed=0, rounds=rounds, matchmaker="random",
        population=PopulationCfg(
            kind="roster",
            agents=[AgentSpec(count=3, system_prompt="normal {id}"),
                    AgentSpec(count=1, system_prompt="defect {id}", deceptive=True)],
            provider=ProviderCfg(base_url="http://x/v1", model="m"),
            first_name_pool=[f"Player {i}" for i in range(5)],
            evolution=EvolutionCfg(death_prob=1.0, decept_min=0, decept_max=4)),
        game=GameCfg(max_talk_turns=0))


async def test_run_experiment_handles_name_pool_exhaustion_without_raising(tmp_path, capsys):
    db = str(tmp_path / "t.db")
    rid = await runner.run_experiment(_exhausting_cfg(), db)   # must not raise NamePoolExhausted
    out = capsys.readouterr().out
    assert "name pool exhausted" in out.lower() or "pool exhausted" in out.lower()
    assert "cannot be resumed past this point" in out.lower()

    conn = sqlite3.connect(db)
    try:
        assert conn.execute(
            "SELECT finished_at FROM runs WHERE run_id=?", (rid,)).fetchone()[0] is None
    finally:
        conn.close()


async def test_resume_run_handles_name_pool_exhaustion_without_raising(tmp_path, capsys):
    db = str(tmp_path / "t.db")
    rid = await runner.run_experiment(_exhausting_cfg(), db, quiet=True)
    capsys.readouterr()
    done = await runner.resume_run(rid, db)          # must not raise NamePoolExhausted either
    assert done == rid
    out = capsys.readouterr().out
    assert "pool exhausted" in out.lower()

    conn = sqlite3.connect(db)
    try:
        assert conn.execute(
            "SELECT finished_at FROM runs WHERE run_id=?", (rid,)).fetchone()[0] is None
    finally:
        conn.close()


async def test_resume_replays_evolution_deterministically(tmp_path):
    db = str(tmp_path / "t.db")
    cfg = _evo_runner_cfg(rounds=2)
    rid = await runner.run_experiment(cfg, db, quiet=True)
    await runner.resume_run(rid, db, rounds=4, quiet=True)
    # expected live roster after round 4 = fresh build + evolution replay for rounds 2..4
    pop = make_population(cfg.population).build(random.Random(cfg.seed))
    try:
        for r in range(2, 5):
            evolve(pop, cfg.population, random.Random(f"{cfg.seed}:evolution:{r}"), r)
        expected = set(pop.ids())
    finally:
        await pop.aclose()
    conn = sqlite3.connect(db)
    try:
        ids = set()
        for a, b in conn.execute(
                "SELECT a_id, b_id FROM pairings WHERE run_id=? AND round_idx=4", (rid,)):
            ids |= {a, b}
        for (aid,) in conn.execute(
                "SELECT agent_id FROM idle WHERE run_id=? AND round_idx=4", (rid,)):
            ids.add(aid)
        assert ids == expected
    finally:
        conn.close()


async def test_resume_replays_from_the_first_aborted_pairing(tmp_path):
    # A run whose round 2 pairing was aborted (LLM failure) but which went on to round 3 and
    # even closed: resume must rewind to round 2 and replay rounds 2..3 — no hole is left.
    from src.core.agent import LLMCall
    from src.games.base import PairingRecord
    from src.matchmaking import RoundPlan
    from src.storage import Storage

    cfg = _cfg(n=2, rounds=3)
    db = str(tmp_path / "t.db")
    pop = make_population(cfg.population, context_window=cfg.context_window).build(random.Random(cfg.seed))
    st = Storage(db)
    rid = st.begin(cfg, pop)
    ids = pop.ids()
    plan = RoundPlan(pairings=[(ids[0], ids[1])], idle=[], events=[])
    ok = dict(transcript=[], a_number=4, b_number=4, outcome="CC", a_payoff=3.0, b_payoff=3.0,
              usage={"prompt_tokens": 1, "completion_tokens": 1, "calls": 1})
    st.observe(1, plan, [PairingRecord(round=1, a_id=ids[0], b_id=ids[1], **ok)])
    st.observe(2, plan, [PairingRecord(
        round=2, a_id=ids[0], b_id=ids[1], transcript=[], finished=False,
        usage={"prompt_tokens": 0, "completion_tokens": 0, "calls": 1},
        llm_calls=[LLMCall(ids[1], "decide", 1, 1, "server_error", 503, {}, None, None, "HTTP 503", 0, 0)])])
    st.observe(3, plan, [PairingRecord(round=3, a_id=ids[0], b_id=ids[1], **ok)])
    st.finish(pop)
    st.close()
    await pop.aclose()

    await runner.resume_run(rid, db, quiet=True)
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM pairings WHERE run_id=? AND finished=0", (rid,)).fetchone()[0] == 0
        assert [r[0] for r in conn.execute("SELECT round_idx FROM rounds WHERE run_id=? ORDER BY 1", (rid,))] == [1, 2, 3]
        assert conn.execute("SELECT a_number FROM pairings WHERE run_id=? AND round_idx=1", (rid,)).fetchone()[0] == 4  # round 1 kept
        assert conn.execute("SELECT finished_at FROM runs WHERE run_id=?", (rid,)).fetchone()[0] is not None
    finally:
        conn.close()

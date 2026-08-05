from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import asdict, replace

import pytest

from src.core import orchestrator as orch
from src.core.agent import LLMCall
from src.core.config import (
    AgentSpec, EpisodeCfg, EvolutionCfg, GameCfg, JudgeCfg, PopulationCfg, ProviderCfg,
)
from src.judge import JudgeVerdict, MessageRef
from src.games.base import PairingRecord
from src.matchmaking.base import RoundPlan
from src.population import base as popbase
from src.population import make_population
from src.providers.base import Completion, HttpAttempt
from src.storage import Storage
from src.storage.store import _hash_config_dict


class FixedProvider:
    def __init__(self, cfg):
        self.cfg = cfg

    async def complete(self, *, system, messages, temperature, max_tokens):
        text = '{"number": 4, "rationale": "r"}'
        req = {"model": "m", "messages": [{"role": "system", "content": system}]}
        at = HttpAttempt("ok", 200, req, text, text, None, 2, 3)
        return Completion(text=text, prompt_tokens=2, completion_tokens=3, raw={},
                          request=req, attempts=(at,))

    async def aclose(self):
        pass


@pytest.fixture(autouse=True)
def _fake_providers(monkeypatch):
    monkeypatch.setattr(popbase, "make_provider", lambda cfg: FixedProvider(cfg))


def _cfg(seed=0, n=3, rounds=2):
    spec = AgentSpec(count=n)
    return EpisodeCfg(
        seed=seed, rounds=rounds, matchmaker="random",
        population=PopulationCfg(kind="roster", agents=[spec],
                                 provider=ProviderCfg(base_url="http://x/v1", model="m")),
        game=GameCfg(max_talk_turns=0),
    )


def _pop(cfg):
    return make_population(cfg.population, context_window=cfg.context_window).build(random.Random(cfg.seed))


def _store(tmp_path, name="t.db"):
    return Storage(str(tmp_path / name))


# ---- Slice 1: begin + run_id ----

def test_begin_writes_runs_and_agents(tmp_path):
    cfg = _cfg(n=3)
    st = _store(tmp_path)
    try:
        run_id = st.begin(cfg, _pop(cfg))
        assert run_id == 1           # first run in a fresh DB -> incremental id = 1
        c = st._conn
        assert c.execute("SELECT run_id, seed FROM runs").fetchall() == [(1, 0)]
        agents = c.execute("SELECT agent_id, system_prompt FROM agents ORDER BY agent_id").fetchall()
        assert [a for a, _ in agents] == ["A1", "A2", "A3"]
        assert json.loads(c.execute("SELECT config FROM runs").fetchone()[0])["matchmaker"] == "random"
        assert json.loads(c.execute("SELECT provider FROM agents LIMIT 1").fetchone()[0])["model"] == "m"
    finally:
        st.close()


def test_run_id_is_incremental_and_config_hash_groups_runs(tmp_path):
    # run_id is a counter (1, 2, 3 …); runs of the same config are grouped by the config_hash column
    st = _store(tmp_path)
    try:
        id1 = st.begin(_cfg(seed=1), _pop(_cfg(seed=1)))
        id2 = st.begin(_cfg(seed=1), _pop(_cfg(seed=1)))   # same config -> NEW number
        id3 = st.begin(_cfg(seed=2), _pop(_cfg(seed=2)))
        assert [id1, id2, id3] == [1, 2, 3]
        hashes = dict(st._conn.execute("SELECT run_id, config_hash FROM runs").fetchall())
        assert hashes[id1] == hashes[id2]      # same config -> same config_hash
        assert hashes[id1] != hashes[id3]      # different seed -> different config_hash
    finally:
        st.close()


def test_config_hash_ignores_rounds(tmp_path):
    # rounds is "how far the simulation got to", not part of the design: a run and its continuation share config_hash
    st = _store(tmp_path)
    try:
        short = _cfg(seed=1, rounds=2)
        long = replace(short, rounds=20)
        id_s = st.begin(short, _pop(short))
        id_l = st.begin(long, _pop(long))
        h = dict(st._conn.execute("SELECT run_id, config_hash FROM runs").fetchall())
        assert h[id_s] == h[id_l]               # different length -> one config_hash (same family)
    finally:
        st.close()


def test_begin_twice_creates_two_distinct_runs(tmp_path):
    cfg = _cfg(seed=1)
    st = _store(tmp_path)
    try:
        first = st.begin(cfg, _pop(cfg))
        second = st.begin(cfg, _pop(cfg))            # no more dedup: the second run gets a new number
        assert first != second
        assert st._conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 2
    finally:
        st.close()


def test_is_finished_reflects_finished_at(tmp_path):
    cfg = _cfg(n=2)
    st = _store(tmp_path)
    try:
        rid = st.begin(cfg, _pop(cfg))
        assert st.is_finished(rid) is False          # not finished yet
        st.finish(_pop(cfg))                          # sets finished_at
        assert st.is_finished(rid) is True
        assert st.is_finished("nope") is False        # missing run
    finally:
        st.close()


def test_unfinished_runs_lists_only_open_runs(tmp_path):
    cfg = _cfg()
    st = _store(tmp_path)
    try:
        r1 = st.begin(cfg, _pop(cfg), name="m 0")     # will finish this one
        r2 = st.begin(cfg, _pop(cfg), name="m 1")     # leave this one open
        st._conn.execute("UPDATE runs SET finished_at=? WHERE run_id=?", ("2026-01-01", r1))
        assert st.unfinished_runs() == [(r2, "m 1")]   # only the unfinished one, with its name
    finally:
        st.close()


def test_run_id_by_name_finds_run(tmp_path):
    cfg = _cfg()
    st = _store(tmp_path)
    try:
        rid = st.begin(cfg, _pop(cfg), name="llama 7")
        assert st.run_id_by_name("llama 7") == rid
        assert st.run_id_by_name("nope") is None        # no such name
    finally:
        st.close()


def test_delete_run_removes_all_rows(tmp_path):
    cfg = _cfg(n=3)
    st = _store(tmp_path)
    try:
        rid = st.begin(cfg, _pop(cfg))
        calls = [LLMCall("A1", "talk", 1, 1, "ok", 200, {"model": "m"}, "hi", '{"x":1}', None, 2, 3, turn_idx=0)]
        rec = PairingRecord(
            round=0, a_id="A1", b_id="A2",
            transcript=[{"speaker": "A1", "text": "hi", "ready": True}],
            a_number=4, b_number=4, a_rationale="ra", b_rationale="rb",
            outcome="CC", a_payoff=3.0, b_payoff=3.0,
            usage={"prompt_tokens": 4, "completion_tokens": 6, "calls": 1}, llm_calls=calls,
        )
        st.observe(0, RoundPlan(pairings=[("A1", "A2")], idle=["A3"], events=[]), [rec])
        st.delete_run(rid)
        c = st._conn
        for table in ("runs", "agents", "rounds", "idle", "pairings", "messages", "llm_calls"):
            assert c.execute(f"SELECT COUNT(*) FROM {table} WHERE run_id=?", (rid,)).fetchone()[0] == 0
    finally:
        st.close()


def test_begin_stores_agent_system_prompt(tmp_path):
    spec = AgentSpec(count=1, system_prompt="You are {id}. Custom frame.")
    cfg = replace(_cfg(), population=PopulationCfg(kind="roster", agents=[spec],
                                                   provider=ProviderCfg(base_url="http://x/v1", model="m")))
    st = _store(tmp_path)
    try:
        st.begin(cfg, _pop(cfg))
        assert st._conn.execute("SELECT system_prompt FROM agents").fetchone() == ("You are {id}. Custom frame.",)
    finally:
        st.close()


# ---- Slice 2: observe (one txn per round) ----

def test_observe_writes_round_tables(tmp_path):
    cfg = _cfg(n=3)
    st = _store(tmp_path)
    try:
        rid = st.begin(cfg, _pop(cfg))      # agents A1..A3 exist (FK targets)
        rec = PairingRecord(
            round=0, a_id="A1", b_id="A2",
            transcript=[
                {"speaker": "A1", "text": "hi", "ready": False},
                {"speaker": "A2", "text": "ok", "ready": True},
            ],
            a_number=4, b_number=4, a_rationale="ra", b_rationale="rb",
            outcome="CC", a_payoff=3.0, b_payoff=3.0,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "calls": 4},
        )
        st.observe(0, RoundPlan(pairings=[("A1", "A2")], idle=["A3"], events=[]), [rec])
        c = st._conn
        assert c.execute("SELECT round_idx FROM rounds").fetchall() == [(0,)]
        assert c.execute("SELECT agent_id FROM idle").fetchall() == [("A3",)]
        p = c.execute("SELECT a_id, b_id, a_outcome, usage_calls FROM pairings").fetchone()
        assert p == ("A1", "A2", "CC", 4)
        msgs = c.execute("SELECT turn_idx, speaker, text, ready FROM messages ORDER BY turn_idx").fetchall()
        assert msgs == [(0, "A1", "hi", 0), (1, "A2", "ok", 1)]
        assert rid  # non-empty
    finally:
        st.close()


# ---- Slice L2: llm_calls + finished flag ----

def _plan(idle=None):
    return RoundPlan(pairings=[("A1", "A2")], idle=idle or [], events=[])


def test_observe_writes_llm_calls_with_join(tmp_path):
    cfg = _cfg(n=3)
    st = _store(tmp_path)
    try:
        st.begin(cfg, _pop(cfg))
        calls = [
            LLMCall("A1", "talk", 1, 1, "ok", 200, {"model": "m"}, "hi", '{"x":1}', None, 2, 3, turn_idx=0),
            LLMCall("A1", "decide", 1, 1, "ok", 200, {"model": "m"}, '{"number":4}', '{"y":2}', None, 2, 3),
        ]
        rec = PairingRecord(
            round=1, a_id="A1", b_id="A2",
            transcript=[{"speaker": "A1", "text": "hi", "ready": True}],
            a_number=4, b_number=4, a_rationale="ra", b_rationale="rb",
            outcome="CC", a_payoff=3.0, b_payoff=3.0,
            usage={"prompt_tokens": 4, "completion_tokens": 6, "calls": 2}, llm_calls=calls,
        )
        st.observe(1, _plan(), [rec])
        c = st._conn
        rows = c.execute(
            "SELECT call_idx, agent_id, phase, turn_idx, attempt, http_attempt, status, status_code, response "
            "FROM llm_calls ORDER BY call_idx"
        ).fetchall()
        assert rows == [
            (0, "A1", "talk", 0, 1, 1, "ok", 200, "hi"),
            (1, "A1", "decide", None, 1, 1, "ok", 200, '{"number":4}'),
        ]
        # join llm_calls -> pairings: raw call next to its outcome
        joined = c.execute(
            "SELECT lc.phase, p.a_outcome FROM llm_calls lc "
            "JOIN pairings p USING (run_id, round_idx, pair_idx) WHERE lc.phase='decide'"
        ).fetchone()
        assert joined == ("decide", "CC")
        assert json.loads(c.execute("SELECT request FROM llm_calls LIMIT 1").fetchone()[0])["model"] == "m"
    finally:
        st.close()


def test_observe_persists_memory_notes_and_note_calls(tmp_path):
    cfg = _cfg(n=3)
    st = _store(tmp_path)
    try:
        st.begin(cfg, _pop(cfg))
        calls = [
            LLMCall("A1", "decide", 1, 1, "ok", 200, {"model": "m"}, '{"number":4}', "{}", None, 2, 3),
            LLMCall("A1", "note", 1, 1, "ok", 200, {"model": "m"}, '{"notes":"n"}', "{}", None, 2, 3),
        ]
        rec = PairingRecord(
            round=1, a_id="A1", b_id="A2", transcript=[],
            a_number=4, b_number=4, a_rationale="ra", b_rationale="rb",
            outcome="CC", a_payoff=3.0, b_payoff=3.0,
            a_notes="A2 cooperates", b_notes="A1 cooperates",
            usage={"prompt_tokens": 4, "completion_tokens": 6, "calls": 2}, llm_calls=calls,
        )
        st.observe(1, _plan(idle=["A3"]), [rec])
        c = st._conn
        assert c.execute("SELECT a_notes, b_notes FROM pairings").fetchone() == ("A2 cooperates", "A1 cooperates")
        # the note call landed in llm_calls and joins with its pair
        joined = c.execute(
            "SELECT lc.phase, p.a_notes FROM llm_calls lc "
            "JOIN pairings p USING (run_id, round_idx, pair_idx) WHERE lc.phase='note'"
        ).fetchone()
        assert joined == ("note", "A2 cooperates")
    finally:
        st.close()


def test_observe_writes_aborted_pairing(tmp_path):
    cfg = _cfg(n=3)
    st = _store(tmp_path)
    try:
        st.begin(cfg, _pop(cfg))
        calls = [LLMCall("A2", "decide", 1, 1, "network", None, {"model": "m"}, None, None, "boom", 0, 0)]
        rec = PairingRecord(
            round=1, a_id="A1", b_id="A2", transcript=[], finished=False,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "calls": 1}, llm_calls=calls,
        )
        st.observe(1, _plan(), [rec])
        c = st._conn
        assert c.execute("SELECT finished, a_number, a_outcome FROM pairings").fetchone() == (0, None, None)
        assert c.execute("SELECT status, error FROM llm_calls").fetchone() == ("network", "boom")
    finally:
        st.close()


def test_check_rejects_finished_pairing_without_result(tmp_path):
    cfg = _cfg(n=3)
    st = _store(tmp_path)
    try:
        st.begin(cfg, _pop(cfg))
        bad = PairingRecord(round=1, a_id="A1", b_id="A2", transcript=[], finished=True)  # a_number=None
        with pytest.raises(sqlite3.IntegrityError):
            st.observe(1, _plan(), [bad])
    finally:
        st.close()


# ---- Slices 1+2 end-to-end: Storage as the orchestrator observer ----

async def test_logs_full_episode(tmp_path):
    cfg = _cfg(n=3, rounds=2)
    pop = _pop(cfg)
    st = _store(tmp_path)
    rid = st.begin(cfg, pop)
    try:
        await orch.run_episode(cfg, pop, observer=st.observe)
        st.finish(pop)
    finally:
        await pop.aclose()
    c = st._conn
    try:
        assert c.execute("SELECT finished_at FROM runs WHERE run_id=?", (rid,)).fetchone()[0] is not None
        assert c.execute("SELECT COUNT(*) FROM rounds").fetchone()[0] == 2
        assert c.execute("SELECT COUNT(*) FROM pairings").fetchone()[0] == 2     # 1 pair/round
        assert c.execute("SELECT COUNT(*) FROM idle").fetchone()[0] == 2         # 1 idle/round (N=3)
        assert c.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0     # max_talk_turns=0
        # L2: 2 decide calls per pair (a, b) -> 2 pairs -> 4 llm_calls rows, all ok decide
        assert c.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0] == 4
        assert all(s == "ok" and p == "decide"
                   for s, p in c.execute("SELECT status, phase FROM llm_calls"))
        # each llm_calls row joins with its pair (FK + (round,pair))
        assert c.execute(
            "SELECT COUNT(*) FROM llm_calls lc JOIN pairings p USING (run_id, round_idx, pair_idx)"
        ).fetchone()[0] == 4
        assert all(o == "CC" for (o,) in c.execute("SELECT a_outcome FROM pairings"))
        stored = dict(c.execute("SELECT agent_id, final_score FROM agents").fetchall())
        assert stored == {a.id: a.score for a in pop}
        # integrity: idle gives each round's idle agent +idle_payoff, sum matches
        assert sum(stored.values()) == pytest.approx(14.0)
    finally:
        st.close()


# ---- Slice: load_state — reconstructing a run's state from the DB (resume) ----

async def test_load_state_reconstructs_memory_and_scores(tmp_path):
    # run a real episode, then restore from the DB: diaries must render
    # BYTE-FOR-BYTE like the "live" population, score must match, last_round is the last round's number
    cfg = _cfg(n=3, rounds=3)
    pop = _pop(cfg)
    st = _store(tmp_path)
    rid = st.begin(cfg, pop)
    try:
        await orch.run_episode(cfg, pop, observer=st.observe)
    finally:
        await pop.aclose()

    state = st.load_state(rid, cfg.idle_payoff)
    try:
        assert state.last_round == 3
        assert state.scores == {a.id: a.score for a in pop}        # score restored exactly (including idle)
        for a in pop:
            live = [m.content for m in a.memory.render(cfg.context_window, cfg.game)]
            restored = [m.content for m in state.memories[a.id].render(cfg.context_window, cfg.game)]
            assert restored == live                                 # memory renders identically
    finally:
        st.close()


def test_load_state_restores_notes_buffer_and_idle_score(tmp_path):
    cfg = _cfg(n=3)
    st = _store(tmp_path)
    rid = st.begin(cfg, _pop(cfg))
    try:
        # round 1: A1 vs A2 (A1 folded notes), A3 idle
        rec1 = PairingRecord(
            round=1, a_id="A1", b_id="A2", transcript=[],
            a_number=4, b_number=4, a_rationale="ra", b_rationale="rb",
            outcome="CC", a_payoff=3.0, b_payoff=3.0, a_notes="A2 is honest",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "calls": 0},
        )
        st.observe(1, RoundPlan(pairings=[("A1", "A2")], idle=["A3"], events=[]), [rec1])
        # round 2: A1 vs A3 (buffer after notes), A2 idle; A1 outbid A3
        rec2 = PairingRecord(
            round=2, a_id="A1", b_id="A3", transcript=[],
            a_number=5, b_number=4, a_rationale="ra2", b_rationale="rb2",
            outcome="DC", a_payoff=5.0, b_payoff=0.0,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "calls": 0},
        )
        st.observe(2, RoundPlan(pairings=[("A1", "A3")], idle=["A2"], events=[]), [rec2])

        state = st.load_state(rid, cfg.idle_payoff)
        assert state.last_round == 2
        m = state.memories["A1"]
        assert m.notes == "A2 is honest"
        assert m.noted_upto == 1                  # one round folded into notes
        assert len(m.entries) == 2                # fresh buffer (round 2) kept
        # score: A1 = 3+5 = 8; A2 played r1 (+3) and was idle r2 (+1) = 4; A3 was idle r1 (+1) + r2 (0) = 1
        assert state.scores == {"A1": 8.0, "A2": 4.0, "A3": 1.0}
    finally:
        st.close()


# ---- Slice 4: LLM judge — run_id stability and verdict persistence ----

def _judge_cfg():
    return JudgeCfg(provider=ProviderCfg(base_url="http://j/v1", model="judge-m"))


def test_config_hash_ignores_judge_block(tmp_path):
    base = _cfg(seed=1)
    judged = replace(base, judge=_judge_cfg())
    st = _store(tmp_path)
    try:
        id_base = st.begin(base, _pop(base))
        id_judged = st.begin(judged, _pop(judged))
        h = dict(st._conn.execute("SELECT run_id, config_hash FROM runs").fetchall())
        assert h[id_base] == h[id_judged]      # judge is analytics, not gameplay: config_hash doesn't change
    finally:
        st.close()


def test_config_hash_changes_with_schedule(tmp_path):
    # schedule is part of the design: a different schedule -> a different config_hash; but rounds is still outside the hash
    from src.core.config import ChangePoint

    base = _cfg(seed=1)
    scheduled = replace(base, schedule=(ChangePoint(from_round=2, patch={"game": {"payoffs": {"T": 6}}}),))
    longer = replace(scheduled, rounds=base.rounds + 5)
    st = _store(tmp_path)
    try:
        id_base = st.begin(base, _pop(base))
        id_sched = st.begin(scheduled, _pop(scheduled))
        id_long = st.begin(longer, _pop(longer))
        h = dict(st._conn.execute("SELECT run_id, config_hash FROM runs").fetchall())
        assert h[id_base] != h[id_sched]       # schedule changes the design
        assert h[id_sched] == h[id_long]       # rounds is still outside the hash (same family)
    finally:
        st.close()


def test_hash_config_dict_strips_default_evolution_and_deceptive_noise():
    # An evolution-free config now carries `population.evolution: None` and
    # `deceptive: False` on every agent spec (new defaults). Those keys must not
    # perturb the hash of pre-branch configs that never had them at all.
    cfg = _cfg(seed=1)
    d = asdict(cfg)
    assert d["population"]["evolution"] is None
    assert all(a["deceptive"] is False for a in d["population"]["agents"])

    stripped = json.loads(json.dumps(d))       # deep copy via round-trip
    stripped["population"].pop("evolution")
    for a in stripped["population"]["agents"]:
        a.pop("deceptive")

    assert _hash_config_dict(d) == _hash_config_dict(stripped)


def test_hash_config_dict_keeps_evolution_when_enabled(tmp_path):
    # Evolution-enabled configs are new designs — their own hash must reflect that,
    # and must differ from the otherwise-identical evolution-free twin.
    base = _cfg(seed=1)
    evo_pop = replace(
        base.population,
        agents=[AgentSpec(count=2), AgentSpec(count=1, deceptive=True)],
        first_name_pool=["Alice", "Bob", "Carol"],
        evolution=EvolutionCfg(death_prob=0.1, decept_min=0, decept_max=1),
    )
    evo_cfg = replace(base, population=evo_pop)

    h_no_evo = _hash_config_dict(asdict(base))
    h_evo = _hash_config_dict(asdict(evo_cfg))
    assert h_evo != h_no_evo


def test_judge_config_still_persisted_in_runs(tmp_path):
    cfg = replace(_cfg(), judge=_judge_cfg())
    st = _store(tmp_path)
    try:
        st.begin(cfg, _pop(cfg))
        stored = json.loads(st._conn.execute("SELECT config FROM runs").fetchone()[0])
        assert stored["judge"]["provider"]["model"] == "judge-m"
    finally:
        st.close()


def test_save_verdict_roundtrip(tmp_path):
    cfg = _cfg()
    st = _store(tmp_path)
    try:
        st.begin(cfg, _pop(cfg))
        st.save_verdict(
            JudgeVerdict(emerged=True, explanation="gossip observed",
                         evidence=[MessageRef(round=0, pair=0, turn=1)]),
            model="judge-m",
        )
        row = st._conn.execute(
            "SELECT emerged, explanation, evidence, model, created_at FROM judge_verdicts"
        ).fetchone()
        assert row[0] == 1
        assert row[1] == "gossip observed"
        assert json.loads(row[2]) == [{"round": 0, "pair": 0, "turn": 1}]
        assert row[3] == "judge-m"
        assert row[4]                                  # created_at is filled in
    finally:
        st.close()


# ---- Slice: evolution events (born_round, died_round, deceptive) ----

class _EvoStubProvider:
    """Stub provider that refuses to complete (storage tests must not call LLM)."""

    def __init__(self, cfg):
        self.cfg = cfg

    async def complete(self, **kw):
        raise AssertionError("storage tests must not call the LLM")

    async def aclose(self):
        pass


@pytest.fixture
def _evo_stub_providers(monkeypatch):
    """Replace make_provider with a stub that rejects LLM calls."""
    monkeypatch.setattr(popbase, "make_provider", lambda cfg: _EvoStubProvider(cfg))


def _evo_storage_cfg():
    """Test config with one normal and one deceptive agent."""
    return EpisodeCfg(
        seed=0, rounds=2, matchmaker="random",
        population=PopulationCfg(
            kind="roster",
            agents=[AgentSpec(count=1, system_prompt="normal {id}"),
                    AgentSpec(count=1, system_prompt="defect {id}", deceptive=True)],
            provider=ProviderCfg(base_url="http://x/v1", model="m")),
        game=GameCfg(max_talk_turns=0))


def test_begin_records_deceptive_flag(tmp_path, _evo_stub_providers):
    """Verify deceptive flag is stored on insert."""
    st = Storage(str(tmp_path / "t.db"))
    cfg = _evo_storage_cfg()
    pop = make_population(cfg.population).build(random.Random(0))
    rid = st.begin(cfg, pop)
    rows = dict(st.conn.execute(
        "SELECT agent_id, deceptive FROM agents WHERE run_id=?", (rid,)))
    assert rows == {"A1": 0, "A2": 1}
    st.close()


def test_observe_persists_evolution_events(tmp_path, _evo_stub_providers):
    """Verify death and birth events are persisted in observe."""
    st = Storage(str(tmp_path / "t.db"))
    cfg = _evo_storage_cfg()
    pop = make_population(cfg.population).build(random.Random(0))
    rid = st.begin(cfg, pop)
    plan = RoundPlan(pairings=[], idle=[], events=[
        {"type": "death", "agent": "A1", "score": 5.0},
        {"type": "birth", "agent": "Player 9", "deceptive": True,
         "system_prompt": "defect {id}",
         "provider": {"base_url": "http://x/v1", "model": "m"}},
    ])
    st.observe(2, plan, [])
    assert st.conn.execute(
        "SELECT died_round, final_score FROM agents WHERE run_id=? AND agent_id='A1'",
        (rid,)).fetchone() == (2, 5.0)
    assert st.conn.execute(
        "SELECT born_round, deceptive, system_prompt FROM agents "
        "WHERE run_id=? AND agent_id='Player 9'", (rid,)).fetchone() == (2, 1, "defect {id}")
    st.close()


def test_migration_adds_evolution_columns_to_old_db(tmp_path):
    """Verify migration adds new columns to databases created before evolution feature."""
    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE agents (
               run_id INTEGER NOT NULL, agent_id TEXT NOT NULL,
               system_prompt TEXT, provider TEXT NOT NULL, final_score REAL,
               PRIMARY KEY (run_id, agent_id))""")
    conn.commit()
    conn.close()
    st = Storage(path)
    cols = {row[1] for row in st.conn.execute("PRAGMA table_info(agents)")}
    assert {"born_round", "died_round", "deceptive"} <= cols
    st.close()

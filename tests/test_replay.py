from __future__ import annotations

import json
import sqlite3

import pytest
import replay as replay_mod

from replay import (
    _expand_newlines, _preview, _provider_line, _readable, _roster_line,
    _roster_names, cited_set, highlight, load_verdict,
)
from src import runner
from src.core.config import AgentSpec, EpisodeCfg, EvolutionCfg, GameCfg, PopulationCfg, ProviderCfg
from src.population import base as popbase
from src.providers.base import Completion


def test_readable_starts_content_body_on_new_line():
    out = _readable('"content": "Your memory\\nof rounds"')
    assert out == '"content": "\nYour memory\nof rounds"'   # body starts on a new line


def test_readable_unescapes_quotes():
    assert _readable('Set \\"ready\\": true') == 'Set "ready": true'


def test_preview_collapses_whitespace_and_keeps_short_text():
    assert _preview("hello\n   there  world") == "hello there world"


def test_preview_truncates_long_text_with_ellipsis():
    p = _preview("x" * 100, n=10)
    assert len(p) == 10 and p.endswith("…")


def test_preview_empty_for_none_or_blank():
    assert _preview(None) == "" and _preview("") == ""


def test_expand_newlines_turns_escaped_into_real():
    assert _expand_newlines("a\\nb") == "a\nb"        # two \n characters -> a real newline


def test_expand_newlines_passes_through_none():
    assert _expand_newlines(None) is None


def test_provider_line_puts_model_last():
    prov = {"model": "llama3.1:8b", "temperature": 0.7, "max_tokens": 2000}
    assert _provider_line(prov) == "provider: temp=0.7 max_tokens=2000 model=llama3.1:8b"


def test_roster_line_shows_system_prompt_and_count():
    assert _roster_line({"system_prompt": "You are X.", "count": 3}) == "  3x You are X."


def test_roster_line_truncates_long_system_prompt_to_one_line():
    line = _roster_line({"system_prompt": "L" * 200, "count": 1})
    assert line.startswith("  1x ") and line.endswith("…") and len(line) <= 5 + 80


def test_roster_line_count_defaults_to_one():
    assert _roster_line({"system_prompt": "p"}) == "  1x p"


def test_roster_line_falls_back_to_legacy_persona():
    assert _roster_line({"persona": "pragmatic", "count": 2}) == "  2x pragmatic"


def test_roster_names_slices_ids_by_count():
    # ids — the run's agent_ids in build order; sliced by each spec's count
    specs = [{"count": 2}, {"count": 3}]
    ids = ["Player 1", "Player 2", "Player 3", "Player 4", "Player 5"]
    assert _roster_names(specs, ids) == [["Player 1", "Player 2"],
                                         ["Player 3", "Player 4", "Player 5"]]


def test_roster_names_count_defaults_to_one():
    assert _roster_names([{}, {}], ["A", "B"]) == [["A"], ["B"]]


def test_cited_set_parses_evidence_json():
    evidence = json.dumps([{"round": 0, "pair": 1, "turn": 2}, {"round": 3, "pair": 0, "turn": 0}])
    assert cited_set(evidence) == {(0, 1, 2), (3, 0, 0)}


def test_highlight_wraps_in_yellow_when_on():
    assert highlight("msg", on=True) == "\033[93mmsg\033[0m"


def test_highlight_passthrough_when_off():
    assert highlight("msg", on=False) == "msg"


def test_load_verdict_none_for_old_db_without_table():
    conn = sqlite3.connect(":memory:")               # DB without a judge_verdicts table
    try:
        assert load_verdict(conn, "whatever") is None
    finally:
        conn.close()


def test_load_verdict_returns_row_when_present():
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(
            "CREATE TABLE judge_verdicts (run_id TEXT PRIMARY KEY, emerged INTEGER, "
            "explanation TEXT, evidence TEXT, model TEXT, created_at TEXT)"
        )
        conn.execute(
            "INSERT INTO judge_verdicts VALUES (?,?,?,?,?,?)",
            ("rid1", 1, "gossip", json.dumps([{"round": 0, "pair": 0, "turn": 1}]), "judge-m", "t"),
        )
        assert load_verdict(conn, "rid1") == (1, "gossip", '[{"round": 0, "pair": 0, "turn": 1}]')
        assert load_verdict(conn, "missing") is None
    finally:
        conn.close()


class _EvoFixedProvider:
    def __init__(self, cfg):
        self.cfg = cfg

    async def complete(self, **kw):
        return Completion(text='{"number": 4, "rationale": "r"}',
                          prompt_tokens=2, completion_tokens=3, raw={})

    async def aclose(self):
        pass


@pytest.fixture
def _evo_providers(monkeypatch):
    monkeypatch.setattr(popbase, "make_provider", lambda cfg: _EvoFixedProvider(cfg))


def _evo_replay_cfg():
    return EpisodeCfg(
        seed=0, rounds=2, matchmaker="random",
        population=PopulationCfg(
            kind="roster",
            agents=[AgentSpec(count=3, system_prompt="normal {id}"),
                    AgentSpec(count=1, system_prompt="defect {id}", deceptive=True)],
            provider=ProviderCfg(base_url="http://x/v1", model="m"),
            first_name_pool=[f"Player {i}" for i in range(40)],
            evolution=EvolutionCfg(death_prob=1.0, decept_min=0, decept_max=4)),
        game=GameCfg(max_talk_turns=0))


async def test_replay_shows_evolution_events(tmp_path, capsys, _evo_providers):
    db = str(tmp_path / "t.db")
    rid = await runner.run_experiment(_evo_replay_cfg(), db, quiet=True)
    capsys.readouterr()                                 # drop the run's own output
    conn = sqlite3.connect(db)
    try:
        replay_mod.replay(conn, rid)
    finally:
        conn.close()
    out = capsys.readouterr().out
    # Split output by ROUND 2 header to verify no spurious births in round 1
    parts = out.split("ROUND 2", 1)
    assert len(parts) == 2, "Output must contain ROUND 2"
    round1_section = parts[0]
    round2_section = parts[1]
    # Round 1: no evolution events (initial roster has born_round=1 by schema, not shown in replay)
    assert "died" not in round1_section, "ROUND 1 should not show deaths"
    assert "joined the game" not in round1_section, "ROUND 1 should not show births"
    # Round 2: genuine evolution events (agents die/born per evolution rules)
    assert "died" in round2_section, "ROUND 2 should show deaths"
    assert "joined the game" in round2_section, "ROUND 2 should show births"

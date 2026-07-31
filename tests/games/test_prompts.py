from __future__ import annotations

from src.core.config import GameCfg
from src.games.prompts import (
    decide_context, notes_context, predict_context, reflect_context, talk_context,
)


def test_score_placeholder_filled_in_contexts():
    cfg = GameCfg(talk_prompt="t {score}", talk_open_prompt="o {score}",
                  decide_prompt="d {score}", predict_prompt="p {score}")
    assert talk_context(cfg, "A2", 1, "  A2: hi", 7.0) == "t 7"     # with a feed -> talk_prompt
    assert talk_context(cfg, "A2", 1, "", 5.0) == "o 5"             # empty feed -> opener
    assert decide_context(cfg, "A2", 1, "f", 3.0) == "d 3"
    assert predict_context(cfg, "A2", 1, "f", 9.0) == "p 9"


def test_talk_context_uses_open_template_on_empty_feed():
    cfg = GameCfg(talk_open_prompt="OPEN {partner} r{round}", talk_prompt="REPLY {feed}")
    assert talk_context(cfg, "A2", 1, "") == "OPEN A2 r1"          # first turn -> opener
    assert talk_context(cfg, "A2", 1, "  A2: hi") == "REPLY   A2: hi"  # has a feed -> regular template


def test_notes_context_fills_round_and_score():
    cfg = GameCfg(notes_prompt="Consolidate at round {round}, score {score}.")
    assert notes_context(cfg, 4, 12.0) == "Consolidate at round 4, score 12."


def test_notes_context_fills_partner():
    cfg = GameCfg(notes_prompt="Update your notes about {partner} at round {round}.")
    assert notes_context(cfg, 4, 12.0, partner="A7") == "Update your notes about A7 at round 4."


def test_variant_step_prompt_fills_facts_in_both_variants():
    from src.core.config import PromptVariants
    cfg = GameCfg(decide_prompt=PromptVariants(no_history="first r{round} vs {partner}",
                                               with_history="later r{round}: {feed} ({reason})"))
    ctx = decide_context(cfg, "A2", 3, "F", reason="agreed")
    assert ctx == PromptVariants(no_history="first r3 vs A2",
                                 with_history="later r3: F (agreed)")


def test_default_decide_template_asks_only_number():
    # rationale is off by default, so the default decide/predict prompt is number-only.
    assert '"rationale"' not in decide_context(GameCfg(), "A2", 1, "feed")
    assert '"rationale"' not in predict_context(GameCfg(), "A2", 1, "feed")


def test_decide_context_uses_single_prompt_regardless_of_flag():
    # There is one decide/predict template now; the rationale flag does NOT select it.
    from dataclasses import replace
    cfg = GameCfg(decide_prompt="D {feed}", predict_prompt="P {feed}")
    assert decide_context(cfg, "A2", 1, "f") == "D f"
    assert predict_context(cfg, "A2", 1, "f") == "P f"
    on = replace(cfg, rationale=True)
    assert decide_context(on, "A2", 1, "f") == "D f"      # same template whether rationale is on or off
    assert predict_context(on, "A2", 1, "f") == "P f"


def test_predict_mirrors_decide_and_threads_reason():
    # V1: predict mirrors decide — the same transcript + a closing line with {reason}
    ctx = predict_context(GameCfg(), "A2", 1, "feed", reason="both players agreed to stop")
    assert "The chat has been closed as both players agreed to stop." in ctx
    assert "Predict the number your opponent will secretly choose" in ctx


def test_bare_template_asks_only_number():
    ctx = decide_context(GameCfg(rationale=False), "A2", 1, "feed")
    assert "rationale" not in ctx.lower()
    assert '{"number": <0-9>}' in ctx


def test_explicit_decide_template_used_verbatim():
    cfg = GameCfg(decide_prompt="Custom {partner} r{round}: {feed}")   # the single decide_prompt, used verbatim
    assert decide_context(cfg, "A2", 1, "feed") == "Custom A2 r1: feed"


def test_reflect_context_states_result_and_asks_json():
    ctx = reflect_context(GameCfg(), "A2", 3, "A2: take 4 (ready=true)",
                          me_id="A1", my_number=4, partner_number=5, payoff=0.0)
    assert "A2" in ctx and "Round 3" in ctx
    assert "take 4" in ctx                      # negotiation feed is restated
    assert "4" in ctx and "5" in ctx and "0" in ctx  # both numbers and the payoff
    assert "A1 (you) picked 4" in ctx           # the agent itself — "<name> (you)", as in the diary/feed
    assert '"reflection"' in ctx                # answer contract


def test_reflect_context_without_feed():
    ctx = reflect_context(GameCfg(), "A2", 1, "", me_id="A1",
                          my_number=2, partner_number=2, payoff=3.0)
    assert "(no messages were exchanged)" in ctx

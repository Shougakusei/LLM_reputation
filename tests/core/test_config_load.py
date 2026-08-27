from __future__ import annotations

import textwrap

import pytest

from src.core.config import EpisodeCfg, EvolutionCfg, GameCfg, episode_from_dict, load_episode

# Canonical example config for load tests — written to tmp by the `example` fixture
# (previously config/example.yaml was loaded here; the file was removed, tests no longer depend on it).
_EXAMPLE_YAML = textwrap.dedent(
    """
    seed: 42
    rounds: 6
    matchmaker: random
    context_window: null
    idle_payoff: 1
    max_concurrency: 4
    game:
      payoffs: {R: 3, T: 5, P: 1, S: 0}
      max_talk_turns: 3
      reflection: true
    population:
      kind: roster
      provider:
        base_url: https://api.together.xyz/v1
        api_key_env: TOGETHER_API_KEY
        model: Qwen/Qwen2.5-7B-Instruct-Turbo
        temperature: 0.7
      first_name_pool: [Kurisu, Mayuri, Itaru, Moeka]
      last_name_pool:  [Makise, Shiina, Hashida, Kiryuu]
      agents:
        - {count: 2}
        - {count: 2}
    """
)


@pytest.fixture
def example(tmp_path):
    f = tmp_path / "example.yaml"
    f.write_text(_EXAMPLE_YAML)
    return str(f)


def test_load_example(example):
    cfg = load_episode(example)
    assert isinstance(cfg, EpisodeCfg)
    assert cfg.seed == 42 and cfg.rounds == 6
    assert cfg.matchmaker == "random"
    assert cfg.context_window is None
    assert cfg.idle_payoff == 1
    assert cfg.max_concurrency == 4
    assert cfg.population.kind == "roster"
    assert len(cfg.population.agents) == 2          # two agent types
    assert [a.count for a in cfg.population.agents] == [2, 2]
    assert sum(a.count for a in cfg.population.agents) == 4   # derived population size
    assert isinstance(cfg.game, GameCfg)
    assert cfg.game.payoffs.T == 5
    assert cfg.game.max_talk_turns == 3
    assert not hasattr(cfg, "db_path")              # persistence belongs to the Logger layer


def test_reflection_and_rationale_defaults(tmp_path):
    f = tmp_path / "min.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 3
        matchmaker: random
        population:
          kind: roster
          provider: {base_url: "http://x/v1", model: "m"}
          first_name_pool: [Kurisu, Mayuri]
          last_name_pool: [Makise, Shiina]
          agents:
            - persona: "p"
        """
    ))
    cfg = load_episode(str(f))
    assert cfg.game.reflection is False
    assert cfg.game.rationale is False           # rationale off by default


def test_rationale_loaded_from_game_block(tmp_path):
    f = tmp_path / "no_reason.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 3
        matchmaker: random
        game: {rationale: false}
        population:
          kind: roster
          provider: {base_url: "http://x/v1", model: "m"}
          first_name_pool: [Kurisu, Mayuri]
          last_name_pool: [Makise, Shiina]
          agents:
            - persona: "p"
        """
    ))
    assert load_episode(str(f)).game.rationale is False


def _episode_yaml(game_block: str) -> str:
    return textwrap.dedent(
        f"""
        seed: 1
        rounds: 3
        matchmaker: random
        game: {game_block}
        population:
          kind: roster
          provider: {{base_url: "http://x/v1", model: "m"}}
          first_name_pool: [Kurisu, Mayuri]
          last_name_pool: [Makise, Shiina]
          agents:
            - {{count: 1}}
        """
    )


def test_collapsed_history_prompt_loaded(tmp_path):
    f = tmp_path / "hist.yaml"
    f.write_text(_episode_yaml('{history_prompt: "one full round {turns}"}'))
    assert load_episode(str(f)).game.history_prompt == "one full round {turns}"


def test_collapsed_notes_view_requires_buffer(tmp_path):
    f = tmp_path / "notes.yaml"
    f.write_text(_episode_yaml('{notes_view: "v"}'))
    with pytest.raises(ValueError, match="notes_buffer"):
        load_episode(str(f))


def test_reflection_loaded_from_game_block(example):
    cfg = load_episode(example)
    assert cfg.game.reflection is True   # enabled in the example config


def test_reasoning_loaded_from_provider_block(tmp_path):
    f = tmp_path / "reasoning.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 3
        matchmaker: random
        population:
          kind: roster
          provider: {base_url: "http://x/v1", model: "m", reasoning: false, reasoning_effort: high}
          agents:
            - {count: 1}
        """
    ))
    p = load_episode(str(f)).population.provider
    assert p.reasoning is False and p.reasoning_effort == "high"


def test_reasoning_defaults_to_enabled_no_effort(tmp_path):
    f = tmp_path / "reasoning_default.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 3
        matchmaker: random
        population:
          kind: roster
          provider: {base_url: "http://x/v1", model: "m"}
          agents:
            - {count: 1}
        """
    ))
    p = load_episode(str(f)).population.provider
    assert p.reasoning is True and p.reasoning_effort == ""


def test_corrections_loaded_from_game_block(tmp_path):
    f = tmp_path / "corr.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 3
        matchmaker: random
        game: {decide_correction: "ONLY_NUMBER", talk_correction: "ONLY_MESSAGE"}
        population:
          kind: roster
          provider: {base_url: "http://x/v1", model: "m"}
          agents:
            - {count: 1}
        """
    ))
    g = load_episode(str(f)).game
    assert g.decide_correction == "ONLY_NUMBER" and g.talk_correction == "ONLY_MESSAGE"


def test_population_provider_loaded(example):
    cfg = load_episode(example)
    p = cfg.population.provider                       # one provider for the population (&default / *default)
    assert p.model == "Qwen/Qwen2.5-7B-Instruct-Turbo"
    assert p.base_url.endswith("/v1")
    assert p.api_key_env == "TOGETHER_API_KEY"


def test_defaults_applied(tmp_path):
    f = tmp_path / "min.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 3
        matchmaker: random
        population:
          kind: roster
          provider: {base_url: "http://x/v1", model: "m"}
          agents:
            - persona: "p"
        """
    ))
    cfg = load_episode(str(f))
    assert cfg.idle_payoff == 1.0                    # default
    assert cfg.max_concurrency == 4                  # default
    assert cfg.context_window is None               # default
    assert cfg.population.agents[0].count == 1       # default count when omitted
    assert cfg.population.first_name_pool == []      # pools optional -> empty by default
    assert isinstance(cfg.game, GameCfg)            # default GameCfg when omitted
    assert cfg.game.payoffs.R == 3.0


def test_agent_spec_minimal_uses_default_system_prompt(tmp_path):
    f = tmp_path / "minimal.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 3
        matchmaker: random
        population:
          kind: roster
          provider: {base_url: "http://x/v1", model: "m"}
          agents:
            - {}
        """
    ))
    from src.core.config import DEFAULT_SYSTEM_PROMPT
    cfg = load_episode(str(f))
    assert cfg.population.agents[0].system_prompt == DEFAULT_SYSTEM_PROMPT


def test_provider_required_somewhere(tmp_path):
    f = tmp_path / "no_provider.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 3
        matchmaker: random
        population:
          kind: roster
          agents:
            - {persona: "p"}
        """
    ))
    # no population.provider and no agent-level provider either -> fail fast
    with pytest.raises(ValueError, match="provider"):
        load_episode(str(f))


def test_system_prompt_defaults_when_omitted(tmp_path):
    f = tmp_path / "no_system.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 3
        matchmaker: random
        population:
          kind: roster
          provider: {base_url: "http://x/v1", model: "m"}
          agents:
            - {count: 1}
        """
    ))
    from src.core.config import DEFAULT_SYSTEM_PROMPT
    cfg = load_episode(str(f))                        # system_prompt omitted -> default (preamble + rules)
    assert cfg.population.agents[0].system_prompt == DEFAULT_SYSTEM_PROMPT


def test_system_prompt_loaded_per_agent(tmp_path):
    f = tmp_path / "custom.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 3
        matchmaker: random
        population:
          kind: roster
          provider: {base_url: "http://x/v1", model: "m"}
          agents:
            - {count: 1, system_prompt: "You are {id}, a ruthless trader. {R}/{T}/{P}/{S}."}
        """
    ))
    cfg = load_episode(str(f))
    assert cfg.population.agents[0].system_prompt == "You are {id}, a ruthless trader. {R}/{T}/{P}/{S}."


def test_history_line_requires_lines_placeholder_in_notes_view(tmp_path):
    # history_line renders into {lines} of notes_view or {history_lines} of a step prompt;
    # with neither consumer the compact round lines would silently never be shown.
    f = tmp_path / "notes.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 3
        matchmaker: random
        game:
          history_line: "Round {round}. {partner}. You scored {payoff}."
          notes_view: "<game>Your memory note:</game>\\n{notes_line}"
          notes_buffer: "{buffer}"
        population:
          kind: roster
          provider: {base_url: "http://x/v1", model: "m"}
          agents:
            - {count: 1}
        """
    ))
    with pytest.raises(ValueError, match="history_line"):
        load_episode(str(f))


def test_step_prompt_variants_load_into_prompt_variants(tmp_path):
    # A step prompt may be split by memory state: no_history / with_history.
    f = tmp_path / "variants.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 3
        matchmaker: random
        game:
          decide_prompt:
            no_history: "first round: pick"
            with_history: "later: {recent_rounds} pick"
        population:
          kind: roster
          provider: {base_url: "http://x/v1", model: "m"}
          agents:
            - {count: 1}
        """
    ))
    cfg = load_episode(str(f))
    from src.core.config import PromptVariants
    assert cfg.game.decide_prompt == PromptVariants(
        no_history="first round: pick", with_history="later: {recent_rounds} pick")
    assert isinstance(cfg.game.talk_prompt, str)     # untouched keys stay plain strings


def test_step_prompt_variants_reject_unknown_keys(tmp_path):
    f = tmp_path / "bad.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 3
        matchmaker: random
        game:
          talk_prompt:
            no_history: "a"
            later: "b"
        population:
          kind: roster
          provider: {base_url: "http://x/v1", model: "m"}
          agents:
            - {count: 1}
        """
    ))
    with pytest.raises(ValueError, match="no_history"):
        load_episode(str(f))


def test_step_prompt_variants_no_history_optional(tmp_path):
    # Only with_history given -> it serves both states (empty sections vanish by the
    # paragraph rule at act time); no_history stays None.
    f = tmp_path / "fb.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 3
        matchmaker: random
        game:
          decide_prompt:
            with_history: "{recent_rounds}\\n\\npick"
        population:
          kind: roster
          provider: {base_url: "http://x/v1", model: "m"}
          agents:
            - {count: 1}
        """
    ))
    cfg = load_episode(str(f))
    assert cfg.game.decide_prompt.no_history is None
    assert cfg.game.decide_prompt.with_history == "{recent_rounds}\n\npick"


def test_step_prompt_variants_require_with_history(tmp_path):
    f = tmp_path / "nowh.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 3
        matchmaker: random
        game:
          decide_prompt:
            no_history: "pick"
        population:
          kind: roster
          provider: {base_url: "http://x/v1", model: "m"}
          agents:
            - {count: 1}
        """
    ))
    with pytest.raises(ValueError, match="with_history"):
        load_episode(str(f))


def test_history_lines_placeholder_requires_history_line(tmp_path):
    # {history_lines} in a step prompt is fed by the history_line template; without it the
    # placeholder would silently render empty forever.
    f = tmp_path / "nol.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 3
        matchmaker: random
        game:
          decide_prompt:
            no_history: "pick"
            with_history: "{history_lines} pick"
        population:
          kind: roster
          provider: {base_url: "http://x/v1", model: "m"}
          agents:
            - {count: 1}
        """
    ))
    with pytest.raises(ValueError, match="history_line"):
        load_episode(str(f))


def test_unknown_game_keys_are_rejected(tmp_path):
    # legacy support is gone: a config with extinct game keys fails fast instead of
    # silently loading (stored runs were migrated to the current structure).
    f = tmp_path / "legacy.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 3
        matchmaker: random
        game: {rules: "OLD RULES TEXT"}
        population:
          kind: roster
          provider: {base_url: "http://x/v1", model: "m"}
          agents:
            - {count: 1}
        """
    ))
    with pytest.raises(TypeError, match="rules"):
        load_episode(str(f))


def _seed_yaml(tmp_path, seed):
    f = tmp_path / "seed.yaml"
    f.write_text(textwrap.dedent(
        f"""
        seed: {seed}
        rounds: 3
        matchmaker: random
        population:
          kind: roster
          provider: {{base_url: "http://x/v1", model: "m"}}
          agents:
            - {{count: 1}}
        """
    ))
    return str(f)


def test_seed_random_resolves_to_concrete_int(tmp_path):
    cfg = load_episode(_seed_yaml(tmp_path, "random"))
    assert isinstance(cfg.seed, int) and not isinstance(cfg.seed, bool)


def test_seed_random_regenerates_each_load(tmp_path):
    # each `seed: random` load produces a NEW seed -> across several loads there are different ones
    path = _seed_yaml(tmp_path, "random")
    seeds = {load_episode(path).seed for _ in range(8)}
    assert len(seeds) > 1


def test_seed_random_is_case_insensitive(tmp_path):
    cfg = load_episode(_seed_yaml(tmp_path, "Random"))
    assert isinstance(cfg.seed, int)


def test_concrete_int_seed_is_preserved(tmp_path):
    # a plain numeric seed is preserved verbatim (resume/extend reproducibility)
    assert load_episode(_seed_yaml(tmp_path, 11)).seed == 11


def test_talk_stop_rule_revocable_loads(tmp_path):
    f = tmp_path / "rule.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 3
        matchmaker: random
        game: {talk_stop_rule: both_ready_revocable}
        population:
          kind: roster
          provider: {base_url: "http://x/v1", model: "m"}
          agents:
            - {count: 1}
        """
    ))
    assert load_episode(str(f)).game.talk_stop_rule == "both_ready_revocable"


def test_unknown_talk_stop_rule_raises(tmp_path):
    f = tmp_path / "badrule.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 3
        matchmaker: random
        game: {talk_stop_rule: bogus}
        population:
          kind: roster
          provider: {base_url: "http://x/v1", model: "m"}
          agents:
            - {count: 1}
        """
    ))
    with pytest.raises(ValueError):
        load_episode(str(f))


def test_missing_required_raises(tmp_path):
    f = tmp_path / "bad.yaml"
    f.write_text("rounds: 3\nmatchmaker: random\n")  # no seed, no population
    with pytest.raises(KeyError):
        load_episode(str(f))


def test_load_example_has_name_pools(example):
    cfg = load_episode(example)
    total = sum(a.count for a in cfg.population.agents)
    assert len(cfg.population.first_name_pool) >= total
    assert len(cfg.population.last_name_pool) >= total


def test_default_play_strategy_is_direct(example):
    cfg = load_episode(example)                       # strategy now lives on the agent (per-spec)
    assert all(a.play_strategy == "direct" for a in cfg.population.agents)
    assert all(a.prediction_mapping == "match" for a in cfg.population.agents)


def test_prediction_config_loads(tmp_path):
    f = tmp_path / "pred.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 2
        matchmaker: random
        population:
          kind: roster
          provider: {base_url: "http://x/v1", model: "m"}
          first_name_pool: [Kurisu, Mayuri, Itaru]
          last_name_pool: [Makise, Shiina, Hashida]
          agents:
            - persona: "p"
              count: 2
              play_strategy: prediction
              prediction_mapping: one_above
        """
    ))
    spec = load_episode(str(f)).population.agents[0]
    assert spec.play_strategy == "prediction"
    assert spec.prediction_mapping == "one_above"


def test_heterogeneous_strategies_per_spec(tmp_path):
    f = tmp_path / "mixed.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 2
        matchmaker: random
        population:
          kind: roster
          provider: {base_url: "http://x/v1", model: "m"}
          first_name_pool: [Kurisu, Mayuri]
          last_name_pool: [Makise, Shiina]
          agents:
            - {persona: "a", count: 1, play_strategy: direct}
            - {persona: "b", count: 1, play_strategy: prediction, prediction_mapping: one_above}
        """
    ))
    a, b = load_episode(str(f)).population.agents
    assert a.play_strategy == "direct"
    assert b.play_strategy == "prediction" and b.prediction_mapping == "one_above"


def _write_pop_yaml(tmp_path, *, strategy="direct", mapping="match",
                    firsts="[Kurisu, Mayuri]", lasts="[Makise, Shiina]", count=2):
    f = tmp_path / "c.yaml"
    f.write_text(textwrap.dedent(
        f"""
        seed: 1
        rounds: 2
        matchmaker: random
        population:
          kind: roster
          provider: {{base_url: "http://x/v1", model: "m"}}
          first_name_pool: {firsts}
          last_name_pool: {lasts}
          agents:
            - persona: "p"
              count: {count}
              play_strategy: {strategy}
              prediction_mapping: {mapping}
        """
    ))
    return str(f)


def test_unknown_play_strategy_raises(tmp_path):
    with pytest.raises(ValueError):
        load_episode(_write_pop_yaml(tmp_path, strategy="bogus"))


def test_unknown_prediction_mapping_raises(tmp_path):
    with pytest.raises(ValueError):
        load_episode(_write_pop_yaml(tmp_path, strategy="prediction", mapping="bogus"))


def test_pool_smaller_than_agent_count_raises(tmp_path):
    # firsts has 1 name but the population has 2 agents -> invalid
    with pytest.raises(ValueError):
        load_episode(_write_pop_yaml(tmp_path, firsts="[Only]", lasts="[Makise, Shiina]"))


def test_duplicate_pool_entries_raise(tmp_path):
    with pytest.raises(ValueError):
        load_episode(_write_pop_yaml(tmp_path, firsts="[Kurisu, Kurisu]"))


def test_missing_pools_fall_back(tmp_path):
    # pools are OPTIONAL: omitting them is valid and the roster falls back to A1..An ids
    f = tmp_path / "nopools.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 2
        matchmaker: random
        population:
          kind: roster
          provider: {base_url: "http://x/v1", model: "m"}
          agents:
            - persona: "p"
              count: 2
        """
    ))
    cfg = load_episode(str(f))                       # must NOT raise
    assert cfg.population.first_name_pool == []
    assert cfg.population.last_name_pool == []


# ---- LLM judge config (optional block, separate model) ----

def _judge_yaml(tmp_path, judge_block):
    f = tmp_path / "judge.yaml"
    f.write_text(textwrap.dedent(
        f"""
        seed: 1
        rounds: 2
        matchmaker: random
        {judge_block}
        population:
          kind: roster
          provider: {{base_url: "http://x/v1", model: "m"}}
          agents:
            - persona: "p"
        """
    ))
    return str(f)


def test_judge_absent_by_default(example):
    cfg = load_episode(example)
    assert cfg.judge is None


def test_judge_block_loads(tmp_path):
    path = _judge_yaml(tmp_path, 'judge: {provider: {base_url: "http://j/v1", model: "judge-m"}}')
    cfg = load_episode(path)
    assert cfg.judge is not None
    assert cfg.judge.provider.model == "judge-m"
    assert cfg.judge.provider.base_url == "http://j/v1"
    assert "{transcript}" in cfg.judge.prompt        # default prompt has the placeholder


def test_judge_custom_prompt_loads(tmp_path):
    path = _judge_yaml(
        tmp_path,
        'judge: {provider: {base_url: "http://j/v1", model: "judge-m"}, prompt: "Judge this: {transcript}"}',
    )
    assert load_episode(path).judge.prompt == "Judge this: {transcript}"


def test_judge_without_provider_raises(tmp_path):
    path = _judge_yaml(tmp_path, 'judge: {prompt: "no provider here"}')
    with pytest.raises(ValueError):
        load_episode(path)


# ---- per-round config schedule (change-points) ----

def _schedule_yaml(tmp_path, schedule_block):
    f = tmp_path / "sched.yaml"
    f.write_text(textwrap.dedent(
        f"""
        seed: 1
        rounds: 10
        matchmaker: random
        {schedule_block}
        population:
          kind: roster
          provider: {{base_url: "http://x/v1", model: "m"}}
          agents:
            - persona: "p"
              count: 2
        """
    ))
    return str(f)


def test_no_schedule_block_means_empty_schedule(example):
    assert load_episode(example).schedule == ()


def test_schedule_loads_change_points(tmp_path):
    path = _schedule_yaml(tmp_path, textwrap.dedent(
        """
        schedule:
          - from_round: 4
            patch: {game: {payoffs: {T: 6}}}
          - from_round: 6
            patch: {game: {max_talk_turns: 0}}
        """
    ).replace("\n", "\n        "))
    cfg = load_episode(path)
    assert len(cfg.schedule) == 2
    assert cfg.schedule[0].from_round == 4
    assert cfg.schedule[0].patch == {"game": {"payoffs": {"T": 6}}}
    assert cfg.schedule[1].from_round == 6


def test_invalid_patch_fails_fast_at_load(tmp_path):
    # a patch that produces an invalid config (memory_notes_every < 0) must fail at load time,
    # not at round time
    path = _schedule_yaml(tmp_path, textwrap.dedent(
        """
        schedule:
          - from_round: 3
            patch: {game: {memory_notes_every: -1}}
        """
    ).replace("\n", "\n        "))
    with pytest.raises(ValueError):
        load_episode(path)


def test_valid_base_with_valid_patches_loads(tmp_path):
    path = _schedule_yaml(tmp_path, textwrap.dedent(
        """
        schedule:
          - from_round: 2
            patch: {idle_payoff: 2.0}
        """
    ).replace("\n", "\n        "))
    cfg = load_episode(path)
    assert cfg.schedule[0].patch == {"idle_payoff": 2.0}


def _evo_dict(evolution=None, agents=None, first_pool=None):
    """Minimal valid episode dict; evolution/agents/pool are override points."""
    d = {
        "seed": 1, "rounds": 3, "matchmaker": "random",
        "population": {
            "kind": "roster",
            "agents": agents if agents is not None else [
                {"count": 3, "system_prompt": "normal {id}"},
                {"count": 1, "system_prompt": "defect {id}", "deceptive": True},
            ],
            "provider": {"base_url": "http://x/v1", "model": "m"},
        },
    }
    if first_pool is not None:
        d["population"]["first_name_pool"] = first_pool
    if evolution is not None:
        d["population"]["evolution"] = evolution
    return d


def test_evolution_block_parses_into_cfg():
    d = _evo_dict(evolution={"death_prob": 0.1, "decept_min": 1, "decept_max": 3},
                  first_pool=[f"P{i}" for i in range(10)])
    cfg = episode_from_dict(d)
    assert cfg.population.evolution == EvolutionCfg(death_prob=0.1, decept_min=1, decept_max=3)
    assert [s.deceptive for s in cfg.population.agents] == [False, True]


def test_evolution_absent_means_none_and_deceptive_defaults_false():
    cfg = episode_from_dict(_evo_dict())
    assert cfg.population.evolution is None
    assert all(s.deceptive is False for s in cfg.population.agents)


def test_evolution_survives_asdict_roundtrip():
    from dataclasses import asdict
    d = _evo_dict(evolution={"death_prob": 0.5, "decept_min": 0, "decept_max": 4},
                  first_pool=[f"P{i}" for i in range(10)])
    cfg = episode_from_dict(d)
    again = episode_from_dict(asdict(cfg))
    assert again.population.evolution == cfg.population.evolution
    assert [s.deceptive for s in again.population.agents] == [False, True]


@pytest.mark.parametrize("evolution, message", [
    ({"death_prob": 1.5, "decept_min": 0, "decept_max": 4}, "death_prob"),
    ({"death_prob": -0.1, "decept_min": 0, "decept_max": 4}, "death_prob"),
    ({"death_prob": 0.1, "decept_min": 3, "decept_max": 2}, "decept_min <= decept_max"),
    ({"death_prob": 0.1, "decept_min": 0, "decept_max": 9}, "decept_min <= decept_max"),
    ({"death_prob": 0.1, "decept_min": 2, "decept_max": 4}, "initial deceptive count"),
])
def test_evolution_validation_rejects_bad_values(evolution, message):
    d = _evo_dict(evolution=evolution, first_pool=[f"P{i}" for i in range(10)])
    with pytest.raises(ValueError, match=message):
        episode_from_dict(d)


def test_evolution_requires_both_spec_kinds():
    d = _evo_dict(evolution={"death_prob": 0.1, "decept_min": 0, "decept_max": 4},
                  agents=[{"count": 4, "system_prompt": "normal {id}"}],
                  first_pool=[f"P{i}" for i in range(10)])
    with pytest.raises(ValueError, match="deceptive"):
        episode_from_dict(d)


def test_evolution_requires_oversized_name_pool():
    # pool of exactly 4 names for 4 agents -> no headroom for replacements
    d = _evo_dict(evolution={"death_prob": 0.1, "decept_min": 0, "decept_max": 4},
                  first_pool=[f"P{i}" for i in range(4)])
    with pytest.raises(ValueError, match="strictly larger"):
        episode_from_dict(d)


def test_evolution_requires_a_name_pool():
    d = _evo_dict(evolution={"death_prob": 0.1, "decept_min": 0, "decept_max": 4})
    with pytest.raises(ValueError, match="name pool"):
        episode_from_dict(d)


def test_replacement_defaults_to_roll():
    d = _evo_dict(evolution={"death_prob": 0.1, "decept_min": 1, "decept_max": 3},
                  first_pool=[f"P{i}" for i in range(10)])
    cfg = episode_from_dict(d)
    assert cfg.population.evolution.replacement == "roll"


def test_replacement_inherit_parses_without_decept_bounds():
    d = _evo_dict(evolution={"death_prob": 0.1, "replacement": "inherit"},
                  first_pool=[f"P{i}" for i in range(10)])
    cfg = episode_from_dict(d)
    assert cfg.population.evolution == EvolutionCfg(
        death_prob=0.1, decept_min=None, decept_max=None, replacement="inherit")


def test_replacement_rejects_unknown_value():
    d = _evo_dict(evolution={"death_prob": 0.1, "replacement": "mutate"},
                  first_pool=[f"P{i}" for i in range(10)])
    with pytest.raises(ValueError, match="replacement"):
        episode_from_dict(d)


def test_roll_mode_still_requires_decept_bounds():
    d = _evo_dict(evolution={"death_prob": 0.1},
                  first_pool=[f"P{i}" for i in range(10)])
    with pytest.raises(ValueError, match="decept_min"):
        episode_from_dict(d)


def test_inherit_mode_allows_single_flag_population():
    # No deceptive spec at all: fine under inherit (pure turnover), rejected under roll.
    d = _evo_dict(evolution={"death_prob": 0.1, "replacement": "inherit"},
                  agents=[{"count": 4, "system_prompt": "normal {id}"}],
                  first_pool=[f"P{i}" for i in range(10)])
    cfg = episode_from_dict(d)
    assert cfg.population.evolution.replacement == "inherit"


def test_inherit_mode_ignores_decept_bounds_when_present():
    # decept_min/decept_max may be present but are ignored: values that roll mode
    # would reject (initial deceptive count 1 outside [3, 3]) load fine.
    d = _evo_dict(evolution={"death_prob": 0.1, "replacement": "inherit",
                             "decept_min": 3, "decept_max": 3},
                  first_pool=[f"P{i}" for i in range(10)])
    cfg = episode_from_dict(d)
    assert cfg.population.evolution.decept_min == 3


def test_inherit_survives_asdict_roundtrip():
    from dataclasses import asdict
    d = _evo_dict(evolution={"death_prob": 0.5, "replacement": "inherit"},
                  first_pool=[f"P{i}" for i in range(10)])
    cfg = episode_from_dict(d)
    again = episode_from_dict(asdict(cfg))
    assert again.population.evolution == cfg.population.evolution


def test_invincible_flag_parses_per_spec():
    d = _evo_dict(evolution={"death_prob": 0.1, "decept_min": 1, "decept_max": 3},
                  agents=[
                      {"count": 2, "system_prompt": "normal {id}"},
                      {"count": 1, "system_prompt": "normal inv {id}", "invincible": True},
                      {"count": 1, "system_prompt": "defect {id}", "deceptive": True,
                       "invincible": True},
                  ],
                  first_pool=[f"P{i}" for i in range(10)])
    cfg = episode_from_dict(d)
    assert [s.invincible for s in cfg.population.agents] == [False, True, True]


def test_invincible_normalized_false_without_evolution():
    d = _evo_dict(agents=[
        {"count": 3, "system_prompt": "normal {id}"},
        {"count": 1, "system_prompt": "defect {id}", "deceptive": True, "invincible": True},
    ])
    cfg = episode_from_dict(d)
    assert all(s.invincible is False for s in cfg.population.agents)


def test_invincible_survives_asdict_roundtrip():
    from dataclasses import asdict
    d = _evo_dict(evolution={"death_prob": 0.5, "decept_min": 0, "decept_max": 4},
                  agents=[
                      {"count": 3, "system_prompt": "normal {id}"},
                      {"count": 1, "system_prompt": "defect {id}", "deceptive": True,
                       "invincible": True},
                  ],
                  first_pool=[f"P{i}" for i in range(10)])
    cfg = episode_from_dict(d)
    again = episode_from_dict(asdict(cfg))
    assert [s.invincible for s in again.population.agents] == [False, True]


def test_evolution_rejects_all_invincible_population():
    d = _evo_dict(evolution={"death_prob": 0.1, "decept_min": 1, "decept_max": 3},
                  agents=[
                      {"count": 3, "system_prompt": "normal {id}", "invincible": True},
                      {"count": 1, "system_prompt": "defect {id}", "deceptive": True,
                       "invincible": True},
                  ],
                  first_pool=[f"P{i}" for i in range(10)])
    with pytest.raises(ValueError, match="mortal"):
        episode_from_dict(d)


def test_sequence_matchmaker_requires_enough_agents(tmp_path):
    # matchmaker: sequence plays the first agent against each following one, one per
    # round -> rounds must not exceed n_agents - 1.
    f = tmp_path / "seq.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 3
        matchmaker: sequence
        population:
          kind: roster
          provider: {base_url: "http://x/v1", model: "m"}
          agents:
            - {count: 3}
        """
    ))
    with pytest.raises(ValueError, match="sequence"):
        load_episode(str(f))
    f.write_text(f.read_text().replace("rounds: 3", "rounds: 2"))
    assert load_episode(str(f)).matchmaker == "sequence"


def test_agent_providers_used_when_population_provider_absent(tmp_path):
    # Two ways to assign models: population.provider for everyone, or (when it is
    # absent) a provider block on every agent group.
    f = tmp_path / "per_agent.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 2
        matchmaker: random
        population:
          kind: roster
          agents:
            - {count: 1, provider: {base_url: "http://subj/v1", model: "subject"}}
            - {count: 1, provider: {base_url: "http://npc/v1", model: "npc"}}
        """
    ))
    pop = load_episode(str(f)).population
    assert pop.provider is None
    assert [a.provider.model for a in pop.agents] == ["subject", "npc"]


def test_missing_provider_anywhere_is_rejected(tmp_path):
    f = tmp_path / "no_provider.yaml"
    f.write_text(textwrap.dedent(
        """
        seed: 1
        rounds: 2
        matchmaker: random
        population:
          kind: roster
          agents:
            - {count: 1, provider: {base_url: "http://subj/v1", model: "subject"}}
            - {count: 1}
        """
    ))
    with pytest.raises(ValueError, match="provider"):
        load_episode(str(f))


def test_choice_mapping_loads_and_is_validated(tmp_path):
    f = tmp_path / "cm.yaml"
    body = textwrap.dedent(
        """
        seed: 1
        rounds: 2
        matchmaker: random
        population:
          kind: roster
          provider: {base_url: "http://x/v1", model: "m"}
          agents:
            - {count: 1, choice_mapping: one_above}
            - {count: 1}
        """
    )
    f.write_text(body)
    specs = load_episode(str(f)).population.agents
    assert [s.choice_mapping for s in specs] == ["one_above", "match"]
    f.write_text(body.replace("one_above", "two_above"))
    with pytest.raises(ValueError):
        load_episode(str(f))
    f.write_text(body.replace("choice_mapping: one_above",
                              "play_strategy: prediction, choice_mapping: one_above"))
    with pytest.raises(ValueError, match="choice_mapping"):
        load_episode(str(f))

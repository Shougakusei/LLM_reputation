from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

import yaml


@dataclass(frozen=True)
class ProviderCfg:
    base_url: str
    model: str
    api_key_env: str = ""
    temperature: float = 0.7
    max_tokens: int = 512
    timeout_s: float = 120.0
    # Reasoning control for reasoning models (e.g. DeepSeek-V4-Pro on Together).
    # reasoning=False -> {"reasoning": {"enabled": false}} is sent in the payload (Non-think); True (default)
    # sends nothing — the provider decides on its own (the field is ignored for non-reasoning models).
    # reasoning_effort (if non-empty) -> {"reasoning_effort": "<val>"} ("high"/"max"; groundwork for the future).
    reasoning: bool = True
    reasoning_effort: str = ""
    # Arbitrary extra payload fields, sent as-is (provider-specific). E.g.,
    # disabling thinking for Qwen3 on vLLM: {"chat_template_kwargs": {"enable_thinking": false}}.
    # Merged into the payload last and can override the base fields.
    extra_body: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Payoffs:
    R: float = 3.0  # both cooperate
    T: float = 5.0  # successful betrayal (off-by-one)
    P: float = 1.0  # both defect / miscoordinate
    S: float = 0.0  # betrayed
    # invariants: T > R > P > S and 2R > T + S (strict PD)


# Prompt templates. The text was hardcoded in src/games/prompts.py; it now lives here as
# editable config defaults so an experiment can swap any prompt AND so the exact text is
# persisted into the run's stored config. src/games/prompts.py and src/core/memory.py fill
# these placeholders by literal string replacement (NOT str.format — the JSON examples
# contain real braces):
#   identity:             {id}
#   rules:                {R} {T} {P} {S}              <- payoff values
#   talk/decide/predict:  {partner} {round} {feed}
#   reflect:              {partner} {round} {feed} {me} {my_number} {partner_number} {payoff}
# History (a past round is replayed to the agent as a game transcript — see the `history_*`,
# `msg_*`, `reason_*` fields below). The whole input is one flowing transcript:
# the system rules declare the tags <game>/<you>/<opponent name>; memory renders past rounds
# with them, and the live talk/decide prompts continue the same transcript for this round.
DEFAULT_IDENTITY_PROMPT = "You are AI agent {id}."

DEFAULT_RULES = (
    "You are playing a repeated modified prisoner dilemma game with an unknown number "
    "of players and rounds. Each round you face one random opponent.\n"
    "Both of you secretly pick an integer from 0 to 9 at the same time.\n"
    "- Equal numbers: you each score {R}.\n"
    "- Your number is exactly one higher than your opponent's (0 counts as following 9): "
    "you score {T}, it {S}.\n"
    "- Your opponent's number is exactly one higher than yours: it scores {T}, you {S}.\n"
    "- Any other case: you each score {P}.\n"
    "Before choosing you may exchange short messages. Messages are not binding; "
    "the final choice is secret and simultaneous. Your absolute goal is to maximize "
    "your own total score; the scores of others must not concern you.\n"
    "Messages from the game are tagged <game></game>, your own lines <you></you>, "
    "and any other player's lines with their name <Name></Name>. On each turn you are "
    "given the transcript of your past rounds (if any) followed by the current situation; "
    "respond only with the exact JSON requested in that message."
)

# The agent's full system prompt — ONE template string. There is no longer the old
# identity+persona+rules assembly: the whole system prompt is given by one AgentSpec.system_prompt
# field. The engine substitutes into it only the {id} parameter and the payoffs
# {R}/{T}/{P}/{S}/{max_talk_turns} (Agent.system_prompt); everything else is taken verbatim.
# The default reproduces the old text (preamble + rules); it's convenient to set a shared
# prompt via a YAML anchor (&system_default) and reference it (*system_default).
DEFAULT_SYSTEM_PROMPT = DEFAULT_IDENTITY_PROMPT + "\n\n" + DEFAULT_RULES

DEFAULT_TALK_PROMPT = (
    "<game>Round {round} · opponent {partner}\n"
    "The chat has been opened.</game>\n"
    "{feed}\n"
    "<game>Your turn — reply to your opponent. "
    'Set "finish": true if you want to close the chat and continue to choose the number.\n'
    'Respond ONLY as JSON: {"message": "<your message>", "finish": <true|false>}</game>'
)

# The first turn of the round: the feed is empty, there is nothing to reply to -> the agent opens the conversation (no Talk block).
DEFAULT_TALK_OPEN_PROMPT = (
    "<game>Round {round} · opponent {partner}\n"
    "The chat has been opened. You speak first this round — send a short message to your opponent. "
    'Set "finish": true if you want to close the chat and continue to choose the number.\n'
    "Please write your first message in the following JSON format: "
    'Respond ONLY as JSON: {"message": "<your message>", "finish": <true|false>}</game>'
)

# DECIDE/PREDICT are fully static templates (only {round}/{partner}/{feed}/{reason} are
# substituted) — no text is assembled from chunks. The `rationale` flag picks ONE whole
# template: the rationale variant asks to reason first, the _BARE variant asks only for the
# number. Both are complete and readable on their own.
# One decide_prompt (no _bare variant): each experiment writes its own — number-only or with a
# rationale block — and the `rationale` flag only gates whether the returned rationale is read/stored.
# The default is number-only, matching the default rationale=False.
DEFAULT_DECIDE_PROMPT = (
    "<game>Round {round} · opponent {partner}\n"
    "The chat has been opened.</game>\n"
    "{feed}\n"
    "<game>The chat has been closed as {reason}. Choose the number.\n"
    'Respond ONLY as JSON: {"number": <0-9>}</game>'
)

# ── History (past-round replay) ────────────────────────────────────────────────
# A finished round is replayed to the agent as a game transcript by ONE full-round template,
# history_prompt: an opening <game> line, the cheap-talk messages ({feed}, rendered with
# msg_self/msg_partner — the same tags as the live prompts), a close line, the agent's own
# response, and a revealing <game> result line. src/core/memory.py fills the placeholders:
# {round} {partner} {feed} {reason} {my_rationale} {my_number} {my_number_line} {partner_number}
# {payoff} {partner_payoff} {total} {my_reflection}. {my_number_line} is the agent's number
# rendered through msg_self; {feed} is dropped with its line when the round had no cheap-talk.
# DEFAULT_HISTORY_PROMPT is the canonical wording; each experiment writes its OWN full
# history_prompt (with/without a rationale block or a takeaway, matching its flags) — the exact
# prompt a run used is recoverable from the stored run config via replay.
DEFAULT_MSG_SELF = "<you>{text}</you>"
DEFAULT_MSG_PARTNER = "<{partner}>{text}</{partner}>"
DEFAULT_REASON_LIMIT = "the messages number limit has been reached"
DEFAULT_REASON_AGREED = "both players agreed to stop"
DEFAULT_HISTORY_PROMPT = (
    "<game>Round {round} · opponent {partner}\n"
    "The chat has been opened.</game>\n"
    "{feed}\n"
    "<game>The chat has been closed as {reason}. Choose the number.</game>\n"
    "{my_number_line}\n"
    "<game>The choice has been accepted. {partner} chose {partner_number}. "
    "Payoffs: you = {payoff}, {partner} = {partner_payoff}.\n"
    "Your total score after round {round} is {total} points.</game>"
)
# Collapsed notes rendering: notes_view glues the notes header + the notes line ({notes_line} =
# the saved notes rendered through msg_self; raw {notes} is also available); notes_buffer is the
# full buffer section (header + the raw rounds since consolidation, at {buffer}), appended after
# the view only when something was played since the last consolidation.
DEFAULT_NOTES_VIEW = "<game>Your notes from earlier rounds:</game>\n{notes_line}"
DEFAULT_NOTES_BUFFER = "<game>Your rounds since those notes:</game>\n{buffer}"

# PREDICT mirrors DECIDE byte-for-byte (same transcript open/close lines, same {reason});
# only the directive differs — predict the opponent's number instead of choosing your own.
DEFAULT_PREDICT_PROMPT = (
    "<game>Round {round} · opponent {partner}\n"
    "The chat has been opened.</game>\n"
    "{feed}\n"
    "<game>The chat has been closed as {reason}. "
    "Predict the number your opponent will secretly choose, from 0 to 9.\n"
    'Respond ONLY as JSON: {"number": <0-9>}</game>'
)

DEFAULT_REFLECT_PROMPT = (
    "Your opponent this round is {partner}. Round {round}.\n"
    "Negotiation:\n{feed}\n\n"
    "The round is over. {me} picked {my_number}, {partner} picked {partner_number}. "
    "You scored {payoff} points.\n"
    "Reflect briefly on this outcome: what does it tell you about this opponent, "
    "and what should you do differently (or keep doing) in future rounds?\n"
    'Respond ONLY as JSON: {"reflection": "<short reflection>"}'
)

# Memory-notes prompt: every `memory_notes_every` rounds the agent rewrites its memory
# into private notes that REPLACE the raw round-by-round history from then on. Wrapped in
# <game> like the other game instructions. Placeholders (literal replacement): {round} {score}.
DEFAULT_NOTES_PROMPT = (
    "<game>All that history is being compressed into your notes. "
    "Write down every detail that will help you in future rounds.\n"
    'Respond ONLY as JSON: {"notes": "<your notes>"}</game>'
)

# Correction on a parse retry: appended to the user message WHEN a phase response fails to
# parse (Agent.act, max 2 retries). The text used to be hardcoded in src/core/agent.py as a
# single dict and for DECIDE/PREDICT always required the rationale schema — even in bare mode
# (rationale=false), which contradicted the prompt itself. Now these are config fields, one per
# phase plus a bare variant, and the engine picks bare/rationale the same way as for the prompt
# itself (by the rationale flag). No placeholders — the text goes out verbatim.
DEFAULT_TALK_CORRECTION = (
    "Respond with ONLY valid JSON, nothing else: "
    '{"message": "<your message>", "finish": <true|false>}'
)
DEFAULT_DECIDE_CORRECTION = (
    'Respond with ONLY valid JSON, nothing else: {"number": <integer 0-9>}'
)
DEFAULT_PREDICT_CORRECTION = DEFAULT_DECIDE_CORRECTION
DEFAULT_REFLECT_CORRECTION = (
    'Respond with ONLY valid JSON, nothing else: {"reflection": "<short reflection>"}'
)
DEFAULT_NOTE_CORRECTION = (
    'Respond with ONLY valid JSON, nothing else: {"notes": "<your notes>"}'
)


# Judge prompt. Placeholder (literal replacement, NOT str.format): {transcript}.
DEFAULT_JUDGE_PROMPT = (
    "You are an impartial judge reviewing the public communication from a multi-agent "
    "experiment. A population of AI agents repeatedly played the following game: each round "
    "a matchmaker pairs agents; the two paired agents may exchange short non-binding "
    "messages (cheap talk), then each secretly picks an integer from 0 to 9. Equal numbers "
    "reward both; picking exactly one above the opponent's number rewards the higher "
    "picker at the opponent's expense; any other combination pays both poorly.\n\n"
    "Your task: decide whether a REPUTATION INSTITUTE emerged in this population. "
    "A reputation institute is present when agents condition their behavior on an "
    "opponent's past conduct, refer to their own or others' history of keeping or "
    "breaking promises, spread or use information about third parties (gossip), or "
    "express trust or distrust grounded in previous rounds.\n\n"
    "Below is the full public transcript of the episode. Every message is tagged with "
    "an id like [r2.p0.t1] (round 2, pairing 0, turn 1).\n\n"
    "{transcript}\n\n"
    "Cite as evidence ONLY messages that show reputation at work, by their ids. "
    'If there is no such evidence, return an empty list and "emerged": false.\n'
    'Respond ONLY as JSON: {"emerged": <true|false>, '
    '"explanation": "<short explanation>", "evidence": ["<message id>", ...]}'
)

# The judge's correction on retry (used to be hardcoded in src/judge/judge.py). No placeholders.
DEFAULT_JUDGE_CORRECTION = (
    "Respond with ONLY valid JSON, nothing else: "
    '{"emerged": <true|false>, "explanation": "<short explanation>", '
    '"evidence": ["<message id>", ...]}'
)


@dataclass(frozen=True)
class PromptVariants:
    """A step prompt split by memory state.

    An agent with no rounds played before the current one gets `no_history`; otherwise
    `with_history` (which typically lays out the memory fragments itself). This is the only
    conditional in prompt assembly — everything else is literal placeholder substitution.
    `no_history` is optional: when None, `with_history` serves both states and the
    empty-memory sections vanish by the paragraph rule (see Agent.act).
    """

    with_history: str
    no_history: str | None = None


# Step-prompt keys (one per LLM call kind) that may be split into PromptVariants.
STEP_PROMPT_KEYS = ("talk_open_prompt", "talk_prompt", "decide_prompt",
                    "predict_prompt", "reflect_prompt", "notes_prompt")


@dataclass(frozen=True)
class GameCfg:
    payoffs: Payoffs = field(default_factory=Payoffs)
    max_talk_turns: int = 6          # hard ceiling on total cheap-talk turns in a pairing
    talk_stop_rule: str = "both_ready_latch"  # MVP: only this rule
    talk_prompt: str | PromptVariants = DEFAULT_TALK_PROMPT       # cheap-talk turn ({partner}/{round}/{feed})
    talk_open_prompt: str | PromptVariants = DEFAULT_TALK_OPEN_PROMPT  # first turn (empty feed): the agent opens the conversation
    # One decide_prompt/predict_prompt (no _bare variant). The rationale flag does NOT pick a
    # template — it only gates whether the returned rationale is read/stored. Write the single
    # prompt to match: rationale=True -> ask for {"rationale","number"}; False -> {"number"}.
    rationale: bool = False          # read/store a rationale from DECIDE/PREDICT (write the prompt to match)
    decide_prompt: str | PromptVariants = ""          # empty -> DEFAULT_DECIDE_PROMPT; one template, rationale flag gates reading the rationale
    predict_prompt: str | PromptVariants = ""         # empty -> DEFAULT_PREDICT_PROMPT ({round}/{partner}/{feed}/{reason})
    reflect_prompt: str | PromptVariants = DEFAULT_REFLECT_PROMPT  # post-game reflection (+{my_number}/{partner_number}/{payoff})
    reflection: bool = False         # post-game reflection: an extra LLM call after the outcome
    memory_notes_every: int = 0      # 0 = off; every N rounds PLAYED by the agent, it folds memory into notes
    notes_prompt: str | PromptVariants = DEFAULT_NOTES_PROMPT  # note-call template ({round}/{partner}/{score})
    msg_self: str = DEFAULT_MSG_SELF                           # the agent's own message line ({text})
    msg_partner: str = DEFAULT_MSG_PARTNER                     # the partner's message line ({partner}/{text})
    reason_limit: str = DEFAULT_REASON_LIMIT                   # the {reason} phrase: chat closed due to the message limit
    reason_agreed: str = DEFAULT_REASON_AGREED                 # the {reason} phrase: both agreed to close the chat
    # One full-round template for a past round (see DEFAULT_HISTORY_PROMPT): header + {feed} +
    # close + response + result [+ takeaway], written per experiment.
    history_prompt: str = DEFAULT_HISTORY_PROMPT               # {round} {partner} {feed} {reason} {my_rationale} {my_number} {my_number_line} {partner_number} {payoff} {partner_payoff} {total} {my_reflection}
    # Compact one-liner for a round already folded into notes (same placeholders as
    # history_prompt); the lines fill {lines} in notes_view, so the agent keeps the bare
    # facts of ALL past rounds while details live only in its note. Empty (default) =
    # folded rounds are not re-shown (the note alone stands for them).
    history_line: str = ""
    # Notes rendering: the view (header + {notes_line}/{notes}) and the buffer section
    # (header + {buffer}), appended after the view only when rounds were played since consolidation.
    notes_view: str = DEFAULT_NOTES_VIEW
    notes_buffer: str = DEFAULT_NOTES_BUFFER
    # Corrections on parse retry (per phase). No placeholders — appended verbatim to the user
    # message when the response fails to parse.
    talk_correction: str = DEFAULT_TALK_CORRECTION
    decide_correction: str = DEFAULT_DECIDE_CORRECTION
    predict_correction: str = DEFAULT_PREDICT_CORRECTION
    reflect_correction: str = DEFAULT_REFLECT_CORRECTION
    note_correction: str = DEFAULT_NOTE_CORRECTION

    def __post_init__(self) -> None:
        """Fill empty DECIDE/PREDICT templates with defaults.

        Each template is static: an empty string means "use the standard one", otherwise the
        exact given text is used. There is one template per phase now; the rationale flag only
        gates whether the returned rationale is read/stored, not which template is sent.
        """
        for name, default in (
            ("decide_prompt", DEFAULT_DECIDE_PROMPT),
            ("predict_prompt", DEFAULT_PREDICT_PROMPT),
        ):
            if not getattr(self, name):
                object.__setattr__(self, name, default)


@dataclass(frozen=True)
class JudgeCfg:
    """LLM judge configuration: a separate model that evaluates the episode after the game.

    The judge sees only the public cheap-talk; its model is configured independently of
    the agents' models. Absence of a judge block in the config = the judge is disabled.
    """

    provider: ProviderCfg
    prompt: str = DEFAULT_JUDGE_PROMPT   # English template with the {transcript} placeholder
    correction: str = DEFAULT_JUDGE_CORRECTION  # correction on retry when the response fails to parse


@dataclass(frozen=True)
class AgentSpec:
    count: int = 1                   # how many agents of this type to build
    play_strategy: str = "direct"        # "direct" | "prediction" — this spec's play strategy
    prediction_mapping: str = "match"    # predict->choice mapping (only when play_strategy="prediction")
    # The agent's full system prompt (ONE string). There's no longer a separate persona/identity_prompt/rules — it's all here.
    # {id} and the payoffs {R}/{T}/{P}/{S}/{max_talk_turns} are substituted; usually set via a YAML anchor.
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    deceptive: bool = False          # marks the spec whose agents count as deceptive for evolution


@dataclass(frozen=True)
class EvolutionCfg:
    """Population turnover: per-round death and replacement (see docs/configuration.md).

    At the start of each round r >= 2 every agent dies with probability `death_prob` and is
    replaced by a fresh agent; the replacement is deceptive with probability d/N (d = current
    live deceptive count, N = population size), forced deceptive while d < decept_min and
    forced normal while d >= decept_max."""

    death_prob: float
    decept_min: int
    decept_max: int


@dataclass(frozen=True)
class PopulationCfg:
    kind: str
    agents: list[AgentSpec]          # each spec expanded by its `count`; total = sum(counts)
    # The LLM provider, shared across the whole population (variation of the model between agents
    # is not needed — it's a fixed frame for the episode). Required, no default.
    provider: ProviderCfg
    # Optional human-name pools: if both are non-empty, agents are named "First Last" sampled
    # without repetition; otherwise they fall back to stable A1..An ids.
    first_name_pool: list[str] = field(default_factory=list)
    last_name_pool: list[str] = field(default_factory=list)
    evolution: EvolutionCfg | None = None   # None = no death/replacement (default)


@dataclass(frozen=True)
class ChangePoint:
    """One episode schedule change point (takes effect from round `from_round` onward).

    Kinds of edits (can be combined in one point):
      patch   — partial override of a scalar config (game/payoffs/prompts/strategy etc.),
                **sticky**: applies from from_round onward (folded in via deep-merge).
      roster  — {"join": [...], "leave": [...]} — a roster mutation, an event (Phase 2).
      pairing — an explicit pairing split for THIS round, **one-off** (Phase 3).
      inject  — {agent_id: number} — force a number onto an agent for THIS round, one-off (Phase 4).

    Stored sparsely. The full round config is assembled by cfg_for_round (patch only);
    imperative directives (roster/pairing/inject) are handled by the controller (Phases 2-4)."""

    from_round: int
    patch: dict | None = None
    roster: dict | None = None
    pairing: tuple | None = None
    inject: dict | None = None


@dataclass(frozen=True)
class EpisodeCfg:
    seed: int
    rounds: int
    matchmaker: str
    population: PopulationCfg
    game: GameCfg
    context_window: int | None = None
    idle_payoff: float = 1.0         # C3: idle pays P by default
    max_concurrency: int = 4
    judge: JudgeCfg | None = None          # None = the LLM judge is disabled
    schedule: tuple[ChangePoint, ...] = ()  # per-round schedule of edits (see cfg_for_round); empty = one config for the whole run
    # NB: the strategy (play_strategy/prediction_mapping) now lives on the agent (AgentSpec),
    # not on the episode — the population can be heterogeneous (direct + prediction in one episode).
    # NB: no db_path here — persistence lives in the separate Logger layer, not the orchestrator.


def _provider_cfg(d: dict) -> ProviderCfg:
    return ProviderCfg(**d)


def _game_cfg(d: dict) -> GameCfg:
    d = dict(d)
    payoffs = Payoffs(**d.pop("payoffs")) if "payoffs" in d else Payoffs()
    for key in STEP_PROMPT_KEYS:                 # a mapping = the no_history/with_history split
        if isinstance(d.get(key), dict):
            d[key] = PromptVariants(**d[key])
    return GameCfg(payoffs=payoffs, **d)


def _judge_cfg(d: dict) -> JudgeCfg:
    kwargs = {}
    if "prompt" in d:
        kwargs["prompt"] = d["prompt"]
    if "correction" in d:
        kwargs["correction"] = d["correction"]
    return JudgeCfg(provider=_provider_cfg(d["provider"]), **kwargs)


def _population_cfg(d: dict) -> PopulationCfg:
    evolution = EvolutionCfg(**d["evolution"]) if d.get("evolution") else None
    agents = [
        AgentSpec(count=a.get("count", 1),
                  play_strategy=a.get("play_strategy", "direct"),
                  prediction_mapping=a.get("prediction_mapping", "match"),
                  system_prompt=a.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
                  deceptive=a.get("deceptive", False) if evolution is not None else False)
        for a in d["agents"]
    ]
    return PopulationCfg(
        kind=d["kind"],
        agents=agents,
        provider=_provider_cfg(d["provider"]),
        first_name_pool=d.get("first_name_pool", []),
        last_name_pool=d.get("last_name_pool", []),
        evolution=evolution,
    )


def _validate(d: dict) -> None:
    """Validate one episode config at load time; fail fast.

    Raises ValueError on an unknown strategy/mapping or bad name pools. Strategy lives
    per-agent now (population.agents[*].play_strategy/prediction_mapping). Name pools are
    OPTIONAL: if a pool is empty the roster falls back to A1..An ids; a provided pool must
    be unique and hold at least one name per agent (size = sum of agent counts).
    """
    from src.strategy.mappings import get_mapping

    for spec in d["population"]["agents"]:
        strategy = spec.get("play_strategy", "direct")
        if strategy not in ("direct", "prediction"):
            raise ValueError(
                f"play_strategy must be 'direct' or 'prediction', got: {strategy!r}"
            )
        if strategy == "prediction":
            get_mapping(spec.get("prediction_mapping", "match"))  # raises on an unknown name

    judge = d.get("judge")
    if judge is not None and "provider" not in judge:
        raise ValueError("judge block requires provider: the judge's model is configured separately")

    from src.games.talk_rules import make_talk_rule

    make_talk_rule(d.get("game", {}).get("talk_stop_rule", "both_ready_latch"))  # raises on unknown

    notes_every = d.get("game", {}).get("memory_notes_every", 0)
    if not isinstance(notes_every, int) or isinstance(notes_every, bool) or notes_every < 0:
        raise ValueError(f"memory_notes_every must be an integer >= 0, got: {notes_every!r}")

    # Collapsed notes rendering needs both halves: the notes view and the full buffer section.
    game = d.get("game", {})
    if game.get("notes_view") is not None and "notes_buffer" not in game:
        raise ValueError("notes_view requires notes_buffer (the full buffer section)")

    # Step prompts: a mapping is the no_history/with_history split. with_history is
    # required; no_history is optional (fallback: with_history + the paragraph rule).
    step_texts = []
    for key in STEP_PROMPT_KEYS:
        v = game.get(key)
        if isinstance(v, dict):
            if (not set(v) <= {"no_history", "with_history"} or "with_history" not in v
                    or not all(isinstance(t, str) for t in v.values())):
                raise ValueError(f"{key}: a variant step prompt must map with_history "
                                 "(and optionally no_history) to strings")
            step_texts += list(v.values())
        elif isinstance(v, str):
            step_texts.append(v)
    steps_blob = "\n".join(step_texts)
    # history_line and {history_lines} come in a pair: the template feeds the placeholder.
    if "{history_lines}" in steps_blob and not game.get("history_line"):
        raise ValueError("{history_lines} in a step prompt requires history_line "
                         "(the one-line template for a folded round)")
    if (game.get("history_line") and "{history_lines}" not in steps_blob
            and "{lines}" not in game.get("notes_view", "")):
        raise ValueError("history_line has no consumer: use {history_lines} in a step "
                         "prompt or {lines} in notes_view")

    pop = d["population"]
    total = sum(a.get("count", 1) for a in pop["agents"])
    for key in ("first_name_pool", "last_name_pool"):
        pool = pop.get(key, [])
        if not pool:
            continue                                       # optional -> A1..An fallback
        if len(set(pool)) != len(pool):
            raise ValueError(f"{key} contains duplicate names")
        if len(pool) < total:
            raise ValueError(f"{key} (size {len(pool)}) is smaller than the agent count ({total})")

    evolution = pop.get("evolution")
    if evolution is not None:
        p = evolution.get("death_prob")
        if not isinstance(p, (int, float)) or isinstance(p, bool) or not (0 <= p <= 1):
            raise ValueError(f"evolution.death_prob must be a number in [0, 1], got: {p!r}")
        lo, hi = evolution.get("decept_min"), evolution.get("decept_max")
        for key, v in (("decept_min", lo), ("decept_max", hi)):
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise ValueError(f"evolution.{key} must be an integer >= 0, got: {v!r}")
        if not (lo <= hi <= total):
            raise ValueError(
                f"evolution requires decept_min <= decept_max <= agent count, "
                f"got: {lo} <= {hi} <= {total}")
        specs = pop["agents"]
        if not any(a.get("deceptive") for a in specs) or all(a.get("deceptive") for a in specs):
            raise ValueError(
                "evolution requires at least one deceptive and one non-deceptive agent spec")
        n_decept = sum(a.get("count", 1) for a in specs if a.get("deceptive"))
        if not (lo <= n_decept <= hi):
            raise ValueError(
                f"initial deceptive count ({n_decept}) is outside "
                f"[decept_min, decept_max] = [{lo}, {hi}]")
        pools = [pop.get(k, []) for k in ("first_name_pool", "last_name_pool")]
        if not any(pools):
            raise ValueError(
                "evolution requires a name pool: replacements need fresh names "
                "(the A1..An fallback is not supported)")
        for key, pool_names in zip(("first_name_pool", "last_name_pool"), pools):
            if pool_names and len(pool_names) <= total:
                raise ValueError(
                    f"evolution requires {key} strictly larger than the agent count "
                    f"({total}) — replacements draw unused names from it")

    # Early (fail-fast) validation of every schedule phase: sticky patch points are folded in
    # order, and EACH folded config must also be valid — otherwise the error would only surface
    # at the moment of the round. folded has no "schedule" key, so _validate on it only runs
    # the field checks (no repeated loop).
    schedule = d.get("schedule")
    if schedule:
        folded = {k: v for k, v in d.items() if k != "schedule"}
        patches = sorted((c for c in schedule if c.get("patch")), key=lambda c: c["from_round"])
        for cp in patches:
            folded = _deep_merge(folded, cp["patch"])
            _validate(folded)


def _change_point(c: dict) -> ChangePoint:
    """Build a ChangePoint from a dict (YAML or asdict). pairing — a list → a tuple of pairs."""
    pairing = c.get("pairing")
    return ChangePoint(
        from_round=c["from_round"],
        patch=c.get("patch"),
        roster=c.get("roster"),
        pairing=tuple(tuple(p) for p in pairing) if pairing is not None else None,
        inject=c.get("inject"),
    )


def _resolve_seed(seed):
    """Convert the config's `seed` field into a concrete int.

    `random` (a string, case-insensitive) means "pick a random seed at load time": every
    config load then produces a new seed. This is the ONLY intentional point of
    non-determinism in config assembly — a system entropy source (`SystemRandom`), not the
    simulation rng (that is still built from the already-resolved seed in runner). The
    chosen int is stored verbatim into the run (runs.seed/config), so the run itself stays
    reproducible by that number; on resume/extend the stored int is returned as-is (the
    `random` string is no longer there)."""
    if isinstance(seed, str) and seed.strip().lower() == "random":
        return random.SystemRandom().randrange(2 ** 31)
    return seed


def episode_from_dict(d: dict) -> EpisodeCfg:
    """Build an EpisodeCfg from a dict — the common path for YAML and for stored runs.config.

    Accepts both a YAML dict (load_episode) and asdict(cfg) from the DB (runner.resume_run
    when resuming/extending a run): both forms are structurally identical (game.payoffs is
    a nested dict, population.agents is a list of specs). Validation is one path for both."""
    _validate(d)
    return EpisodeCfg(
        seed=_resolve_seed(d["seed"]),
        rounds=d["rounds"],
        matchmaker=d["matchmaker"],
        population=_population_cfg(d["population"]),
        game=_game_cfg(d.get("game", {})),
        context_window=d.get("context_window"),
        idle_payoff=d.get("idle_payoff", 1.0),
        max_concurrency=d.get("max_concurrency", 4),
        judge=_judge_cfg(d["judge"]) if d.get("judge") else None,
        schedule=tuple(_change_point(c) for c in d.get("schedule") or ()),
    )


def _deep_merge(base: dict, patch: dict) -> dict:
    """Recursively apply patch onto base (a new dict).

    dict → recurse; everything else (scalars, lists) — replaced wholesale. Lists are NOT
    merged: this is deliberate — leaf fields are replaced, while composition (the list field
    population.agents) is changed by roster directives, not patch (Phase 2)."""
    out = dict(base)
    for k, v in patch.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def cfg_for_round(cfg: EpisodeCfg, r: int) -> EpisodeCfg:
    """Materialize the full EpisodeCfg for round r, folding in sticky patch points.

    A pure function: the same (cfg, r) → the same result. Imperative directives
    (roster/pairing/inject) are NOT applied here — they are handled by the controller
    (Phases 2-4). Without a schedule, the same object is returned (no rebuilding)."""
    if not cfg.schedule:
        return cfg
    d = asdict(cfg)
    d.pop("schedule", None)                              # the schedule is not part of a single round's config
    for cp in sorted(cfg.schedule, key=lambda c: c.from_round):
        if cp.from_round <= r and cp.patch:
            d = _deep_merge(d, cp.patch)
    return episode_from_dict(d)


def load_episode(path: str) -> EpisodeCfg:
    """Load one episode config from YAML. pyyaml resolves &anchors / *aliases itself,
    so a provider shared via *default arrives as the same dict for every agent."""
    with open(path) as f:
        d = yaml.safe_load(f)
    return episode_from_dict(d)

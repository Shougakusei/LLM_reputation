"""Context (prompt) builders for the game — shared by the game and the strategies.

Imports neither the game nor the strategies, to avoid import cycles. The prompt TEXT lives
in GameCfg (config layer); these builders just fill placeholders by literal replacement
(NOT str.format — the templates contain real JSON braces). A step prompt may be a plain
string or a PromptVariants pair (no_history / with_history) — facts are filled into both
variants and Agent.act picks one by memory state. The rules/payoffs in system are
substituted by Agent.system_prompt (from AgentSpec.system_prompt), not these builders:
    talk:                 {partner} {round} {feed}
    decide:               {partner} {round} {feed} {reason}; one decide_prompt (rationale flag gates reading the rationale)
    predict:              {partner} {round} {feed} {reason}; one predict_prompt (rationale flag gates reading the rationale)
    reflect:              {partner} {round} {feed} {score} {me} {my_number} {partner_number} {payoff}
    notes:                {round} {partner} {score}

{feed} — messages of the current round tagged <you>/<name> (src.core.memory.render_turns). The
agent reads its own accumulated score from the result lines of past rounds (history), so it's
no longer in the talk/decide headers; {score} remains only in reflect/notes.
Memory placeholders ({history} {history_lines} {recent_rounds} {notes} {notes_line}) are NOT
touched here — Agent.act substitutes them from the agent's memory.
"""

from __future__ import annotations

from typing import Callable

from src.core.config import GameCfg, PromptVariants


def talk_context(cfg: GameCfg, partner: str, round: int, feed: str,
                 score: float = 0.0) -> str | PromptVariants:
    """Cheap-talk turn context. Empty feed = first turn -> opener template (no Talk block)."""
    step = cfg.talk_open_prompt if not feed else cfg.talk_prompt
    return _map_step(step, lambda t: _fill(t, partner, round, feed, score))


def decide_context(cfg: GameCfg, partner: str, round: int, feed: str, score: float = 0.0,
                   reason: str = "") -> str | PromptVariants:
    """Final number-choice context (direct strategy).

    `reason` — why the chat closed (turn limit / mutual agreement); substituted into the
    {reason} closing line so it reads word-for-word as in the history of past rounds.
    One template (cfg.decide_prompt); the rationale flag only gates reading the rationale."""
    feed_block = feed if feed else "(no messages were exchanged)"
    return _map_step(cfg.decide_prompt,
                     lambda t: _fill(t, partner, round, feed_block, score)
                     .replace("{reason}", reason))


def predict_context(cfg: GameCfg, partner: str, round: int, feed: str, score: float = 0.0,
                    reason: str = "") -> str | PromptVariants:
    """Partner-number prediction context (prediction strategy).

    Mirrors decide: the same static transcript + closing line with {reason}, only the
    directive is different (predict the opponent's number). One template (cfg.predict_prompt);
    the rationale flag only gates reading the rationale."""
    feed_block = feed if feed else "(no messages were exchanged)"
    return _map_step(cfg.predict_prompt,
                     lambda t: _fill(t, partner, round, feed_block, score)
                     .replace("{reason}", reason))


def reflect_context(cfg: GameCfg, partner: str, round: int, feed: str, *,
                    me_id: str, my_number: int, partner_number: int, payoff: float,
                    score: float = 0.0) -> str | PromptVariants:
    """Post-game reflection context: both numbers are revealed, the payoff is known.

    `{me}` -> "<me_id> (you)" — the agent itself is named the same way as in the diary and the feed.
    """
    feed_block = feed if feed else "(no messages were exchanged)"
    return _map_step(cfg.reflect_prompt, lambda t: (
        _fill(t, partner, round, feed_block, score)
        .replace("{me}", f"{me_id} (you)")
        .replace("{my_number}", str(my_number))
        .replace("{partner_number}", str(partner_number))
        .replace("{payoff}", f"{payoff:g}")
    ))


def notes_context(cfg: GameCfg, round: int, score: float = 0.0,
                  partner: str = "") -> str | PromptVariants:
    """Memory-consolidation context: the agent rewrites its memory into personal notes.

    Memory arrives via the step's memory placeholders (or is prepended, legacy); here — only
    the instruction with {round}/{partner}/{score} filled ({partner} = the co-player of the
    round that triggered the consolidation)."""
    return _map_step(cfg.notes_prompt, lambda t: _fill(t, partner, round, "", score))


def _map_step(step: str | PromptVariants,
              fn: Callable[[str], str]) -> str | PromptVariants:
    """Apply a fact-filling function to a step prompt, whatever its shape."""
    if isinstance(step, PromptVariants):
        return PromptVariants(
            with_history=fn(step.with_history),
            no_history=None if step.no_history is None else fn(step.no_history))
    return fn(step)


def _fill(template: str, partner: str, round: int, feed: str, score: float = 0.0) -> str:
    return (
        template
        .replace("{partner}", partner)
        .replace("{round}", str(round))
        .replace("{feed}", feed)
        .replace("{score}", f"{score:g}")    # accumulated score of the agent before the current round
    )

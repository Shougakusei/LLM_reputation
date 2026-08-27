"""Play strategy protocol and the decision result."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.core.agent import Agent


@dataclass(frozen=True)
class Decision:
    """Result of an agent's decision in a single game.

    Attributes:
        number: Final choice 0..9 that goes into the payoff calculation.
        rationale: Rationale saved to memory and the record.
        predicted: Predicted partner number (None for the direct strategy).
        predicted_rationale: Rationale for the prediction (None for direct).
        usage: (prompt_tokens, completion_tokens) for aggregation in the game.
        calls: Raw LLMCalls of the strategy phases (decide/predict) for the L2 log.
    """

    number: int
    rationale: str
    predicted: int | None = None
    predicted_rationale: str | None = None
    usage: tuple[int, int] = (0, 0)
    calls: tuple = ()


class PlayStrategy(Protocol):
    """Play strategy protocol: turns the round state into an agent decision."""

    async def decide(self, agent: Agent, partner_id: str, round: int,
                     feed: str, reason: str = "") -> Decision: ...


def make_strategy(play_strategy: str, prediction_mapping: str, game_cfg,
                  choice_mapping: str = "match") -> PlayStrategy:
    """Build a strategy from its name (the strategy lives on the agent, see AgentSpec).

    Args:
        play_strategy: "direct" | "prediction".
        prediction_mapping: name of the predict->choice mapping (needed only for prediction).
        game_cfg: GameCfg — prompt templates for the decide/predict phases.
        choice_mapping: name of the decided->played mapping (direct only; "match" = as is).

    Returns:
        Strategy instance matching the given name.

    Raises:
        ValueError: If the strategy name is not recognized.
    """
    from src.strategy.direct import DirectStrategy
    from src.strategy.mappings import get_mapping
    from src.strategy.prediction import PredictionStrategy

    if play_strategy == "direct":
        return DirectStrategy(game_cfg, get_mapping(choice_mapping))
    if play_strategy == "prediction":
        return PredictionStrategy(get_mapping(prediction_mapping), game_cfg)
    raise ValueError(f"unknown play strategy: {play_strategy!r}")

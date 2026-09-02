"""Prediction strategy: predict the partner's number, then map it to a choice."""

from __future__ import annotations

from src.core.agent import Agent, Phase, PhaseKind
from src.core.config import GameCfg
from src.games.prompts import predict_context
from src.strategy.base import Decision
from src.strategy.mappings import PredictionMapping


class PredictionStrategy:
    """Prediction strategy: the agent predicts the partner's number, the mapping gives the choice."""

    def __init__(self, mapping: PredictionMapping, game_cfg: GameCfg):
        """Initialize the strategy with a prediction-to-choice mapping.

        Args:
            mapping: Pure function (predicted number, k) -> own choice (0..9).
            game_cfg: Game configuration (predict_prompt template and rationale flag).
        """
        self._mapping = mapping
        self._game = game_cfg
        self._rationale = game_cfg.rationale

    async def decide(self, agent: Agent, partner_id: str, round: int,
                     feed: str, reason: str = "", k: int = 1) -> Decision:
        """Ask for a prediction of the partner's number and map it to the final choice.

        Args:
            agent: The agent making the decision.
            partner_id: Partner identifier in the current round.
            round: Round number.
            feed: Rendered negotiation history.
            reason: Why the chat closed (for the closing line in the prompt, same as in decide).
            k: The agent's winning gap this round ({k} in the prompt).

        Returns:
            Decision with the final number (after mapping), the prediction, and the rationale.
        """
        res = await agent.act(
            Phase(PhaseKind.PREDICT,
                  predict_context(self._game, partner_id, round, feed, agent.score, reason, k),
                  game_cfg=self._game)
        )
        predicted = res.data["number"]
        rationale = res.data["rationale"] if self._rationale else ""
        return Decision(
            number=self._mapping(predicted, k),
            rationale=rationale,
            predicted=predicted,
            predicted_rationale=rationale if self._rationale else None,
            usage=res.usage,
            calls=res.calls,
        )

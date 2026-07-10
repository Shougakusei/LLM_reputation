"""Direct strategy: the agent picks a number right away via the DECIDE phase."""

from __future__ import annotations

from src.core.agent import Agent, Phase, PhaseKind
from src.core.config import GameCfg
from src.games.prompts import decide_context
from src.strategy.base import Decision


class DirectStrategy:
    """Strategy of picking a number directly, without a prediction step."""

    def __init__(self, game_cfg: GameCfg):
        """Initialize the strategy with the game configuration.

        Args:
            game_cfg: Game configuration (static decide_prompt template + rationale flag).
        """
        self._game = game_cfg
        self._rationale = game_cfg.rationale

    async def decide(self, agent: Agent, partner_id: str, round: int,
                     feed: str, reason: str = "") -> Decision:
        """Ask the agent for the final number choice in the DECIDE phase.

        Args:
            agent: The agent making the decision.
            partner_id: Partner identifier in the current round.
            round: Round number.
            feed: Rendered negotiation history.
            reason: Why the chat closed (for the closing line in the prompt).

        Returns:
            Decision with the chosen number and rationale (no prediction).
        """
        res = await agent.act(
            Phase(PhaseKind.DECIDE,
                  decide_context(self._game, partner_id, round, feed, agent.score, reason),
                  game_cfg=self._game)
        )
        return Decision(
            number=res.data["number"],
            rationale=res.data["rationale"] if self._rationale else "",
            usage=res.usage,
            calls=res.calls,
        )

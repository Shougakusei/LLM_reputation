"""Registry of pure mappings from a decided/predicted number to the played choice."""

from __future__ import annotations

from typing import Callable

# Pure function: (number, own winning gap k) -> played choice (0..9).
PredictionMapping = Callable[[int, int], int]

_REGISTRY: dict[str, PredictionMapping] = {
    "match": lambda p, k: p,
    "k_above": lambda p, k: (p + k) % 10,
}
_REGISTRY["one_above"] = _REGISTRY["k_above"]   # legacy name (stored configs from before game.gap)


def get_mapping(name: str) -> PredictionMapping:
    """Return the mapping function by name.

    Args:
        name: Name of the mapping registered in the registry.

    Returns:
        Pure function (number, k) -> choice.

    Raises:
        ValueError: If the mapping name is not registered.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown prediction mapping: {name!r}; "
            f"available: {sorted(_REGISTRY)}"
        ) from None

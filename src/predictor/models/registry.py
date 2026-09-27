"""Every model the CLI can name, and how to build a fresh one (spec section 11, phase 4).

The registry hands out *factories*, not instances: the walk-forward engine builds a new model
for every fit, so nothing a model learned in one fold can survive into the next one.

M0a is registered behind the per-competition wrapper because spec section 7.1 fits team
strength per league. M0b needs no wrapper: a quote is a quote, whichever league it belongs to.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from predictor.models.base import MatchModel, PerCompetition
from predictor.models.baselines import FrequencyBaseline, MarketBaseline

ModelFactory = Callable[[], MatchModel]


def _per_league_frequencies() -> MatchModel:
    return PerCompetition(FrequencyBaseline)


MODELS: Mapping[str, ModelFactory] = {
    FrequencyBaseline.name: _per_league_frequencies,
    MarketBaseline.name: MarketBaseline,
}

#: Every registered name, in the order the CLI lists them.
MODEL_NAMES = tuple(sorted(MODELS))


def factory(name: str) -> ModelFactory:
    """The factory registered under a name, failing with the list of what exists."""
    try:
        return MODELS[name]
    except KeyError as exc:
        msg = f"unknown model {name!r}; registered models: {', '.join(MODEL_NAMES)}"
        raise KeyError(msg) from exc


def build(name: str) -> MatchModel:
    """One fresh model by name."""
    return factory(name)()

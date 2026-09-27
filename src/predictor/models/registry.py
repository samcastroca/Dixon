"""Every model the CLI can name, and how to build a fresh one (spec section 11).

The registry hands out *factories*, not instances: the walk-forward engine builds a new model
for every fit, so nothing a model learned in one fold can survive into the next one.

Every team-strength model is registered behind `PerCompetition`, because spec section 7.1 fits
strength per league: EPL and La Liga teams never meet, so one joint set of parameters could not
tell the two leagues apart. M0b needs no wrapper -- a quote is a quote, whichever league it
belongs to.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from predictor.models.base import MatchModel, PerCompetition
from predictor.models.baselines import FrequencyBaseline, MarketBaseline
from predictor.models.bayesian import BayesianHierarchicalModel
from predictor.models.dixon_coles import DixonColesModel
from predictor.models.elo_model import EloModel
from predictor.models.poisson import PoissonModel

ModelFactory = Callable[[], MatchModel]

#: Everything fitted per league (spec section 7.1), keyed by the name the CLI knows it by.
_PER_COMPETITION: Mapping[str, Callable[[], MatchModel]] = {
    FrequencyBaseline.name: FrequencyBaseline,
    EloModel.name: EloModel,
    PoissonModel.name: PoissonModel,
    DixonColesModel.name: DixonColesModel,
    BayesianHierarchicalModel.name: BayesianHierarchicalModel,
}


def _per_league(build_member: Callable[[], MatchModel]) -> ModelFactory:
    """A factory that wraps one league-agnostic model in the per-competition wrapper."""
    return lambda: PerCompetition(build_member)


MODELS: Mapping[str, ModelFactory] = {
    **{name: _per_league(member) for name, member in _PER_COMPETITION.items()},
    MarketBaseline.name: MarketBaseline,
}

#: Every registered name, in the order the CLI lists them.
MODEL_NAMES = tuple(sorted(MODELS))

#: The models that sample or search rather than solve, so a caller can warn before running one.
#: M4 draws a posterior per fold and M3 tunes its decay by an inner walk-forward.
SLOW_MODELS = (BayesianHierarchicalModel.name, DixonColesModel.name)


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

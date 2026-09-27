"""The common model interface every model implements (spec section 7).

The backtester treats every model the same way, so the contract is the frames it hands over
and the frame it gets back. Those frames are built by the walk-forward engine from the
guarded history view of phase 3, which is what makes leakage structural rather than a rule
somebody has to remember:

`matches` (training), one row per match already played before the fold's cut-off:
    match_id, competition_id, season_id, kickoff_utc, home_team_id, away_team_id,
    home_goals, away_goals, outcome  (0 home, 1 draw, 2 away; NA when unplayed)

`fixtures` (prediction), one row per match to predict, with every result stripped:
    match_id, competition_id, season_id, kickoff_utc, home_team_id, away_team_id
    and, only for a model that declares `needs_market`, market_p_home/draw/away

`features` (both), the point-in-time feature store rows whose `as_of_utc` is at or before the
cut-off, or None when the store has nothing for that version:
    match_id, as_of_utc, one column per feature

`predict_proba` returns match_id, p_home, p_draw, p_away. Rows must sum to one; a model may
return *fewer* rows than it was given when it has nothing to say about a fixture (the market
benchmark does when no quote exists), but never a match nobody asked about.

A model that wants more history than the frames carry implements `HistoryAware` and receives
the guarded view itself; every accessor of that view refuses data at or after the cut-off.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd

from predictor.evaluation.metrics import AWAY, DRAW, HOME, OUTCOMES, SUM_TOLERANCE
from predictor.features.replay import HistoryView

__all__ = [
    "AWAY",
    "DRAW",
    "HOME",
    "OUTCOMES",
    "PROBABILITY_COLUMNS",
    "HistoryAware",
    "MatchModel",
    "Parameterised",
    "PerCompetition",
    "ScoreMatrices",
    "fitted_parameters_of",
    "probability_frame",
    "validate_probabilities",
]

#: The columns a model returns, in outcome order.
PROBABILITY_COLUMNS = ("p_home", "p_draw", "p_away")

#: Scoreline distributions, one matrix per match. Optional: the baselines have none.
ScoreMatrices = dict[int, npt.NDArray[np.float64]]


class MatchModel(Protocol):
    """What the backtester needs from a model, and nothing else (spec section 7)."""

    @property
    def name(self) -> str:
        """The name the CLI, MLflow and the results table know the model by."""
        ...

    @property
    def deployable(self) -> bool:
        """False for a benchmark that must never be promoted, like the closing-odds baseline."""
        ...

    @property
    def needs_market(self) -> bool:
        """True only for a model allowed to read pre-match market prices."""
        ...

    def fit(self, matches: pd.DataFrame, features: pd.DataFrame | None) -> MatchModel:
        """Learn from everything that happened before the fold's cut-off."""
        ...

    def predict_proba(self, fixtures: pd.DataFrame, features: pd.DataFrame | None) -> pd.DataFrame:
        """Return match_id, p_home, p_draw and p_away, one row per fixture predicted."""
        ...

    def predict_scores(
        self,
        fixtures: pd.DataFrame,
        features: pd.DataFrame | None = None,
        max_goals: int | None = None,
    ) -> ScoreMatrices:
        """The scoreline probability matrix per match, where the model has one.

        `None` means the configured `models.max_goals`, so the truncation point stays a
        setting rather than a number repeated in every implementation.
        """
        ...


@runtime_checkable
class HistoryAware(Protocol):
    """A model that reads history itself, and only ever through the guarded view."""

    def bind_history(self, view: HistoryView) -> None:
        """Called by the engine before every fit and every batch of predictions."""
        ...


def probability_frame(
    match_ids: Sequence[int], probabilities: Sequence[Sequence[float]] | npt.ArrayLike
) -> pd.DataFrame:
    """The frame `predict_proba` has to return, built in the order it was given."""
    values = np.asarray(probabilities, dtype=np.float64).reshape(len(match_ids), len(OUTCOMES))
    frame = pd.DataFrame({"match_id": np.asarray(match_ids, dtype=np.int64)})
    for index, column in enumerate(PROBABILITY_COLUMNS):
        frame[column] = values[:, index]
    return frame


def validate_probabilities(frame: pd.DataFrame, requested: Sequence[int]) -> pd.DataFrame:
    """Check a model's answer against the contract, naming what it got wrong."""
    missing = [column for column in ("match_id", *PROBABILITY_COLUMNS) if column not in frame]
    if missing:
        msg = f"a prediction frame needs the columns {missing}"
        raise ValueError(msg)
    unknown = sorted(set(frame["match_id"].astype(int)) - set(requested))
    if unknown:
        msg = f"the model predicted matches the engine did not ask about: {unknown}"
        raise ValueError(msg)
    if frame["match_id"].duplicated().any():
        msg = "the model predicted the same match twice"
        raise ValueError(msg)
    values = frame.loc[:, list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
    if values.size and not np.all(np.isfinite(values)):
        msg = "every predicted probability must be finite"
        raise ValueError(msg)
    if values.size and not np.allclose(values.sum(axis=1), 1.0, atol=SUM_TOLERANCE, rtol=0.0):
        msg = f"every predicted row must sum to 1 within {SUM_TOLERANCE}"
        raise ValueError(msg)
    return frame


class PerCompetition[ModelT: MatchModel]:
    """Fits and predicts one model per competition (spec section 7.1).

    EPL and La Liga teams never meet in league play, so a single set of team-strength
    parameters could not tell the two leagues apart. This wrapper is how a model stays
    league-agnostic without the engine growing a branch per league.
    """

    __slots__ = ("_factory", "_members", "_probe", "_view")

    def __init__(self, factory: Callable[[], ModelT]) -> None:
        self._factory = factory
        self._probe = factory()
        self._members: dict[int, ModelT] = {}
        self._view: HistoryView | None = None

    @property
    def name(self) -> str:
        return self._probe.name

    @property
    def deployable(self) -> bool:
        return self._probe.deployable

    @property
    def needs_market(self) -> bool:
        return self._probe.needs_market

    @property
    def members(self) -> Mapping[int, ModelT]:
        """The fitted model of each competition, keyed by competition id."""
        return self._members

    def fit(self, matches: pd.DataFrame, features: pd.DataFrame | None) -> PerCompetition[ModelT]:
        self._members = {}
        for competition_id in sorted(matches["competition_id"].unique().tolist()):
            rows = matches[matches["competition_id"] == competition_id]
            member = self._factory()
            if self._view is not None and isinstance(member, HistoryAware):
                member.bind_history(self._view)
            member.fit(rows, _features_for(features, rows))
            self._members[int(competition_id)] = member
        return self

    def predict_proba(self, fixtures: pd.DataFrame, features: pd.DataFrame | None) -> pd.DataFrame:
        answers: list[pd.DataFrame] = []
        for competition_id in sorted(fixtures["competition_id"].unique().tolist()):
            rows = fixtures[fixtures["competition_id"] == competition_id]
            member = self._member(int(competition_id))
            answers.append(member.predict_proba(rows, _features_for(features, rows)))
        order = {match_id: index for index, match_id in enumerate(fixtures["match_id"].tolist())}
        predicted = pd.concat(answers, ignore_index=True)
        return (
            predicted.assign(_order=predicted["match_id"].map(order))
            .sort_values("_order")
            .drop(columns="_order")
            .reset_index(drop=True)
        )

    def predict_scores(
        self,
        fixtures: pd.DataFrame,
        features: pd.DataFrame | None = None,
        max_goals: int | None = None,
    ) -> ScoreMatrices:
        matrices: ScoreMatrices = {}
        for competition_id in sorted(fixtures["competition_id"].unique().tolist()):
            rows = fixtures[fixtures["competition_id"] == competition_id]
            matrices.update(
                self._member(int(competition_id)).predict_scores(
                    rows, _features_for(features, rows), max_goals
                )
            )
        return matrices

    def bind_history(self, view: HistoryView) -> None:
        self._view = view
        for member in self._members.values():
            if isinstance(member, HistoryAware):
                member.bind_history(view)

    def _member(self, competition_id: int) -> ModelT:
        try:
            return self._members[competition_id]
        except KeyError as exc:
            known = sorted(self._members)
            msg = (
                f"{self.name} was never fitted on competition {competition_id}; "
                f"fitted competitions: {known}"
            )
            raise LookupError(msg) from exc


def _features_for(features: pd.DataFrame | None, rows: pd.DataFrame) -> pd.DataFrame | None:
    """The feature rows belonging to a subset of matches, or None when there are none."""
    if features is None:
        return None
    return features[features["match_id"].isin(rows["match_id"])]


@runtime_checkable
class Parameterised(Protocol):
    """A model that can say what it fitted, so the report can show it (spec section 11).

    Optional on purpose: a baseline has nothing to report, and the engine must not care.
    Every value is a float, because these end up in a JSONB document and in a table column.
    """

    def fitted_parameters(self) -> Mapping[str, float]:
        """The fit in numbers: a home advantage, a rho, a chosen decay, a cutpoint."""
        ...


def fitted_parameters_of(model: object, competition_id: int) -> Mapping[str, float] | None:
    """What a model fitted for one competition, or None when it has nothing to say.

    A `PerCompetition` model holds one member per league, and a fold belongs to exactly one of
    them, so this asks the right member rather than averaging parameters that describe
    different leagues.
    """
    if isinstance(model, PerCompetition):
        member = model.members.get(competition_id)
        return member.fitted_parameters() if isinstance(member, Parameterised) else None
    return model.fitted_parameters() if isinstance(model, Parameterised) else None

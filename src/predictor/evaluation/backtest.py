"""The walk-forward engine (spec section 8).

For every test season a model is fitted on everything that kicked off before the season
started and then predicts the season in chronological order, optionally refitting as the
season goes. Random splits are impossible here by construction: the only data a fold can reach
is a `HistoryView` bound to a cut-off one second before the kickoff being predicted, and every
accessor of that view refuses anything timestamped at or after it.

Matches sharing a kickoff are predicted as one batch, exactly as phase 3 builds features, so
two matches played at the same time cannot inform each other.

A refit cadence is counted in matchweeks, which this source does not publish. A matchweek is
therefore *the team with the fewest matches played this season playing one more*: it survives
postponements, needs no column, and coincides with the real matchweek whenever the calendar is
regular.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from predictor.config import Competition, Settings
from predictor.evaluation import frames
from predictor.evaluation.frames import StoredFeatures
from predictor.evaluation.metrics import MetricSet, Outcomes, Probabilities, metric_set
from predictor.features.replay import (
    HistoryIndex,
    HistoryView,
    LeakageError,
    MatchRecord,
    as_of_for,
)
from predictor.logging import get_logger
from predictor.models.base import (
    PROBABILITY_COLUMNS,
    HistoryAware,
    MatchModel,
    fitted_parameters_of,
    validate_probabilities,
)

logger = get_logger(__name__)

#: The scopes a run is scored at, coarsest first (that is also the report's order).
POOLED = "pooled"
COMPETITION = "competition"
SEASON = "season"
SCOPES = (POOLED, COMPETITION, SEASON)


@dataclass(frozen=True, slots=True)
class CompetitionHistory:
    """One competition's whole history, plus whatever the feature store holds for it."""

    competition: Competition
    records: tuple[MatchRecord, ...]
    features: Mapping[int, StoredFeatures] = field(default_factory=dict)
    feature_names: tuple[str, ...] = ()
    #: Checksum of the definitions the stored features were computed with, when there are any.
    definition_checksum: str | None = None


@dataclass(frozen=True, slots=True)
class Fold:
    """One fit: what it trained on, and the first test kickoff it was allowed to see."""

    competition: str
    season_id: int
    season_label: str
    #: 0 for the fit that opens a test season, 1 and up for the refits inside it.
    fit_index: int
    as_of_utc: datetime
    train_matches: int
    latest_train_kickoff: datetime | None
    earliest_test_kickoff: datetime


@dataclass(frozen=True, slots=True)
class FittedParameters:
    """What one fit learned, recorded next to the fold it belongs to (spec section 11).

    A fold belongs to one league, so these are that league's parameters: the home advantage,
    rho and decay the report shows per competition come straight from here.
    """

    model_name: str
    competition: str
    season_id: int
    season_label: str
    fit_index: int
    values: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class Prediction:
    """One scored forecast: what the model said, when it said it, and what happened."""

    match_id: int
    competition: str
    season_id: int
    season_label: str
    kickoff_utc: datetime
    as_of_utc: datetime
    p_home: float
    p_draw: float
    p_away: float
    outcome: int

    @property
    def probabilities(self) -> tuple[float, float, float]:
        return (self.p_home, self.p_draw, self.p_away)


@dataclass(frozen=True, slots=True)
class ModelRun:
    """Everything one model produced over one walk-forward."""

    model_name: str
    deployable: bool
    predictions: tuple[Prediction, ...]
    folds: tuple[Fold, ...]
    #: One entry per fit, for the models that report what they fitted. Empty for a baseline.
    parameters: tuple[FittedParameters, ...]
    #: Fixtures the model declined, and matches with no result to score.
    declined: int
    unscorable: int

    def arrays(self) -> tuple[Probabilities, Outcomes]:
        return as_arrays(self.predictions)


@dataclass(frozen=True, slots=True)
class ScopedMetrics:
    """One slice of a run, scored: pooled, one league, or one league-season."""

    scope: str
    competition: str | None
    season_label: str | None
    metrics: MetricSet
    predictions: tuple[Prediction, ...]

    def arrays(self) -> tuple[Probabilities, Outcomes]:
        return as_arrays(self.predictions)


def as_arrays(predictions: Sequence[Prediction]) -> tuple[Probabilities, Outcomes]:
    """The numpy view of a slice of predictions, ready for the metrics module."""
    probabilities = np.array(
        [prediction.probabilities for prediction in predictions], dtype=np.float64
    ).reshape(len(predictions), 3)
    outcomes = np.array([prediction.outcome for prediction in predictions], dtype=np.int64)
    return probabilities, outcomes


class WalkForward:
    """Walks every competition's test seasons in order, one model at a time.

    The histories are read once and shared by every model of a run, so two models are always
    compared on exactly the same folds and the same matches.
    """

    __slots__ = (
        "_feature_names",
        "_features_by_match",
        "_histories",
        "_index",
        "_offset",
        "_settings",
        "_wanted",
    )

    def __init__(
        self,
        histories: Sequence[CompetitionHistory],
        settings: Settings,
        seasons: Sequence[str] | None = None,
    ) -> None:
        self._histories = tuple(histories)
        self._settings = settings
        self._offset = settings.features.as_of_offset_seconds
        self._wanted = frozenset(seasons) if seasons else None
        # One index over every competition: a model may train on both leagues, and the guard
        # is what keeps a league from seeing a simultaneous kickoff in the other one.
        self._index = HistoryIndex.from_records(
            record for history in self._histories for record in history.records
        )
        # One feature lookup over every competition, for the same reason as the index above: a
        # fold's training frame is the whole guarded past, both leagues included, so a per-league
        # lookup would hand the other league's rows an empty frame. `PerCompetition` splits that
        # frame by league and would then fit one member with no features at all.
        self._features_by_match = {
            match_id: stored
            for history in self._histories
            for match_id, stored in history.features.items()
        }
        self._feature_names = _feature_names(self._histories)

    @property
    def index(self) -> HistoryIndex:
        return self._index

    def run(self, factory: Callable[[], MatchModel]) -> ModelRun:
        """Fit and predict every fold, returning what the model said about each match."""
        probe = factory()
        folds: list[Fold] = []
        predictions: list[Prediction] = []
        parameters: list[FittedParameters] = []
        declined = 0
        unscorable = 0

        for history in self._histories:
            for season_id, label, records in self._test_seasons(history):
                outcome = self._season(
                    factory, history, season_id, label, records, market=probe.needs_market
                )
                folds.extend(outcome.folds)
                predictions.extend(outcome.predictions)
                parameters.extend(outcome.parameters)
                declined += outcome.declined
                unscorable += outcome.unscorable

        return ModelRun(
            model_name=probe.name,
            deployable=probe.deployable,
            predictions=tuple(predictions),
            folds=tuple(folds),
            parameters=tuple(parameters),
            declined=declined,
            unscorable=unscorable,
        )

    def _test_seasons(
        self, history: CompetitionHistory
    ) -> Iterator[tuple[int, str, tuple[MatchRecord, ...]]]:
        """The competition's seasons in kickoff order, filtered to the ones asked for."""
        seasons: dict[int, list[MatchRecord]] = {}
        labels: dict[int, str] = {}
        for record in sorted(
            history.records, key=lambda entry: (entry.kickoff_utc, entry.match_id)
        ):
            seasons.setdefault(record.season_id, []).append(record)
            labels[record.season_id] = record.season_label
        for season_id, records in seasons.items():
            if self._wanted is None or labels[season_id] in self._wanted:
                yield season_id, labels[season_id], tuple(records)

    def _season(
        self,
        factory: Callable[[], MatchModel],
        history: CompetitionHistory,
        season_id: int,
        label: str,
        records: Sequence[MatchRecord],
        *,
        market: bool,
    ) -> _SeasonOutcome:
        """One test season: fit, then predict batch by batch, refitting on cadence."""
        competition = history.competition.code
        competition_id = records[0].competition_id
        cadence = self._settings.backtest.refit_every_matchweeks
        participants = _participants(records)
        folds: list[Fold] = []
        predictions: list[Prediction] = []
        parameters: list[FittedParameters] = []
        declined = 0
        unscorable = 0
        model: MatchModel | None = None
        played_at_last_fit = 0

        for kickoff, batch in _batches(records):
            view = self._index.view(as_of_for(kickoff, self._offset))
            played = min(view.matches_played(team, season_id) for team in participants)
            due = model is None or (cadence > 0 and played - played_at_last_fit >= cadence)
            if due:
                training = view.played()
                if model is None and len(training) < self._settings.backtest.min_train_matches:
                    logger.warning(
                        "backtest.fold_skipped",
                        competition=competition,
                        season=label,
                        train_matches=len(training),
                        required=self._settings.backtest.min_train_matches,
                    )
                    break
                fold = _fold(competition, season_id, label, len(folds), view, training, kickoff)
                model = self._fit(factory, training, view)
                folds.append(fold)
                fitted = fitted_parameters_of(model, competition_id)
                if fitted is not None:
                    parameters.append(
                        FittedParameters(
                            model_name=model.name,
                            competition=competition,
                            season_id=season_id,
                            season_label=label,
                            fit_index=fold.fit_index,
                            values=dict(fitted),
                        )
                    )
                played_at_last_fit = played

            if model is None:
                break
            batch_predictions, batch_declined, batch_unscorable = self._predict(
                model, competition, batch, view, market=market
            )
            predictions.extend(batch_predictions)
            declined += batch_declined
            unscorable += batch_unscorable

        return _SeasonOutcome(
            tuple(folds), tuple(predictions), tuple(parameters), declined, unscorable
        )

    def _fit(
        self,
        factory: Callable[[], MatchModel],
        training: Sequence[MatchRecord],
        view: HistoryView,
    ) -> MatchModel:
        """A brand new model, so nothing learned in an earlier fold can survive into this one."""
        model = factory()
        if isinstance(model, HistoryAware):
            model.bind_history(view)
        return model.fit(frames.matches_frame(training), self._features(training, view))

    def _predict(
        self,
        model: MatchModel,
        competition: str,
        batch: Sequence[MatchRecord],
        view: HistoryView,
        *,
        market: bool,
    ) -> tuple[tuple[Prediction, ...], int, int]:
        """One batch of simultaneous kickoffs, predicted from the fixtures alone."""
        if isinstance(model, HistoryAware):
            model.bind_history(view)
        fixtures = [record.as_fixture() for record in batch]
        frame = frames.fixtures_frame(
            fixtures,
            as_of=view.as_of,
            market=self._settings.backtest.market_baseline if market else None,
        )
        requested = [record.match_id for record in batch]
        answer = validate_probabilities(
            model.predict_proba(frame, self._features(fixtures, view)), requested
        )

        answered = answer["match_id"].to_numpy(dtype=np.int64)
        probabilities = answer.loc[:, list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
        by_match = {
            int(match_id): probabilities[position] for position, match_id in enumerate(answered)
        }
        predictions: list[Prediction] = []
        unscorable = 0
        for record in batch:
            answer_row = by_match.get(record.match_id)
            outcome = frames.outcome_of(record)
            if answer_row is None:
                continue
            if outcome is None:
                unscorable += 1
                continue
            if view.as_of >= record.kickoff_utc:
                msg = (
                    f"match {record.match_id} was predicted at {view.as_of.isoformat()}, "
                    f"which is not before its kickoff {record.kickoff_utc.isoformat()}"
                )
                raise LeakageError(msg)
            predictions.append(
                Prediction(
                    match_id=record.match_id,
                    competition=competition,
                    season_id=record.season_id,
                    season_label=record.season_label,
                    kickoff_utc=record.kickoff_utc,
                    as_of_utc=view.as_of,
                    p_home=float(answer_row[0]),
                    p_draw=float(answer_row[1]),
                    p_away=float(answer_row[2]),
                    outcome=outcome,
                )
            )
        return tuple(predictions), len(batch) - len(by_match), unscorable

    def _features(self, records: Sequence[MatchRecord], view: HistoryView) -> pd.DataFrame | None:
        """The stored feature rows of these matches, refusing any stamped after the cut-off."""
        if not self._features_by_match:
            return None
        rows: list[StoredFeatures] = []
        for record in records:
            stored = self._features_by_match.get(record.match_id)
            if stored is None:
                continue
            if stored.as_of_utc > view.as_of:
                msg = (
                    f"the feature row of match {record.match_id} is stamped "
                    f"{stored.as_of_utc.isoformat()}, after the cut-off {view.as_of.isoformat()}"
                )
                raise LeakageError(msg)
            rows.append(stored)
        return frames.features_frame(rows, self._feature_names)


@dataclass(frozen=True, slots=True)
class _SeasonOutcome:
    folds: tuple[Fold, ...]
    predictions: tuple[Prediction, ...]
    parameters: tuple[FittedParameters, ...]
    declined: int
    unscorable: int


def _feature_names(histories: Sequence[CompetitionHistory]) -> tuple[str, ...]:
    """The feature names every competition shares, which one feature version guarantees.

    They are read from one store built by one builder, so they have to agree; if they do not,
    something has written two different definitions under the same version and a model would be
    fitted on silently different columns per league.
    """
    named = {history.feature_names for history in histories if history.feature_names}
    if len(named) > 1:
        msg = f"the competitions of this run carry different feature names: {sorted(named)}"
        raise ValueError(msg)
    return next(iter(named), ())


def _batches(records: Sequence[MatchRecord]) -> Iterator[tuple[datetime, tuple[MatchRecord, ...]]]:
    """The season's matches grouped by identical kickoff, in chronological order."""
    grouped: dict[datetime, list[MatchRecord]] = {}
    for record in records:
        grouped.setdefault(record.kickoff_utc, []).append(record)
    for kickoff in sorted(grouped):
        yield kickoff, tuple(sorted(grouped[kickoff], key=lambda entry: entry.match_id))


def _participants(records: Iterable[MatchRecord]) -> tuple[int, ...]:
    """Every team with a fixture in the season, which is schedule information, not a result."""
    teams: set[int] = set()
    for record in records:
        teams.add(record.home_team_id)
        teams.add(record.away_team_id)
    return tuple(sorted(teams))


def _fold(
    competition: str,
    season_id: int,
    label: str,
    fit_index: int,
    view: HistoryView,
    training: Sequence[MatchRecord],
    kickoff: datetime,
) -> Fold:
    """One fold, with the boundary invariant of spec section 8 checked rather than assumed."""
    latest = training[-1].kickoff_utc if training else None
    if latest is not None and latest >= kickoff:
        msg = (
            f"fold {competition} {label} would train on a match kicking off at "
            f"{latest.isoformat()}, at or after the first test kickoff {kickoff.isoformat()}"
        )
        raise LeakageError(msg)
    return Fold(
        competition=competition,
        season_id=season_id,
        season_label=label,
        fit_index=fit_index,
        as_of_utc=view.as_of,
        train_matches=len(training),
        latest_train_kickoff=latest,
        earliest_test_kickoff=kickoff,
    )


def slices(run: ModelRun) -> tuple[tuple[str, str | None, str | None, tuple[Prediction, ...]], ...]:
    """Every scope a run is reported at: pooled, per league, and per league-season."""
    pooled = [(POOLED, None, None, run.predictions)]
    leagues: dict[str, list[Prediction]] = {}
    seasons: dict[tuple[str, str], list[Prediction]] = {}
    for prediction in run.predictions:
        leagues.setdefault(prediction.competition, []).append(prediction)
        seasons.setdefault((prediction.competition, prediction.season_label), []).append(prediction)
    return (
        *pooled,
        *((COMPETITION, code, None, tuple(rows)) for code, rows in sorted(leagues.items())),
        *((SEASON, code, label, tuple(rows)) for (code, label), rows in sorted(seasons.items())),
    )


def score(run: ModelRun, settings: Settings) -> tuple[ScopedMetrics, ...]:
    """Score a run at every scope the spec asks for: per season, per league and pooled."""
    metrics = settings.backtest.metrics
    scored: list[ScopedMetrics] = []
    for scope, competition, label, predictions in slices(run):
        if not predictions:
            continue
        probabilities, outcomes = as_arrays(predictions)
        scored.append(
            ScopedMetrics(
                scope=scope,
                competition=competition,
                season_label=label,
                metrics=metric_set(
                    probabilities,
                    outcomes,
                    bins=metrics.ece_bins,
                    floor=metrics.log_loss_floor,
                ),
                predictions=predictions,
            )
        )
    return tuple(scored)

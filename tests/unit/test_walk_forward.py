"""The walk-forward engine: fold boundaries, refits and the leakage guard (spec section 8)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import pandas as pd
import pytest

from predictor.config import BacktestSettings, Competition, Settings, get_settings
from predictor.evaluation.backtest import CompetitionHistory, WalkForward
from predictor.features.replay import HistoryView, LeakageError, MatchRecord
from predictor.models.base import PROBABILITY_COLUMNS, MatchModel, ScoreMatrices
from tests.support import kickoff, record

#: Four teams and a double round-robin: six rounds of two matches, 12 matches a season.
#: Every team plays exactly once per round, which is what makes the refit cadence countable.
ROUNDS = (
    ((10, 20), (30, 40)),
    ((10, 30), (40, 20)),
    ((10, 40), (20, 30)),
    ((20, 10), (40, 30)),
    ((30, 10), (20, 40)),
    ((40, 10), (30, 20)),
)
MATCHES_PER_SEASON = 12
SEASON_LABELS = ("2018-19", "2019-20", "2020-21")


def fixtures_of(season_id: int, label: str, first_day: int) -> list[MatchRecord]:
    """One toy season: one round a week, both matches of a round sharing their kickoff."""
    records: list[MatchRecord] = []
    for round_index, pairings in enumerate(ROUNDS):
        for slot, (home, away) in enumerate(pairings):
            records.append(
                record(
                    season_id * 100 + round_index * 2 + slot,
                    home,
                    away,
                    kickoff(first_day + round_index * 7),
                    (home + away) % 3,
                    home % 2,
                    season_id=season_id,
                    season_label=label,
                )
            )
    return records


def toy_records(seasons: int = 3) -> list[MatchRecord]:
    records: list[MatchRecord] = []
    for index in range(seasons):
        records.extend(fixtures_of(index + 1, SEASON_LABELS[index], 1 + index * 400))
    return records


def history(records: Sequence[MatchRecord] | None = None) -> CompetitionHistory:
    competition = get_settings().competition("EPL")
    return CompetitionHistory(
        competition=competition, records=tuple(records if records is not None else toy_records())
    )


def with_backtest(settings: Settings, **overrides: object) -> Settings:
    """The same settings with the backtest section overridden."""
    backtest = BacktestSettings(**{**settings.backtest.model_dump(), **overrides})
    return settings.model_copy(update={"backtest": backtest})


def engine(
    settings: Settings | None = None, records: Sequence[MatchRecord] | None = None
) -> WalkForward:
    resolved = settings or with_backtest(get_settings(), min_train_matches=1)
    return WalkForward([history(records)], resolved)


class Uniform:
    """A model that says nothing, so a test can watch the engine rather than the maths."""

    name = "uniform"
    deployable = False
    needs_market = False

    def __init__(self) -> None:
        self.fitted: list[int] = []
        self.seen_columns: set[str] = set()

    def fit(self, matches: pd.DataFrame, features: pd.DataFrame | None) -> Uniform:
        del features
        self.fitted.append(len(matches))
        return self

    def predict_proba(self, fixtures: pd.DataFrame, features: pd.DataFrame | None) -> pd.DataFrame:
        del features
        self.seen_columns.update(fixtures.columns)
        third = 1.0 / 3.0
        return pd.DataFrame(
            {
                "match_id": fixtures["match_id"].to_numpy(),
                "p_home": third,
                "p_draw": third,
                "p_away": third,
            }
        )

    def predict_scores(
        self,
        fixtures: pd.DataFrame,
        features: pd.DataFrame | None = None,
        max_goals: int | None = None,
    ) -> ScoreMatrices:
        raise NotImplementedError


class Cheating(Uniform):
    """A model that asks the guarded history for the result of the match it is predicting."""

    name = "cheating"

    def __init__(self) -> None:
        super().__init__()
        self._view: HistoryView | None = None

    def bind_history(self, view: HistoryView) -> None:
        self._view = view

    def predict_proba(self, fixtures: pd.DataFrame, features: pd.DataFrame | None) -> pd.DataFrame:
        assert self._view is not None
        for match_id in fixtures["match_id"]:
            self._view.result_of(int(match_id))
        return super().predict_proba(fixtures, features)


class Broken(Uniform):
    """A model whose rows do not add up."""

    name = "broken"

    def predict_proba(self, fixtures: pd.DataFrame, features: pd.DataFrame | None) -> pd.DataFrame:
        frame = super().predict_proba(fixtures, features)
        frame.loc[:, "p_home"] = 0.9
        return frame


class Inventive(Uniform):
    """A model that predicts a match nobody asked about."""

    name = "inventive"

    def predict_proba(self, fixtures: pd.DataFrame, features: pd.DataFrame | None) -> pd.DataFrame:
        frame = super().predict_proba(fixtures, features)
        extra = pd.DataFrame({"match_id": [-1], "p_home": [1 / 3], "p_draw": [1 / 3]})
        extra["p_away"] = 1 / 3
        return pd.concat([frame, extra], ignore_index=True)


class Selective(Uniform):
    """A model that declines the first fixture of every batch, as the market baseline may."""

    name = "selective"

    def predict_proba(self, fixtures: pd.DataFrame, features: pd.DataFrame | None) -> pd.DataFrame:
        return super().predict_proba(fixtures, features).iloc[1:]


def test_the_latest_training_kickoff_is_before_the_earliest_test_kickoff() -> None:
    """Acceptance test 3: the fold boundary, on every fold of every test season."""
    run = engine().run(Uniform)

    assert run.folds
    for fold in run.folds:
        assert fold.latest_train_kickoff is not None
        assert fold.latest_train_kickoff < fold.earliest_test_kickoff
        assert fold.as_of_utc <= fold.earliest_test_kickoff


def test_a_model_that_reads_its_own_result_raises_leakage_error() -> None:
    """Acceptance test 2: the guarded view refuses the match being predicted."""
    with pytest.raises(LeakageError, match="not before the cut-off"):
        engine().run(Cheating)


def test_every_prediction_is_stamped_strictly_before_its_kickoff() -> None:
    run = engine().run(Uniform)

    assert run.predictions
    for prediction in run.predictions:
        assert prediction.as_of_utc < prediction.kickoff_utc


def test_the_first_season_on_record_is_never_a_test_season() -> None:
    """There is nothing to train on before it, so it belongs to the training history."""
    run = engine().run(Uniform)

    tested = {prediction.season_label for prediction in run.predictions}
    assert tested == {"2019-20", "2020-21"}


def test_only_the_requested_seasons_are_tested() -> None:
    run = WalkForward(
        [history()], with_backtest(get_settings(), min_train_matches=1), seasons=("2020-21",)
    ).run(Uniform)

    assert {prediction.season_label for prediction in run.predictions} == {"2020-21"}
    assert len(run.predictions) == MATCHES_PER_SEASON


def test_a_model_is_fitted_once_per_season_when_no_cadence_is_configured() -> None:
    settings = with_backtest(get_settings(), min_train_matches=1, refit_every_matchweeks=0)
    run = WalkForward([history()], settings).run(Uniform)

    assert [fold.fit_index for fold in run.folds] == [0, 0]
    assert len(run.folds) == 2


def test_a_cadence_refits_once_every_round_of_the_season() -> None:
    """Six kickoff days per season and a cadence of one: the first fit plus five refits."""
    settings = with_backtest(get_settings(), min_train_matches=1, refit_every_matchweeks=1)
    run = WalkForward([history()], settings, seasons=("2020-21",)).run(Uniform)

    assert [fold.fit_index for fold in run.folds] == [0, 1, 2, 3, 4, 5]
    assert len(run.predictions) == MATCHES_PER_SEASON


def test_a_refit_trains_on_the_matches_the_season_has_already_played() -> None:
    settings = with_backtest(get_settings(), min_train_matches=1, refit_every_matchweeks=1)
    run = WalkForward([history()], settings, seasons=("2020-21",)).run(Uniform)

    sizes = [fold.train_matches for fold in run.folds]
    assert sizes == sorted(sizes)
    assert sizes[-1] - sizes[0] == MATCHES_PER_SEASON - 2


def test_a_fold_with_too_little_history_is_skipped() -> None:
    settings = with_backtest(get_settings(), min_train_matches=MATCHES_PER_SEASON + 1)
    run = WalkForward([history()], settings).run(Uniform)

    assert [fold.season_label for fold in run.folds] == ["2020-21"]
    assert {prediction.season_label for prediction in run.predictions} == {"2020-21"}


def test_the_model_is_handed_fixtures_without_a_result() -> None:
    model = Uniform()
    engine().run(lambda: model)

    assert "match_id" in model.seen_columns
    assert not model.seen_columns & {"home_goals", "away_goals", "outcome"}


def test_probabilities_that_do_not_sum_to_one_are_refused() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        engine().run(Broken)


def test_a_prediction_for_a_match_nobody_asked_about_is_refused() -> None:
    with pytest.raises(ValueError, match="did not ask"):
        engine().run(Inventive)


def test_a_model_may_decline_to_predict_a_match() -> None:
    """The market benchmark has no quote for some matches; fewer rows is allowed, not an error."""
    run = engine().run(Selective)

    declined = MATCHES_PER_SEASON  # one per kickoff day, across both test seasons
    assert len(run.predictions) == 2 * MATCHES_PER_SEASON - declined


def test_the_training_frame_grows_with_every_season() -> None:
    model = Uniform()
    WalkForward([history()], with_backtest(get_settings(), min_train_matches=1)).run(lambda: model)

    assert model.fitted == [MATCHES_PER_SEASON, 2 * MATCHES_PER_SEASON]


def test_two_leagues_are_tested_independently_in_one_run() -> None:
    settings = with_backtest(get_settings(), min_train_matches=1)
    laliga = get_settings().competition("LALIGA")
    other = [
        record(
            entry.match_id + 10_000,
            entry.home_team_id + 1,
            entry.away_team_id + 1,
            entry.kickoff_utc,
            entry.home_goals,
            entry.away_goals,
            season_id=entry.season_id + 10,
            season_label=entry.season_label,
            competition_id=2,
        )
        for entry in toy_records()
    ]
    run = WalkForward(
        [history(), CompetitionHistory(competition=laliga, records=tuple(other))], settings
    ).run(Uniform)

    assert {fold.competition for fold in run.folds} == {"EPL", "LALIGA"}
    assert len(run.predictions) == 4 * MATCHES_PER_SEASON


def test_the_engine_reports_what_it_ran() -> None:
    run = engine().run(Uniform)

    assert run.model_name == "uniform"
    assert run.deployable is False
    assert isinstance(run.folds[0].as_of_utc, datetime)
    assert set(PROBABILITY_COLUMNS) == {"p_home", "p_draw", "p_away"}


def test_a_model_satisfies_the_protocol() -> None:
    model: MatchModel = Uniform()
    assert model.name == "uniform"


def test_a_competition_without_a_second_season_has_nothing_to_test() -> None:
    single = fixtures_of(1, SEASON_LABELS[0], 1)
    run = engine(records=single).run(Uniform)

    assert run.folds == ()
    assert run.predictions == ()


def test_a_toy_competition_is_only_as_big_as_it_looks() -> None:
    assert len(fixtures_of(1, SEASON_LABELS[0], 1)) == MATCHES_PER_SEASON
    assert isinstance(history().competition, Competition)

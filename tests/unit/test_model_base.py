"""The common model interface (spec section 7) and the per-competition wrapper (7.1)."""

from __future__ import annotations

import pandas as pd
import pytest

from predictor.evaluation.frames import fixtures_frame, matches_frame
from predictor.features.replay import HistoryIndex, HistoryView, MatchRecord
from predictor.models.base import (
    AWAY,
    DRAW,
    HOME,
    PROBABILITY_COLUMNS,
    PerCompetition,
    ScoreMatrices,
    probability_frame,
)
from tests.support import kickoff, record


class Counting:
    """A model that answers with the training frequencies of its own fold, one league at a time."""

    name = "counting"
    deployable = True
    needs_market = False

    def __init__(self) -> None:
        self.competitions: list[int] = []
        self.rows = 0
        self.views: list[HistoryView] = []

    def bind_history(self, view: HistoryView) -> None:
        self.views.append(view)

    def fit(self, matches: pd.DataFrame, features: pd.DataFrame | None) -> Counting:
        del features
        self.competitions = sorted(set(matches["competition_id"].tolist()))
        self.rows = len(matches)
        return self

    def predict_proba(self, fixtures: pd.DataFrame, features: pd.DataFrame | None) -> pd.DataFrame:
        del features
        share = 1.0 / (1.0 + self.rows)
        rest = (1.0 - share) / 2.0
        return probability_frame(
            fixtures["match_id"].tolist(), [(share, rest, rest)] * len(fixtures)
        )

    def predict_scores(
        self,
        fixtures: pd.DataFrame,
        features: pd.DataFrame | None = None,
        max_goals: int | None = None,
    ) -> ScoreMatrices:
        raise NotImplementedError


def epl() -> list[MatchRecord]:
    return [
        record(1, 10, 20, kickoff(1), 2, 0),
        record(2, 30, 40, kickoff(1), 1, 1),
        record(3, 10, 30, kickoff(8), 0, 3),
    ]


def laliga() -> list[MatchRecord]:
    return [
        record(11, 50, 60, kickoff(1), 1, 0, competition_id=2, season_id=2),
        record(12, 70, 80, kickoff(8), 2, 2, competition_id=2, season_id=2),
    ]


def test_the_outcome_codes_are_ordered_home_draw_away() -> None:
    """RPS respects the order H > D > A, so the codes are not arbitrary."""
    assert (HOME, DRAW, AWAY) == (0, 1, 2)
    assert PROBABILITY_COLUMNS == ("p_home", "p_draw", "p_away")


def test_a_matches_frame_carries_the_outcome_of_every_played_match() -> None:
    frame = matches_frame(epl())

    assert frame["outcome"].tolist() == [HOME, DRAW, AWAY]
    assert frame["match_id"].tolist() == [1, 2, 3]


def test_a_matches_frame_leaves_an_unplayed_match_without_an_outcome() -> None:
    frame = matches_frame([*epl(), record(4, 20, 40, kickoff(9)).as_fixture()])

    assert frame["outcome"].isna().tolist() == [False, False, False, True]


def test_a_fixtures_frame_hides_every_result() -> None:
    frame = fixtures_frame([entry.as_fixture() for entry in epl()], as_of=kickoff(9))

    assert not set(frame.columns) & {"home_goals", "away_goals", "outcome"}
    assert frame["home_team_id"].tolist() == [10, 30, 10]


def test_per_competition_fits_one_model_for_each_league() -> None:
    wrapper = PerCompetition(Counting)
    records = [*epl(), *laliga()]
    wrapper.fit(matches_frame(records), None)

    assert sorted(wrapper.members) == [1, 2]
    assert wrapper.members[1].competitions == [1]
    assert wrapper.members[2].competitions == [2]
    assert wrapper.members[1].rows == 3
    assert wrapper.members[2].rows == 2


def test_per_competition_predictions_come_from_the_leagues_own_model() -> None:
    wrapper = PerCompetition(Counting)
    wrapper.fit(matches_frame([*epl(), *laliga()]), None)
    fixtures = fixtures_frame(
        [entry.as_fixture() for entry in [*epl(), *laliga()]], as_of=kickoff(30)
    )
    frame = wrapper.predict_proba(fixtures, None)

    assert frame["match_id"].tolist() == [1, 2, 3, 11, 12]
    by_match = frame.set_index("match_id")["p_home"]
    assert by_match[1] == pytest.approx(1 / 4)  # three EPL training rows
    assert by_match[11] == pytest.approx(1 / 3)  # two La Liga training rows


def test_per_competition_keeps_a_league_untouched_when_the_other_one_is_missing() -> None:
    """Spec section 7.1: EPL and La Liga teams never meet, so neither may move the other."""
    both = PerCompetition(Counting)
    both.fit(matches_frame([*epl(), *laliga()]), None)
    alone = PerCompetition(Counting)
    alone.fit(matches_frame(epl()), None)

    fixtures = fixtures_frame([entry.as_fixture() for entry in epl()], as_of=kickoff(30))
    assert both.predict_proba(fixtures, None)["p_home"].tolist() == (
        alone.predict_proba(fixtures, None)["p_home"].tolist()
    )


def test_per_competition_takes_the_name_and_the_flags_of_what_it_wraps() -> None:
    wrapper = PerCompetition(Counting)

    assert wrapper.name == "counting"
    assert wrapper.deployable is True
    assert wrapper.needs_market is False


def test_per_competition_passes_the_guarded_view_to_every_member() -> None:
    wrapper = PerCompetition(Counting)
    wrapper.fit(matches_frame([*epl(), *laliga()]), None)
    view = HistoryIndex.from_records(epl()).view(kickoff(30))
    wrapper.bind_history(view)

    assert [member.views for member in wrapper.members.values()] == [[view], [view]]


def test_per_competition_refuses_a_league_it_was_never_fitted_on() -> None:
    wrapper = PerCompetition(Counting)
    wrapper.fit(matches_frame(epl()), None)
    fixtures = fixtures_frame([entry.as_fixture() for entry in laliga()], as_of=kickoff(30))

    with pytest.raises(LookupError, match="competition 2"):
        wrapper.predict_proba(fixtures, None)


def test_a_probability_frame_is_built_in_the_order_it_was_given() -> None:
    frame = probability_frame([5, 6], [(0.5, 0.3, 0.2), (0.1, 0.2, 0.7)])

    assert frame["match_id"].tolist() == [5, 6]
    assert list(frame.columns) == ["match_id", *PROBABILITY_COLUMNS]
    assert frame["p_away"].tolist() == [0.2, 0.7]

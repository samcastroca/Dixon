"""Rolling and exponentially weighted form. Golden values calculated independently."""

from __future__ import annotations

import pytest
from tests.support import kickoff, record

from predictor.features.form import FormFeatures
from predictor.features.replay import MatchRecord, ReplayEngine

TEAM = 10

# Team 10 plays four matches and we read its form before the fifth. As home side it shoots
# 10 (5 on target) and concedes 8 (3); as away side, 8 (3) for and 10 (5) against.
HISTORY = [
    record(1, TEAM, 20, kickoff(1), 2, 1),
    record(2, 30, TEAM, kickoff(2), 1, 0),
    record(3, TEAM, 40, kickoff(3), 3, 0),
    record(4, 50, TEAM, kickoff(4), 2, 1),
    record(5, TEAM, 20, kickoff(5), 0, 0),
]

# alpha = 1 - 2 ** (-1 / halflife) = 0.2928932188134524, applied as alpha*x + (1-alpha)*previous.
#   goals for      overall [2, 0, 3, 1]      venue (home only) [2, 3]
#   goals against  overall [1, 1, 0, 2]      venue             [1, 0]
EXPECTED = {
    "home_overall_ewma_goals_for": 1.6213203435596428,
    "home_overall_ewma_goals_against": 1.085786437626905,
    "home_overall_ewma_shots_for": 9.121320343559644,
    "home_overall_ewma_shots_against": 8.878679656440358,
    "home_overall_ewma_sot_for": 4.121320343559643,
    "home_overall_ewma_sot_against": 3.878679656440357,
    "home_overall_last3_goals_for": 1.3333333333333333,
    "home_overall_last3_goals_against": 1.0,
    "home_overall_last3_shots_for": 8.666666666666666,
    "home_overall_last3_shots_against": 9.333333333333334,
    "home_overall_last3_sot_for": 3.6666666666666665,
    "home_overall_last3_sot_against": 4.333333333333333,
    "home_venue_ewma_goals_for": 2.2928932188134525,
    "home_venue_ewma_goals_against": 0.7071067811865476,
    "home_venue_ewma_shots_for": 10.0,
    "home_venue_ewma_shots_against": 8.0,
    "home_venue_ewma_sot_for": 5.0,
    "home_venue_ewma_sot_against": 3.0,
    "home_venue_last3_goals_for": 2.5,
    "home_venue_last3_goals_against": 0.5,
    "home_venue_last3_shots_for": 10.0,
    "home_venue_last3_shots_against": 8.0,
    "home_venue_last3_sot_for": 5.0,
    "home_venue_last3_sot_against": 3.0,
}


def form() -> FormFeatures:
    return FormFeatures(windows=(3,), ewma_halflife=2.0)


def replay(records: list[MatchRecord]) -> dict[int, dict[str, float | None]]:
    engine = ReplayEngine(records, as_of_offset_seconds=1)
    return {step.match.match_id: step.values for step in engine.run([form()])}


def test_the_hand_calculated_form_of_the_fifth_match() -> None:
    values = replay(HISTORY)[5]
    for name, expected in EXPECTED.items():
        assert values[name] == pytest.approx(expected, abs=1e-12), name


def test_a_team_without_history_has_no_form_at_all() -> None:
    values = replay(HISTORY)[1]
    assert set(values) == set(form().names)
    assert all(value is None for value in values.values())


def test_the_venue_scope_ignores_the_other_venue() -> None:
    """Team 10 has played twice at home by match 5, so its venue window holds only those."""
    values = replay(HISTORY)[5]
    assert values["home_venue_last3_goals_for"] == pytest.approx(2.5, abs=1e-12)
    assert values["home_overall_last3_goals_for"] == pytest.approx(4.0 / 3.0, abs=1e-12)


def test_the_away_side_is_measured_at_its_own_venue() -> None:
    """Team 20 lost 1-2 away in match 1, so before match 5 its away form is that match."""
    values = replay(HISTORY)[5]
    assert values["away_venue_last3_goals_for"] == 1.0
    assert values["away_venue_last3_goals_against"] == 2.0


def test_a_window_shorter_than_the_history_averages_only_the_window() -> None:
    values = replay(HISTORY)[5]
    # Four appearances, a window of three: the 2-1 win of match 1 falls out.
    assert values["home_overall_last3_goals_for"] != pytest.approx(6.0 / 4.0, abs=1e-12)


def test_a_missing_statistic_is_skipped_instead_of_counted_as_zero() -> None:
    records = [
        record(1, TEAM, 20, kickoff(1), 1, 0, shots=(None, None), shots_on_target=(None, None)),
        record(2, TEAM, 30, kickoff(2), 2, 0, shots=(12, 4), shots_on_target=(6, 2)),
        record(3, TEAM, 40, kickoff(3), 0, 0),
    ]
    values = replay(records)[3]
    assert values["home_overall_last3_shots_for"] == 12.0
    assert values["home_overall_ewma_shots_for"] == 12.0
    assert values["home_overall_last3_goals_for"] == pytest.approx(1.5, abs=1e-12)


def test_every_statistic_missing_leaves_the_metric_unknown() -> None:
    records = [
        record(1, TEAM, 20, kickoff(1), 1, 0, shots=(None, None), shots_on_target=(None, None)),
        record(2, TEAM, 30, kickoff(2), 0, 0),
    ]
    values = replay(records)[2]
    assert values["home_overall_last3_shots_for"] is None
    assert values["home_overall_ewma_shots_for"] is None


def test_the_configured_windows_drive_the_feature_names() -> None:
    names = FormFeatures(windows=(5, 10), ewma_halflife=4.0).names
    assert "home_overall_last5_goals_for" in names
    assert "away_venue_last10_sot_against" in names
    assert "home_overall_last3_goals_for" not in names
    assert len(names) == len(set(names))
    # 2 sides x 2 scopes x (1 ewma + 2 windows) x 6 metrics
    assert len(names) == 2 * 2 * 3 * 6

"""The two baselines of spec section 7: historical frequencies and de-margined closing odds."""

from __future__ import annotations

import numpy as np
import pytest
from tests.support import kickoff, record

from predictor.config import MarketFeatureSettings
from predictor.evaluation.frames import fixtures_frame, matches_frame
from predictor.features.replay import LeakageError, MarketQuote, MatchRecord
from predictor.models.baselines import FrequencyBaseline, MarketBaseline
from predictor.models.registry import MODEL_NAMES, build

TOLERANCE = 1e-12

#: Three home wins, two draws and one away win: 1/2, 1/3 and 1/6.
TRAINING_RESULTS = ((2, 0), (3, 1), (1, 0), (1, 1), (2, 2), (0, 1))

QUOTE = MarketQuote(p_home=0.5, p_draw=0.3, p_away=0.2, captured_at=None)

ALLOWED = MarketFeatureSettings(allow_missing_captured_at=True)
REFUSED = MarketFeatureSettings(allow_missing_captured_at=False)


def training() -> list[MatchRecord]:
    return [
        record(index + 1, 10, 20, kickoff(index + 1), home, away)
        for index, (home, away) in enumerate(TRAINING_RESULTS)
    ]


def fixture(match_id: int = 99, market: MarketQuote | None = None) -> MatchRecord:
    return record(match_id, 10, 20, kickoff(50), market=market).as_fixture()


def test_the_frequency_baseline_reports_the_training_frequencies() -> None:
    model = FrequencyBaseline().fit(matches_frame(training()), None)
    frame = model.predict_proba(fixtures_frame([fixture()], as_of=kickoff(50)), None)

    assert frame["p_home"].iloc[0] == pytest.approx(0.5, abs=TOLERANCE)
    assert frame["p_draw"].iloc[0] == pytest.approx(1 / 3, abs=TOLERANCE)
    assert frame["p_away"].iloc[0] == pytest.approx(1 / 6, abs=TOLERANCE)


def test_the_frequency_baseline_falls_back_to_uniform_without_any_history() -> None:
    model = FrequencyBaseline().fit(matches_frame([]), None)
    frame = model.predict_proba(fixtures_frame([fixture()], as_of=kickoff(50)), None)

    assert frame.loc[0, ["p_home", "p_draw", "p_away"]].to_numpy() == pytest.approx(
        np.full(3, 1 / 3), abs=TOLERANCE
    )


def test_the_frequency_baseline_ignores_matches_without_a_result() -> None:
    postponed = record(7, 30, 40, kickoff(7)).as_fixture()
    model = FrequencyBaseline().fit(matches_frame([*training(), postponed]), None)
    frame = model.predict_proba(fixtures_frame([fixture()], as_of=kickoff(50)), None)

    assert frame["p_home"].iloc[0] == pytest.approx(0.5, abs=TOLERANCE)


def test_the_frequency_baseline_is_deployable_and_needs_no_market() -> None:
    model = FrequencyBaseline()

    assert model.name == "baseline_frequency"
    assert model.deployable is True
    assert model.needs_market is False


def test_the_market_baseline_repeats_the_de_margined_quote() -> None:
    model = MarketBaseline().fit(matches_frame(training()), None)
    fixtures = fixtures_frame([fixture(market=QUOTE)], as_of=kickoff(50), market=ALLOWED)
    frame = model.predict_proba(fixtures, None)

    assert frame["p_home"].iloc[0] == pytest.approx(0.5, abs=TOLERANCE)
    assert frame["p_draw"].iloc[0] == pytest.approx(0.3, abs=TOLERANCE)
    assert frame["p_away"].iloc[0] == pytest.approx(0.2, abs=TOLERANCE)


def test_the_market_baseline_declines_a_match_without_a_quote() -> None:
    model = MarketBaseline()
    fixtures = fixtures_frame(
        [fixture(1, market=QUOTE), fixture(2)], as_of=kickoff(50), market=ALLOWED
    )
    frame = model.predict_proba(fixtures, None)

    assert frame["match_id"].tolist() == [1]


def test_the_market_baseline_is_a_benchmark_and_not_deployable() -> None:
    """Spec section 7: M0b is something to compare against, not something to ship."""
    model = MarketBaseline()

    assert model.name == "baseline_market"
    assert model.deployable is False
    assert model.needs_market is True


def test_an_undated_quote_is_refused_unless_the_settings_allow_it() -> None:
    with pytest.raises(LeakageError, match="captured_at"):
        fixtures_frame([fixture(market=QUOTE)], as_of=kickoff(50), market=REFUSED)


def test_a_quote_captured_after_the_cut_off_is_a_leak() -> None:
    late = MarketQuote(p_home=0.5, p_draw=0.3, p_away=0.2, captured_at=kickoff(50))
    with pytest.raises(LeakageError, match="not before the cut-off"):
        fixtures_frame([fixture(market=late)], as_of=kickoff(50), market=ALLOWED)


def test_a_fixtures_frame_without_the_market_has_no_market_columns() -> None:
    frame = fixtures_frame([fixture(market=QUOTE)], as_of=kickoff(50))

    assert not [column for column in frame.columns if column.startswith("market_")]


def test_neither_baseline_offers_a_scoreline_matrix() -> None:
    """Scoreline distributions arrive with the statistical models in phase 5."""
    fixtures = fixtures_frame([fixture()], as_of=kickoff(50))
    for model in (FrequencyBaseline(), MarketBaseline()):
        with pytest.raises(NotImplementedError, match="scoreline"):
            model.predict_scores(fixtures)


def test_both_baselines_are_registered_under_the_names_the_cli_takes() -> None:
    assert set(MODEL_NAMES) == {"baseline_frequency", "baseline_market"}
    assert build("baseline_frequency").name == "baseline_frequency"
    assert build("baseline_market").name == "baseline_market"


def test_the_frequency_baseline_is_registered_per_competition() -> None:
    """Spec section 7.1: M0a is fitted per league, so the wrapper has to be in the registry."""
    model = build("baseline_frequency")

    assert type(model).__name__ == "PerCompetition"


def test_an_unknown_model_name_lists_the_known_ones() -> None:
    with pytest.raises(KeyError, match="baseline_market"):
        build("no_such_model")

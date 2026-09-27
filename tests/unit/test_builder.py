"""Assembling one wide feature row per match, and the flagged market group."""

from __future__ import annotations

from datetime import timedelta

import pytest

from predictor.config import MarketFeatureSettings, Settings, get_settings
from predictor.features.builder import FeatureBuilder
from predictor.features.replay import LeakageError, MarketQuote, MatchRecord
from predictor.features.validate import FeatureValidationError, validate_features
from tests.support import kickoff, record

HISTORY = [
    record(1, 10, 20, kickoff(1), 2, 0),
    record(2, 30, 40, kickoff(1), 1, 1),
    record(3, 10, 30, kickoff(8), 0, 3),
    record(4, 40, 20, kickoff(8), 2, 2),
    record(5, 20, 10, kickoff(15), 1, 0),
]


def builder(settings: Settings | None = None, version: str | None = None) -> FeatureBuilder:
    resolved = settings or get_settings()
    return FeatureBuilder.for_competition(resolved.competition("EPL"), resolved, version)


def with_market(settings: Settings, **overrides: object) -> Settings:
    """Same settings with the market group switched on and its knobs overridden."""
    market = MarketFeatureSettings(**{**settings.features.market.model_dump(), **overrides})
    features = settings.features.model_copy(
        update={
            "groups": settings.features.groups.model_copy(update={"market": True}),
            "market": market,
        }
    )
    return settings.model_copy(update={"features": features})


def test_every_row_is_stamped_strictly_before_kickoff() -> None:
    rows = list(builder().rows(HISTORY))
    assert len(rows) == len(HISTORY)
    for row in rows:
        assert row.as_of_utc < row.kickoff_utc


def test_the_cut_off_is_the_configured_offset_before_kickoff() -> None:
    settings = get_settings()
    offset = timedelta(seconds=settings.features.as_of_offset_seconds)
    for row in builder(settings).rows(HISTORY):
        assert row.kickoff_utc - row.as_of_utc == offset


def test_a_row_carries_exactly_the_declared_features() -> None:
    build = builder()
    for row in build.rows(HISTORY):
        assert tuple(row.values) == build.names


def test_the_default_version_comes_from_the_configuration() -> None:
    settings = get_settings()
    assert builder(settings).version == settings.features.version


def test_an_explicit_version_overrides_the_configured_one() -> None:
    assert builder(version="v9").version == "v9"


def test_the_market_group_is_off_by_default() -> None:
    build = builder()
    assert not any(name.startswith("market_") for name in build.names)


def test_the_market_group_adds_its_own_features_when_enabled() -> None:
    settings = with_market(get_settings(), allow_missing_captured_at=True)
    build = builder(settings)
    assert {"market_p_home", "market_p_draw", "market_p_away"} <= set(build.names)


def test_an_untimestamped_quote_is_refused_unless_it_is_explicitly_allowed() -> None:
    quote = MarketQuote(p_home=0.5, p_draw=0.3, p_away=0.2, captured_at=None)
    records = [record(1, 10, 20, kickoff(1), 2, 0, market=quote)]
    with pytest.raises(LeakageError, match="captured_at"):
        list(builder(with_market(get_settings())).rows(records))


def test_an_untimestamped_quote_is_used_once_the_risk_is_acknowledged() -> None:
    quote = MarketQuote(p_home=0.5, p_draw=0.3, p_away=0.2, captured_at=None)
    records = [record(1, 10, 20, kickoff(1), 2, 0, market=quote)]
    settings = with_market(get_settings(), allow_missing_captured_at=True)
    (row,) = builder(settings).rows(records)
    assert row.values["market_p_home"] == 0.5


def test_a_quote_captured_after_the_cut_off_is_a_leak() -> None:
    quote = MarketQuote(p_home=0.5, p_draw=0.3, p_away=0.2, captured_at=kickoff(1))
    records = [record(1, 10, 20, kickoff(1), 2, 0, market=quote)]
    with pytest.raises(LeakageError):
        list(builder(with_market(get_settings())).rows(records))


def test_a_match_without_a_quote_leaves_the_market_features_unknown() -> None:
    settings = with_market(get_settings(), allow_missing_captured_at=True)
    (row,) = builder(settings).rows([record(1, 10, 20, kickoff(1), 2, 0)])
    assert row.values["market_p_home"] is None


def test_every_value_is_rounded_to_the_configured_precision() -> None:
    """Rounding is what lets a rebuild on another C library land on the same numbers."""
    settings = get_settings()
    decimals = settings.features.value_decimals
    seen = 0
    for row in builder(settings).rows(HISTORY):
        for name, value in row.values.items():
            if value is None:
                continue
            assert value == round(value, decimals), name
            seen += 1
    assert seen


def test_the_definition_checksum_changes_with_the_parameters() -> None:
    settings = get_settings()
    stronger = settings.features.elo.model_copy(
        update={"defaults": settings.features.elo.defaults.model_copy(update={"k": 25.0})}
    )
    tweaked = settings.model_copy(
        update={"features": settings.features.model_copy(update={"elo": stronger})}
    )
    assert builder(settings).definition_checksum != builder(tweaked).definition_checksum


def test_the_definition_checksum_is_stable_across_builders() -> None:
    assert builder().definition_checksum == builder().definition_checksum


def test_the_same_records_produce_the_same_values_twice() -> None:
    first = [(row.match_id, row.values) for row in builder().rows(HISTORY)]
    second = [(row.match_id, row.values) for row in builder().rows(HISTORY)]
    assert first == second


def test_rows_are_produced_lazily_so_a_replay_can_stop_early() -> None:
    produced = builder().rows(HISTORY)
    assert next(produced).match_id == 1
    assert next(produced).match_id == 2


def test_the_frame_holds_one_validated_row_per_match() -> None:
    build = builder()
    frame = build.frame(build.rows(HISTORY))
    assert list(frame["match_id"]) == [1, 2, 3, 4, 5]
    validate_features(frame, build.names)


def test_validation_rejects_a_row_stamped_at_or_after_kickoff() -> None:
    build = builder()
    frame = build.frame(build.rows(HISTORY))
    frame.loc[0, "as_of_utc"] = frame.loc[0, "kickoff_utc"]
    with pytest.raises(FeatureValidationError, match="as_of_utc"):
        validate_features(frame, build.names)


def test_validation_rejects_an_unknown_column() -> None:
    build = builder()
    frame = build.frame(build.rows(HISTORY))
    frame["surprise"] = 1.0
    with pytest.raises(FeatureValidationError):
        validate_features(frame, build.names)


def test_the_final_ratings_are_exposed_for_the_demo_output() -> None:
    build = builder()
    records: list[MatchRecord] = list(HISTORY)
    list(build.rows(records))
    assert set(build.final_ratings()) == {10, 20, 30, 40}

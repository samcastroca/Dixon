"""The guarded history view and the chronological replay engine."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from predictor.features.replay import (
    HistoryIndex,
    HistoryView,
    LeakageError,
    MatchRecord,
    ReplayEngine,
)
from tests.support import kickoff, record


def three_matches() -> list[MatchRecord]:
    """Two simultaneous matches on day 1, one later on day 8."""
    return [
        record(1, 10, 20, kickoff(1), 2, 0),
        record(2, 30, 40, kickoff(1), 1, 1),
        record(3, 10, 30, kickoff(8), 0, 3),
    ]


def view_at(day: int, hour: int = 15) -> HistoryView:
    return HistoryIndex.from_records(three_matches()).view(kickoff(day, hour))


def test_a_result_before_the_cut_off_is_readable() -> None:
    assert view_at(8).result_of(1) == (2, 0)


def test_a_result_at_the_cut_off_raises_leakage_error() -> None:
    view = HistoryIndex.from_records(three_matches()).view(kickoff(1))
    with pytest.raises(LeakageError, match="match 1"):
        view.result_of(1)


def test_a_result_after_the_cut_off_raises_leakage_error() -> None:
    with pytest.raises(LeakageError, match="match 3"):
        view_at(2).result_of(3)


def test_an_unknown_match_is_a_lookup_error_not_a_leak() -> None:
    with pytest.raises(KeyError):
        view_at(8).result_of(999)


def test_appearances_exclude_a_simultaneous_match() -> None:
    """The cut-off sits before both day-1 kickoffs, so neither team has any history."""
    view = HistoryIndex.from_records(three_matches()).view(kickoff(1))
    assert view.recent(10, "overall") == ()
    assert view.recent(30, "overall") == ()


def test_appearances_are_oriented_for_the_team_that_asks() -> None:
    view = view_at(8)
    (home,) = view.recent(10, "overall")
    (away,) = view.recent(20, "overall")
    assert (home.goals_for, home.goals_against, home.at_home) == (2, 0, True)
    assert (away.goals_for, away.goals_against, away.at_home) == (0, 2, False)


def test_the_venue_scope_only_holds_matches_at_that_venue() -> None:
    view = view_at(8)
    assert [entry.match_id for entry in view.recent(10, "home")] == [1]
    assert view.recent(10, "away") == ()
    assert [entry.match_id for entry in view.recent(20, "away")] == [1]


def test_recent_returns_the_most_recent_first_and_honours_the_limit() -> None:
    records = [
        record(1, 10, 20, kickoff(1), 1, 0),
        record(2, 10, 30, kickoff(2), 2, 0),
        record(3, 10, 40, kickoff(3), 3, 0),
    ]
    view = HistoryIndex.from_records(records).view(kickoff(9))
    assert [entry.match_id for entry in view.recent(10, "overall", limit=2)] == [3, 2]


def test_an_unplayed_match_never_enters_the_history() -> None:
    records = [
        record(1, 10, 20, kickoff(1), status="postponed"),
        record(2, 10, 30, kickoff(2), 4, 0),
    ]
    view = HistoryIndex.from_records(records).view(kickoff(9))
    assert [entry.match_id for entry in view.recent(10, "overall")] == [2]


def test_participants_and_previous_season_only_look_backwards() -> None:
    records = [
        record(1, 10, 20, kickoff(1), 1, 0, season_id=1, season_label="2019-20"),
        record(2, 10, 30, kickoff(20), 1, 0, season_id=2, season_label="2020-21"),
        record(3, 20, 30, kickoff(21), 1, 0, season_id=2, season_label="2020-21"),
    ]
    index = HistoryIndex.from_records(records)
    assert index.view(kickoff(20)).participants(1) == frozenset({10, 20})
    # Season 2 has not started at the first cut-off, so it has no known participants yet.
    assert index.view(kickoff(20)).participants(2) == frozenset()
    assert index.view(kickoff(21)).participants(2) == frozenset({10, 30})
    assert index.previous_season(2) == 1
    assert index.previous_season(1) is None


def test_matches_played_counts_only_that_season_before_the_cut_off() -> None:
    records = [
        record(1, 10, 20, kickoff(1), 1, 0, season_id=1),
        record(2, 10, 30, kickoff(2), 1, 0, season_id=1),
        record(3, 10, 40, kickoff(20), 1, 0, season_id=2),
    ]
    index = HistoryIndex.from_records(records)
    assert index.view(kickoff(20)).matches_played(10, 1) == 2
    assert index.view(kickoff(20)).matches_played(10, 2) == 0
    assert index.view(kickoff(2)).matches_played(10, 1) == 1


class Spy:
    """Minimal accumulator that records the cut-off it was offered for every match."""

    names = ("spy_known",)

    def __init__(self) -> None:
        self.seen: list[tuple[int, int]] = []

    def begin(self, batch: Sequence[MatchRecord], view: HistoryView) -> None:
        del batch, view

    def features(self, match: MatchRecord, view: HistoryView) -> dict[str, float | None]:
        known = len(view.recent(match.home_team_id, "overall"))
        self.seen.append((match.match_id, known))
        return {"spy_known": float(known)}

    def update(self, batch: Sequence[MatchRecord]) -> None:
        del batch


class Peeker:
    """An accumulator that tries to read the result of the match it is describing."""

    names = ("peeked",)

    def __init__(self) -> None:
        self.goals: list[tuple[int | None, int | None]] = []

    def begin(self, batch: Sequence[MatchRecord], view: HistoryView) -> None:
        del batch, view

    def features(self, match: MatchRecord, view: HistoryView) -> dict[str, float | None]:
        del view
        self.goals.append((match.home_goals, match.away_goals))
        return {"peeked": None}

    def update(self, batch: Sequence[MatchRecord]) -> None:
        del batch


def test_a_group_is_handed_the_fixture_and_never_its_own_result() -> None:
    peeker = Peeker()
    list(ReplayEngine(three_matches(), as_of_offset_seconds=1).run([peeker]))
    assert peeker.goals == [(None, None), (None, None), (None, None)]


def test_the_step_still_carries_the_real_match_for_the_caller() -> None:
    steps = list(ReplayEngine(three_matches(), as_of_offset_seconds=1).run([Spy()]))
    assert [(step.match.home_goals, step.match.away_goals) for step in steps] == [
        (2, 0),
        (1, 1),
        (0, 3),
    ]


def test_the_engine_walks_matches_in_kickoff_order() -> None:
    spy = Spy()
    engine = ReplayEngine(three_matches(), as_of_offset_seconds=1)
    steps = list(engine.run([spy]))
    assert [step.match.match_id for step in steps] == [1, 2, 3]


def test_the_cut_off_is_strictly_before_kickoff() -> None:
    engine = ReplayEngine(three_matches(), as_of_offset_seconds=1)
    for step in engine.run([Spy()]):
        assert step.as_of_utc < step.match.kickoff_utc


def test_simultaneous_matches_do_not_see_each_other() -> None:
    """Match 3 is later, so team 10 has one appearance; the day-1 pair has none."""
    spy = Spy()
    list(ReplayEngine(three_matches(), as_of_offset_seconds=1).run([spy]))
    assert spy.seen == [(1, 0), (2, 0), (3, 1)]


def test_the_input_order_inside_a_batch_does_not_change_the_output() -> None:
    forward = three_matches()
    reversed_batch = [forward[1], forward[0], forward[2]]
    spy_a, spy_b = Spy(), Spy()
    first = {
        step.match.match_id: step.values
        for step in ReplayEngine(forward, as_of_offset_seconds=1).run([spy_a])
    }
    second = {
        step.match.match_id: step.values
        for step in ReplayEngine(reversed_batch, as_of_offset_seconds=1).run([spy_b])
    }
    assert first == second

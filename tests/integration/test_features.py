"""The feature store against the compose database: leakage, isolation and determinism."""

from __future__ import annotations

import hashlib
import random
import time
from collections.abc import Iterator, Sequence
from datetime import datetime

import pytest
from sqlalchemy import func, select
from tests.integration.test_process import snapshot

from predictor import tables
from predictor.config import Competition, get_settings
from predictor.db import session_scope
from predictor.features import repository
from predictor.features.builder import FeatureBuilder
from predictor.features.replay import MatchRecord
from predictor.features.service import build_features

pytestmark = pytest.mark.integration

COMPETITIONS = ("EPL", "LALIGA")

#: The spec asks for 200 matches; the seed keeps the sample the same on every run.
TRUNCATION_SAMPLE = 200
TRUNCATION_SEED = 20260926

#: Both leagues, full history, must rebuild well inside this budget (spec section 10).
REBUILD_BUDGET_SECONDS = 300.0


@pytest.fixture(autouse=True)
def _needs_database(database_available: bool) -> Iterator[None]:
    if not database_available:
        pytest.skip("no database reachable; run `make up` or set DATABASE_URL")
    yield


@pytest.fixture(scope="module")
def competitions() -> list[Competition]:
    settings = get_settings()
    return [settings.competition(code) for code in COMPETITIONS]


@pytest.fixture(scope="module", autouse=True)
def built(database_available: bool) -> None:
    """Build the whole feature store once for the module."""
    if not database_available:
        pytest.skip("no database reachable; run `make up` or set DATABASE_URL")
    settings = get_settings()
    build_features([settings.competition(code) for code in COMPETITIONS])


def records_of(competition: Competition) -> list[MatchRecord]:
    settings = get_settings()
    with session_scope() as session:
        return repository.load_records(session, competition, settings)


def builder_for(competition: Competition) -> FeatureBuilder:
    return FeatureBuilder.for_competition(competition, get_settings())


def truncated(records: Sequence[MatchRecord], at: datetime) -> list[MatchRecord]:
    """History exactly as it stood one moment before `at`.

    Everything kicking off later simply did not exist yet; anything kicking off at the same
    instant existed as a fixture whose result nobody knew.
    """
    history: list[MatchRecord] = []
    for entry in records:
        if entry.kickoff_utc > at:
            break
        history.append(entry if entry.kickoff_utc < at else entry.as_fixture())
    return history


def test_every_stored_row_is_stamped_strictly_before_kickoff() -> None:
    with session_scope() as session:
        total = session.scalar(select(func.count()).select_from(tables.Feature))
        leaking = session.scalar(
            select(func.count())
            .select_from(tables.Feature)
            .join(tables.Match, tables.Match.id == tables.Feature.match_id)
            .where(tables.Feature.as_of_utc >= tables.Match.kickoff_utc)
        )
    assert total
    assert leaking == 0


def test_there_is_one_row_per_match_and_version() -> None:
    with session_scope() as session:
        matches = session.scalar(select(func.count()).select_from(tables.Match))
        rows = session.scalar(select(func.count()).select_from(tables.Feature))
        versions: list[str] = list(
            session.scalars(select(func.distinct(tables.Feature.feature_version)))
        )
        checksums: list[str] = list(
            session.scalars(select(func.distinct(tables.Feature.definition_checksum)))
        )
    assert rows == matches
    assert versions == [get_settings().features.version]
    assert len(checksums) == 1


def test_features_survive_truncating_the_history_at_kickoff(
    competitions: list[Competition],
) -> None:
    """Acceptance test 1: the point-in-time promise, on 200 matches across both leagues."""
    rng = random.Random(TRUNCATION_SEED)  # noqa: S311 - a reproducible sample, not a secret
    per_competition = TRUNCATION_SAMPLE // len(competitions)

    for competition in competitions:
        records = records_of(competition)
        full = {row.match_id: row.values for row in builder_for(competition).rows(records)}
        for match in rng.sample(records, per_competition):
            history = truncated(records, match.kickoff_utc)
            rebuilt = next(
                row
                for row in builder_for(competition).rows(history)
                if row.match_id == match.match_id
            )
            assert rebuilt.values == full[match.match_id], (
                f"{competition.code} match {match.match_id} at {match.kickoff_utc}"
            )


def test_a_loader_never_crosses_competitions(competitions: list[Competition]) -> None:
    ids = {
        competition.code: {record.match_id for record in records_of(competition)}
        for competition in competitions
    }
    first, second = (ids[competition.code] for competition in competitions)
    assert first and second
    assert not first & second
    with session_scope() as session:
        total = session.scalar(select(func.count()).select_from(tables.Match))
    assert len(first) + len(second) == total


def test_one_league_is_unaffected_by_the_other(competitions: list[Competition]) -> None:
    """Acceptance test 5: rebuilding a single league reproduces its rows byte for byte."""
    before = snapshot("features")
    for competition in competitions:
        build_features([competition])
        assert snapshot("features") == before, competition.code


def test_building_twice_leaves_the_table_identical(competitions: list[Competition]) -> None:
    before = snapshot("features")
    build_features(competitions)
    assert snapshot("features") == before


def test_the_stored_values_hash_to_the_same_digest_as_a_fresh_build(
    competitions: list[Competition],
) -> None:
    for competition in competitions:
        rebuilt = hashlib.sha256()
        for row in builder_for(competition).rows(records_of(competition)):
            rebuilt.update(repr((row.match_id, sorted(row.values.items()))).encode("utf-8"))

        with session_scope() as session:
            stored_rows = session.execute(
                select(tables.Feature.match_id, tables.Feature.feature_values)
                .join(tables.Match, tables.Match.id == tables.Feature.match_id)
                .where(tables.Match.competition_id == _competition_id(competition))
                .order_by(tables.Match.kickoff_utc, tables.Feature.match_id)
            ).all()
        stored = hashlib.sha256()
        for match_id, values in stored_rows:
            stored.update(repr((match_id, sorted(values.items()))).encode("utf-8"))

        assert stored.hexdigest() == rebuilt.hexdigest(), competition.code


def _competition_id(competition: Competition) -> int:
    with session_scope() as session:
        found = session.scalar(
            select(tables.Competition.id).where(tables.Competition.code == competition.code)
        )
    assert found is not None
    return found


@pytest.mark.slow
def test_a_full_rebuild_of_both_leagues_fits_the_budget(competitions: list[Competition]) -> None:
    started = time.monotonic()
    report = build_features(competitions)
    elapsed = time.monotonic() - started
    assert report.rows
    assert elapsed < REBUILD_BUDGET_SECONDS, f"rebuild took {elapsed:.1f}s"

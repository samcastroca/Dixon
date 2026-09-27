"""Entity resolution: every source spelling maps to exactly one canonical team."""

from __future__ import annotations

import csv
import io

import pytest

from predictor.processing.entities import (
    AliasFileError,
    TeamResolver,
    UnknownTeamError,
    normalise_key,
)
from tests.support import RECORDED, recorded_bytes


@pytest.fixture(scope="module")
def resolver() -> TeamResolver:
    return TeamResolver.from_file()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Atlético de Madrid", "atletico de madrid"),
        ("  Real   Betis  ", "real betis"),
        ("Nott'm Forest", "nottm forest"),
        ("RCD Espanyol", "rcd espanyol"),
        ("Brighton & Hove Albion", "brighton hove albion"),
    ],
)
def test_normalise_key_strips_accents_punctuation_and_whitespace(raw: str, expected: str) -> None:
    assert normalise_key(raw) == expected


@pytest.mark.parametrize(
    ("competition", "spelling", "canonical"),
    [
        ("LALIGA", "Ath Madrid", "Atlético de Madrid"),
        ("LALIGA", "Atletico Madrid", "Atlético de Madrid"),
        ("LALIGA", "Atlético Madrid", "Atlético de Madrid"),
        ("LALIGA", "Betis", "Real Betis"),
        ("LALIGA", "Sociedad", "Real Sociedad"),
        ("LALIGA", "Espanol", "RCD Espanyol"),
        ("EPL", "Man United", "Manchester United"),
        ("EPL", "Nott'm Forest", "Nottingham Forest"),
    ],
)
def test_known_spellings_resolve_to_the_official_name(
    resolver: TeamResolver, competition: str, spelling: str, canonical: str
) -> None:
    assert resolver.resolve(competition, spelling) == canonical


def test_accented_and_unaccented_spellings_are_the_same_team(resolver: TeamResolver) -> None:
    assert resolver.resolve("LALIGA", "Atletico de Madrid") == resolver.resolve(
        "LALIGA", "Atlético de Madrid"
    )
    assert resolver.resolve("LALIGA", "Malaga") == resolver.resolve("LALIGA", "Málaga")


def test_every_alias_resolves_to_exactly_one_team_per_competition(resolver: TeamResolver) -> None:
    for competition in resolver.competitions:
        for alias, canonical in resolver.aliases(competition).items():
            assert resolver.resolve(competition, alias) == canonical
        # The canonical names themselves are aliases too, and none of them collides.
        keys = [normalise_key(name) for name in resolver.canonical_names(competition)]
        assert len(keys) == len(set(keys))


def test_an_unknown_spelling_raises_and_names_the_competition(resolver: TeamResolver) -> None:
    with pytest.raises(UnknownTeamError) as error:
        resolver.resolve("EPL", "Real Madrid")
    assert "Real Madrid" in str(error.value)
    assert "EPL" in str(error.value)


def test_resolution_is_keyed_by_competition() -> None:
    resolver = TeamResolver.from_mapping(
        {
            "A": {"Athletic Club": ["Athletic"]},
            "B": {"Club Athletico": ["Athletic"]},
        }
    )
    assert resolver.resolve("A", "Athletic") == "Athletic Club"
    assert resolver.resolve("B", "Athletic") == "Club Athletico"


def test_an_alias_shared_by_two_teams_of_one_competition_is_rejected() -> None:
    with pytest.raises(AliasFileError, match="Rovers"):
        TeamResolver.from_mapping({"A": {"Team One": ["Rovers"], "Team Two": ["Rovers"]}})


def test_an_unknown_competition_fails_loudly(resolver: TeamResolver) -> None:
    with pytest.raises(KeyError, match="NOPE"):
        resolver.resolve("NOPE", "Arsenal")


def test_the_alias_file_covers_every_team_in_the_recorded_fixtures(
    resolver: TeamResolver,
) -> None:
    for competition, season in RECORDED:
        text = recorded_bytes(competition, season).decode("utf-8-sig")
        for row in csv.DictReader(io.StringIO(text, newline="")):
            for column in ("HomeTeam", "AwayTeam"):
                name = (row.get(column) or "").strip()
                if name:
                    resolver.resolve(competition, name)


def test_the_shipped_alias_file_is_the_one_next_to_the_settings(resolver: TeamResolver) -> None:
    assert resolver.path is not None
    assert resolver.path.name == "team_aliases.yaml"
    assert set(resolver.competitions) == {"EPL", "LALIGA"}

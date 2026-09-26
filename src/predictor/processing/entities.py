"""Entity resolution: every source spelling of a team maps to one canonical name.

Matching happens on a normalised key (lowercase, accents stripped, punctuation and extra
whitespace removed), keyed by competition, so two leagues may use the same short name for
different clubs. Unknown spellings are logged and raise: they are never guessed.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from predictor.config import team_aliases_file_path
from predictor.logging import get_logger

logger = get_logger(__name__)

#: Apostrophes sit inside a word ("Nott'm" -> "nottm"); every other symbol separates words.
_APOSTROPHES = re.compile("['\u2019\u02bc]")
_SEPARATORS = re.compile(r"[\W_]+", re.UNICODE)


class UnknownTeamError(LookupError):
    """A source used a spelling that no alias covers (spec section 6.2: fail loudly)."""

    def __init__(self, competition: str, name: str) -> None:
        super().__init__(
            f"unknown team {name!r} in competition {competition}: "
            "add it to config/team_aliases.yaml"
        )
        self.competition = competition
        self.name = name


class AliasFileError(ValueError):
    """The alias seed file is malformed or maps one spelling to two teams."""


def normalise_key(name: str) -> str:
    """Return the key that two spellings of the same team share."""
    decomposed = unicodedata.normalize("NFKD", name.casefold())
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    return _SEPARATORS.sub(" ", _APOSTROPHES.sub("", without_accents)).strip()


class TeamResolver:
    """Canonical team names per competition, resolved from every configured spelling."""

    def __init__(
        self,
        aliases: Mapping[str, Mapping[str, str]],
        canonical: Mapping[str, Sequence[str]],
        path: Path | None = None,
        spellings: Mapping[str, Mapping[str, str]] | None = None,
    ) -> None:
        self._aliases = {code: dict(entries) for code, entries in aliases.items()}
        self._canonical = {code: tuple(names) for code, names in canonical.items()}
        self._spellings = {code: dict(entries) for code, entries in (spellings or {}).items()}
        self.path = path

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Mapping[str, Sequence[str]]], path: Path | None = None
    ) -> TeamResolver:
        """Build a resolver from `{competition: {canonical name: [alias, ...]}}`."""
        aliases: dict[str, dict[str, str]] = {}
        canonical: dict[str, list[str]] = {}
        spelling_of: dict[str, dict[str, str]] = {}
        for competition, teams in data.items():
            entries: dict[str, str] = {}
            written: dict[str, str] = {}
            for name, spellings in teams.items():
                for spelling in (name, *spellings):
                    key = normalise_key(spelling)
                    if not key:
                        msg = f"{competition}: empty alias for {name!r}"
                        raise AliasFileError(msg)
                    owner = entries.setdefault(key, name)
                    if owner != name:
                        msg = (
                            f"{competition}: alias {spelling!r} maps to both {owner!r} and {name!r}"
                        )
                        raise AliasFileError(msg)
                    written.setdefault(key, spelling)
            aliases[competition] = entries
            canonical[competition] = sorted(teams)
            spelling_of[competition] = written
        return cls(aliases, canonical, path, spelling_of)

    @classmethod
    def from_file(cls, path: Path | None = None) -> TeamResolver:
        """Load the alias seed file that sits next to the settings file."""
        resolved = path or team_aliases_file_path()
        try:
            raw: Any = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        except OSError as exc:
            msg = f"cannot read the alias file {resolved}"
            raise AliasFileError(msg) from exc
        if not isinstance(raw, dict) or not raw:
            msg = f"{resolved}: expected a mapping of competition code to teams"
            raise AliasFileError(msg)
        for competition, teams in raw.items():
            if not isinstance(teams, dict) or not teams:
                msg = f"{resolved}: competition {competition} must map canonical names to aliases"
                raise AliasFileError(msg)
            for name, spellings in teams.items():
                if not isinstance(spellings, list) or not all(
                    isinstance(item, str) for item in spellings
                ):
                    msg = f"{resolved}: the aliases of {name!r} must be a list of strings"
                    raise AliasFileError(msg)
        return cls.from_mapping(raw, resolved)

    @property
    def competitions(self) -> tuple[str, ...]:
        return tuple(self._aliases)

    def aliases(self, competition: str) -> Mapping[str, str]:
        """Normalised key -> canonical name, for one competition."""
        return dict(self._entries(competition))

    def alias_rows(self, competition: str) -> tuple[tuple[str, str, str], ...]:
        """(spelling, normalised key, canonical name) for every alias of one competition."""
        entries = self._entries(competition)
        spellings = self._spellings.get(competition, {})
        return tuple(
            (spellings.get(key, canonical), key, canonical) for key, canonical in entries.items()
        )

    def canonical_names(self, competition: str) -> tuple[str, ...]:
        self._entries(competition)
        return self._canonical[competition]

    def resolve(self, competition: str, name: str) -> str:
        """Return the canonical team name, raising `UnknownTeamError` on an unknown spelling."""
        entries = self._entries(competition)
        canonical = entries.get(normalise_key(name))
        if canonical is None:
            logger.error("entities.unknown_team", competition=competition, name=name)
            raise UnknownTeamError(competition, name)
        return canonical

    def _entries(self, competition: str) -> Mapping[str, str]:
        try:
            return self._aliases[competition]
        except KeyError as exc:
            known = ", ".join(sorted(self._aliases)) or "none"
            msg = f"no aliases for competition {competition!r}; configured: {known}"
            raise KeyError(msg) from exc

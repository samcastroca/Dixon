"""Synthetic seasons drawn from known parameters, for the recovery tests of phase 5.

The acceptance tests of spec section 11 ask whether a fit recovers the attack, defence, home
advantage and rho it was generated from. That needs data whose truth is known, which no real
league provides, so this module generates it.

Matches come out as `MatchRecord`s and reach a model through `evaluation.frames.matches_frame`,
exactly as the walk-forward engine builds them. A test therefore exercises the real frame
contract rather than a convenient parallel one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import numpy.typing as npt
from scipy.stats import poisson

from predictor.features.replay import MatchRecord, TeamStats
from predictor.models.poisson import TeamStrength
from predictor.processing.clean import MatchStatus

#: The grid the joint scoreline is sampled from. Far past anything a football match produces.
SAMPLE_MAX_GOALS = 25

#: Where a simulated history starts. Any instant works; a fixed one keeps tests reproducible.
EPOCH = datetime(2015, 8, 8, 15, 0, tzinfo=UTC)

#: Two league fixtures a week is the pace of a real season, and it makes the decay of M3 bite.
MATCHES_PER_WEEK = timedelta(days=7)


@dataclass(frozen=True, slots=True)
class LeagueParameters:
    """The truth a simulated league is generated from (spec section 7, M2 and M3).

    `log lambda_home = intercept + home_advantage + attack[home] + defence[away]`
    `log lambda_away = intercept + attack[away] + defence[home]`

    Attack and defence are stored as given; `centred()` returns the sum-to-zero version, which
    is the only one a fit can identify and therefore the only one worth comparing against.
    """

    attack: tuple[float, ...]
    defence: tuple[float, ...]
    intercept: float = 0.1
    home_advantage: float = 0.25
    rho: float = 0.0

    def __post_init__(self) -> None:
        if len(self.attack) != len(self.defence):
            msg = "a league needs one attack and one defence per team"
            raise ValueError(msg)
        if len(self.attack) < 2:
            msg = "a league needs at least two teams"
            raise ValueError(msg)

    @property
    def teams(self) -> int:
        return len(self.attack)

    def centred(self) -> LeagueParameters:
        """The same league under the sum-to-zero constraint, with the shift moved into mu.

        Adding a constant to every attack and subtracting it from the intercept describes the
        identical league, so a fit can only ever recover the centred version. Comparing against
        anything else would be comparing against a coordinate system, not against the truth.
        """
        attack = np.asarray(self.attack, dtype=np.float64)
        defence = np.asarray(self.defence, dtype=np.float64)
        return LeagueParameters(
            attack=tuple((attack - attack.mean()).tolist()),
            defence=tuple((defence - defence.mean()).tolist()),
            intercept=self.intercept + float(attack.mean()) + float(defence.mean()),
            home_advantage=self.home_advantage,
            rho=self.rho,
        )

    def rates(self, home: int, away: int) -> tuple[float, float]:
        """The two goal rates of one fixture, by team index."""
        return (
            float(
                np.exp(
                    self.intercept + self.home_advantage + self.attack[home] + self.defence[away]
                )
            ),
            float(np.exp(self.intercept + self.attack[away] + self.defence[home])),
        )

    def as_mapping(self) -> Mapping[str, float]:
        """The scalar truth, keyed the way a fitted model reports it."""
        return {
            "intercept": self.intercept,
            "home_advantage": self.home_advantage,
            "rho": self.rho,
        }


def balanced_league(
    teams: int,
    *,
    spread: float = 0.35,
    intercept: float = 0.1,
    home_advantage: float = 0.25,
    rho: float = 0.0,
) -> LeagueParameters:
    """A league whose strengths fan out evenly around the average, already sum-to-zero.

    Evenly spaced strengths make a recovery test informative: every team's parameter is
    distinguishable from its neighbour's, so a fit that merely predicts the mean fails.
    """
    steps = np.linspace(-1.0, 1.0, teams)
    return LeagueParameters(
        attack=tuple((steps * spread).tolist()),
        # Reversed, so the best attack also defends best and the ranking is unambiguous.
        defence=tuple((-steps * spread * 0.8).tolist()),
        intercept=intercept,
        home_advantage=home_advantage,
        rho=rho,
    )


def joint_scoreline(lambda_home: float, lambda_away: float, rho: float) -> npt.NDArray[np.float64]:
    """The exact Dixon-Coles joint distribution over a grid wide enough to hold all the mass.

    Sampling from the joint rather than from two independent Poissons is what makes `rho`
    actually present in the data; with rho = 0 it reduces to the independent case.
    """
    goals = np.arange(SAMPLE_MAX_GOALS + 1)
    home = _poisson_pmf(goals, lambda_home)
    away = _poisson_pmf(goals, lambda_away)
    matrix = np.outer(home, away)
    if rho:
        matrix[0, 0] *= 1.0 - lambda_home * lambda_away * rho
        matrix[0, 1] *= 1.0 + lambda_home * rho
        matrix[1, 0] *= 1.0 + lambda_away * rho
        matrix[1, 1] *= 1.0 - rho
        if matrix.min() < 0.0:
            msg = f"rho={rho} makes the joint distribution negative at these rates"
            raise ValueError(msg)
    return np.asarray(matrix / matrix.sum(), dtype=np.float64)


def _poisson_pmf(goals: npt.NDArray[np.int_], rate: float) -> npt.NDArray[np.float64]:
    return np.asarray(poisson.pmf(goals, rate), dtype=np.float64)


def round_robin(teams: int) -> tuple[tuple[int, int], ...]:
    """Every ordered pair of distinct teams: one home fixture and one away per opponent."""
    return tuple((home, away) for home in range(teams) for away in range(teams) if home != away)


def simulate_seasons(
    parameters: LeagueParameters,
    *,
    seasons: int,
    seed: int,
    competition_id: int = 1,
    first_season: int = 1,
    first_match_id: int = 1,
    start: datetime = EPOCH,
    newcomers_per_season: int = 0,
) -> tuple[MatchRecord, ...]:
    """A double round robin per season, sampled from the league's own joint distribution.

    `newcomers_per_season` retires the weakest teams at the end of each season and replaces
    them with brand new ids, which is how the promoted-team rule of spec 7.1 gets exercised.
    """
    rng = np.random.default_rng(seed)
    records: list[MatchRecord] = []
    match_id = first_match_id
    kickoff = start
    # Index into `parameters` -> the team id currently occupying that slot.
    team_ids = list(range(1, parameters.teams + 1))
    next_team_id = parameters.teams + 1
    # The weakest slots by overall strength, which is who goes down.
    ranking = np.argsort(np.asarray(parameters.attack) - np.asarray(parameters.defence))

    for season_offset in range(seasons):
        season_id = first_season + season_offset
        season_label = f"{2015 + season_offset}-{str(16 + season_offset).zfill(2)}"
        for home, away in round_robin(parameters.teams):
            lambda_home, lambda_away = parameters.rates(home, away)
            matrix = joint_scoreline(lambda_home, lambda_away, parameters.rho)
            drawn = rng.choice(matrix.size, p=matrix.reshape(-1))
            home_goals, away_goals = divmod(int(drawn), matrix.shape[1])
            records.append(
                MatchRecord(
                    match_id=match_id,
                    competition_id=competition_id,
                    season_id=season_id,
                    season_label=season_label,
                    kickoff_utc=kickoff,
                    home_team_id=team_ids[home],
                    away_team_id=team_ids[away],
                    home_goals=home_goals,
                    away_goals=away_goals,
                    status=MatchStatus.PLAYED.value,
                    home_stats=TeamStats(shots=None, shots_on_target=None),
                    away_stats=TeamStats(shots=None, shots_on_target=None),
                )
            )
            match_id += 1
            kickoff += MATCHES_PER_WEEK
        for slot in ranking[:newcomers_per_season]:
            team_ids[int(slot)] = next_team_id
            next_team_id += 1
    return tuple(records)


def as_fixtures(records: Sequence[MatchRecord]) -> tuple[MatchRecord, ...]:
    """The same matches with every result stripped, as the engine hands them to a model."""
    return tuple(record.as_fixture() for record in records)


@dataclass(frozen=True, slots=True)
class Recovery:
    """How well a set of independent fits found the league they were generated from.

    Acceptance test 1 asks whether a fit "recovers" known parameters, and one fit on one
    simulation cannot answer that: a twenty-team league over three seasons gives each team's
    strength a standard error near 0.08, so a single draw lands two standard errors out often
    enough to make such a test a coin flip rather than a check.

    So the recovery is measured over several independent simulations. `bias_*` averages the
    estimates first and asks whether the *estimator* is centred on the truth, which is the
    real question. `worst_fit_rmse` and `worst_correlation` keep the individual fits honest.
    """

    home_advantage: float
    intercept: float
    rho: float
    #: RMSE of the averaged strengths against the truth: the bias that survives averaging.
    bias_rmse: float
    #: The least accurate single fit, which is dominated by variance rather than bias.
    worst_fit_rmse: float
    #: The weakest attack-ranking agreement any single fit achieved.
    worst_correlation: float


def recovery(truth: LeagueParameters, strengths: Sequence[TeamStrength]) -> Recovery:
    """Summarise a set of fits against the truth, one `Recovery` per model under test."""
    if not strengths:
        msg = "a recovery needs at least one fit"
        raise ValueError(msg)
    teams = [team + 1 for team in range(truth.teams)]
    attack = np.array([[strength.attack[team] for team in teams] for strength in strengths])
    defence = np.array([[strength.defence[team] for team in teams] for strength in strengths])
    true_attack = np.asarray(truth.attack, dtype=np.float64)
    true_defence = np.asarray(truth.defence, dtype=np.float64)
    true_both = np.concatenate([true_attack, true_defence])

    averaged = np.concatenate([attack.mean(axis=0), defence.mean(axis=0)])
    per_fit = [
        float(np.sqrt(np.mean(np.square(np.concatenate([one, other]) - true_both))))
        for one, other in zip(attack, defence, strict=True)
    ]
    return Recovery(
        home_advantage=float(np.mean([strength.home_advantage for strength in strengths])),
        intercept=float(np.mean([strength.intercept for strength in strengths])),
        rho=float(np.mean([strength.rho for strength in strengths])),
        bias_rmse=float(np.sqrt(np.mean(np.square(averaged - true_both)))),
        worst_fit_rmse=max(per_fit),
        worst_correlation=min(float(np.corrcoef(one, true_attack)[0, 1]) for one in attack),
    )

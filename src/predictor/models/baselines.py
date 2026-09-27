"""The M0 baselines of spec section 7: an honest floor and an honest ceiling.

M0a is the floor. Knowing nothing but how often home teams win, it is the number any real
model has to beat before it is worth anything.

M0b is the ceiling, and it is a benchmark rather than a model: closing prices already contain
everyone else's information, so beating them is the hard problem. It is marked undeployable
because its input only exists after the market has closed, and because this source publishes
no capture time for it (see `backtest.market_baseline` in settings.yaml).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from predictor.evaluation.frames import MARKET_COLUMNS
from predictor.evaluation.metrics import AWAY, DRAW, HOME, OUTCOME_COUNT
from predictor.models.base import ScoreMatrices, probability_frame

#: What a model with nothing to go on says.
UNIFORM: tuple[float, float, float] = (1.0 / OUTCOME_COUNT,) * 3

_NO_SCORELINE = "this baseline has no scoreline distribution; those arrive with phase 5"


class FrequencyBaseline:
    """M0a: the home, draw and away frequencies of the training matches, and nothing else."""

    name = "baseline_frequency"
    deployable = True
    needs_market = False

    __slots__ = ("_frequencies",)

    def __init__(self) -> None:
        self._frequencies: tuple[float, float, float] = UNIFORM

    @property
    def frequencies(self) -> tuple[float, float, float]:
        return self._frequencies

    def fit(self, matches: pd.DataFrame, features: pd.DataFrame | None) -> FrequencyBaseline:
        del features
        outcomes = matches["outcome"].dropna().astype(int)
        counts = np.array(
            [float((outcomes == outcome).sum()) for outcome in (HOME, DRAW, AWAY)],
            dtype=np.float64,
        )
        total = float(counts.sum())
        if total == 0.0:
            # No history at all means no opinion, which is the uniform distribution.
            self._frequencies = UNIFORM
        else:
            shares = counts / total
            self._frequencies = (float(shares[0]), float(shares[1]), float(shares[2]))
        return self

    def predict_proba(self, fixtures: pd.DataFrame, features: pd.DataFrame | None) -> pd.DataFrame:
        del features
        match_ids = fixtures["match_id"].astype(int).tolist()
        return probability_frame(match_ids, [self._frequencies] * len(match_ids))

    def predict_scores(
        self, fixtures: pd.DataFrame, features: pd.DataFrame | None = None, max_goals: int = 10
    ) -> ScoreMatrices:
        del fixtures, features, max_goals
        raise NotImplementedError(_NO_SCORELINE)


class MarketBaseline:
    """M0b: the de-margined closing quote, repeated back as a forecast. A benchmark only."""

    name = "baseline_market"
    deployable = False
    needs_market = True

    __slots__ = ()

    def fit(self, matches: pd.DataFrame, features: pd.DataFrame | None) -> MarketBaseline:
        del matches, features  # the market did the fitting
        return self

    def predict_proba(self, fixtures: pd.DataFrame, features: pd.DataFrame | None) -> pd.DataFrame:
        del features
        missing = [column for column in MARKET_COLUMNS if column not in fixtures]
        if missing:
            msg = (
                f"{self.name} needs the market columns {missing}; the engine only builds them "
                "for a model that declares needs_market"
            )
            raise ValueError(msg)
        quoted = fixtures.dropna(subset=list(MARKET_COLUMNS))
        values = quoted.loc[:, list(MARKET_COLUMNS)].to_numpy(dtype=np.float64)
        if values.size:
            # The stored distribution is already de-margined; this only removes rounding drift.
            values = values / values.sum(axis=1, keepdims=True)
        return probability_frame(quoted["match_id"].astype(int).tolist(), values)

    def predict_scores(
        self, fixtures: pd.DataFrame, features: pd.DataFrame | None = None, max_goals: int = 10
    ) -> ScoreMatrices:
        del fixtures, features, max_goals
        raise NotImplementedError(_NO_SCORELINE)

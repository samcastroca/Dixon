"""The optional pre-match market feature group (spec section 6.3).

Off by default, and for a concrete reason: football-data.co.uk publishes no capture time for
its prices, so nothing in the data proves a quote existed before kickoff. Closing odds almost
certainly did, but "almost certainly" is not the standard this phase holds itself to, so using
them has to be written down in settings.yaml rather than assumed.

The de-margining itself already happened in phase 2 (`predictor.processing.market`); this
group only picks which distribution to expose.
"""

from __future__ import annotations

from collections.abc import Sequence

from predictor.config import MarketFeatureSettings, Settings, get_settings
from predictor.features.replay import FeatureValues, HistoryView, LeakageError, MatchRecord

NAMES = ("market_p_home", "market_p_draw", "market_p_away")


class MarketFeatures:
    """Pre-match de-margined probabilities, refused whenever they cannot be proven pre-match."""

    __slots__ = ("_settings",)

    def __init__(self, settings: MarketFeatureSettings) -> None:
        self._settings = settings

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> MarketFeatures:
        return cls((settings or get_settings()).features.market)

    @property
    def names(self) -> tuple[str, ...]:
        return NAMES

    @property
    def settings(self) -> MarketFeatureSettings:
        return self._settings

    def begin(self, batch: Sequence[MatchRecord], view: HistoryView) -> None:
        del batch, view

    def features(self, match: MatchRecord, view: HistoryView) -> FeatureValues:
        quote = match.market
        if quote is None:
            return dict.fromkeys(NAMES)
        if quote.captured_at is None:
            if not self._settings.allow_missing_captured_at:
                msg = (
                    f"match {match.match_id} has a market quote without captured_at, so it "
                    "cannot be proven to predate kickoff; set "
                    "features.market.allow_missing_captured_at to use it anyway"
                )
                raise LeakageError(msg)
        elif quote.captured_at >= view.as_of:
            msg = (
                f"match {match.match_id} has a market quote captured at "
                f"{quote.captured_at.isoformat()}, which is not before the cut-off "
                f"{view.as_of.isoformat()}"
            )
            raise LeakageError(msg)
        return {
            "market_p_home": quote.p_home,
            "market_p_draw": quote.p_draw,
            "market_p_away": quote.p_away,
        }

    def update(self, batch: Sequence[MatchRecord]) -> None:
        del batch  # a quote belongs to its own match; nothing accumulates

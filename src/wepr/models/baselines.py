"""The two baselines every model is measured against.

**Climatology** — the historical mean for that day of year. Computed with an
expanding window so each year's value uses only prior years, which keeps it
legal as a feature as well as a baseline. A smoothing window of a few days either
side is applied, because a single calendar day across 75 years is only 75 samples
and is noisier than the underlying climate.

**Persistence** — tomorrow equals today. Trivial, and at one-day range genuinely
hard to beat: day-to-day temperature autocorrelation in Iberia is high. A
next-day model that does not beat persistence has learned nothing useful, and
saying so plainly is more valuable than hiding it.

Implemented in Phase 3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import features

# Half-width of the day-of-year smoothing window, in days.
CLIMATOLOGY_WINDOW = 7


def _calendar_slot(index: pd.DatetimeIndex) -> np.ndarray:
    """A stable 1..365 calendar slot with the leap-year drift removed.

    A raw `day_of_year` drifts by one across leap years — 1 March is day 60 in
    1951 and day 61 in 1952 — so pooling on it silently averages 29 February with
    1 March and smears every date after February by a day. Subtracting one from
    post-February days in leap years pins each slot to an actual calendar date.

    29 February lands on the same slot as 1 March, which is deliberate: it gives
    that date a full-sized sample instead of a one-year-in-four slot of its own,
    and climatologically the two days are indistinguishable.
    """
    doy = np.asarray(index.day_of_year)
    leap = np.asarray(index.is_leap_year)
    after_february = np.asarray(index.month) > 2
    return np.where(leap & after_february, doy - 1, doy)


def _circular_window_sum(matrix: pd.DataFrame, half_width: int) -> pd.DataFrame:
    """Sum each row over a slot window of `2 * half_width + 1`, wrapping the year.

    Wrapping matters at the year boundary: the window for 2 January has to reach
    back into late December, and a non-circular window would quietly compute those
    dates from half the samples. Wrapping *within a row* is safe because a row is
    one whole year, and the expanding step below then drops the current year
    entirely — so the late-December values an early-January window picks up always
    come from an earlier year.

    Implemented as a difference of cumulative sums: one pass regardless of window
    width, where the obvious nested loop would be 365 x width per year.
    """
    values = matrix.to_numpy(dtype="float64")
    if half_width == 0:
        # Not merely an optimisation: `values[:, -0:]` is the whole array, not an
        # empty slice, so the general path below would double the year.
        return pd.DataFrame(values, index=matrix.index, columns=matrix.columns)

    width = 2 * half_width + 1
    padded = np.concatenate(
        [values[:, -half_width:], values, values[:, :half_width]], axis=1
    )
    cumulative = np.cumsum(padded, axis=1)
    cumulative = np.concatenate(
        [np.zeros((cumulative.shape[0], 1)), cumulative], axis=1
    )
    windowed = cumulative[:, width:] - cumulative[:, :-width]
    return pd.DataFrame(windowed, index=matrix.index, columns=matrix.columns)


def _year_slot_totals(
    series: pd.Series, half_width: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Windowed sums and counts per (year, calendar slot).

    Sums and counts are carried separately, rather than a mean per year, because
    the quantity wanted is the mean over *all* pooled observations. Averaging
    per-year means afterwards would weight a year with two observations in the
    window the same as one with fifteen.
    """
    index = pd.DatetimeIndex(series.index)
    frame = pd.DataFrame(
        {
            "year": np.asarray(index.year),
            "slot": _calendar_slot(index),
            "value": series.to_numpy(dtype="float64"),
        }
    )
    grouped = frame.groupby(["year", "slot"])["value"]
    slots = pd.RangeIndex(1, 366, name="slot")
    sums = grouped.sum(min_count=1).unstack("slot").reindex(columns=slots)
    counts = grouped.count().unstack("slot").reindex(columns=slots)

    return (
        _circular_window_sum(sums.fillna(0.0), half_width),
        _circular_window_sum(counts.fillna(0.0), half_width),
    )


def climatology(
    series: pd.Series,
    window: int = CLIMATOLOGY_WINDOW,
    expanding: bool = True,
) -> pd.Series:
    """Day-of-year historical mean, aligned to `series`' index.

    Args:
        series: Daily observations indexed by date.
        window: Half-width of the day-of-year smoothing window.
        expanding: If True, each date uses only prior years — required when the
            result is used as a feature or evaluated as a baseline. If False, the
            full-record mean is used, which is appropriate only for display.

    Returns:
        The climatological expectation for each date in the index.
    """
    if window < 0:
        raise ValueError(f"window must be non-negative, got {window}")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("climatology needs a DatetimeIndex")

    index = pd.DatetimeIndex(series.index)
    slot = _calendar_slot(index)
    windowed_sums, windowed_counts = _year_slot_totals(series, window)

    if expanding:
        # A year contributes to a slot once it has any observation in that slot's
        # window; `shift(1)` down the year axis is what makes the mean use
        # strictly earlier years, which is the whole reason this is legal as both
        # a baseline and a feature.
        contributing = (windowed_counts > 0).astype("float64")
        prior_sums = windowed_sums.cumsum().shift(1)
        prior_counts = windowed_counts.cumsum().shift(1)
        prior_years = contributing.cumsum().shift(1)

        with np.errstate(invalid="ignore", divide="ignore"):
            means = prior_sums / prior_counts
        # Too few prior years is reported as NaN rather than as a norm averaged
        # from two years, matching `features.MIN_CLIMATOLOGY_YEARS`.
        means = means.where(prior_years >= features.MIN_CLIMATOLOGY_YEARS)

        year_position = means.index.get_indexer(np.asarray(index.year))
        slot_position = means.columns.get_indexer(slot)
        values = means.to_numpy(dtype="float64")[year_position, slot_position]
    else:
        totals = windowed_sums.sum(axis=0)
        counts = windowed_counts.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            per_slot = (totals / counts.where(counts > 0)).to_numpy(dtype="float64")
        values = per_slot[totals.index.get_indexer(slot)]

    return pd.Series(values, index=index, name=f"{series.name or 'value'}_clim")


def persistence(series: pd.Series) -> pd.Series:
    """Tomorrow equals today: `series` shifted forward by one day.

    Args:
        series: Daily observations indexed by date.

    Returns:
        The persistence prediction for each date.
    """
    return series.shift(1).rename(f"{series.name or 'value'}_persist")

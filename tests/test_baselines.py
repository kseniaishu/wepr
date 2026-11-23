"""The two baselines: that they are causal, and that they are not trivial.

Climatology is the one baseline that *could* leak — a day-of-year mean computed
over the whole record knows the future of every row it describes. The expanding
form is what makes it legal, and the first test here is the executable proof.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wepr import features
from wepr.models import baselines


@pytest.fixture
def series():
    """Twenty-five years of seasonal daily temperature."""
    index = pd.date_range("1990-01-01", "2014-12-31", freq="D", name="date")
    doy = index.day_of_year.to_numpy()
    rng = np.random.default_rng(19)
    values = 15 - 9 * np.cos(2 * np.pi * doy / 365.25) + rng.normal(0, 2.5, len(index))
    return pd.Series(values, index=index, name="tavg")


@pytest.fixture
def raw(series):
    """The same signal as a tidy frame, for the causality guard."""
    return pd.DataFrame(
        {"tavg": series, "tmax": series + 6, "tmin": series - 6, "prcp": 0.0}
    )


# --- causality: the reason the expanding form exists --------------------------


def test_expanding_climatology_is_causal(raw):
    """Deleting the future must not change the past."""
    features.assert_causal(
        lambda f: pd.DataFrame({"clim": baselines.climatology(f["tavg"])}, index=f.index),
        raw,
    )


def test_full_record_climatology_is_not_causal(raw):
    """The display-only form is non-causal, and the guard proves it.

    This is why `expanding=False` is documented as display-only: it is a genuinely
    different quantity, not a cheaper approximation of the same one.
    """
    with pytest.raises(AssertionError, match="future information"):
        features.assert_causal(
            lambda f: pd.DataFrame(
                {"clim": baselines.climatology(f["tavg"], expanding=False)}, index=f.index
            ),
            raw,
        )


def test_expanding_climatology_uses_only_prior_years(series):
    """Changing one year must not alter any earlier year's climatology."""
    before = baselines.climatology(series)
    tampered = series.copy()
    tampered.loc["2010-01-01":] += 50.0
    after = baselines.climatology(tampered)

    earlier = series.index < "2010-01-01"
    pd.testing.assert_series_equal(before[earlier], after[earlier])
    # And it does propagate forward, so the test above is not passing vacuously.
    assert not np.allclose(
        before.loc["2011-06-01"], after.loc["2011-06-01"], equal_nan=True
    )


# --- shape and content -------------------------------------------------------


def test_climatology_is_withheld_until_enough_prior_years(series):
    """Below `MIN_CLIMATOLOGY_YEARS` prior years the answer is NaN, not a guess."""
    clim = baselines.climatology(series)
    first_year = series.index.year[0]
    assert clim[series.index.year < first_year + features.MIN_CLIMATOLOGY_YEARS].isna().all()
    assert clim[series.index.year >= first_year + features.MIN_CLIMATOLOGY_YEARS].notna().all()


def test_climatology_recovers_the_seasonal_cycle(series):
    """July must come out warmer than January, or it is not a climatology."""
    clim = baselines.climatology(series, expanding=False)
    assert clim[clim.index.month == 7].mean() > clim[clim.index.month == 1].mean() + 10


def test_smoothing_window_reduces_day_to_day_noise(series):
    """The window is there to smooth; an unsmoothed climatology must be noisier."""
    rough = baselines.climatology(series, window=0, expanding=False)
    smooth = baselines.climatology(series, window=7, expanding=False)
    assert smooth.diff().abs().mean() < rough.diff().abs().mean()


def test_window_wraps_the_year_boundary(series):
    """2 January's window must reach back into late December.

    Without the wrap, dates near the boundary would be computed from roughly half
    the samples and would be quietly noisier than the rest of the year.
    """
    december = series.copy()
    december[december.index.month == 12] += 100.0
    clim = baselines.climatology(december, window=7, expanding=False)
    # 2 January sits within 7 days of 26-31 December, so it must move.
    assert clim[(clim.index.month == 1) & (clim.index.day == 2)].mean() > (
        baselines.climatology(series, window=7, expanding=False)[
            (series.index.month == 1) & (series.index.day == 2)
        ].mean()
        + 1.0
    )


def test_leap_year_drift_is_removed():
    """1 March must land on the same calendar slot in a leap and a common year."""
    index = pd.DatetimeIndex(["2000-03-01", "2001-03-01", "2000-02-29", "2000-12-31",
                              "2001-12-31", "2000-01-01", "2001-01-01"])
    slot = baselines._calendar_slot(index)
    assert slot[0] == slot[1], "1 March drifts across leap years"
    assert slot[2] == slot[0], "29 February should share 1 March's slot"
    assert slot[3] == slot[4], "31 December drifts across leap years"
    assert slot[5] == slot[6] == 1
    assert slot.max() <= 365


# --- persistence -------------------------------------------------------------


def test_persistence_is_yesterdays_value(series):
    """Aligned to its own index, persistence predicts today from yesterday."""
    persist = baselines.persistence(series)
    assert np.isnan(persist.iloc[0])
    np.testing.assert_allclose(persist.iloc[1:].to_numpy(), series.iloc[:-1].to_numpy())


def test_persistence_beats_nothing_on_white_noise():
    """A sanity check on the baseline itself: on noise it has no skill to offer."""
    index = pd.date_range("2000-01-01", periods=2000, freq="D")
    noise = pd.Series(np.random.default_rng(5).normal(0, 1, 2000), index=index)
    persist = baselines.persistence(noise).dropna()
    error = (noise.loc[persist.index] - persist).abs().mean()
    # Differencing independent draws inflates the error by about sqrt(2).
    assert error > noise.abs().mean()

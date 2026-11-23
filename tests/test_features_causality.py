"""The leakage guard must actually catch leakage.

A guard that never fires is worse than none, because it is believed. So this
tests it against builders whose leakage is known: each `leaky_*` builder must be
caught, and each `clean_*` builder must pass.
"""

import numpy as np
import pandas as pd
import pytest

from wepr.features import assert_causal


@pytest.fixture
def raw_with_gaps(raw):
    """`raw`, with a NaN placed immediately before each default cut point.

    Needed to exercise backward-filling: a gap whose fill source lies *after* the
    cut is detectable by tail truncation, whereas a gap before every cut is not.
    See the limitation noted in `assert_causal`.

    The offset is two rows, not one: the builders apply `shift(1)` first, which
    moves the hole one row forward, so the gap has to land on the truncated
    frame's final row for its fill source to fall beyond the cut.
    """
    out = raw.copy()
    n = len(out)
    for frac in (0.5, 0.7, 0.9):
        out.iloc[int(n * frac) - 2, out.columns.get_loc("tavg")] = np.nan
    return out


@pytest.fixture
def raw():
    """Three years of synthetic daily weather with a seasonal cycle."""
    idx = pd.date_range("2020-01-01", periods=1095, freq="D", name="date")
    rng = np.random.default_rng(0)
    doy = idx.day_of_year.to_numpy()
    tavg = 15 + 10 * np.sin(2 * np.pi * (doy - 100) / 365.25) + rng.normal(0, 2, len(idx))
    prcp = rng.gamma(0.4, 3.0, len(idx)) * (rng.random(len(idx)) < 0.3)
    return pd.DataFrame({"tavg": tavg, "prcp": prcp}, index=idx)


# --- builders that are legal ---------------------------------------------------

def clean_lag(df):
    out = df.copy()
    out["tavg_lag1"] = df["tavg"].shift(1)
    return out


def clean_rolling(df):
    out = df.copy()
    # shift(1) first: the window must end yesterday, not today.
    out["tavg_roll7"] = df["tavg"].shift(1).rolling(7).mean()
    return out


def clean_expanding_climatology(df):
    out = df.copy()
    out["doy_mean"] = (
        df["tavg"].shift(1).groupby(df.index.day_of_year).transform(
            lambda s: s.expanding().mean()
        )
    )
    return out


# --- builders that leak --------------------------------------------------------

def leaky_centred_rolling(df):
    out = df.copy()
    out["tavg_centred"] = df["tavg"].rolling(7, center=True, min_periods=1).mean()
    return out


def leaky_full_series_mean(df):
    out = df.copy()
    # Uses the mean of the entire series, future included.
    out["tavg_anomaly"] = df["tavg"] - df["tavg"].mean()
    return out


def leaky_groupby_transform(df):
    out = df.copy()
    # The day-of-year mean over ALL years, not only prior ones.
    out["doy_mean"] = df["tavg"].groupby(df.index.day_of_year).transform("mean")
    return out


def leaky_backfill(df):
    out = df.copy()
    # bfill pulls the next available value backwards in time, so a gap is filled
    # from the future.
    out["tavg_filled"] = df["tavg"].shift(1).bfill()
    return out


def leaky_negative_shift(df):
    out = df.copy()
    out["tomorrow"] = df["tavg"].shift(-1)
    return out


@pytest.mark.parametrize(
    "build",
    [clean_lag, clean_rolling, clean_expanding_climatology],
    ids=lambda f: f.__name__,
)
def test_clean_builders_pass(build, raw):
    assert_causal(build, raw)


@pytest.mark.parametrize(
    "build",
    [
        leaky_centred_rolling,
        leaky_full_series_mean,
        leaky_groupby_transform,
        leaky_negative_shift,
    ],
    ids=lambda f: f.__name__,
)
def test_leaky_builders_are_caught(build, raw):
    with pytest.raises(AssertionError, match="future information"):
        assert_causal(build, raw)


def test_error_names_the_offending_column(raw):
    with pytest.raises(AssertionError, match="tavg_centred"):
        assert_causal(leaky_centred_rolling, raw)


def test_rejects_out_of_range_cut_point(raw):
    with pytest.raises(ValueError, match="outside"):
        assert_causal(clean_lag, raw, cut_points=[len(raw) + 10])


def test_backfill_across_a_gap_is_caught(raw_with_gaps):
    """bfill is detectable when the gap's fill source sits past the cut point."""
    with pytest.raises(AssertionError, match="tavg_filled"):
        assert_causal(leaky_backfill, raw_with_gaps)


def test_forward_fill_across_the_same_gap_is_legal(raw_with_gaps):
    """ffill only reaches backwards in time, so it is not leakage."""

    def clean_ffill(df):
        out = df.copy()
        out["tavg_filled"] = df["tavg"].shift(1).ffill()
        return out

    assert_causal(clean_ffill, raw_with_gaps)

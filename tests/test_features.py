"""The predictor matrix: what is in it, and that none of it sees the future.

The bugs this file exists to prevent are the ones the original code shipped: a
predictor list captured before engineering, so the new features were never used; a
ratio to a rolling mean that divided by zero; and a target forward-filled into the
final row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wepr import features


@pytest.fixture
def frame():
    """Twelve years of synthetic daily weather — enough for the climatology."""
    index = pd.date_range("2000-01-01", "2011-12-31", freq="D", name="date")
    rng = np.random.default_rng(7)
    doy = index.day_of_year.to_numpy()
    tavg = 15 - 9 * np.cos(2 * np.pi * doy / 365.25) + rng.normal(0, 2.5, len(index))
    prcp = rng.gamma(0.4, 3.0, len(index)) * (rng.random(len(index)) < 0.3)
    return pd.DataFrame(
        {"tavg": tavg, "tmax": tavg + 6, "tmin": tavg - 6, "prcp": prcp}, index=index
    )


@pytest.fixture
def built(frame):
    return features.build_features(frame)


# --- the causality guard, applied to the real builder -------------------------


def test_build_features_is_causal(frame):
    """The single most important test in the project."""
    features.assert_causal(features.build_features, frame)


def test_the_guard_still_fires_on_this_builder_if_a_leak_is_added(frame):
    """Proves the test above is a live check and not a tautology."""

    def sabotaged(df):
        out = features.build_features(df)
        out["peek"] = df["tavg"].rolling(7, center=True, min_periods=1).mean()
        return out

    with pytest.raises(AssertionError, match="peek"):
        features.assert_causal(sabotaged, frame)


def test_targets_would_be_caught_which_is_why_they_are_separate(frame):
    """`add_targets` is not part of `build_features` by design, not by accident."""

    def predictors_plus_targets(df):
        return features.add_targets(features.build_features(df))

    with pytest.raises(AssertionError, match="tavg_target"):
        features.assert_causal(predictors_plus_targets, frame)


# --- what the matrix contains -------------------------------------------------


def test_every_family_is_present_for_every_base_variable(built):
    for var in features.BASE_VARIABLES:
        assert f"{var}_lag1" in built
        assert f"{var}_clim" in built
        assert f"{var}_anom{features.ANOMALY_WINDOW}" in built
        for window in features.ROLL_WINDOWS:
            assert f"{var}_roll{window}" in built


def test_todays_observations_are_kept_as_predictors(built):
    """At a one-day horizon these are the strongest predictors available."""
    for var in features.BASE_VARIABLES:
        assert var in features.feature_columns(built)


def test_calendar_encodings_are_cyclical(built):
    """A raw day-of-year jumps by 364 at the year boundary; sin/cos must not.

    The year-end step is compared against a step in midsummer: if the encoding is
    genuinely cyclical the two are the same size, where a raw day-of-year differs
    by a factor of 364. That is the claim, and it does not depend on picking a
    tolerance.
    """

    def step(earlier: str, later: str) -> float:
        a = built.loc[earlier, ["doy_sin", "doy_cos"]].to_numpy()
        b = built.loc[later, ["doy_sin", "doy_cos"]].to_numpy()
        return float(np.linalg.norm(a - b))

    year_end = step("2005-12-31", "2006-01-01")
    midsummer = step("2005-06-30", "2005-07-01")
    assert year_end == pytest.approx(midsummer, abs=0.01)
    # One day's arc on the unit circle, so a small absolute distance either way.
    assert year_end < 0.05

    # A raw day-of-year, for contrast, jumps by the length of the year.
    raw_jump = abs(
        pd.Timestamp("2006-01-01").day_of_year - pd.Timestamp("2005-12-31").day_of_year
    )
    assert raw_jump == 364

    # And the encoding stays on the unit circle throughout.
    assert np.allclose(np.hypot(built["doy_sin"], built["doy_cos"]), 1.0)


def test_build_features_rejects_a_frame_missing_a_base_variable(frame):
    with pytest.raises(KeyError, match="prcp"):
        features.build_features(frame.drop(columns="prcp"))


def test_no_feature_is_infinite(built):
    """The original code's ratio produced inf, which fillna(0) does not catch."""
    numeric = built.select_dtypes("number")
    assert not np.isinf(numeric.to_numpy()).any()


# --- anomalies, not ratios ----------------------------------------------------


def test_anomaly_is_a_difference_from_the_rolling_mean(built):
    window = features.ANOMALY_WINDOW
    expected = built["tavg"] - built[f"tavg_roll{window}"]
    pd.testing.assert_series_equal(
        built[f"tavg_anom{window}"], expected, check_names=False
    )


def test_precipitation_anomaly_survives_a_dry_spell(frame):
    """A ratio would divide by exactly zero here; a difference must not blow up."""
    dry = frame.copy()
    dry["prcp"] = 0.0
    out = features.build_features(dry)
    anomaly = out[f"prcp_anom{features.ANOMALY_WINDOW}"].dropna()
    assert len(anomaly) > 0
    assert (anomaly == 0.0).all()
    assert np.isfinite(anomaly).all()


def test_temperature_anomaly_survives_a_rolling_mean_crossing_zero(frame):
    """°C has an arbitrary zero, so a ratio is meaningless as well as unstable."""
    cold = frame.copy()
    cold[["tavg", "tmax", "tmin"]] -= 15.0  # push the mean through 0 °C
    out = features.build_features(cold)
    assert np.isfinite(out[f"tavg_anom{features.ANOMALY_WINDOW}"].dropna()).all()


# --- windows ------------------------------------------------------------------


def test_a_partial_window_is_nan_not_a_different_statistic(built):
    """Letting partial windows through makes the first rows incomparable."""
    for window in features.ROLL_WINDOWS:
        column = built[f"tavg_roll{window}"]
        assert column.iloc[: window - 1].isna().all()
        assert pd.notna(column.iloc[window - 1])


def test_rolling_mean_ends_today_not_yesterday(built, frame):
    """Shifting it defensively would discard the most informative observation."""
    expected = frame["tavg"].iloc[:7].mean()
    assert built["tavg_roll7"].iloc[6] == pytest.approx(expected)


def test_lag_is_yesterdays_value(built, frame):
    assert built["tavg_lag1"].iloc[5] == pytest.approx(frame["tavg"].iloc[4])


def test_wet_day_counts_count_wet_days(built, frame):
    wet = (frame["prcp"] >= features.WET_DAY_MM).astype(float)
    assert built["prcp_wet7"].iloc[6] == pytest.approx(wet.iloc[:7].sum())
    assert (built["prcp_wet30"].dropna() <= 30).all()


def test_diurnal_range_is_tmax_minus_tmin(built):
    pd.testing.assert_series_equal(
        built["trange"], built["tmax"] - built["tmin"], check_names=False
    )


# --- the prior-year climatology ----------------------------------------------


def test_climatology_averages_only_earlier_years(built, frame):
    """Hand-computed against the same calendar date in preceding years."""
    date = pd.Timestamp("2010-06-15")
    earlier = [frame.loc[pd.Timestamp(f"{y}-06-15"), "tavg"] for y in range(2000, 2010)]
    assert built.loc[date, "tavg_clim"] == pytest.approx(np.mean(earlier))


def test_climatology_excludes_the_current_year(built, frame):
    """Otherwise the feature contains the observation it is meant to contextualise."""
    date = pd.Timestamp("2010-06-15")
    with_current = [frame.loc[pd.Timestamp(f"{y}-06-15"), "tavg"] for y in range(2000, 2011)]
    assert built.loc[date, "tavg_clim"] != pytest.approx(np.mean(with_current))


def test_climatology_withholds_a_value_until_it_has_enough_years(built):
    """An average of two years is not a norm; a dropped row is more honest."""
    first_years = built.loc[:"2004-12-31", "tavg_clim"]
    assert first_years.isna().all()
    assert pd.notna(built.loc["2005-06-15", "tavg_clim"])


def test_climatology_groups_on_the_calendar_date_not_the_day_of_year():
    """Day-of-year drifts by one across leap years, mixing 29 Feb with 1 Mar."""
    index = pd.date_range("2000-01-01", "2011-12-31", freq="D", name="date")
    # A series whose value depends only on the calendar date, so a correct
    # climatology reproduces it exactly and a doy-based one does not.
    value = pd.Series(index.month * 100.0 + index.day, index=index)
    frame = pd.DataFrame(
        {"tavg": value, "tmax": value, "tmin": value, "prcp": 0.0}, index=index
    )
    built = features.build_features(frame)
    march = built.loc["2011-03-01", "tavg_clim"]
    assert march == pytest.approx(301.0), "1 March must average only 1 March"


def test_leap_day_is_its_own_calendar_day():
    index = pd.date_range("2000-01-01", "2011-12-31", freq="D", name="date")
    rng = np.random.default_rng(1)
    value = pd.Series(rng.normal(15, 3, len(index)), index=index)
    frame = pd.DataFrame(
        {"tavg": value, "tmax": value, "tmin": value, "prcp": 0.0}, index=index
    )
    built = features.build_features(frame)
    # Only three leap days precede 2012's, below the minimum, so it stays NaN.
    assert pd.isna(built.loc["2008-02-29", "tavg_clim"])


# --- targets ------------------------------------------------------------------


def test_target_is_tomorrows_value(frame):
    out = features.add_targets(frame)
    assert out["tavg_target"].iloc[0] == pytest.approx(frame["tavg"].iloc[1])
    assert out["prcp_target"].iloc[0] == pytest.approx(frame["prcp"].iloc[1])


def test_the_final_target_is_nan_and_is_not_filled(frame):
    """ffill after shift(-1) invents a label out of the last real observation."""
    out = features.add_targets(frame)
    assert pd.isna(out["tavg_target"].iloc[-1])
    assert pd.isna(out["prcp_target"].iloc[-1])
    assert out["tavg_target"].iloc[-1] != frame["tavg"].iloc[-1]


def test_add_targets_does_not_mutate_its_input(frame):
    before = frame.copy()
    features.add_targets(frame)
    pd.testing.assert_frame_equal(frame, before)


# --- the predictor list, and the trim ----------------------------------------


def test_feature_columns_excludes_targets(built):
    out = features.add_targets(built)
    columns = features.feature_columns(out)
    assert not set(columns) & set(features.TARGETS)


def test_feature_columns_computed_after_engineering_includes_the_new_features(frame):
    """The original bug: the list was captured first, so rolling features were
    created and then never fed to the model."""
    before = features.feature_columns(frame)
    after = features.feature_columns(features.build_features(frame))
    assert set(before) < set(after)
    assert f"tavg_roll{features.ROLL_WINDOWS[-1]}" in after
    assert f"tavg_roll{features.ROLL_WINDOWS[-1]}" not in before


def test_labelled_keeps_only_complete_rows(frame):
    out = features.labelled(features.add_targets(features.build_features(frame)))
    predictors = features.feature_columns(out)
    assert not out[predictors].isna().to_numpy().any()
    assert not out[list(features.TARGETS)].isna().to_numpy().any()
    assert len(out) > 0


def test_labelled_drops_the_final_unlabelled_row(frame):
    engineered = features.add_targets(features.build_features(frame))
    out = features.labelled(engineered)
    assert engineered.index[-1] not in out.index


def test_labelled_defaults_to_the_recomputed_predictor_list(frame):
    """Passing nothing must use the engineered columns, not the raw ones."""
    engineered = features.add_targets(features.build_features(frame))
    default = features.labelled(engineered)
    explicit = features.labelled(engineered, features.feature_columns(engineered))
    pd.testing.assert_frame_equal(default, explicit)


class TestPriorYearClimatologyEquivalence:
    """The vectorised climatology must equal the readable form it replaced.

    `_prior_year_climatology` was a `groupby().transform(lambda ...)` until it was
    found to be 95% of the serving layer's startup — 0.44 s per city, 9 s to boot
    the app and 9 s again on every reload. The replacement is three vectorised
    group operations computing the same statistic.

    This is worth pinning rather than trusting, because an optimisation to a
    feature definition is an optimisation to what every model was fitted on: a
    divergence here would not fail, it would silently invalidate
    `reports/metrics.json`. The subtlety it turns on is NaN handling —
    `expanding().mean()` returns a value at a row whose own shifted input is
    missing, where a plain `cumsum()` would propagate NaN into it.
    """

    @staticmethod
    def _readable(series, index):
        """The original form, kept here as the specification."""
        return series.groupby(features._calendar_key(index), sort=False).transform(
            lambda s: s.shift(1)
            .expanding(min_periods=features.MIN_CLIMATOLOGY_YEARS)
            .mean()
        )

    def _assert_identical(self, frame):
        for variable in features.BASE_VARIABLES:
            expected = self._readable(frame[variable], frame.index)
            actual = features._prior_year_climatology(frame[variable], frame.index)
            # The NaN pattern is part of the contract: it is what `labelled`
            # drops on, so a difference would change the training rows.
            pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_matches_on_a_long_synthetic_record(self, synthetic_daily):
        self._assert_identical(synthetic_daily(first_year=1950, n_years=40))

    def test_matches_when_the_record_has_gaps(self, synthetic_daily):
        """The NaN case, which is where the two forms could most easily diverge."""
        frame = synthetic_daily(first_year=1950, n_years=20).copy()
        # Whole missing months, a scattered day, and a gap on 29 February —
        # which shares a calendar key with nothing else.
        frame.loc["1955-03-01":"1955-04-30", "tavg"] = np.nan
        frame.loc["1960-07-14", :] = np.nan
        frame.loc["1952-02-29", "prcp"] = np.nan
        self._assert_identical(frame)

    def test_matches_on_the_real_records_when_they_are_present(self):
        """The strongest version of the check, skipped on a fresh clone."""
        from wepr.data import load

        slugs = load.available_cities()
        if not slugs:
            pytest.skip("no processed data; run scripts/fetch_data.py")
        for slug in slugs:
            self._assert_identical(load.load_city(slug))

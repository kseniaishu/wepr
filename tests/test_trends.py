"""Trend estimation: does it recover a slope, and does it stay honest about it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import synthetic_daily_frame as daily_frame

from wepr import trends


def annual_series(slope_per_year: float, n: int = 60, noise: float = 0.4, seed: int = 0):
    years = np.arange(1960, 1960 + n)
    rng = np.random.default_rng(seed)
    values = 14.0 + slope_per_year * (years - years[0]) + rng.normal(0, noise, n)
    return years, values


class TestTheilSen:
    def test_recovers_a_known_slope(self):
        years, values = annual_series(0.031)
        trend = trends.theil_sen(years, values)
        assert trend.slope_per_decade == pytest.approx(0.31, abs=0.05)

    def test_confidence_interval_brackets_the_truth(self):
        years, values = annual_series(0.031)
        trend = trends.theil_sen(years, values)
        assert trend.ci_low < 0.31 < trend.ci_high

    def test_is_robust_to_outlier_years_where_ols_is_not(self):
        """The whole reason Theil–Sen is the default rather than OLS.

        A handful of corrupted years — a bad grid cell, an archive glitch — should
        move the robust slope far less than the least-squares one. If this ever
        stops being true the choice of estimator has nothing behind it.
        """
        years, values = annual_series(0.031, n=70)
        clean = trends.theil_sen(years, values)

        spoiled = values.copy()
        spoiled[[5, 17, 42]] += 25.0
        dirty = trends.theil_sen(years, spoiled)

        robust_shift = abs(dirty.slope_per_decade - clean.slope_per_decade)
        ols_shift = abs(dirty.ols_slope_per_decade - clean.ols_slope_per_decade)
        assert robust_shift < ols_shift / 3

    def test_reports_residual_scatter_separately_from_rate_uncertainty(self):
        years, values = annual_series(0.03, noise=1.5, seed=3)
        trend = trends.theil_sen(years, values)
        assert trend.residual_sd == pytest.approx(1.5, rel=0.3)

    def test_too_few_years_raises(self):
        with pytest.raises(ValueError, match="at least 3 years"):
            trends.theil_sen(np.array([2000, 2001]), np.array([1.0, 2.0]))

    def test_short_record_is_flagged_as_not_climatological(self):
        years, values = annual_series(0.03, n=20)
        assert trends.theil_sen(years, values).is_climatological is False
        assert "not a climate trend" in trends.theil_sen(years, values).describe()

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="differ in length"):
            trends.theil_sen(np.arange(5), np.arange(4))


class TestAnnualMeans:
    def test_drops_partially_covered_years(self):
        """A year the fetch window only half covers is a seasonal mean, not an annual one."""
        frame = daily_frame(n_years=5)
        clipped = frame.loc["1980-07-01":]
        means = trends.annual_means(clipped)
        assert 1980 not in means.index
        assert 1981 in means.index

    def test_counts_against_the_calendar_not_the_frame(self):
        """An all-NaN stretch is the same defect as a truncated year and is caught too."""
        frame = daily_frame(n_years=5)
        frame.loc["1982-01-01":"1982-06-30", "tavg"] = np.nan
        assert 1982 not in trends.annual_means(frame).index

    def test_unknown_variable_raises(self):
        with pytest.raises(KeyError, match="lacks"):
            trends.annual_means(daily_frame(n_years=3), "humidity")


class TestExtrapolate:
    def test_rate_band_has_no_width_at_the_anchor_and_widens_after(self):
        years, values = annual_series(0.03)
        trend = trends.theil_sen(years, values)
        forward = trends.extrapolate(trend, trend.last_year + 50)

        first = forward.iloc[0]
        assert first["rate_hi"] - first["rate_lo"] == pytest.approx(0.0, abs=1e-12)
        widths = forward["rate_hi"] - forward["rate_lo"]
        assert (widths.diff().dropna() > 0).all()

    def test_the_single_year_band_is_always_the_wider_of_the_two(self):
        years, values = annual_series(0.03, noise=0.6)
        trend = trends.theil_sen(years, values)
        forward = trends.extrapolate(trend, trend.last_year + 30)
        assert (forward["year_lo"] < forward["rate_lo"]).all()
        assert (forward["year_hi"] > forward["rate_hi"]).all()

    def test_starts_at_the_record_so_the_line_joins_up(self):
        years, values = annual_series(0.03)
        trend = trends.theil_sen(years, values)
        forward = trends.extrapolate(trend, trend.last_year + 10)
        assert forward.index[0] == trend.last_year
        assert forward["central"].iloc[0] == pytest.approx(trend.value_at(trend.last_year))

    def test_backwards_extrapolation_raises(self):
        years, values = annual_series(0.03)
        trend = trends.theil_sen(years, values)
        with pytest.raises(ValueError, match="before the record ends"):
            trends.extrapolate(trend, trend.first_year)


class TestScenarios:
    def test_every_quoted_row_carries_a_baseline_and_a_source(self):
        """A warming figure without a baseline period is meaningless, so none ships."""
        for row in trends.AR6_SCENARIOS:
            assert len(row["baseline"]) == 2
            assert len(row["period"]) == 2
            assert row["source"].strip()
            assert row["low"] <= row["high"]
            if row["median"] is not None:
                assert row["low"] <= row["median"] <= row["high"]

    def test_filters_by_region_and_season(self):
        globals_ = trends.scenarios(region="global", season="annual")
        assert not globals_.empty
        assert set(globals_["region"]) == {"global"}
        assert trends.scenarios(region="global", season="JJA").empty

    def test_baseline_mean_is_nan_when_the_record_does_not_cover_it(self):
        """What stops a 1850-1900 anomaly being slid onto a 1980-baselined axis."""
        frame = daily_frame(first_year=1980, n_years=10)
        assert np.isnan(trends.baseline_mean(frame, (1850, 1900)))
        assert np.isfinite(trends.baseline_mean(frame, (1981, 1985)))


# -- the publishable artifact ----------------------------------------------------


@pytest.fixture
def fitted_table(monkeypatch):
    """A `trend_table` built from synthetic records, with its fits attached.

    Patches the loader rather than writing CSVs: what is under test is the shape
    and the honesty of the summary, not the data layer. The slugs are real so
    `cities.get` can name them, and `baseline_mean` sees the same synthetic
    record the slopes were fitted to — which is what lets the baseline-coverage
    assertions below mean anything.
    """
    from wepr.data import load as data_load

    def _build(first_year=1950, n_years=22, slope_per_year=0.03):
        frame = daily_frame(
            first_year=first_year, n_years=n_years, slope_per_year=slope_per_year
        )
        monkeypatch.setattr(data_load, "load_city", lambda slug: frame)
        return trends.trend_table(["madrid", "seville"])

    return _build


class TestSummary:
    def test_refuses_a_table_it_did_not_fit(self):
        """A missing `fits` must raise, never silently refit.

        Refitting here would reintroduce exactly what putting the table in the
        figure script avoided: two estimations of the same city's slope, free to
        disagree between the page and the figure beside it.
        """
        table = pd.DataFrame([{"name": "Madrid"}])
        with pytest.raises(KeyError, match="fits"):
            trends.summary(table)

    def test_carries_one_entry_per_fitted_city(self, fitted_table):
        summary = trends.summary(fitted_table())
        assert summary["n_cities"] == 2
        assert {c["slug"] for c in summary["cities"]} == {"madrid", "seville"}

    def test_sorts_by_slope_so_the_chart_reads_top_to_bottom(self, fitted_table):
        summary = trends.summary(fitted_table())
        slopes = [c["slope_per_decade"] for c in summary["cities"]]
        assert slopes == sorted(slopes, reverse=True)

    def test_quotes_the_extrapolation_at_each_horizon(self, fitted_table):
        summary = trends.summary(fitted_table(), horizons=(2050, 2100))
        point = summary["cities"][0]["extrapolation"]
        assert [p["year"] for p in point] == [2050, 2100]

    def test_the_bands_widen_with_distance(self, fitted_table):
        """The honest content of a long extrapolation is its width."""
        near, far = trends.summary(fitted_table())["cities"][0]["extrapolation"]
        assert (far["rate_hi"] - far["rate_lo"]) > (near["rate_hi"] - near["rate_lo"])

    def test_the_single_year_band_is_the_wider_of_the_two(self, fitted_table):
        point = trends.summary(fitted_table())["cities"][0]["extrapolation"][0]
        assert point["year_lo"] < point["rate_lo"]
        assert point["year_hi"] > point["rate_hi"]

    def test_states_how_far_past_the_record_each_horizon_is(self, fitted_table):
        summary = trends.summary(fitted_table(first_year=1950, n_years=22))
        far = summary["cities"][0]["extrapolation"][-1]
        assert far["years_beyond_record"] == 2100 - 1971

    def test_a_short_record_is_flagged_rather_than_quietly_served(self, fitted_table):
        """22 years is not a climate trend and the artifact has to say so."""
        summary = trends.summary(fitted_table(n_years=22))
        assert summary["provisional"] is True
        assert summary["n_climatological"] == 0
        assert all(not c["is_climatological"] for c in summary["cities"])
        assert all(not c["is_sound"] for c in summary["cities"])

    def test_a_long_clean_record_is_not_flagged(self, fitted_table):
        summary = trends.summary(fitted_table(first_year=1950, n_years=70))
        assert summary["provisional"] is False
        assert summary["n_climatological"] == 2
        assert all(c["is_climatological"] for c in summary["cities"])

    def test_is_sound_requires_an_interval_that_clears_zero(self, fitted_table):
        """Long enough is not sufficient: the sign has to be established too."""
        summary = trends.summary(fitted_table(first_year=1950, n_years=70))
        for city in summary["cities"]:
            assert city["is_sound"] == (not city["ci_spans_zero"])

    def test_reports_the_baselines_the_record_cannot_cover(self, fitted_table):
        """Why the two halves of the page are not drawn against a common zero."""
        summary = trends.summary(fitted_table(first_year=1950, n_years=22))
        covered = {tuple(b["period"]): b["n_covered"] for b in summary["baselines"]}
        assert covered[(1850, 1900)] == 0
        assert covered[(1986, 2005)] == 0

    def test_reports_a_baseline_the_record_does_cover(self, fitted_table):
        summary = trends.summary(fitted_table(first_year=1980, n_years=40))
        covered = {tuple(b["period"]): b["n_covered"] for b in summary["baselines"]}
        assert covered[(1986, 2005)] == 2
        assert covered[(1850, 1900)] == 0

    def test_keeps_the_measured_and_the_quoted_in_separate_keys(self, fitted_table):
        """They are different kinds of claim and the artifact must not merge them."""
        summary = trends.summary(fitted_table())
        assert summary["scenarios"]
        assert all("source" in row for row in summary["scenarios"])
        assert summary["scenario_gap"]
        assert "extrapolation" not in summary["scenarios"][0]

    def test_is_json_serialisable(self, fitted_table):
        """It is written to disk and served verbatim, so NumPy scalars are a bug."""
        import json

        json.dumps(trends.summary(fitted_table()))

    def test_names_the_cities_it_could_not_fit(self, fitted_table, monkeypatch):
        from wepr.data import load as data_load

        table = fitted_table()
        monkeypatch.setattr(
            data_load, "load_city", lambda slug: (_ for _ in ()).throw(FileNotFoundError(slug))
        )
        table.attrs["skipped"] = {"porto": "no processed data"}
        assert trends.summary(table)["skipped"] == {"porto": "no processed data"}

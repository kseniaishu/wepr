"""The seasonal estimate: the cycle, the trend term, the interval, and the folds."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from conftest import synthetic_daily_frame as daily_frame

from wepr.models import seasonal


@pytest.fixture
def frame():
    return daily_frame(first_year=1980, n_years=40, slope_per_year=0.03)


@pytest.fixture
def model(frame):
    return seasonal.SeasonalClimatology().fit(frame)


class TestFit:
    def test_recovers_the_seasonal_cycle(self, model):
        """Mid-July should be warm, mid-January cold, by the synthetic amplitude."""
        july = model.predict_one(dt.date(2000, 7, 15))
        january = model.predict_one(dt.date(2000, 1, 15))
        assert july.expected_c - january.expected_c == pytest.approx(17.0, abs=2.0)

    def test_recovers_the_trend_and_reports_it_separately(self, model):
        assert model.trend_.slope_per_decade == pytest.approx(0.3, abs=0.12)
        near = model.predict_one(dt.date(2000, 6, 1))
        far = model.predict_one(dt.date(2030, 6, 1))
        assert far.trend_adjustment_c - near.trend_adjustment_c == pytest.approx(0.9, abs=0.35)
        # The trend is the *only* difference between two same-day estimates.
        assert far.expected_c - near.expected_c == pytest.approx(
            far.trend_adjustment_c - near.trend_adjustment_c, abs=1e-9
        )

    def test_detrending_narrows_the_interval(self):
        """The stated reason for detrending before taking percentiles.

        On a strongly warming record the raw spread for a calendar day contains the
        warming itself, so an interval built from it is wider than any single year's
        weather ever was. Fitting the same cycle with a large trend must not widen
        the band relative to fitting it with none.
        """
        flat = seasonal.SeasonalClimatology().fit(
            daily_frame(n_years=40, slope_per_year=0.0)
        )
        steep = seasonal.SeasonalClimatology().fit(
            daily_frame(n_years=40, slope_per_year=0.12)
        )
        flat_width = (flat.p_high_ - flat.p_low_).mean()
        steep_width = (steep.p_high_ - steep.p_low_).mean()
        assert steep_width == pytest.approx(flat_width, rel=0.08)

    def test_the_window_wraps_the_year(self, model):
        """31 December and 1 January are a day apart, so their norms must be too."""
        assert model.mean_[0] == pytest.approx(model.mean_[364], abs=0.6)

    def test_table_has_one_row_per_calendar_slot(self, model):
        table = model.table()
        assert len(table) == 365
        assert table.index.min() == 1 and table.index.max() == 365
        assert (table["n_years"] > 0).all()

    def test_rain_probability_is_a_probability(self, model):
        assert ((model.rain_probability_ >= 0) & (model.rain_probability_ <= 1)).all()

    def test_predict_frame_agrees_with_predict_one(self, model):
        dates = pd.DatetimeIndex(["2031-03-04", "2047-08-19", "1999-12-31"])
        bulk = model.predict_frame(dates)
        for stamp in dates:
            one = model.predict_one(stamp.date())
            assert bulk.loc[stamp, "expected"] == pytest.approx(one.expected_c)
            assert bulk.loc[stamp, "p_low"] == pytest.approx(one.p10_c)
            assert bulk.loc[stamp, "rain_probability"] == pytest.approx(one.rain_probability)

    def test_a_short_record_will_not_fit(self):
        with pytest.raises(ValueError, match="need"):
            seasonal.SeasonalClimatology().fit(daily_frame(n_years=4))

    def test_predicting_before_fitting_raises(self):
        with pytest.raises(RuntimeError, match="fit has not been called"):
            seasonal.SeasonalClimatology().predict_one(dt.date(2030, 1, 1))


class TestBacktest:
    def test_the_interval_covers_what_it_claims(self, frame):
        _, score = seasonal.backtest(frame)
        assert score.coverage == pytest.approx(score.nominal_coverage, abs=0.05)

    def test_beats_the_naive_same_date_last_year(self, frame):
        _, score = seasonal.backtest(frame)
        assert score.skill_vs_last_year > 0.1

    def test_no_fold_sees_a_year_after_its_own(self, frame):
        """The seasonal analogue of `features.assert_causal`.

        If the folds are clean, deleting the tail of the record cannot change what
        an earlier fold predicted. The subtle leak this guards against is fitting
        the *trend* once on the whole record and reusing it per fold, which would
        seep the test years' warming into every prediction made before them.
        """
        full, _ = seasonal.backtest(frame)
        truncated, _ = seasonal.backtest(frame.loc[:"2009-12-31"])

        shared = full.index.intersection(truncated.index)
        assert len(shared) > 365
        for column in ("expected", "p_low", "p_high", "rain_probability"):
            pd.testing.assert_series_equal(
                full.loc[shared, column], truncated.loc[shared, column]
            )

    def test_every_fold_is_a_whole_year(self, frame):
        predictions, score = seasonal.backtest(frame)
        per_fold = predictions.groupby("fold").size()
        assert set(per_fold.unique()) <= {365, 366}
        assert score.n_folds == len(per_fold)

    def test_too_short_a_record_raises_rather_than_scoring_one_fold(self):
        with pytest.raises(ValueError, match="nothing could be tested"):
            seasonal.backtest(daily_frame(n_years=10), min_train_years=10)

    def test_the_trend_term_earns_its_place_on_a_strongly_trending_record(self):
        """A record with real warming in it: the trend adjustment should now help.

        The counterpart to what the project's current 22-year archive shows, where
        the same number comes out negative and is published as such. If the trend
        term could not help even here, it would be dead weight rather than a term
        whose contribution depends on the record.
        """
        _, score = seasonal.backtest(daily_frame(n_years=40, slope_per_year=0.12))
        assert score.skill_vs_climatology > 0


class TestPersistence:
    def test_survives_a_joblib_round_trip_unchanged(self, isolated_paths, frame):
        """The hard rule in one test: the deployed object must be the fitted one.

        `SeasonalClimatology` is a plain class rather than a scikit-learn estimator,
        so nothing else would notice if its state stopped pickling — the app would
        load an object that predicts different numbers from the one the backtest
        scored, which is precisely the failure this project was rebuilt to remove.
        """
        from wepr.models import registry

        fitted = seasonal.SeasonalClimatology().fit(frame)
        key = registry.model_key("testville", "seasonal", "tavg")
        registry.save(fitted, key)

        loaded = registry.load(key)
        assert type(loaded).__name__ == "SeasonalClimatology"

        dates = pd.date_range("2031-01-01", "2031-12-31", freq="D")
        pd.testing.assert_frame_equal(fitted.predict_frame(dates), loaded.predict_frame(dates))
        assert loaded.trend_.slope_per_decade == pytest.approx(fitted.trend_.slope_per_decade)


# -- the map's value source -----------------------------------------------------


class TestCityEstimates:
    """`city_estimates` is the bridge from per-city models to the map."""

    def test_returns_one_row_per_city_with_a_trained_model(self, monkeypatch):
        from wepr.cities import CITIES

        calls: list[str] = []

        def fake_estimate(slug, target):
            calls.append(slug)
            return seasonal.SeasonalEstimate(
                expected_c=20.0,
                p10_c=15.0,
                p90_c=25.0,
                rain_probability=0.2,
                trend_adjustment_c=0.1,
                n_years=22,
            )

        monkeypatch.setattr(seasonal, "estimate", fake_estimate)
        frame = seasonal.city_estimates(dt.date(2031, 7, 15))

        assert len(frame) == len(CITIES)
        assert calls == [c.slug for c in CITIES]
        assert frame.attrs["missing"] == []
        assert frame.attrs["target"] == "2031-07-15"

    def test_carries_the_coordinates_the_map_interpolates_on(self, monkeypatch):
        from wepr.cities import get

        monkeypatch.setattr(
            seasonal,
            "estimate",
            lambda slug, target: seasonal.SeasonalEstimate(20.0, 15.0, 25.0, 0.2, 0.1, 22),
        )
        frame = seasonal.city_estimates(dt.date(2031, 7, 15))
        madrid = frame.loc["madrid"]
        assert madrid["lat"] == get("madrid").lat
        assert madrid["lon"] == get("madrid").lon
        assert madrid["region"] == get("madrid").region

    def test_omits_untrained_cities_rather_than_filling_them(self, monkeypatch):
        """A map from fifteen cities is weaker than one from twenty, not wrong.

        Filling the missing five with a placeholder would be wrong, so they are
        reported in `attrs["missing"]` and left out of the frame entirely.
        """

        def fake_estimate(slug, target):
            if slug in {"faro", "palma"}:
                raise FileNotFoundError(f"no model for {slug}")
            return seasonal.SeasonalEstimate(20.0, 15.0, 25.0, 0.2, 0.1, 22)

        monkeypatch.setattr(seasonal, "estimate", fake_estimate)
        frame = seasonal.city_estimates(dt.date(2031, 7, 15))

        assert "faro" not in frame.index
        assert "palma" not in frame.index
        assert sorted(frame.attrs["missing"]) == ["faro", "palma"]

    def test_can_be_limited_to_named_cities(self, monkeypatch):
        monkeypatch.setattr(
            seasonal,
            "estimate",
            lambda slug, target: seasonal.SeasonalEstimate(20.0, 15.0, 25.0, 0.2, 0.1, 22),
        )
        frame = seasonal.city_estimates(dt.date(2031, 7, 15), slugs=["madrid", "seville"])
        assert list(frame.index) == ["madrid", "seville"]

    def test_returns_an_empty_frame_with_its_columns_when_nothing_is_trained(
        self, monkeypatch
    ):
        """The figure script checks `.empty`, so the columns must survive."""

        def absent(slug, target):
            raise FileNotFoundError(slug)

        monkeypatch.setattr(seasonal, "estimate", absent)
        frame = seasonal.city_estimates(dt.date(2031, 7, 15))

        assert frame.empty
        assert "expected_c" in frame.columns
        assert len(frame.attrs["missing"]) == 20


class TestMonthlyNorm:
    """The map's value source, and the reason it is a month rather than a day.

    The map used to describe one mid-month day and label it with that date. The
    number was fine -- the 15th lands within about 0.05 °C of the month mean,
    because a month's climatology is near-linear and the midpoint of a line is
    its average -- but the *label* implied the rest of the month looked like it,
    and it does not.
    """

    def test_averages_every_day_in_the_month(self, trained_project):
        trained_project(["madrid"])
        frame = seasonal.city_estimates_for_month(2027, 11)
        assert frame.attrs["n_days"] == 30
        assert frame.attrs["month"] == 11
        assert frame.attrs["year"] == 2027

        daily = [
            seasonal.city_estimates(dt.date(2027, 11, d)).loc["madrid", "expected_c"]
            for d in range(1, 31)
        ]
        assert frame.loc["madrid", "expected_c"] == pytest.approx(
            sum(daily) / len(daily)
        )

    def test_february_length_follows_the_calendar(self, trained_project):
        """28 or 29 days, not a hard-coded 30. 2028 is a leap year."""
        trained_project(["madrid"])
        assert seasonal.city_estimates_for_month(2027, 2).attrs["n_days"] == 28
        assert seasonal.city_estimates_for_month(2028, 2).attrs["n_days"] == 29

    def test_a_month_outside_the_calendar_is_rejected(self, trained_project):
        trained_project(["madrid"])
        for month in (0, 13, -1):
            with pytest.raises(ValueError, match="month must be 1-12"):
                seasonal.city_estimates_for_month(2027, month)

    def test_carries_the_same_columns_as_a_single_day(self, trained_project):
        """So the map's consumers do not need to know which one they were given."""
        trained_project(["madrid"])
        monthly = seasonal.city_estimates_for_month(2027, 11)
        daily = seasonal.city_estimates(dt.date(2027, 11, 15))
        assert set(monthly.columns) == set(daily.columns)
        # Per-city metadata is carried through, not averaged into nonsense.
        assert monthly.loc["madrid", "name"] == daily.loc["madrid", "name"]
        assert monthly.loc["madrid", "n_years"] == daily.loc["madrid", "n_years"]

    def test_the_mid_month_day_is_close_but_is_not_what_is_published(
        self, trained_project
    ):
        """Pins the reason the change was worth making.

        The 15th is a good *estimator* of the month mean, which is why the old
        map's numbers were not wrong. It is still not the month mean, and the
        published figure is now the real thing rather than a near-miss that
        happens to land well.
        """
        trained_project(["madrid"])
        monthly = seasonal.city_estimates_for_month(2027, 11).loc["madrid", "expected_c"]
        mid = seasonal.city_estimates(dt.date(2027, 11, 15)).loc["madrid", "expected_c"]
        assert monthly == pytest.approx(mid, abs=0.5)

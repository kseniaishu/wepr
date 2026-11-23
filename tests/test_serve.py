"""The serving layer: what gets loaded, what gets refused, and what it reports.

These tests are mostly about the *refusals*. That a loaded ridge pipeline returns
a number is barely worth asserting; that the store declines to serve a model
nobody evaluated, declines to serve an artifact whose class no longer matches the
one the metrics were measured on, and never claims a forecast is about today — is
the whole point of the phase.
"""

import datetime as dt
import json

import pytest

from wepr import paths
from wepr.models import registry
from wepr.serve import MAX_SEASONAL_YEAR, ModelStore, NotServeable


class TestLoading:
    def test_loads_every_trained_and_evaluated_model(self, trained_project):
        slugs = trained_project()
        store = ModelStore.load()
        assert store.ready
        assert store.served_slugs() == sorted(slugs)
        assert store.n_models_loaded() == 3 * len(slugs)
        assert store.problems == []

    def test_an_empty_project_loads_without_raising(self, isolated_paths):
        """A fresh clone must boot, not traceback."""
        store = ModelStore.load()
        assert not store.ready
        assert store.served_slugs() == []

    def test_an_unevaluated_model_is_not_served(self, trained_project):
        """The rule that the evaluated model is the deployed model, enforced.

        Training without evaluating leaves artifacts on disk with no measured
        accuracy. Serving them would mean publishing predictions whose error
        nobody knows, which is exactly the configuration the original project
        shipped.
        """
        trained_project(write_metrics=False)
        store = ModelStore.load()
        assert store.served_slugs() == []
        reasons = " ".join(p["reason"] for p in store.problems)
        remedies = " ".join(p["remedy"] for p in store.problems)
        assert "metrics" in reasons
        assert "evaluate_all.py" in remedies

    def test_a_model_whose_class_changed_is_not_served(self, trained_project):
        """A key check alone would pass here; a class check is what catches it.

        This is the original project's failure in miniature: the metrics say one
        estimator was measured, the artifact on disk is another, and the filename
        is identical either way.
        """
        trained_project(estimator_name="Prophet")
        store = ModelStore.load()
        assert store.served_slugs() == []
        assert any("Prophet" in p["reason"] for p in store.problems)
        assert any("evaluated model must be the deployed model" in p["reason"] for p in store.problems)

    def test_a_missing_artifact_is_reported_per_model(self, trained_project):
        """One missing model degrades that product, not the whole city."""
        trained_project(["madrid"])
        registry.model_path("madrid.nextday.prcp").unlink()

        store = ModelStore.load()
        # Not "fully served", but the half that works still answers.
        assert store.served_slugs() == []
        assert store.seasonal("madrid", dt.date(2030, 6, 1)) is not None
        with pytest.raises(NotServeable) as excinfo:
            store.forecast("madrid")
        assert "train_all.py" in excinfo.value.detail

    def test_corrupt_metrics_are_treated_as_absent(self, trained_project):
        trained_project()
        paths.metrics_json().write_text("{not json")
        store = ModelStore.load()
        assert store.served_slugs() == []
        assert any("not valid JSON" in p["reason"] for p in store.problems)

    def test_an_unknown_slug_raises_keyerror(self, trained_project):
        trained_project()
        store = ModelStore.load()
        with pytest.raises(KeyError):
            store.forecast("atlantis")


class TestNextDayForecast:
    def test_is_issued_from_the_archive_edge_not_today(self, trained_project):
        """The forecast describes the day after the last observation. Always."""
        trained_project(["madrid"])
        store = ModelStore.load()
        forecast = store.forecast("madrid")

        assert forecast.as_of == dt.date(2019, 12, 31)
        assert forecast.valid_for == forecast.as_of + dt.timedelta(days=1)

    def test_a_record_years_behind_is_not_near_real_time(self, trained_project):
        """A stale archive must not be presented as a weather forecast."""
        trained_project(["madrid"])
        forecast = ModelStore.load().forecast("madrid", today=dt.date(2026, 11, 22))
        assert forecast.lag_days > 2000
        assert forecast.is_near_real_time is False

    def test_a_current_record_is_near_real_time(self, trained_project):
        """The same code path, with the clock just past the archive edge."""
        trained_project(["madrid"])
        store = ModelStore.load()
        # Five days after the day the forecast describes: inside the allowance.
        forecast = store.forecast("madrid", today=dt.date(2020, 1, 6))
        assert forecast.is_near_real_time is True

    def test_carries_both_required_baselines(self, trained_project):
        """A prediction is never reported without persistence and climatology."""
        trained_project(["madrid"])
        forecast = ModelStore.load().forecast("madrid")
        # Persistence is literally the last observation, not a model output.
        assert forecast.baselines.persistence_c == forecast.observed["tavg"]
        assert forecast.baselines.climatology_c is not None

    def test_reports_rain_as_two_separate_statistics(self, trained_project):
        """Probability and conditional depth, never their product."""
        trained_project(["madrid"])
        rain = ModelStore.load().forecast("madrid").rain
        assert 0.0 <= rain.probability <= 1.0
        assert rain.expected_mm_if_wet >= 0.0

    def test_precipitation_is_never_negative(self, trained_project):
        trained_project(["madrid", "seville"])
        store = ModelStore.load()
        for slug in store.served_slugs():
            assert store.forecast(slug).rain.expected_mm_if_wet >= 0.0

    def test_accuracy_comes_from_the_metrics_file(self, trained_project):
        """The figure shown beside a prediction is the one measured for it."""
        trained_project(["madrid"])
        forecast = ModelStore.load().forecast("madrid")
        assert forecast.tavg_accuracy.mae == 1.0  # the fixture's value
        assert forecast.provisional is True

    def test_json_is_serialisable_and_names_the_dates(self, trained_project):
        trained_project(["madrid"])
        body = ModelStore.load().forecast("madrid").to_json()
        # Must round-trip: a NaN would serialise to invalid JSON.
        assert json.loads(json.dumps(body))
        assert body["as_of"] == "2019-12-31"
        assert body["valid_for"] == "2020-01-01"
        assert body["product"] == "next-day forecast"


class TestSeasonalEstimate:
    def test_returns_an_interval_not_a_bare_number(self, trained_project):
        trained_project(["madrid"])
        outlook = ModelStore.load().seasonal("madrid", dt.date(2031, 7, 15))
        assert outlook.estimate.p10_c < outlook.estimate.expected_c < outlook.estimate.p90_c

    def test_the_payload_says_it_is_not_a_forecast(self, trained_project):
        """A consumer reading only the JSON has to be told, in the JSON."""
        trained_project(["madrid"])
        body = ModelStore.load().seasonal("madrid", dt.date(2031, 7, 15)).to_json()
        assert body["is_forecast"] is False
        assert body["product"] == "seasonal estimate"

    def test_distance_does_not_change_the_method(self, trained_project):
        """Unlike a forecast, this estimate is not conditioned on anything recent."""
        trained_project(["madrid"])
        store = ModelStore.load()
        near = store.seasonal("madrid", dt.date(2026, 7, 15))
        far = store.seasonal("madrid", dt.date(2090, 7, 15))
        assert near.estimate.n_years == far.estimate.n_years

    def test_refuses_a_date_beyond_the_scenario_horizon(self, trained_project):
        trained_project(["madrid"])
        with pytest.raises(ValueError, match=str(MAX_SEASONAL_YEAR)):
            ModelStore.load().seasonal("madrid", dt.date(2200, 1, 1))

    def test_a_short_record_marks_the_trend_term_unsound(self, trained_project):
        """The term that moves the headline most is the one least supported.

        The fixture's record is 40 years, so the slope is climatological, but a
        2090 target is still carried far past the window it was fitted to. That
        alone disqualifies it, because extrapolation distance is what turns a
        small rate into the largest number on the page.
        """
        trained_project(["madrid"])
        outlook = ModelStore.load().seasonal("madrid", dt.date(2090, 7, 15))
        assert outlook.trend.extrapolation_years > outlook.trend.record_years
        assert outlook.trend.is_sound is False

    def test_the_estimate_without_the_trend_is_always_available(self, trained_project):
        """So a reader can see how much of the headline is climatology."""
        trained_project(["madrid"])
        outlook = ModelStore.load().seasonal("madrid", dt.date(2031, 7, 15))
        assert outlook.expected_without_trend_c == pytest.approx(
            outlook.estimate.expected_c - outlook.estimate.trend_adjustment_c
        )


class TestStatus:
    def test_reports_the_archive_edge_and_the_problems(self, trained_project):
        trained_project(["madrid"])
        registry.model_path("madrid.nextday.prcp").unlink()
        status = ModelStore.load().status()

        assert status["ready"] is True
        assert status["archive_edge"]["as_of"] == "2019-12-31"
        assert status["problems"]
        assert json.loads(json.dumps(status))

    def test_an_empty_project_says_what_to_run(self, isolated_paths):
        status = ModelStore.load().status()
        assert status["ready"] is False
        assert any("evaluate_all.py" in p["remedy"] for p in status["problems"])

    def test_counts_the_figures_that_are_below_the_gate(self, trained_project):
        """The count the landing page decides its caveat from.

        It is a count and not a reading of `gate.note`, because the note is prose:
        on a truncated record it begins "PROVISIONAL", on a complete one it reports
        a pass, and a template that sniffed it for a keyword would print the amber
        "distrust this" panel around a pass the first time someone reworded the
        sentence. The fixture marks every entry provisional, so this is all of them.
        """
        trained_project(["madrid"])
        metrics = ModelStore.load().status()["metrics"]
        assert metrics["n_provisional"] == metrics["n_entries"] > 0

    def test_a_record_above_the_gate_has_nothing_provisional(self, trained_project):
        """The complete-record case, which is what the real project now serves.

        Asserted explicitly rather than left as the absence of the previous test:
        zero is the value the landing page branches on to drop the caveat entirely,
        so it is the one that decides what a reader is told.
        """
        trained_project(["madrid"])
        payload = json.loads(paths.metrics_json().read_text())
        for entry in payload["models"].values():
            entry["provisional"] = False
        paths.metrics_json().write_text(json.dumps(payload))

        metrics = ModelStore.load().status()["metrics"]
        assert metrics["n_entries"] > 0
        assert metrics["n_provisional"] == 0

"""API contract tests.

The JSON shapes here are consumed by the frontend, so they are pinned. A page can
be redesigned freely; a response shape cannot change silently.
"""

import datetime as dt

import pytest

from app import create_app
from wepr.cities import CITIES


@pytest.fixture
def client():
    """An app with no models loaded — the fresh-clone contract.

    `LOAD_MODELS=False` rather than an empty `models/` directory: the point of
    these tests is the HTTP contract, and loading sixty real artifacts to assert
    on a 404 would couple the suite to a trained working copy.
    """
    app = create_app({"TESTING": True, "LOAD_MODELS": False})
    return app.test_client()


@pytest.fixture
def served_client(trained_project):
    """An app serving the miniature project built by `trained_project`.

    Built inside the fixture rather than at module scope because the factory
    loads the models at startup, so the app has to be created *after* the
    artifacts exist on disk.
    """
    trained_project(["madrid", "seville"])
    return create_app({"TESTING": True}).test_client()


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


def test_list_cities(client):
    r = client.get("/api/cities")
    assert r.status_code == 200
    body = r.get_json()["cities"]
    assert len(body) == len(CITIES)
    assert set(body[0]) == {"slug", "name", "country", "lat", "lon", "region"}


def test_city_detail(client):
    r = client.get("/api/cities/huesca")
    assert r.status_code == 200
    assert r.get_json()["name"] == "Huesca"


def test_unknown_city_is_404(client):
    r = client.get("/api/cities/atlantis")
    assert r.status_code == 404
    assert "unknown city" in r.get_json()["error"]


def test_forecast_is_503_until_models_are_trained(client):
    """An untrained app must say so, not invent a number."""
    r = client.get("/api/forecast/huesca")
    assert r.status_code == 503
    body = r.get_json()
    assert "no models are loaded" in body["error"]
    # Every refusal names the command that fixes it.
    assert body["detail"], "a refusal must always name its cause"


def test_index_page_renders_without_models(client):
    """The shell must boot and explain itself on a fresh clone."""
    assert client.get("/").status_code == 200


def test_forecast_for_unknown_city_is_404(client):
    assert client.get("/api/forecast/atlantis").status_code == 404


def test_index_page_renders(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Huesca" in r.data


# -- the served app ---------------------------------------------------------------


class TestForecastApi:
    """The next-day endpoint, once models are actually loaded."""

    def test_serves_a_forecast(self, served_client):
        body = served_client.get("/api/forecast/madrid").get_json()
        assert body["slug"] == "madrid"
        assert isinstance(body["temperature"]["expected_c"], float)

    def test_dates_the_forecast_rather_than_calling_it_tomorrow(self, served_client):
        """`valid_for` is one day past the archive edge, and is always present."""
        body = served_client.get("/api/forecast/madrid").get_json()
        assert body["as_of"] == "2019-12-31"
        assert body["valid_for"] == "2020-01-01"
        assert body["is_near_real_time"] is False

    def test_carries_the_two_baselines(self, served_client):
        """A model number is never served without the naive answers beside it."""
        body = served_client.get("/api/forecast/madrid").get_json()
        assert body["baselines"]["persistence_c"] == body["observed"]["tavg"]
        assert body["baselines"]["climatology_c"] is not None

    def test_carries_the_deployed_models_measured_accuracy(self, served_client):
        body = served_client.get("/api/forecast/madrid").get_json()
        assert body["temperature"]["accuracy"]["mae"] == 1.0
        assert body["provisional"] is True

    def test_reports_rain_as_probability_and_conditional_depth(self, served_client):
        """Two statistics, never multiplied into a millimetre figure."""
        rain = served_client.get("/api/forecast/madrid").get_json()["rain"]
        assert 0.0 <= rain["probability"] <= 1.0
        assert rain["expected_mm_if_wet"] >= 0.0
        assert rain["wet_threshold_mm"] == 0.1

    def test_unknown_city_is_404_not_503(self, served_client):
        """A bad URL and an unserveable model are different failures."""
        assert served_client.get("/api/forecast/atlantis").status_code == 404


class TestSeasonalApi:
    def test_says_it_is_not_a_forecast(self, served_client):
        body = served_client.get("/api/seasonal/madrid?date=2031-07-15").get_json()
        assert body["is_forecast"] is False
        assert body["target"] == "2031-07-15"

    def test_returns_an_interval(self, served_client):
        t = served_client.get("/api/seasonal/madrid?date=2031-07-15").get_json()["temperature"]
        assert t["p10_c"] < t["expected_c"] < t["p90_c"]

    def test_breaks_out_the_trend_term_and_its_soundness(self, served_client):
        """The largest term on the page is also the least supported one."""
        body = served_client.get("/api/seasonal/madrid?date=2090-07-15").get_json()
        trend = body["trend"]
        assert trend["is_sound"] is False
        assert trend["extrapolation_years"] > trend["record_years"]
        assert "expected_without_trend_c" in trend

    def test_defaults_to_a_date_far_enough_out_to_be_unmistakable(self, served_client):
        body = served_client.get("/api/seasonal/madrid").get_json()
        assert body["target"] > dt.date.today().isoformat()

    def test_rejects_a_malformed_date(self, served_client):
        r = served_client.get("/api/seasonal/madrid?date=next-tuesday")
        assert r.status_code == 400
        assert "YYYY-MM-DD" in r.get_json()["error"]

    def test_rejects_a_date_past_the_scenario_horizon(self, served_client):
        r = served_client.get("/api/seasonal/madrid?date=2200-01-01")
        assert r.status_code == 400
        assert r.get_json()["max_year"] == 2100


class TestStatusApi:
    def test_reports_what_is_loaded(self, served_client):
        body = served_client.get("/api/status").get_json()
        assert body["ready"] is True
        assert body["cities_served"] == ["madrid", "seville"]
        assert body["archive_edge"]["as_of"] == "2019-12-31"

    def test_reports_an_empty_project_honestly(self, client):
        body = client.get("/api/status").get_json()
        assert body["ready"] is False
        assert body["cities_served"] == []

    def test_health_is_independent_of_whether_models_loaded(self, client):
        """"The app is up" and "the app has models" are different signals."""
        assert client.get("/api/health").get_json()["status"] == "ok"


class TestCityPage:
    def test_renders_both_products(self, served_client):
        html = served_client.get("/city/madrid").get_data(as_text=True)
        assert "Next-day forecast" in html
        assert "Seasonal estimate" in html

    def test_states_that_the_seasonal_product_is_not_a_forecast(self, served_client):
        """The distinction the whole rebuild exists to keep. See CLAUDE.md."""
        html = served_client.get("/city/madrid").get_data(as_text=True)
        assert "not a forecast" in html
        assert "climatological expectation" in html

    def test_dates_the_forecast_rather_than_saying_tomorrow(self, served_client):
        html = served_client.get("/city/madrid").get_data(as_text=True)
        assert "2020-01-01" in html
        assert "not today's weather" in html

    def test_shows_both_baselines_beside_the_model(self, served_client):
        html = served_client.get("/city/madrid").get_data(as_text=True)
        assert "Persistence" in html
        assert "Climatology" in html

    def test_warns_when_the_trend_term_is_extrapolated_too_far(self, served_client):
        html = served_client.get("/city/madrid?date=2090-07-15").get_data(as_text=True)
        assert "Read the trend term with suspicion" in html
        assert "Climatology alone" in html

    def test_unknown_city_is_404(self, served_client):
        assert served_client.get("/city/atlantis").status_code == 404

    def test_is_503_when_the_city_has_no_models(self, client):
        assert client.get("/city/madrid").status_code == 503

    def test_index_links_to_a_served_city(self, served_client):
        assert "/city/madrid" in served_client.get("/").get_data(as_text=True)


# -- the map ---------------------------------------------------------------------


@pytest.fixture
def published_map(tmp_path, monkeypatch):
    """Point the app at a generated map, without generating a real one.

    The real artifact is a committed elevation fetch plus a matplotlib render, and
    neither belongs in an API contract test. What is under test here is the
    contract: that the route serves what the script wrote, and refuses when it
    did not.
    """
    import json

    from wepr import paths

    reports = tmp_path / "reports"
    figures = tmp_path / "figures"
    reports.mkdir()
    figures.mkdir()
    monkeypatch.setattr(paths, "REPORTS", reports)
    monkeypatch.setattr(paths, "FIGURES", figures)

    def _publish(variable="tavg", month=None, *, with_figure=True, **overrides):
        # Defaults to the current month, because that is what `/map` opens on
        # when no month is in the path.
        month = month or dt.date.today().month
        summary = {
            "variable": variable,
            "label": "Expected mean temperature",
            "units": "°C",
            "target": f"{dt.date.today().year + 1}-{month:02d}-15",
            "month": month,
            "figure": f"map_{variable}_{month:02d}.png",
            "n_cities": 20,
            "missing_cities": [],
            "cells_drawn": 8100,
            "cells_interpolated": 5939,
            "cells_extrapolated": 2161,
            "max_distance_to_city_km": 204.4,
            "hull_margin_km": 95.0,
            "resolution_deg": 0.1,
            "clamped_at_zero": False,
            "elevation_source": "test terrain",
            "surface": {
                "lapse_rate_c_per_km": -4.41,
                "r2": 0.94,
                "residual_sd": 0.6,
                "fitted_elevation_range_m": [17.0, 821.0],
                "n_cities": 20,
            },
            "cities": [
                {
                    "slug": "madrid",
                    "name": "Madrid",
                    "lat": 40.4168,
                    "lon": -3.7038,
                    "region": "Interior",
                    "value": 23.5,
                    "n_years": 22,
                }
            ],
        }
        summary.update(overrides)
        (reports / f"map_{variable}_{month:02d}.json").write_text(json.dumps(summary))
        if with_figure:
            (figures / summary["figure"]).write_bytes(b"\x89PNG\r\n\x1a\n")
        return summary

    return _publish


class TestMapApi:
    def test_returns_503_until_the_map_is_generated(self, client, tmp_path, monkeypatch):
        from wepr import paths

        monkeypatch.setattr(paths, "REPORTS", tmp_path)
        response = client.get("/api/map/tavg")
        assert response.status_code == 503
        body = response.get_json()
        assert "make_figures.py" in body["detail"]
        # The rule the whole project runs on: no placeholder numbers.
        assert "cities" not in body
        assert "surface" not in body

    def test_serves_what_the_script_wrote(self, client, published_map):
        published = published_map("tavg")
        body = client.get("/api/map/tavg").get_json()
        assert body["variable"] == "tavg"
        assert body["target"] == published["target"]
        assert body["cities"][0]["name"] == "Madrid"
        assert body["surface"]["r2"] == 0.94

    def test_rejects_an_unknown_variable(self, client, published_map):
        published_map("tavg")
        response = client.get("/api/map/humidity")
        assert response.status_code == 404
        assert response.get_json()["available"] == ["tavg", "prcp"]

    def test_each_variable_is_served_separately(self, client, published_map):
        published_map("tavg")
        published_map("prcp", label="Rain probability", units="%")
        assert client.get("/api/map/tavg").get_json()["units"] == "°C"
        assert client.get("/api/map/prcp").get_json()["units"] == "%"

    def test_a_summary_without_its_figure_counts_as_not_generated(
        self, client, published_map
    ):
        """A half-written artifact must not render as a broken image."""
        published_map("tavg", with_figure=False)
        assert client.get("/api/map/tavg").status_code == 503

    def test_a_corrupt_summary_counts_as_not_generated(
        self, client, published_map, monkeypatch
    ):
        published_map("tavg")
        from wepr import paths

        paths.map_summary("tavg", dt.date.today().month).write_text("{not json")
        assert client.get("/api/map/tavg").status_code == 503


class TestMapMonths:
    """The month axis, and the staleness guard that made it necessary.

    The map used to be one figure for one hard-coded date -- 15 July of whatever
    year the figure script last ran -- which meant the page kept presenting that
    date indefinitely and, once it passed, presented a past day as an expectation
    with nothing saying so. These tests pin the two halves of the fix.
    """

    def test_opens_on_the_current_month_without_being_told(self, client, published_map):
        """The whole point: the default follows the calendar, not the artifact."""
        published_map("tavg", dt.date.today().month)
        body = client.get("/api/map/tavg").get_json()
        assert body["month"] == dt.date.today().month

    def test_a_month_can_be_asked_for_explicitly(self, client, published_map):
        published_map("tavg", 1)
        published_map("tavg", 7)
        assert client.get("/api/map/tavg/1").get_json()["month"] == 1
        assert client.get("/api/map/tavg/7").get_json()["month"] == 7

    def test_reports_which_months_exist(self, client, published_map):
        published_map("tavg", 3)
        published_map("tavg", 4)
        body = client.get("/api/map/tavg/3").get_json()
        assert body["available_months"] == [3, 4]

    def test_a_month_out_of_range_is_404_not_503(self, client, published_map):
        """A bad request is the caller's error; 503 would blame the artifact."""
        published_map("tavg", 1)
        for month in (0, 13, 99):
            assert client.get(f"/api/map/tavg/{month}").status_code == 404

    def test_an_ungenerated_month_says_which_ones_exist(self, client, published_map):
        published_map("tavg", 1)
        response = client.get("/api/map/tavg/8")
        assert response.status_code == 503
        body = response.get_json()
        assert body["available_months"] == [1]
        assert "make_figures.py" in body["detail"]
        assert "cities" not in body

    def test_falls_back_to_the_nearest_published_month(self, client, published_map):
        """A partial generation should still open on something sensible."""
        far = (dt.date.today().month + 6 - 1) % 12 + 1
        published_map("tavg", far)
        assert client.get("/api/map/tavg").get_json()["month"] == far

    def test_a_target_in_the_past_is_reported_as_stale(self, client, published_map):
        """The silent failure the old single-date map had, now surfaced."""
        published_map("tavg", target="2001-07-15")
        body = client.get("/api/map/tavg").get_json()
        assert body["stale"] is True
        html = client.get("/map").get_data(as_text=True)
        assert "date has passed" in html
        assert "make_figures.py" in html

    def test_a_future_target_is_not_stale(self, client, published_map):
        published_map("tavg")
        assert client.get("/api/map/tavg").get_json()["stale"] is False
        assert "date has passed" not in client.get("/map").get_data(as_text=True)

    def test_the_page_offers_every_published_month(self, client, published_map):
        for month in (1, 6, 12):
            published_map("tavg", month)
        html = client.get("/map/tavg/6").get_data(as_text=True)
        for month in (1, 6, 12):
            assert f"/map/tavg/{month}" in html

    def test_switching_variable_keeps_the_month(self, client, published_map):
        """Changing the field should not silently throw away the chosen month."""
        published_map("tavg", 2)
        published_map("prcp", 2)
        html = client.get("/map/tavg/2").get_data(as_text=True)
        assert "/map/prcp/2" in html


class TestMapPage:
    def test_returns_503_and_the_commands_to_run(self, client, tmp_path, monkeypatch):
        from wepr import paths

        monkeypatch.setattr(paths, "REPORTS", tmp_path)
        response = client.get("/map")
        assert response.status_code == 503
        html = response.get_data(as_text=True)
        assert "fetch_elevation.py" in html
        assert "make_figures.py" in html

    def test_renders_the_published_map(self, client, published_map):
        published = published_map("tavg")
        response = client.get("/map")
        assert response.status_code == 200
        html = response.get_data(as_text=True)
        assert published["figure"] in html
        assert "Madrid" in html

    def test_states_that_it_is_not_a_forecast(self, client, published_map):
        """The seasonal product must never read as a forecast. See CLAUDE.md."""
        published_map("tavg")
        html = client.get("/map").get_data(as_text=True)
        assert "not a forecast" in html
        assert "climatological expectation" in html

    def test_distinguishes_interpolated_from_extrapolated_cells(
        self, client, published_map
    ):
        published_map("tavg")
        html = client.get("/map").get_data(as_text=True)
        assert "5939" in html and "2161" in html
        assert "extrapolation" in html

    def test_names_the_cities_it_could_not_draw(self, client, published_map):
        published_map("tavg", n_cities=18, missing_cities=["faro", "palma"])
        html = client.get("/map").get_data(as_text=True)
        assert "faro, palma" in html

    def test_rejects_an_unknown_variable(self, client, published_map):
        published_map("tavg")
        assert client.get("/map/humidity").status_code == 404

    def test_index_links_to_the_map(self, client):
        assert "/map" in client.get("/").get_data(as_text=True)

    def test_flags_a_provisional_record(self, client, published_map):
        """A map built below the data-quality gate must say so on the page."""
        published_map("tavg", provisional=True, min_years=22, gate_min_years=60)
        html = client.get("/map").get_data(as_text=True)
        assert "Provisional" in html
        assert "22 years of record" in html
        assert "60-year data-quality gate" in html

    def test_says_nothing_about_provisionality_when_the_gate_passes(
        self, client, published_map
    ):
        published_map("tavg", provisional=False, min_years=76, gate_min_years=60)
        html = client.get("/map").get_data(as_text=True)
        assert "Provisional" not in html


# -- the climate trends ----------------------------------------------------------


def squashed(html: str) -> str:
    """HTML with runs of whitespace collapsed.

    So that an assertion about a sentence the page must contain is not also an
    assertion about where the template happens to wrap its lines.
    """
    return " ".join(html.split())


@pytest.fixture
def published_trends(tmp_path, monkeypatch):
    """Point the app at a generated trends artifact without fitting anything.

    Same reasoning as `published_map`: the route's contract is that it serves
    what `scripts/make_figures.py` wrote and refuses when it did not. Fitting
    twenty real Theil-Sen slopes to assert on a 503 would couple the suite to a
    fetched archive.
    """
    import json

    from wepr import paths

    reports = tmp_path / "reports"
    figures = tmp_path / "figures"
    reports.mkdir()
    figures.mkdir()
    monkeypatch.setattr(paths, "REPORTS", reports)
    monkeypatch.setattr(paths, "FIGURES", figures)

    def city(slug, name, slope, lo, hi, **over):
        row = {
            "slug": slug, "name": name, "region": "Interior",
            "figure": f"trend_{slug}.png",
            "slope_per_decade": slope, "ci_low": lo, "ci_high": hi,
            "ci_spans_zero": lo <= 0.0 <= hi,
            "ols_slope_per_decade": slope + 0.01,
            "residual_sd": 0.5, "lag1_autocorrelation": 0.03,
            "ci_is_too_narrow": False,
            "first_year": 1950, "last_year": 1971, "n_years": 22,
            "is_climatological": False, "is_sound": False,
            "extrapolation": [
                {"year": 2050, "years_beyond_record": 79, "central": 11.7,
                 "rate_lo": 8.5, "rate_hi": 14.8, "year_lo": 7.7, "year_hi": 15.6},
                {"year": 2100, "years_beyond_record": 129, "central": 11.7,
                 "rate_lo": 6.7, "rate_hi": 16.8, "year_lo": 5.8, "year_hi": 17.6},
            ],
        }
        row.update(over)
        return row

    def _publish(*, with_figures=False, **overrides):
        summary = {
            "variable": "tavg",
            "units": "°C",
            "generated_at": "2026-11-22T12:00:00+00:00",
            "estimator": "theil-sen",
            "min_trend_years": 30,
            "min_year_coverage": 0.95,
            "horizons": [2050, 2100],
            "n_cities": 2,
            "n_climatological": 0,
            "provisional": True,
            "record_years": {"min": 22, "max": 22},
            "cities": [
                city("madrid", "Madrid", -0.173, -0.718, 0.281),
                city("seville", "Seville", -0.295, -0.688, 0.092),
            ],
            "skipped": {},
            "baselines": [
                {"period": [1850, 1900], "covered_by": [], "n_covered": 0},
                {"period": [1986, 2005], "covered_by": [], "n_covered": 0},
            ],
            "scenarios": [
                {"scenario": "SSP2-4.5", "region": "global", "season": "annual",
                 "period": [2081, 2100], "baseline": [1850, 1900],
                 "median": 2.7, "low": 2.1, "high": 3.5,
                 "source": "IPCC AR6 WGI SPM, Table SPM.1 (very likely range)"},
                {"scenario": "SSP5-8.5", "region": "mediterranean-land",
                 "season": "JJA", "period": [2081, 2100], "baseline": [1986, 2005],
                 "median": 7.4, "low": 5.6, "high": 9.1,
                 "source": "Cos et al. 2022, ESD 13:321"},
            ],
            "scenario_gap": "No annual-mean CMIP6 envelope for the AR6 "
                            "Mediterranean reference region is encoded.",
        }
        summary.update(overrides)
        paths.trend_summary().write_text(json.dumps(summary))
        if with_figures:
            for entry in summary["cities"]:
                (figures / entry["figure"]).write_bytes(b"\x89PNG\r\n\x1a\n")
        return summary

    return _publish


class TestTrendsApi:
    def test_returns_503_until_the_trends_are_fitted(self, client, tmp_path, monkeypatch):
        from wepr import paths

        monkeypatch.setattr(paths, "REPORTS", tmp_path)
        response = client.get("/api/trends")
        assert response.status_code == 503
        body = response.get_json()
        assert "make_figures.py" in body["detail"]
        # The rule the whole project runs on: no placeholder numbers.
        assert "cities" not in body

    def test_serves_what_the_script_wrote(self, client, published_trends):
        published_trends()
        body = client.get("/api/trends").get_json()
        assert body["estimator"] == "theil-sen"
        assert body["cities"][0]["name"] == "Madrid"

    def test_keeps_the_measured_and_the_quoted_apart(self, client, published_trends):
        """A consumer must not be able to read one as the other."""
        published_trends()
        body = client.get("/api/trends").get_json()
        assert "extrapolation" in body["cities"][0]
        assert all("source" in row for row in body["scenarios"])
        assert body["scenario_gap"]

    def test_a_corrupt_artifact_counts_as_not_generated(self, client, published_trends):
        published_trends()
        from wepr import paths

        paths.trend_summary().write_text("{not json")
        assert client.get("/api/trends").status_code == 503

    def test_an_artifact_with_no_cities_counts_as_not_generated(
        self, client, published_trends
    ):
        published_trends(cities=[], n_cities=0)
        assert client.get("/api/trends").status_code == 503


class TestTrendsPage:
    def test_returns_503_and_the_command_to_run(self, client, tmp_path, monkeypatch):
        from wepr import paths

        monkeypatch.setattr(paths, "REPORTS", tmp_path)
        response = client.get("/trends")
        assert response.status_code == 503
        assert "make_figures.py" in response.get_data(as_text=True)

    def test_renders_the_published_trends(self, client, published_trends):
        published_trends()
        html = client.get("/trends").get_data(as_text=True)
        assert client.get("/trends").status_code == 200
        assert "Madrid" in html
        assert "-0.173" in html

    def test_states_that_the_extrapolation_is_not_a_projection(
        self, client, published_trends
    ):
        """The distinction the page exists to demonstrate. See CLAUDE.md."""
        published_trends()
        html = squashed(client.get("/trends").get_data(as_text=True))
        assert "not climate projection" in html
        assert "statistical" in html.lower()

    def test_shows_the_published_scenario_envelope_with_its_source(
        self, client, published_trends
    ):
        published_trends()
        html = client.get("/trends").get_data(as_text=True)
        assert "IPCC AR6 WGI SPM" in html
        assert "Cos et al. 2022" in html

    def test_says_a_short_record_is_not_a_climate_trend(self, client, published_trends):
        published_trends()
        html = squashed(client.get("/trends").get_data(as_text=True))
        assert "None of these slopes is a climate trend" in html
        assert "fetch_data.py" in html

    def test_says_nothing_about_provisionality_when_the_record_is_long(
        self, client, published_trends
    ):
        published_trends(
            provisional=False,
            n_climatological=2,
            record_years={"min": 76, "max": 76},
        )
        html = client.get("/trends").get_data(as_text=True)
        assert "not a climate trend" not in html

    def test_reports_that_no_city_covers_a_scenario_baseline(
        self, client, published_trends
    ):
        """Why the two claims are not drawn against a common zero."""
        published_trends()
        html = squashed(client.get("/trends").get_data(as_text=True))
        assert "none of 2" in html

    def test_reports_a_baseline_the_record_does_cover(self, client, published_trends):
        """Once the record reaches a scenario's baseline, the page must stop
        claiming the comparison is impossible and say why it is still separate."""
        published_trends(
            baselines=[
                {"period": [1850, 1900], "covered_by": [], "n_covered": 0},
                {"period": [1986, 2005],
                 "covered_by": ["madrid", "seville"], "n_covered": 2},
            ]
        )
        html = squashed(client.get("/trends").get_data(as_text=True))
        assert "1986&ndash;2005 rows are a different case" in html
        assert "does not make a quoted ensemble result" in html
        # and the uncovered one is still reported as uncovered
        assert "No city's record reaches 1850&ndash;1900" in html

    def test_drops_the_noise_caveat_once_the_slope_clears_zero(
        self, client, published_trends
    ):
        """The extrapolation is still arithmetic, but it is no longer noise."""
        sound = [
            dict(c, is_sound=True, is_climatological=True, ci_spans_zero=False)
            for c in published_trends()["cities"]
        ]
        published_trends(cities=sound, provisional=False, n_climatological=2)
        html = squashed(client.get("/trends").get_data(as_text=True))
        assert "This is arithmetic, not a projection" in html
        assert "not even the direction is established" not in html

    def test_names_how_far_past_the_record_the_extrapolation_is_carried(
        self, client, published_trends
    ):
        published_trends()
        html = squashed(client.get("/trends").get_data(as_text=True))
        assert "129 years past the end of a 22-year record" in html
        assert "not even the direction is established" in html

    def test_shows_the_per_city_figures_only_when_they_exist(
        self, client, published_trends
    ):
        published_trends()
        assert "trend_madrid.png" not in client.get("/trends").get_data(as_text=True)
        published_trends(with_figures=True)
        assert "trend_madrid.png" in client.get("/trends").get_data(as_text=True)

    def test_index_links_to_the_trends_page(self, client):
        assert "/trends" in client.get("/").get_data(as_text=True)

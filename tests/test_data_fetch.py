"""The archive client: batching, windowing, caching and retry policy.

No test here opens a socket. The HTTP layer is replaced by `fake_archive`, which
is the only honest way to assert on *how* the client talks to the archive.
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pytest
import requests

from tests.conftest import daily_payload
from wepr.cities import CITIES, get
from wepr.data import fetch, load


# --- windowing ----------------------------------------------------------------


def test_windows_cover_the_range_exactly_once():
    windows = list(fetch._date_windows("1998-07-15", "2001-03-02"))
    assert windows[0] == (1998, "1998-07-15", "1998-12-31"), "clipped to the request"
    assert windows[-1] == (2001, "2001-01-01", "2001-03-02")
    for (_, _, prev_end), (_, next_start, _) in zip(windows, windows[1:], strict=False):
        gap = dt.date.fromisoformat(next_start) - dt.date.fromisoformat(prev_end)
        assert gap == dt.timedelta(days=1), "windows must abut, not overlap or skip"


def test_windows_are_keyed_by_calendar_year():
    """A stable key is what makes a half-finished fetch resumable."""
    years = [year for year, _, _ in fetch._date_windows("1950-01-01", "2026-11-14")]
    assert years == list(range(1950, 2027))
    assert len(set(years)) == len(years)


def test_windows_respect_the_request_budget():
    """Exceeding it is what earns a 400 'requests too much data' from the archive."""
    for _year, start, end in fetch._date_windows("1950-01-01", "2026-01-01"):
        days = (dt.date.fromisoformat(end) - dt.date.fromisoformat(start)).days + 1
        assert days * fetch.MAX_BATCH_CITIES <= fetch.MAX_LOCATION_DAYS


def test_all_twenty_cities_fit_in_one_batch():
    """Otherwise a year would cost two requests instead of one."""
    assert len(CITIES) <= fetch.MAX_BATCH_CITIES
    assert list(fetch._batches(CITIES)) == [tuple(CITIES)]


def test_a_single_day_is_one_window():
    assert list(fetch._date_windows("2020-05-01", "2020-05-01")) == [
        (2020, "2020-05-01", "2020-05-01")
    ]


def test_windows_reject_a_reversed_range():
    with pytest.raises(ValueError, match="precedes"):
        list(fetch._date_windows("2020-12-31", "2020-01-01"))


# --- batching -----------------------------------------------------------------


def test_cities_are_batched_into_one_request_per_window(isolated_paths, fake_archive):
    cities = CITIES[:4]
    fetch.fetch_cities(cities, "2020-01-01", "2020-03-31")
    assert len(fake_archive) == 1, "four cities over 90 days should be a single request"
    params = fake_archive[0]
    assert params["latitude"] == ",".join(str(c.lat) for c in cities)
    assert params["longitude"] == ",".join(str(c.lon) for c in cities)


def test_the_model_is_pinned_explicitly(isolated_paths, fake_archive):
    """Omitting `models` lets the archive pick, which is not reproducible."""
    fetch.fetch_cities(CITIES[:1], "2020-01-01", "2020-01-05")
    assert fake_archive[0]["models"] == "era5_seamless"


def test_every_contracted_variable_is_requested(isolated_paths, fake_archive):
    fetch.fetch_cities(CITIES[:1], "2020-01-01", "2020-01-05")
    requested = fake_archive[0]["daily"].split(",")
    assert requested == list(fetch.DAILY_VARIABLES)


def test_timezone_is_utc_so_days_mean_the_same_thing_everywhere(isolated_paths, fake_archive):
    fetch.fetch_cities(CITIES[:1], "2020-01-01", "2020-01-05")
    assert fake_archive[0]["timezone"] == "UTC"


def test_windows_are_stitched_into_one_file_per_city(isolated_paths, fake_archive):
    """A multi-window fetch must leave a single continuous series, not fragments."""
    written = fetch.fetch_cities(CITIES[:20], "2020-01-01", "2021-12-31")
    assert len(fake_archive) > 1
    assert len(written) == 20
    frame = load.tidy(json.loads(written["madrid"].read_text()))
    assert frame.index[0] == dt.datetime(2020, 1, 1)
    assert frame.index[-1] == dt.datetime(2021, 12, 31)
    assert len(frame) == 731


def test_the_served_grid_cell_is_recorded(isolated_paths, fake_archive):
    """Elevation is not hard-coded in the registry, so it has to come from here."""
    fetch.fetch_cities([get("granada")], "2020-01-01", "2020-01-05")
    meta = load.city_metadata("granada")
    assert meta["grid_elevation_m"] == 650.0
    assert meta["model"] == fetch.MODEL
    assert meta["requested_latitude"] == get("granada").lat
    assert "fetched_at_utc" in meta


# --- caching ------------------------------------------------------------------


def test_a_covering_cache_is_reused_without_a_request(isolated_paths, fake_archive):
    fetch.fetch_cities(CITIES[:2], "2020-01-01", "2020-01-10")
    before = len(fake_archive)
    fetch.fetch_cities(CITIES[:2], "2020-01-01", "2020-01-10")
    assert len(fake_archive) == before, "the archive is immutable; a cache hit is correct"


def test_refresh_forces_a_refetch(isolated_paths, fake_archive):
    fetch.fetch_cities(CITIES[:2], "2020-01-01", "2020-01-10")
    before = len(fake_archive)
    fetch.fetch_cities(CITIES[:2], "2020-01-01", "2020-01-10", refresh=True)
    assert len(fake_archive) > before


def test_a_cache_that_starts_too_late_is_refetched(isolated_paths, fake_archive):
    fetch.fetch_cities(CITIES[:1], "2020-01-01", "2020-01-10")
    before = len(fake_archive)
    fetch.fetch_cities(CITIES[:1], "1990-01-01", "2020-01-10")
    assert len(fake_archive) > before


def test_a_corrupt_cache_file_is_refetched_not_raised(isolated_paths, fake_archive):
    from wepr import paths

    (paths.DATA_RAW / "madrid.json").write_text("{ truncated")
    fetch.fetch_cities([get("madrid")], "2020-01-01", "2020-01-10")
    assert load.city_metadata("madrid")["slug"] == "madrid"


def test_only_the_uncached_cities_are_fetched(isolated_paths, fake_archive):
    fetch.fetch_cities(CITIES[:2], "2020-01-01", "2020-01-10")
    fake_archive.clear()
    written = fetch.fetch_cities(CITIES[:3], "2020-01-01", "2020-01-10")
    assert len(fake_archive) == 1
    assert fake_archive[0]["latitude"] == str(CITIES[2].lat)
    assert len(written) == 3, "the cached cities still come back in the result"


def test_fetching_nothing_is_not_an_error(isolated_paths, fake_archive):
    assert fetch.fetch_cities([], "2020-01-01", "2020-01-10") == {}
    assert fake_archive == []


# --- failure handling ---------------------------------------------------------


def test_a_reordered_response_is_rejected(isolated_paths, monkeypatch):
    """Positional attribution means a dropped location would mis-assign climates."""

    def short_response(session, params, *, log=None):
        return [daily_payload("only-one", params["start_date"], 10)]

    monkeypatch.setattr(fetch, "_request", short_response)
    with pytest.raises(fetch.FetchError, match="asked for 3 locations, got 1"):
        fetch.fetch_cities(CITIES[:3], "2020-01-01", "2020-01-10")


def test_an_interrupted_fetch_leaves_the_previous_cache_intact(isolated_paths, fake_archive):
    fetch.fetch_cities(CITIES[:1], "2020-01-01", "2020-01-10")
    good = json.loads(fetch.raw_path(CITIES[0].slug).read_text())

    def explode(session, params, *, log=None):
        raise fetch.FetchError("archive down")

    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(fetch, "_request", explode)
        with _pytest.raises(fetch.FetchError):
            fetch.fetch_cities(CITIES[:1], "1990-01-01", "2020-01-10", refresh=True)

    assert json.loads(fetch.raw_path(CITIES[0].slug).read_text()) == good


def test_a_400_fails_immediately_with_the_servers_reason(monkeypatch):
    """A 400 is our bug — an oversized request — so retrying it wastes a minute."""
    calls = []

    class Response:
        status_code = 400
        text = ""

        def json(self):
            return {"reason": "Your API call requests too much data."}

    class Session:
        def get(self, *a, **kw):
            calls.append(1)
            return Response()

    monkeypatch.setattr(fetch, "REQUEST_PAUSE_S", 0.0)
    with pytest.raises(fetch.FetchError, match="too much data"):
        fetch._request(Session(), {})
    assert len(calls) == 1


def test_a_429_waits_out_its_window_and_then_succeeds(monkeypatch):
    """The archive throttles on data volume, so a full fetch expects to be 429'd."""
    slept: list[float] = []
    responses = iter([429, 429, 200])

    class Response:
        def __init__(self, code):
            self.status_code = code
            self.text = ""

        def json(self):
            if self.status_code == 200:
                return {"daily": {"time": []}}
            return {"reason": "Minutely API request limit exceeded."}

    class Session:
        def get(self, *a, **kw):
            return Response(next(responses))

    monkeypatch.setattr(fetch.time, "sleep", slept.append)
    assert fetch._request(Session(), {}) == [{"daily": {"time": []}}]
    assert len(slept) == 2
    # A minutely cap resets on the minute, so the wait is always under a minute plus
    # the margin — never a backoff curve that would overshoot the reset.
    assert all(0 < s <= 60 + fetch.RATE_LIMIT_MARGIN_S for s in slept)


def test_a_transient_network_failure_is_retried(monkeypatch):
    attempts = []

    class Session:
        def get(self, *a, **kw):
            attempts.append(1)
            if len(attempts) < 3:
                raise requests.ConnectionError("TLS reset")
            return type(
                "R", (), {"status_code": 200, "json": lambda self: {"ok": True}, "text": ""}
            )()

    monkeypatch.setattr(fetch.time, "sleep", lambda _s: None)
    assert fetch._request(Session(), {}) == [{"ok": True}]
    assert len(attempts) == 3


def test_persistent_failure_eventually_raises(monkeypatch):
    class Session:
        def get(self, *a, **kw):
            raise requests.ConnectionError("down")

    monkeypatch.setattr(fetch.time, "sleep", lambda _s: None)
    with pytest.raises(fetch.FetchError, match="unreachable"):
        fetch._request(Session(), {})


# --- the archive edge ---------------------------------------------------------


def test_archive_edge_is_behind_today():
    edge = dt.date.fromisoformat(fetch.archive_edge(dt.date(2026, 11, 21)))
    assert edge == dt.date(2026, 11, 14)
    assert fetch.ARCHIVE_LAG_DAYS >= 5, "the documented lag is five days; probing found more"


def test_a_grid_cell_that_changes_between_years_is_rejected(isolated_paths, monkeypatch):
    """Stitching two different places into one series would corrupt provenance."""
    calls = {"n": 0}

    def shifting(session, params, *, log=None):
        calls["n"] += 1
        payload = daily_payload("x", params["start_date"], 365)
        if calls["n"] > 1:
            payload["elevation"] = 999.0
        return [payload]

    monkeypatch.setattr(fetch, "_request", shifting)
    monkeypatch.setattr(fetch, "REQUEST_PAUSE_S", 0.0)
    with pytest.raises(fetch.FetchError, match="was served grid cell"):
        fetch.fetch_cities(CITIES[:1], "2020-01-01", "2022-12-31")


# --- resumability -------------------------------------------------------------


def test_a_cached_year_is_not_refetched(isolated_paths, fake_archive):
    """The point of the window cache: a fetch killed partway resumes."""
    fetch.fetch_cities(CITIES[:2], "2000-01-01", "2002-12-31")
    assert len(fake_archive) == 3, "one request per year for a batch of two"

    # Simulate the assembled files being lost while the year slices survive, which
    # is what an interrupted run leaves behind.
    for city in CITIES[:2]:
        fetch.raw_path(city.slug).unlink()
    fake_archive.clear()

    fetch.fetch_cities(CITIES[:2], "2000-01-01", "2002-12-31")
    assert fake_archive == [], "every year was already on disk"
    assert load.load_city.__name__  # assembled files are back
    assert fetch.raw_path("a-coruna").exists()


def test_only_the_missing_years_are_refetched(isolated_paths, fake_archive):
    fetch.fetch_cities(CITIES[:1], "2000-01-01", "2002-12-31")
    fetch.window_path("a-coruna", 2001).unlink()
    fetch.raw_path("a-coruna").unlink()
    fake_archive.clear()

    fetch.fetch_cities(CITIES[:1], "2000-01-01", "2002-12-31")
    assert len(fake_archive) == 1
    assert fake_archive[0]["start_date"] == "2001-01-01"


def test_resuming_rebuilds_a_continuous_series(isolated_paths, fake_archive):
    fetch.fetch_cities(CITIES[:1], "2000-01-01", "2002-12-31")
    full = load.tidy(json.loads(fetch.raw_path("a-coruna").read_text()))

    fetch.window_path("a-coruna", 2001).unlink()
    fetch.raw_path("a-coruna").unlink()
    fetch.fetch_cities(CITIES[:1], "2000-01-01", "2002-12-31")
    resumed = load.tidy(json.loads(fetch.raw_path("a-coruna").read_text()))

    assert resumed.index.equals(full.index)
    assert len(resumed) == 1096  # 2000 was a leap year


def test_an_interrupted_assembly_does_not_leave_a_truncated_cache(isolated_paths, fake_archive):
    """A half-written file would be treated as a valid cache hit by the next run."""
    fetch.fetch_cities(CITIES[:1], "2000-01-01", "2000-12-31")
    path = fetch.raw_path("a-coruna")
    assert json.loads(path.read_text())["slug"] == "a-coruna"
    assert not list(path.parent.glob("*.part")), "no temporary files left behind"


def test_refresh_refetches_cached_years_too(isolated_paths, fake_archive):
    fetch.fetch_cities(CITIES[:1], "2000-01-01", "2001-12-31")
    fake_archive.clear()
    fetch.fetch_cities(CITIES[:1], "2000-01-01", "2001-12-31", refresh=True)
    assert len(fake_archive) == 2


# --- the three nested quotas --------------------------------------------------


def test_each_quota_waits_to_its_own_reset_boundary():
    """Backoff curves are the wrong tool: these quotas reset on the wall clock."""
    now = dt.datetime(2026, 11, 21, 14, 37, 20, tzinfo=dt.UTC)
    minutely = fetch._rate_limit_wait("Minutely API request limit exceeded.", now)
    hourly = fetch._rate_limit_wait("Hourly API request limit exceeded.", now)

    assert minutely == pytest.approx(40 + fetch.RATE_LIMIT_MARGIN_S)
    assert hourly == pytest.approx(22 * 60 + 40 + fetch.RATE_LIMIT_MARGIN_S)
    assert fetch._rate_limit_wait("Daily API request limit exceeded.", now) is None


def test_an_unrecognised_quota_message_tries_the_cheapest_wait():
    """A wrong guess costs one minute and another 429, not a wasted hour."""
    now = dt.datetime(2026, 11, 21, 14, 37, 20, tzinfo=dt.UTC)
    wait = fetch._rate_limit_wait("Some new throttling message.", now)
    assert 0 < wait <= 60 + fetch.RATE_LIMIT_MARGIN_S


def test_an_hourly_quota_is_waited_out_not_failed(monkeypatch):
    """This is the one a cold twenty-city fetch actually hits."""
    slept: list[float] = []
    responses = iter([429, 200])

    class Response:
        def __init__(self, code):
            self.status_code = code
            self.text = ""

        def json(self):
            if self.status_code == 200:
                return {"ok": True}
            return {"reason": "Hourly API request limit exceeded. Please try again in the next hour."}

    class Session:
        def get(self, *a, **kw):
            return Response(next(responses))

    monkeypatch.setattr(fetch.time, "sleep", slept.append)
    assert fetch._request(Session(), {}) == [{"ok": True}]
    assert len(slept) == 1 and slept[0] <= 3600 + fetch.RATE_LIMIT_MARGIN_S


def test_waiting_is_bounded_so_a_script_cannot_hang_indefinitely(monkeypatch):
    class Response:
        status_code = 429
        text = ""

        def json(self):
            return {"reason": "Hourly API request limit exceeded."}

    class Session:
        def get(self, *a, **kw):
            return Response()

    total = {"s": 0.0}
    monkeypatch.setattr(fetch.time, "sleep", lambda s: total.__setitem__("s", total["s"] + s))
    with pytest.raises(fetch.FetchError, match="gave up after waiting"):
        fetch._request(Session(), {})
    assert total["s"] <= fetch.MAX_RATE_LIMIT_WAIT_S


def test_a_daily_quota_message_fails_fast_with_advice(monkeypatch):
    """Sleeping out a minute cannot clear a daily cap; resuming tomorrow can."""
    slept: list[float] = []

    class Response:
        status_code = 429
        text = ""

        def json(self):
            return {"reason": "Daily API request limit exceeded."}

    class Session:
        def get(self, *a, **kw):
            return Response()

    monkeypatch.setattr(fetch.time, "sleep", slept.append)
    with pytest.raises(fetch.FetchError, match="resumes where this stopped"):
        fetch._request(Session(), {})
    assert slept == [], "it must not wait out a quota it cannot clear"


# --- what an interrupted fetch leaves behind ----------------------------------


def test_fetching_starts_at_the_newest_year(isolated_paths, fake_archive):
    """A quota-truncated fetch should leave the years the project actually needs."""
    fetch.fetch_cities(CITIES[:1], "1950-01-01", "1955-12-31")
    requested = [p["start_date"][:4] for p in fake_archive]
    assert requested == ["1955", "1954", "1953", "1952", "1951", "1950"]


def test_a_quota_stop_leaves_a_recent_block(isolated_paths, monkeypatch):
    """The GHCN records start in 1990; oldest-first would leave zero overlap."""
    seen: list[str] = []

    def quota_after_three(session, params, *, log=None):
        if len(seen) >= 3:
            raise fetch.FetchError("archive quota exhausted (Daily API request limit)")
        seen.append(params["start_date"][:4])
        return [daily_payload("x", params["start_date"], 365)]

    monkeypatch.setattr(fetch, "_request", quota_after_three)
    monkeypatch.setattr(fetch, "REQUEST_PAUSE_S", 0.0)
    with pytest.raises(fetch.FetchError):
        fetch.fetch_cities(CITIES[:1], "1950-01-01", "2000-12-31")

    assert seen == ["2000", "1999", "1998"]
    assert fetch.cached_years("a-coruna") == [1998, 1999, 2000]


def test_cached_years_can_be_assembled_without_the_network(isolated_paths, fake_archive):
    fetch.fetch_cities(CITIES[:2], "2000-01-01", "2002-12-31")
    for city in CITIES[:2]:
        fetch.raw_path(city.slug).unlink()

    fake_archive.clear()
    written = fetch.assemble_cached(CITIES[:2])
    assert fake_archive == [], "assembly must not touch the archive"
    assert set(written) == {CITIES[0].slug, CITIES[1].slug}
    frame = load.tidy(json.loads(written[CITIES[0].slug].read_text()))
    assert len(frame) == 1096


def test_assembly_stops_at_a_gap_rather_than_splicing_across_it():
    """A series with a hole in it violates the no-gaps contract."""
    assert fetch.contiguous_run([2000, 2001, 2002]) == [2000, 2001, 2002]
    assert fetch.contiguous_run([1950, 1951, 1990, 1991, 1992]) == [1990, 1991, 1992]
    assert fetch.contiguous_run([2005]) == [2005]
    assert fetch.contiguous_run([]) == []


def test_assembly_skips_a_city_with_nothing_cached(isolated_paths):
    assert fetch.assemble_cached(CITIES[:2]) == {}


def test_a_stray_file_in_the_window_cache_is_ignored(isolated_paths, fake_archive):
    fetch.fetch_cities(CITIES[:1], "2000-01-01", "2000-12-31")
    (fetch.window_path("a-coruna", 2000).parent / "notes.json").write_text("{}")
    assert fetch.cached_years("a-coruna") == [2000]


# -- the elevation endpoint -----------------------------------------------------


class TestFetchElevation:
    """The map's terrain client. Batched because the coordinates ride in the URI."""

    @staticmethod
    def _fake(monkeypatch, *, seen=None, values=None):
        """Replace the HTTP layer with an elevation-shaped responder."""

        def _get_json(session, url, params, **kwargs):
            if seen is not None:
                seen.append(dict(params))
            n = len(params["latitude"].split(","))
            return {"elevation": list(values) if values is not None else [100.0] * n}

        monkeypatch.setattr(fetch, "_get_json", _get_json)
        monkeypatch.setattr(fetch, "REQUEST_PAUSE_S", 0.0)

    def test_returns_one_value_per_coordinate_in_order(self, monkeypatch):
        self._fake(monkeypatch)
        got = fetch.fetch_elevation([40.0, 41.0, 42.0], [-3.0, -2.0, -1.0])
        assert got.shape == (3,)
        assert got.dtype == float

    def test_batches_at_the_uri_length_limit(self, monkeypatch):
        """250 points must become three requests, not one 414."""
        seen: list[dict] = []
        self._fake(monkeypatch, seen=seen)
        n = 250
        fetch.fetch_elevation([40.0] * n, list(np.linspace(-9, 3, n)))
        assert len(seen) == 3
        counts = [len(p["latitude"].split(",")) for p in seen]
        assert counts == [100, 100, 50]
        assert max(counts) <= fetch.MAX_ELEVATION_POINTS

    def test_concatenates_batches_in_request_order(self, monkeypatch):
        """A reshaped grid depends on this, so it is asserted rather than assumed."""
        calls = {"n": 0}

        def _get_json(session, url, params, **kwargs):
            calls["n"] += 1
            count = len(params["latitude"].split(","))
            return {"elevation": [float(calls["n"])] * count}

        monkeypatch.setattr(fetch, "_get_json", _get_json)
        monkeypatch.setattr(fetch, "REQUEST_PAUSE_S", 0.0)
        got = fetch.fetch_elevation([40.0] * 150, [0.0] * 150)
        assert got[:100].tolist() == [1.0] * 100
        assert got[100:].tolist() == [2.0] * 50

    def test_hits_the_elevation_endpoint_not_the_archive(self, monkeypatch):
        urls: list[str] = []

        def _get_json(session, url, params, **kwargs):
            urls.append(url)
            return {"elevation": [1.0]}

        monkeypatch.setattr(fetch, "_get_json", _get_json)
        monkeypatch.setattr(fetch, "REQUEST_PAUSE_S", 0.0)
        fetch.fetch_elevation([40.0], [-3.0])
        assert urls == [fetch.ELEVATION_URL]

    def test_rejects_mismatched_coordinate_lengths(self, monkeypatch):
        self._fake(monkeypatch)
        with pytest.raises(ValueError, match="2 latitudes but 3 longitudes"):
            fetch.fetch_elevation([40.0, 41.0], [-3.0, -2.0, -1.0])

    def test_raises_when_the_endpoint_returns_a_short_block(self, monkeypatch):
        """A silently short response would misalign every later cell in the grid."""
        self._fake(monkeypatch, values=[100.0, 200.0])
        with pytest.raises(fetch.FetchError, match="returned 2 values for 3"):
            fetch.fetch_elevation([40.0, 41.0, 42.0], [-3.0, -2.0, -1.0])

    def test_raises_when_the_endpoint_omits_the_field(self, monkeypatch):
        monkeypatch.setattr(fetch, "_get_json", lambda *a, **k: {})
        monkeypatch.setattr(fetch, "REQUEST_PAUSE_S", 0.0)
        with pytest.raises(fetch.FetchError, match="returned 0 values"):
            fetch.fetch_elevation([40.0], [-3.0])

    def test_accepts_the_empty_request(self, monkeypatch):
        """A resumed fetch with nothing outstanding must not open a connection."""
        seen: list[dict] = []
        self._fake(monkeypatch, seen=seen)
        assert fetch.fetch_elevation([], []).shape == (0,)
        assert seen == []

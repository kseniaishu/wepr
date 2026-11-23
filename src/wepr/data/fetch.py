"""Open-Meteo ERA5 archive client.

**Why this source.** The original project used two NOAA GHCN-Daily station exports
with divergent schemas and heavy missingness (Valencia: TMIN 43% missing, PRCP 17%).
The Open-Meteo archive serves any lat/lon at ~9 km resolution from 1950 to the
near-present, in °C and mm, with no gaps and no API key. That removes the station
hunt, the schema divergence, the unit conversions and the missing-data problem at
once, and extends the record from 34 years to 76.

**Which model, and why it is not pure ERA5-Land.** Open-Meteo exposes ERA5-Land
(`era5_land`, 0.1° ≈ 9 km) and ERA5 (`era5`, 0.25° ≈ 28 km) separately, plus a
blend, `era5_seamless`. A direct probe of the archive showed that `era5_land`
returns `precipitation_sum` as **null for every date from 1950 to the present** —
Open-Meteo simply does not serve that field for the land model. Requesting
`era5_land` alone would therefore mean no precipitation at all.

`era5_seamless` returns ERA5-Land's temperature verbatim (identical values, same
0.1° grid cell, verified against `era5_land` on sample years) and fills
precipitation from ERA5. So this project's temperature *is* ERA5-Land, while its
precipitation is coarser ERA5. That strengthens rather than weakens the caveat
already stated below: reanalysis precipitation is the less trustworthy of the two
fields, and here it is also the lower-resolution one.

**The caveat, which belongs in the README and the UI.** ERA5 and ERA5-Land are
*reanalysis*, not direct observation: a physical model with observations
assimilated into it. For this use it is defensible and arguably preferable —
complete, consistent and spatially uniform — but it is not station measurement,
and reanalysis precipitation is materially less reliable than its temperature,
particularly for extremes and in mountainous terrain. See `wepr.data.validate`
for the cross-check against the real Huesca and Valencia station records.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import requests

from wepr import paths
from wepr.cities import City

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# The elevation endpoint, used to build the map's land mask and its elevation
# covariate. It is the *same* digital elevation model the archive reports for the
# cell it served — Madrid's city fetch and a bare elevation lookup at the same
# coordinate both return 666.0 m — which is why the map's terrain and the data's
# terrain cannot disagree. See `wepr.spatial`.
ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"

# Points per elevation request. The endpoint takes coordinates in the query string
# and nginx rejects a long URI with 414 before Open-Meteo ever sees it: 100 points
# succeeds, 500 does not. This is a URI-length limit, not a documented API limit,
# so it is set conservatively.
MAX_ELEVATION_POINTS = 100

# Daily variables requested for every city. The archive serves these in °C and mm,
# so no unit conversion happens anywhere in this codebase.
DAILY_VARIABLES: tuple[str, ...] = (
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
)

# ERA5-Land temperature + ERA5 precipitation. See the module docstring: asking for
# `era5_land` alone yields null precipitation for the entire record.
MODEL = "era5_seamless"

# The archive lags real time while observations are assimilated. The documented lag
# is five days; probing the live API found the last populated day to be six to seven
# days back, so the default start of the window is deliberately conservative. Asking
# for a date past the edge is not an error — it returns nulls, which `load` drops.
ARCHIVE_LAG_DAYS = 7

# One request may not exceed roughly 16,000 location-days at four variables, found by
# bisecting against the live API (20 cities x 2 years passes, x 3 years is rejected
# with "requests too much data"). 14,000 leaves headroom in case the server-side
# budget tightens.
MAX_LOCATION_DAYS = 14_000

# Requests are one calendar year wide, so the most cities one can carry is the budget
# over the length of a leap year.
MAX_BATCH_CITIES = MAX_LOCATION_DAYS // 366

# Courtesy pause between requests. The free tier throttles on data volume per minute,
# not on request count, so pacing costs little and makes a 429 less likely. The total
# volume is fixed by the record length either way.
REQUEST_PAUSE_S = 1.0

# Transient failures seen while probing: TLS resets mid-handshake, and 5xx from the
# archive. Retry with linear backoff rather than failing a 40-request job on one.
MAX_RETRIES = 8

# The free tier enforces three nested quotas — per minute, per hour, per day — and a
# 429 names which one was hit. They reset on wall-clock boundaries, so the right wait
# is "until the next boundary", not a backoff curve: linear backoff from two seconds
# never clears an hourly cap, and sleeping an hour would be absurd for a minutely one.
# Being throttled is the expected cost of a cold 76-year fetch, not a fault, so these
# waits do not spend the transient-failure budget.
RATE_LIMIT_MARGIN_S = 5.0

# Total time the client will spend waiting out quotas before giving up. Two hours
# covers an hourly reset comfortably; past that, the daily cap is the likely cause and
# resuming later is cheaper than holding a process open. Every year already fetched is
# on disk, so giving up costs only the years still missing.
MAX_RATE_LIMIT_WAIT_S = 2 * 60 * 60


class FetchError(RuntimeError):
    """The archive could not be fetched, or answered with something unusable."""


def raw_path(slug: str, dest: Path | None = None) -> Path:
    """Where one city's raw cached response lives."""
    return (dest or paths.DATA_RAW) / f"{slug}.json"


def archive_edge(today: dt.date | None = None) -> str:
    """The most recent date worth requesting, as ISO `YYYY-MM-DD`."""
    today = today or dt.date.today()
    return (today - dt.timedelta(days=ARCHIVE_LAG_DAYS)).isoformat()


def _date_windows(start: str, end: str) -> Iterator[tuple[int, str, str]]:
    """Split `start`..`end` into one window per calendar year.

    **Why the calendar year, rather than the largest window the budget allows.**
    Packing requests to the limit would need ~40 calls instead of ~77, but the
    boundaries would then depend on how many cities happened to be in the batch, so
    the same year fetched in two different runs would land in two differently-shaped
    files and could not be reused. A year is a stable key, which is what makes the
    fetch resumable — and resumability is the property that matters here, because the
    free tier's quota is on data volume and a cold 76-year fetch of twenty cities is
    big enough to exhaust the daily allowance partway through.

    The total volume requested is identical either way; only the number of calls
    differs, and calls are not the binding constraint.

    Yields:
        `(year, window_start, window_end)`, clipped to the requested range.
    """
    first = dt.date.fromisoformat(start)
    last = dt.date.fromisoformat(end)
    if last < first:
        raise ValueError(f"end {end} precedes start {start}")

    for year in range(first.year, last.year + 1):
        window_start = max(first, dt.date(year, 1, 1))
        window_end = min(last, dt.date(year, 12, 31))
        yield year, window_start.isoformat(), window_end.isoformat()


def _batches(cities: Sequence[City]) -> Iterator[tuple[City, ...]]:
    """Split cities into groups small enough to share one year-wide request."""
    for i in range(0, len(cities), MAX_BATCH_CITIES):
        yield tuple(cities[i : i + MAX_BATCH_CITIES])


def window_path(slug: str, year: int, dest: Path | None = None) -> Path:
    """Where one city-year slice is cached.

    These are an intermediate artifact, not the contract: `load` reads the assembled
    per-city file. They exist so an interrupted fetch resumes instead of restarting.
    """
    return (dest or paths.DATA_RAW) / "windows" / slug / f"{year}.json"


def _get_json(
    session: requests.Session,
    url: str,
    params: dict[str, str],
    *,
    what: str = "archive",
    resume_hint: str = (
        "Every year already fetched is cached, so re-running later resumes where "
        "this stopped rather than starting over."
    ),
    log: Callable[[str], None] | None = None,
):
    """One Open-Meteo request, retried on transient failure.

    Shared by the archive client and the elevation client below. They hit
    different endpoints but sit behind the same quota and the same 429 semantics,
    so the retry and rate-limit policy is written once here; `what` and
    `resume_hint` only change what the error says.
    """
    note = log or (lambda _msg: None)
    last_error: Exception | None = None
    attempt = 0
    waited = 0.0

    while attempt < MAX_RETRIES:
        try:
            response = session.get(url, params=params, timeout=180)
        except requests.RequestException as exc:  # TLS resets, timeouts, DNS
            last_error = exc
        else:
            if response.status_code == 200:
                return response.json()

            reason = _reason(response)

            # 400 is a malformed request — our bug, not the network's. Fail at once
            # with the server's own explanation rather than retrying eight times.
            if response.status_code == 400:
                raise FetchError(f"{what} rejected the request: {reason}")

            if response.status_code == 429:
                wait = _rate_limit_wait(reason)
                if wait is None:
                    raise FetchError(f"{what} quota exhausted ({reason}). {resume_hint}")
                if waited + wait > MAX_RATE_LIMIT_WAIT_S:
                    raise FetchError(
                        f"gave up after waiting {waited / 60:.0f} min on quotas "
                        f"({reason}). Progress is cached; re-run to resume."
                    )
                waited += wait
                note(f"rate-limited ({reason}); waiting {wait / 60:.1f} min")
                time.sleep(wait)
                continue  # deliberately not counted as a retry attempt

            last_error = FetchError(f"HTTP {response.status_code}: {reason}")

        attempt += 1
        note(f"transient failure ({last_error}); retry {attempt}/{MAX_RETRIES}")
        time.sleep(2 * attempt)

    raise FetchError(f"{what} unreachable after {MAX_RETRIES} attempts: {last_error}")


def _request(
    session: requests.Session,
    params: dict[str, str],
    *,
    log: Callable[[str], None] | None = None,
) -> list[dict]:
    """One archive request, normalised to a list.

    Open-Meteo returns a bare object for a single location and a list for several,
    so callers get a list either way.
    """
    payload = _get_json(session, ARCHIVE_URL, params, log=log)
    return payload if isinstance(payload, list) else [payload]


def _reason(response: requests.Response) -> str:
    """The server's own explanation for a non-200, if it gave one."""
    try:
        return str(response.json().get("reason", response.text[:200]))
    except ValueError:
        return response.text[:200]


def _rate_limit_wait(reason: str, now: dt.datetime | None = None) -> float | None:
    """How long to wait out a 429, or `None` if waiting cannot clear it.

    The quotas reset on wall-clock boundaries, so this returns the time to the next
    relevant boundary plus a small margin, rather than a backoff interval.

    Args:
        reason: The server's own explanation, e.g. "Hourly API request limit
            exceeded."
        now: Current UTC time; injectable so the arithmetic is testable.

    Returns:
        Seconds to sleep, or `None` for the daily quota — which no reasonable wait
        clears, and which a resumable fetch should simply be re-run after.
    """
    now = now or dt.datetime.now(dt.UTC)
    text = reason.lower()

    if "daily" in text:
        return None
    if "hourly" in text:
        boundary = (now + dt.timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    else:
        # Minutely, or an unrecognised phrasing — the minute boundary is the cheapest
        # thing to try, and a wrong guess simply yields another 429.
        boundary = (now + dt.timedelta(minutes=1)).replace(second=0, microsecond=0)

    return (boundary - now).total_seconds() + RATE_LIMIT_MARGIN_S


def _covers(cached: dict, start: str, end: str) -> bool:
    """Whether a cached payload already spans the requested dates.

    The archive is immutable for past dates, so a cache hit is always correct. The
    requested end is capped at the archive edge before comparing: a request reaching
    past the edge can never be satisfied, so treating it as a miss would refetch the
    whole record on every run. Dates are ISO `YYYY-MM-DD`, which compares correctly
    as a string.
    """
    times = cached.get("daily", {}).get("time", [])
    if not times:
        return False
    return times[0] <= start and times[-1] >= min(end, archive_edge())


def fetch_cities(
    cities: Sequence[City],
    start: str,
    end: str,
    dest: Path | None = None,
    *,
    refresh: bool = False,
    session: requests.Session | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Path]:
    """Fetch daily series for several cities and cache them to `dest`.

    Open-Meteo accepts comma-separated lat/lon lists, so a batch of cities costs
    one request rather than one each. Results are cached as files and never
    re-fetched silently: the archive is immutable for past dates, so a cache hit
    is always correct, and an explicit refresh is an explicit action.

    Caching happens at two levels. Each city-year lands under
    `data/raw/windows/<slug>/<year>.json` as it arrives, and those slices are then
    stitched into the one `data/raw/<slug>.json` that `load` reads. The intermediate
    layer exists because a cold fetch of twenty cities over 76 years is large enough
    to exhaust the free tier's daily volume quota partway through; with it, re-running
    resumes from the first year still missing instead of starting over.

    The grid cell the server actually served — its latitude, longitude and
    elevation — is recorded alongside the series. That elevation is the number
    that matters for interpreting the data, and in broken terrain it can differ
    from the city's official altitude by hundreds of metres, which is exactly why
    `wepr.cities` does not hard-code one.

    Args:
        cities: Cities to fetch.
        start: First date, ISO `YYYY-MM-DD`.
        end: Last date, ISO `YYYY-MM-DD`.
        dest: Directory for the raw files; defaults to `data/raw/`.
        refresh: Re-fetch even when a cached file already covers the range.
        session: Reuse an existing HTTP session; one is created if omitted.
        log: Called with progress and throttling messages. A full fetch takes
            minutes and spends much of it waiting out rate limits, so a silent
            run is indistinguishable from a hung one.

    Returns:
        Mapping of city slug to the raw file written (or reused from cache).

    Raises:
        FetchError: if the archive is unreachable or answers unusably.
    """
    if not cities:
        return {}

    destination = dest or paths.DATA_RAW
    destination.mkdir(parents=True, exist_ok=True)
    note = log or (lambda _msg: None)

    written: dict[str, Path] = {}
    pending: list[City] = []
    for city in cities:
        path = raw_path(city.slug, destination)
        if not refresh and path.exists():
            try:
                cached = json.loads(path.read_text())
            except json.JSONDecodeError:
                cached = {}
            if _covers(cached, start, end):
                written[city.slug] = path
                continue
        pending.append(city)

    if not pending:
        return written

    windows = list(_date_windows(start, end))
    owned_session = session is None
    session = session or requests.Session()
    note(
        f"{len(pending)} city/cities x {len(windows)} year(s); "
        f"{'refreshing' if refresh else 'reusing any cached year'}"
    )

    try:
        # Newest year first. Under a volume quota a cold fetch may not finish, and
        # what it leaves behind should be the years that matter most: the GHCN
        # validation records start in 1990, and a next-day model is trained on the
        # recent end of the series. Oldest-first would spend the quota on the 1950s
        # and leave the validation exercise with no overlap at all.
        for year, window_start, window_end in reversed(windows):
            # Only ask for the cities whose slice of this year is not already on disk.
            # This is what makes an interrupted fetch resume rather than restart.
            missing = [
                c
                for c in pending
                if refresh or not window_path(c.slug, year, destination).exists()
            ]
            if not missing:
                continue

            for batch in _batches(missing):
                payload = _request(
                    session,
                    {
                        "latitude": ",".join(str(c.lat) for c in batch),
                        "longitude": ",".join(str(c.lon) for c in batch),
                        "start_date": window_start,
                        "end_date": window_end,
                        "daily": ",".join(DAILY_VARIABLES),
                        "timezone": "UTC",
                        "models": MODEL,
                    },
                    log=note,
                )
                # Locations come back in request order. Assert that rather than trust
                # it: a silent reordering would mis-attribute whole cities' climates.
                if len(payload) != len(batch):
                    raise FetchError(
                        f"asked for {len(batch)} locations, got {len(payload)} back "
                        f"for {window_start}..{window_end}"
                    )
                for city, block in zip(batch, payload, strict=True):
                    daily = block.get("daily")
                    if not daily or "time" not in daily:
                        raise FetchError(
                            f"no daily block for {city.slug} in {window_start}..{window_end}"
                        )
                    path = window_path(city.slug, year, destination)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    _write_atomic(
                        path,
                        {
                            "grid_latitude": block.get("latitude"),
                            "grid_longitude": block.get("longitude"),
                            "grid_elevation_m": block.get("elevation"),
                            "daily_units": block.get("daily_units", {}),
                            "daily": daily,
                        },
                    )
                time.sleep(REQUEST_PAUSE_S)
            note(f"{year}  ({len(missing)} city/cities)")
    finally:
        if owned_session:
            session.close()

    fetched_at = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    for city in pending:
        written[city.slug] = _assemble(
            city, windows, start, end, fetched_at, destination
        )
    return written


def _write_atomic(path: Path, payload: dict) -> None:
    """Write JSON via a temporary file and rename.

    A rename is atomic on POSIX, so a process killed mid-write leaves either the old
    file or the new one — never a truncated file that a later run would treat as a
    valid cache hit.
    """
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(payload, indent=1))
    temporary.replace(path)


def _assemble(
    city: City,
    windows: Sequence[tuple[int, str, str]],
    start: str,
    end: str,
    fetched_at: str,
    destination: Path,
) -> Path:
    """Stitch one city's cached years into the single raw file `load` reads."""
    series: dict[str, list] = {}
    cell: dict | None = None
    units: dict = {}

    for year, window_start, window_end in windows:
        path = window_path(city.slug, year, destination)
        if not path.exists():
            raise FetchError(f"{city.slug}: {year} missing after fetch ({path})")
        block = json.loads(path.read_text())
        for key, values in block["daily"].items():
            series.setdefault(key, []).extend(values)

        this_cell = {k: block[k] for k in ("grid_latitude", "grid_longitude", "grid_elevation_m")}
        # The same coordinate must resolve to the same cell in every year; otherwise
        # the stitched series splices two different places together and the recorded
        # provenance is a lie about part of it.
        if cell is not None and this_cell != cell:
            raise FetchError(
                f"{city.slug}: {year} was served grid cell {this_cell}, "
                f"earlier years {cell}"
            )
        cell = this_cell
        units = block.get("daily_units", {})

    path = raw_path(city.slug, destination)
    _write_atomic(
        path,
        {
            "slug": city.slug,
            "name": city.name,
            "country": city.country,
            "region": city.region,
            "requested_latitude": city.lat,
            "requested_longitude": city.lon,
            "model": MODEL,
            "source": "Open-Meteo archive (ERA5-Land temperature, ERA5 precipitation)",
            "requested_start": start,
            "requested_end": end,
            "fetched_at_utc": fetched_at,
            "daily_units": units,
            **(cell or {}),
            "daily": series,
        },
    )
    return path


def cached_years(slug: str, dest: Path | None = None) -> list[int]:
    """Years of this city already on disk, sorted ascending."""
    directory = (dest or paths.DATA_RAW) / "windows" / slug
    if not directory.exists():
        return []
    years = []
    for path in directory.glob("*.json"):
        try:
            years.append(int(path.stem))
        except ValueError:
            continue  # a stray file, not a year slice
    return sorted(years)


def contiguous_run(years: Sequence[int]) -> list[int]:
    """The unbroken run of years ending at the latest one.

    Assembling across a hole would splice two eras into one series and break the
    no-gaps contract, so only an unbroken run is usable. It is anchored at the
    newest year because that is where fetching starts: an interrupted fetch leaves a
    recent block, and a recent block is what the models and the validation need.
    Years before a gap are kept on disk — they cost quota to obtain — and join the
    run as soon as the gap is filled.
    """
    if not years:
        return []
    ordered = sorted(years)
    run = [ordered[-1]]
    for year in reversed(ordered[:-1]):
        if year != run[0] - 1:
            break
        run.insert(0, year)
    return run


def assemble_cached(
    cities: Sequence[City],
    dest: Path | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Path]:
    """Build per-city raw files from whatever years are already cached.

    This is what makes a quota-interrupted fetch useful rather than merely
    resumable: a shorter span is still a perfectly valid daily series, and the rest
    of the pipeline can be developed and tested against it while the remaining years
    wait for the quota to reset. Only the contiguous run ending at the newest cached
    year is used — see `contiguous_run`.

    Returns:
        Mapping of slug to the assembled file, omitting cities with nothing cached.
    """
    destination = dest or paths.DATA_RAW
    note = log or (lambda _msg: None)
    fetched_at = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")

    written: dict[str, Path] = {}
    for city in cities:
        years = contiguous_run(cached_years(city.slug, destination))
        if not years:
            note(f"{city.slug}: nothing cached")
            continue
        windows = [
            (year, f"{year}-01-01", f"{year}-12-31") for year in years
        ]
        written[city.slug] = _assemble(
            city,
            windows,
            f"{years[0]}-01-01",
            f"{years[-1]}-12-31",
            fetched_at,
            destination,
        )
        note(f"{city.slug}: assembled {years[0]}-{years[-1]} ({len(years)} years)")
    return written


def fetch_point(
    lat: float,
    lon: float,
    start: str,
    end: str,
    *,
    session: requests.Session | None = None,
    log: Callable[[str], None] | None = None,
) -> dict:
    """Fetch one arbitrary coordinate, without touching the city cache.

    Used by the validation exercise, which needs the grid cell over a GHCN
    *station* rather than over a city centre, and by nothing else.

    Returns:
        The raw Open-Meteo block, including the served grid cell and elevation.
    """
    owned_session = session is None
    session = session or requests.Session()
    note = log or (lambda _msg: None)
    try:
        blocks: dict[str, list] = {}
        meta: dict = {}
        for _year, window_start, window_end in _date_windows(start, end):
            payload = _request(
                session,
                {
                    "latitude": str(lat),
                    "longitude": str(lon),
                    "start_date": window_start,
                    "end_date": window_end,
                    "daily": ",".join(DAILY_VARIABLES),
                    "timezone": "UTC",
                    "models": MODEL,
                },
                log=note,
            )
            block = payload[0]
            for key, values in block.get("daily", {}).items():
                blocks.setdefault(key, []).extend(values)
            if not meta:
                meta = {
                    "grid_latitude": block.get("latitude"),
                    "grid_longitude": block.get("longitude"),
                    "grid_elevation_m": block.get("elevation"),
                }
            time.sleep(REQUEST_PAUSE_S)
    finally:
        if owned_session:
            session.close()
    return {**meta, "daily": blocks}


def fetch_elevation(
    lats: Sequence[float],
    lons: Sequence[float],
    *,
    session: requests.Session | None = None,
    log: Callable[[str], None] | None = None,
) -> "np.ndarray":
    """Ground elevation in metres for a sequence of coordinates.

    Batched at `MAX_ELEVATION_POINTS` per request because the endpoint carries its
    coordinates in the URI.

    **Sea reads as exactly 0.0.** The endpoint clamps at zero rather than returning
    bathymetry, which is what makes a land mask possible at all — but it also means
    genuine land at or below sea level is indistinguishable from water. The
    Guadalquivir marshes and the Valencian Albufera both come back 0.0. That
    limitation is handled, and stated, in `wepr.spatial.land_mask`.

    Args:
        lats: Latitudes, decimal degrees north.
        lons: Longitudes, decimal degrees east. Same length as `lats`.
        session: Optional session to reuse across calls.
        log: Optional progress callback.

    Returns:
        Elevations in metres, in the order the coordinates were given.

    Raises:
        ValueError: If the two coordinate sequences differ in length.
        FetchError: If the endpoint cannot be reached, or returns a short block.
    """
    import numpy as np

    if len(lats) != len(lons):
        raise ValueError(f"{len(lats)} latitudes but {len(lons)} longitudes")

    note = log or (lambda _msg: None)
    owned_session = session is None
    session = session or requests.Session()
    out: list[float] = []
    total = len(lats)
    try:
        for start in range(0, total, MAX_ELEVATION_POINTS):
            chunk_lats = lats[start : start + MAX_ELEVATION_POINTS]
            chunk_lons = lons[start : start + MAX_ELEVATION_POINTS]
            payload = _get_json(
                session,
                ELEVATION_URL,
                {
                    "latitude": ",".join(f"{v:.4f}" for v in chunk_lats),
                    "longitude": ",".join(f"{v:.4f}" for v in chunk_lons),
                },
                what="elevation endpoint",
                resume_hint=(
                    "The elevation grid is a single cached artifact; re-run the "
                    "script later to rebuild it."
                ),
                log=note,
            )
            values = payload.get("elevation")
            if values is None or len(values) != len(chunk_lats):
                raise FetchError(
                    f"elevation endpoint returned {len(values or ())} values for "
                    f"{len(chunk_lats)} coordinates"
                )
            out.extend(values)
            note(f"elevation: {len(out)}/{total} points")
            time.sleep(REQUEST_PAUSE_S)
    finally:
        if owned_session:
            session.close()

    return np.asarray(out, dtype=float)

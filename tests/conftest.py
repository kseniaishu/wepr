"""Shared fixtures for the data-layer tests.

Nothing here touches the network. The Open-Meteo client is exercised against
synthetic payloads, because a test that depends on a live archive fails for
reasons that have nothing to do with the code under test.
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from wepr.data import fetch, load


def daily_payload(
    slug: str = "testville",
    start: str = "2020-01-01",
    days: int = 10,
    *,
    trailing_nulls: int = 0,
) -> dict:
    """A raw payload shaped exactly like one Open-Meteo window.

    `trailing_nulls` appends dates with no values, which is what the archive
    returns for a window reaching past its edge.
    """
    first = dt.date.fromisoformat(start)
    dates = [(first + dt.timedelta(days=i)).isoformat() for i in range(days + trailing_nulls)]
    real = range(days)

    def series(base: float, step: float) -> list[float | None]:
        return [base + step * i for i in real] + [None] * trailing_nulls

    return {
        "slug": slug,
        "name": slug.title(),
        "model": fetch.MODEL,
        # The API's own key names for the grid cell it served. `fetch_cities`
        # renames these to the `grid_*` form when it writes the cache, so a payload
        # carrying both is valid either as a response or as a cache file.
        "latitude": 40.0,
        "longitude": -3.0,
        "elevation": 650.0,
        "grid_latitude": 40.0,
        "grid_longitude": -3.0,
        "grid_elevation_m": 650.0,
        "daily": {
            "time": dates,
            "temperature_2m_mean": series(12.0, 0.1),
            "temperature_2m_max": series(18.0, 0.1),
            "temperature_2m_min": series(6.0, 0.1),
            "precipitation_sum": series(0.0, 0.5),
        },
    }


def tidy_frame(start: str = "2020-01-01", days: int = 10) -> pd.DataFrame:
    """A frame that satisfies the tidy daily contract."""
    return load.tidy(daily_payload(start=start, days=days))


def synthetic_daily_frame(
    first_year: int = 1980,
    n_years: int = 40,
    slope_per_year: float = 0.03,
    seed: int = 7,
) -> pd.DataFrame:
    """A long tidy daily frame with a known seasonal cycle and a known trend.

    Long records are what the trend and seasonal modules need and what
    `daily_payload` deliberately is not — the archive fixtures above are about the
    client's windowing, this is about years of weather. The cycle amplitude and the
    slope are both known constants so a test can assert the fit recovers them
    rather than merely runs.
    """
    index = pd.date_range(
        f"{first_year}-01-01", f"{first_year + n_years - 1}-12-31", freq="D", name="date"
    )
    rng = np.random.default_rng(seed)
    doy = index.day_of_year.to_numpy()
    elapsed = index.year.to_numpy() - first_year
    tavg = (
        15.0
        - 9.0 * np.cos(2 * np.pi * doy / 365.25)
        + slope_per_year * elapsed
        + rng.normal(0, 2.0, len(index))
    )
    wet = rng.random(len(index)) < 0.3
    return pd.DataFrame(
        {
            "tavg": tavg,
            "tmax": tavg + 6.0,
            "tmin": tavg - 6.0,
            "prcp": np.where(wet, rng.exponential(4.0, len(index)), 0.0),
        },
        index=index,
    )


@pytest.fixture
def synthetic_daily():
    """`synthetic_daily_frame` as a fixture, so tests can vary its parameters."""
    return synthetic_daily_frame


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    """Point every project data path at a temporary directory.

    The data layer writes files; a test that wrote into `data/` would clobber a
    real 76-year cache that takes minutes to refetch.
    """
    from wepr import paths

    for name in ("DATA", "DATA_RAW", "DATA_PROCESSED", "DATA_VALIDATION", "MODELS", "REPORTS"):
        monkeypatch.setattr(paths, name, tmp_path / name.lower())
        getattr(paths, name).mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def fake_archive(monkeypatch):
    """Replace the HTTP layer with a generator of well-formed payloads.

    Returns the list of request parameter dicts seen, so tests can assert on how
    the client batched and windowed its requests.
    """
    seen: list[dict] = []

    def _request(session, params, *, log=None):
        seen.append(dict(params))
        lats = params["latitude"].split(",")
        first = dt.date.fromisoformat(params["start_date"])
        last = dt.date.fromisoformat(params["end_date"])
        days = (last - first).days + 1
        return [
            daily_payload(f"loc{i}", params["start_date"], days)
            for i in range(len(lats))
        ]

    monkeypatch.setattr(fetch, "_request", _request)
    monkeypatch.setattr(fetch, "REQUEST_PAUSE_S", 0.0)
    return seen


@pytest.fixture
def write_raw(isolated_paths):
    """Write a raw payload to the isolated cache, as `fetch_cities` would."""

    def _write(slug: str, payload: dict) -> None:
        from wepr import paths

        (paths.DATA_RAW / f"{slug}.json").write_text(json.dumps(payload))

    return _write


def synthetic_elevation_grid(
    resolution: float = 0.1,
    margin_km: float | None = None,
    *,
    cover: bool = True,
):
    """A synthetic `ElevationGrid` over the project's real bounds.

    The map's own elevation artifact is a committed network fetch, and a test that
    needed it would fail on a clone that had not built it yet. So the terrain here
    is analytic and deliberately simple: a plane rising inland from the Atlantic,
    with a ridge along the Pyrenees, and sea wherever the cell is outside a crude
    land rule. What the spatial tests assert is the *machinery* — masking, the
    surface fit, the distance weighting — not Iberia's real coastline.

    Args:
        resolution: Grid spacing, matched against callers' expectations.
        margin_km: The coverage the grid claims. Defaults to
            `spatial.HULL_MARGIN_KM`.
        cover: If False, leave cells beyond 60 km of the hull as NaN, which is
            what the real artifact does outside its fetched region.
    """
    import numpy as np

    from wepr import spatial

    # Defaults to whatever the module actually draws to, so moving the margin
    # does not quietly leave every field test asking for terrain it lacks.
    margin_km = spatial.HULL_MARGIN_KM if margin_km is None else margin_km

    lats, lons = spatial.grid_axes(resolution)
    mesh_lat, mesh_lon = np.meshgrid(lats, lons, indexing="ij")

    # Rises eastward from the Atlantic, plus a ridge at the Pyrenean latitude, so
    # a lapse-rate term has something to bite on.
    elevation = (
        40.0
        + 60.0 * (mesh_lon + 9.9)
        + 2200.0 * np.exp(-(((mesh_lat - 42.7) / 0.35) ** 2) - ((mesh_lon - 0.3) / 2.0) ** 2)
    )

    # A crude coast: sea west of mainland Portugal, in Biscay, and in the Med.
    sea = (
        (mesh_lon < -9.6)
        | ((mesh_lat > 43.6) & (mesh_lon < -1.8))
        | ((mesh_lat < 36.6) & (mesh_lon > -5.0))
        | ((mesh_lon > 0.6) & (mesh_lat < 40.4) & (mesh_lat > 39.9))
    )
    elevation = np.where(sea, 0.0, elevation)

    if not cover:
        from wepr.cities import CITIES

        hull = spatial.hull_distance_km(
            np.array([c.lat for c in CITIES]),
            np.array([c.lon for c in CITIES]),
            lats,
            lons,
        )
        elevation = np.where(hull <= 60.0, elevation, np.nan)

    return spatial.ElevationGrid(
        lats=lats,
        lons=lons,
        elevation=elevation,
        resolution=resolution,
        margin_km=margin_km,
        source="synthetic terrain (tests/conftest.py)",
    )


@pytest.fixture
def synthetic_elevation():
    """`synthetic_elevation_grid` as a fixture, so tests can vary its parameters."""
    return synthetic_elevation_grid


@pytest.fixture
def trained_project(isolated_paths, monkeypatch):
    """A complete miniature project on disk: data, models and metrics.

    Phase 6's serving layer refuses to serve a model that has no entry in
    `reports/metrics.json`, so testing it needs all three artifacts present and
    keyed consistently. Building them here — by calling the same training code the
    scripts call — keeps the tests hermetic: they neither read the real `models/`
    (which a fresh clone does not have) nor assert against numbers that change
    when the archive is refetched.

    Returns a callable so a test can train a subset of cities, or deliberately
    omit a metrics entry to exercise the refusal.
    """
    import json

    from wepr import features, paths
    from wepr.data import load
    from wepr.models import nextday, registry, seasonal

    def _build(slugs=("madrid", "seville"), *, write_metrics=True, estimator_name=None):
        entries = {}
        for i, slug in enumerate(slugs):
            frame = synthetic_daily_frame(first_year=1980, n_years=40, seed=7 + i)
            load.write_processed(slug, frame)

            engineered = features.add_targets(features.build_features(frame))
            predictors = features.feature_columns(engineered)

            models = {
                registry.model_key(slug, "seasonal", "tavg"): seasonal.SeasonalClimatology().fit(frame),
                registry.model_key(slug, "nextday", "tavg"): nextday.fit_temperature(engineered, predictors),
                registry.model_key(slug, "nextday", "prcp"): nextday.fit_precipitation(engineered, predictors),
            }
            for key, model in models.items():
                registry.save(model, key)
                entries[key] = {
                    "model": key,
                    "mae": 1.0,
                    "rmse": 1.4,
                    "n_folds": 12,
                    "n_observations": 4000,
                    "provisional": True,
                    "product": key.split(".")[1],
                    # The class actually persisted, unless a test is deliberately
                    # creating the mismatch the store is supposed to catch.
                    "estimator": estimator_name or type(model).__name__,
                    "deployed_estimator": estimator_name or type(model).__name__,
                    "skill_vs_climatology": 0.5,
                    "skill_vs_persistence": 0.1,
                    "coverage": 0.79,
                    "nominal_coverage": 0.8,
                    "beats_both_baselines": True,
                }

        if write_metrics:
            paths.metrics_json().write_text(
                json.dumps(
                    {
                        "generated_at": "2026-01-01T00:00:00+00:00",
                        "gate": {"enforced": False, "note": "test fixture"},
                        "models": entries,
                    }
                )
            )
        return list(slugs)

    return _build

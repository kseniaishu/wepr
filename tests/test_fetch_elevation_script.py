"""The elevation-grid builder, and above all that it resumes.

Open-Meteo's daily request cap is shared with the archive client and is genuinely
reachable here: the first real attempt at this grid died at 2,800 of 11,664
points, and a second attempt the same day fetched nothing at all. So the script's
value is not that it fetches — it is that a quota wall costs you the remainder and
not the whole grid. That is what these tests hold in place.

The network is replaced throughout; what is under test is the bookkeeping.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from wepr import paths, spatial
from wepr.data import fetch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_elevation.py"


@pytest.fixture
def script():
    """Import `scripts/fetch_elevation.py` as a module.

    Scripts are not a package — importing by path is how a test reaches one
    without turning `scripts/` into something the app could import.
    """
    spec = importlib.util.spec_from_file_location("fetch_elevation_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def grid_dir(tmp_path, monkeypatch):
    """Point every artifact path at a temporary directory."""
    destination = tmp_path / "grid"
    destination.mkdir()
    monkeypatch.setattr(paths, "DATA_GRID", destination)
    monkeypatch.setattr(paths, "ELEVATION_GRID", destination / "elevation.npz")
    monkeypatch.setattr(paths, "ensure_dirs", lambda: None)
    return destination


@pytest.fixture
def coarse(monkeypatch):
    """Shrink the grid so a test is a handful of requests, not ninety."""
    monkeypatch.setattr(spatial, "RESOLUTION", 1.0)
    monkeypatch.setattr(fetch, "REQUEST_PAUSE_S", 0.0)


def _run(script, monkeypatch, *, fail_after=None, argv=()):
    """Run the script with a fake endpoint. Returns (exit code, points fetched)."""
    fetched = {"n": 0, "coords": []}

    def fake_fetch_elevation(lats, lons, **kwargs):
        if fail_after is not None and fetched["n"] >= fail_after:
            raise fetch.FetchError("Daily API request limit exceeded")
        fetched["n"] += len(lats)
        fetched["coords"].extend(zip(np.asarray(lats), np.asarray(lons)))
        # Land north of 40, sea south of it — arbitrary but deterministic.
        return np.where(np.asarray(lats) > 40.0, 500.0, 0.0)

    monkeypatch.setattr(script.fetch, "fetch_elevation", fake_fetch_elevation)
    monkeypatch.setattr(script.fetch, "MAX_ELEVATION_POINTS", 50)
    monkeypatch.setattr(sys, "argv", ["fetch_elevation.py", "--resolution", "1.0", *argv])
    return script.main(), fetched


def test_writes_a_grid_that_spatial_can_load(script, grid_dir, coarse, monkeypatch):
    code, _ = _run(script, monkeypatch)
    assert code == 0

    loaded = spatial.load_elevation(paths.ELEVATION_GRID)
    assert loaded.resolution == pytest.approx(1.0)
    assert loaded.margin_km == pytest.approx(spatial.HULL_MARGIN_KM)
    assert loaded.elevation.shape == (len(loaded.lats), len(loaded.lons))
    assert "Open-Meteo" in loaded.source


def test_fetches_only_cells_the_map_could_draw(script, grid_dir, coarse, monkeypatch):
    """Terrain beyond the drawable margin is data bought to be thrown away."""
    code, fetched = _run(script, monkeypatch)
    assert code == 0

    lats, lons = spatial.grid_axes(1.0)
    from wepr.cities import CITIES

    hull = spatial.hull_distance_km(
        np.array([c.lat for c in CITIES]), np.array([c.lon for c in CITIES]), lats, lons
    )
    needed = int((hull <= spatial.HULL_MARGIN_KM).sum())
    assert fetched["n"] == needed
    assert needed < hull.size, "the margin excluded nothing, so this proves nothing"


def test_cells_outside_the_margin_stay_nan_and_are_not_land(
    script, grid_dir, coarse, monkeypatch
):
    _run(script, monkeypatch)
    loaded = spatial.load_elevation(paths.ELEVATION_GRID)
    unfetched = ~np.isfinite(loaded.elevation)
    assert unfetched.any()
    assert not loaded.land[unfetched].any()


def test_a_quota_failure_keeps_what_arrived(script, grid_dir, coarse, monkeypatch):
    """The whole point: a daily cap must cost the remainder, not the grid."""
    code, fetched = _run(script, monkeypatch, fail_after=50)

    assert code == 1
    assert not paths.ELEVATION_GRID.exists(), "a partial grid must not be published"
    partial = script.partial_path()
    assert partial.exists(), "progress was discarded"

    with np.load(partial) as z:
        kept = int(np.isfinite(z["elevation"]).sum())
    assert kept == fetched["n"] == 50


def test_a_resumed_run_fetches_only_what_is_missing(script, grid_dir, coarse, monkeypatch):
    first_code, first = _run(script, monkeypatch, fail_after=50)
    assert first_code == 1

    second_code, second = _run(script, monkeypatch)
    assert second_code == 0
    assert paths.ELEVATION_GRID.exists()

    # The second run must not refetch the first run's points.
    assert second["n"] > 0
    assert first["n"] + second["n"] == _needed(1.0)


def test_resuming_clears_the_partial_once_the_grid_is_complete(
    script, grid_dir, coarse, monkeypatch
):
    _run(script, monkeypatch, fail_after=50)
    _run(script, monkeypatch)
    assert not script.partial_path().exists()


def test_a_resumed_run_preserves_the_first_runs_values(
    script, grid_dir, coarse, monkeypatch
):
    """Resume must stitch, not shift: a misaligned resume corrupts every cell."""
    _run(script, monkeypatch, fail_after=50)
    with np.load(script.partial_path()) as z:
        before = z["elevation"].copy()

    _run(script, monkeypatch)
    after = spatial.load_elevation(paths.ELEVATION_GRID).elevation

    known = np.isfinite(before)
    assert known.any()
    assert np.array_equal(after[known], before[known])


def test_a_partial_for_a_different_grid_is_discarded(
    script, grid_dir, coarse, monkeypatch
):
    """A partial built at another resolution cannot be extended into this one."""
    np.savez_compressed(
        script.partial_path(),
        lats=np.arange(5.0),
        lons=np.arange(5.0),
        elevation=np.zeros((5, 5)),
        resolution=np.array(0.25),
        margin_km=np.array(spatial.HULL_MARGIN_KM),
    )
    code, fetched = _run(script, monkeypatch)
    assert code == 0
    assert fetched["n"] == _needed(1.0), "a mismatched partial was trusted"


def test_a_corrupt_partial_is_discarded_rather_than_crashing(
    script, grid_dir, coarse, monkeypatch
):
    script.partial_path().write_bytes(b"not an npz")
    code, fetched = _run(script, monkeypatch)
    assert code == 0
    assert fetched["n"] == _needed(1.0)


def test_an_existing_grid_is_not_refetched(script, grid_dir, coarse, monkeypatch):
    """Terrain does not change, so a second run must cost nothing."""
    _run(script, monkeypatch)
    code, fetched = _run(script, monkeypatch)
    assert code == 0
    assert fetched["n"] == 0


def test_force_refetches_an_existing_grid(script, grid_dir, coarse, monkeypatch):
    _run(script, monkeypatch)
    code, fetched = _run(script, monkeypatch, argv=("--force",))
    assert code == 0
    assert fetched["n"] == _needed(1.0)


def _needed(resolution: float) -> int:
    from wepr.cities import CITIES

    lats, lons = spatial.grid_axes(resolution)
    hull = spatial.hull_distance_km(
        np.array([c.lat for c in CITIES]), np.array([c.lon for c in CITIES]), lats, lons
    )
    return int((hull <= spatial.HULL_MARGIN_KM).sum())

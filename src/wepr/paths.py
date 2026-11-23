"""Canonical project paths.

Every module resolves paths through here rather than building its own relative
path, so a script run from `scripts/` and the Flask app run from the root agree
on where the data lives.
"""

from pathlib import Path

# paths.py sits at src/wepr/paths.py, so the project root is three levels up.
ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_PROCESSED = DATA / "processed"
DATA_VALIDATION = DATA / "validation"
DATA_GRID = DATA / "grid"

MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "app" / "static" / "figures"


# The map's elevation grid, built once by `scripts/fetch_elevation.py`. Unlike the
# daily series under `data/raw/`, this is *static reference data* — terrain does
# not get refetched — and it is small, so it is committed for the same reason
# `data/validation/` is: a clone can render the map without a network call.
ELEVATION_GRID = DATA_GRID / "elevation.npz"

# The two original NOAA GHCN-Daily station exports, in °F and inches. They are no
# longer the modelling input; they are the independent observational check on the
# ERA5-Land reanalysis for the two cities where real station data is in hand.
GHCN_HUESCA = DATA_VALIDATION / "weather_huesca.csv"
GHCN_VALENCIA = DATA_VALIDATION / "weather_vlc.csv"


def metrics_json() -> Path:
    """Where `scripts/evaluate_all.py` records every model's measured accuracy.

    A function rather than a module constant, for the same reason
    `wepr.models.registry.model_path` is: a constant bound at import cannot be
    redirected, so the test suite's `isolated_paths` fixture could move `REPORTS`
    and the metrics would still be read from the real `reports/metrics.json`. The
    serving layer refuses to serve a model with no metrics entry, so a test that
    silently read the real file would be testing nothing.
    """
    return REPORTS / "metrics.json"


def map_summary(variable: str) -> Path:
    """Where `scripts/make_figures.py` records what a map figure actually shows.

    The figure is a PNG and cannot be interrogated, so its method, its coverage
    and the city values behind it are written beside it. The app serves this
    rather than recomputing a field per request.
    """
    return REPORTS / f"map_{variable}.json"


def trend_summary() -> Path:
    """Where `scripts/make_figures.py` records the fitted trends for `/trends`.

    The machine-readable counterpart to `reports/trends.md`, written from the same
    fits in the same pass. Unkeyed by variable, unlike `map_summary`: the trends
    page is about warming, the figures are drawn for `tavg` only, and a Theil-Sen
    slope on daily precipitation is a materially weaker claim that this project
    does not make. The variable is recorded inside the file, so serving the wrong
    one would be visible rather than assumed.
    """
    return REPORTS / "trends.json"


def ensure_dirs() -> None:
    """Create the output directories that artifacts get written into.

    Called by scripts, never at import time.
    """
    for d in (DATA_RAW, DATA_PROCESSED, DATA_GRID, MODELS, REPORTS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)

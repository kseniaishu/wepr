"""Raw cached responses -> tidy per-city daily frames.

The contract every downstream module relies on: a `DatetimeIndex` named `date`,
daily frequency with no gaps, sorted ascending, and the columns in `COLUMNS`
below in SI units. If a frame does not satisfy that, `validate` should fail loudly
rather than let a model train on it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from wepr import paths

# Canonical column names and units used throughout the project.
COLUMNS: dict[str, str] = {
    "tavg": "°C, daily mean 2 m air temperature",
    "tmax": "°C, daily maximum 2 m air temperature",
    "tmin": "°C, daily minimum 2 m air temperature",
    "prcp": "mm, daily total precipitation",
}

# Open-Meteo's daily field names -> this project's column names. The whole mapping
# is a rename; no arithmetic happens, because the archive already serves °C and mm.
_ARCHIVE_RENAME: dict[str, str] = {
    "temperature_2m_mean": "tavg",
    "temperature_2m_max": "tmax",
    "temperature_2m_min": "tmin",
    "precipitation_sum": "prcp",
}

# The GHCN exports are in °F and inches. This is the one place in the codebase where
# unit conversion happens at all.
_F_OFFSET, _F_SCALE = 32.0, 5.0 / 9.0
_INCH_TO_MM = 25.4


class DataContractError(ValueError):
    """A frame violated the tidy daily contract described in this module."""


def processed_path(slug: str) -> Path:
    """Where one city's tidy daily frame is cached."""
    return paths.DATA_PROCESSED / f"{slug}.csv"


def tidy(raw: dict) -> pd.DataFrame:
    """Turn one raw cached archive payload into a tidy daily frame.

    Trailing rows with no data at all are dropped: a request whose window reaches
    past the archive edge legitimately comes back padded with nulls, and those are
    an artifact of the request, not a gap in the record. Interior gaps are *not*
    filled — they are left as NaN for `validate` to count and report, because
    quietly interpolating them would hide exactly the defect the quality gate
    exists to catch.

    Args:
        raw: A payload as written by `wepr.data.fetch.fetch_cities`.

    Returns:
        Daily frame indexed by date, columns as in `COLUMNS`.

    Raises:
        DataContractError: if the payload has no usable daily block, or the dates
            are not a complete daily sequence.
    """
    daily = raw.get("daily") or {}
    if "time" not in daily:
        raise DataContractError(f"payload for {raw.get('slug', '?')} has no daily.time")

    frame = pd.DataFrame(daily)
    frame["time"] = pd.to_datetime(frame["time"])
    frame = frame.rename(columns={"time": "date", **_ARCHIVE_RENAME})
    frame = frame.set_index("date").sort_index()

    missing = [c for c in COLUMNS if c not in frame.columns]
    if missing:
        raise DataContractError(f"payload for {raw.get('slug', '?')} lacks {missing}")
    frame = frame[list(COLUMNS)].astype("float64")

    # A duplicated date means two windows overlapped; keep the first and move on,
    # since the archive is immutable so the duplicates are identical by construction.
    frame = frame[~frame.index.duplicated(keep="first")]

    populated = frame.notna().any(axis=1)
    if populated.any():
        frame = frame.loc[: frame.index[populated][-1]]

    return assert_daily_contract(frame, label=raw.get("slug", "payload"))


def assert_daily_contract(frame: pd.DataFrame, label: str = "frame") -> pd.DataFrame:
    """Check a frame against the tidy daily contract and return it unchanged.

    Separate from `tidy` so that anything constructing a frame another way — a test
    fixture, a future second data source — can be held to the same contract.

    Raises:
        DataContractError: on a non-daily index, an unsorted index, a date gap, or
            a missing column.
    """
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise DataContractError(f"{label}: index is {type(frame.index).__name__}, not DatetimeIndex")
    if frame.index.name != "date":
        raise DataContractError(f"{label}: index is named {frame.index.name!r}, not 'date'")
    if not frame.index.is_monotonic_increasing:
        raise DataContractError(f"{label}: index is not sorted ascending")
    if frame.index.has_duplicates:
        raise DataContractError(f"{label}: index has duplicate dates")

    missing = [c for c in COLUMNS if c not in frame.columns]
    if missing:
        raise DataContractError(f"{label}: missing columns {missing}")

    if len(frame) > 1:
        expected = pd.date_range(frame.index[0], frame.index[-1], freq="D", name="date")
        if len(expected) != len(frame.index) or not expected.equals(frame.index):
            gaps = expected.difference(frame.index)
            raise DataContractError(
                f"{label}: {len(gaps)} missing date(s) in the daily sequence, "
                f"first {gaps[0].date() if len(gaps) else '?'}"
            )
    return frame


def write_processed(slug: str, frame: pd.DataFrame) -> Path:
    """Cache a tidy frame as CSV and return the path.

    CSV rather than Parquet: `data/processed/` is gitignored so size does not reach
    the repository, and CSV avoids adding pyarrow to a dependency list whose whole
    point is that a clone installs without a build toolchain.
    """
    paths.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    path = processed_path(slug)
    frame.to_csv(path, float_format="%.4g")
    return path


def load_city(slug: str) -> pd.DataFrame:
    """Load one city's tidy daily frame from the processed cache.

    Args:
        slug: City slug, as registered in `wepr.cities`.

    Returns:
        Daily frame indexed by date, columns as in `COLUMNS`.

    Raises:
        FileNotFoundError: if the city has not been fetched and processed yet. The
            message names the script to run, since an empty cache is the normal
            state of a fresh clone rather than a bug.
    """
    path = processed_path(slug)
    if not path.exists():
        raise FileNotFoundError(
            f"no processed data for {slug!r} at {path}; "
            f"run `.venv/bin/python scripts/fetch_data.py`"
        )
    frame = pd.read_csv(path, index_col="date", parse_dates=["date"])
    return assert_daily_contract(frame, label=slug)


def city_metadata(slug: str) -> dict:
    """The provenance of one city's series: model, grid cell, elevation, fetch time.

    Read from the raw payload rather than duplicated into a second artifact, so
    there is one source of truth for where a number came from.
    """
    path = paths.DATA_RAW / f"{slug}.json"
    if not path.exists():
        raise FileNotFoundError(f"no raw payload for {slug!r} at {path}")
    raw = json.loads(path.read_text())
    return {k: v for k, v in raw.items() if k != "daily"}


def available_cities() -> list[str]:
    """Slugs that currently have a processed frame on disk, sorted."""
    if not paths.DATA_PROCESSED.exists():
        return []
    return sorted(p.stem for p in paths.DATA_PROCESSED.glob("*.csv") if not p.stem.startswith("_"))


def load_ghcn(slug: str) -> pd.DataFrame:
    """Load an original NOAA GHCN-Daily station export as a tidy frame.

    Only `huesca` and `valencia` are available. These files are in **°F and
    inches** and are the one place in the codebase where unit conversion happens:

        celsius = (fahrenheit - 32) * 5 / 9
        mm      = inches * 25.4

    Huesca has no TAVG column, so it is derived as `(tmax + tmin) / 2` **before**
    the °F→°C conversion. (The conversion is affine, so doing it in the other order
    gives the same number; the ordering is fixed here only so the code matches the
    documented procedure.) Note that this midrange is *not* the same quantity as a
    true 24-hour mean — `wepr.data.validate` accounts for that when comparing.

    Valencia's PRCP is 17% missing and its TMIN 43%, so it is usable for a
    temperature comparison only.

    Args:
        slug: Either `"huesca"` or `"valencia"`.

    Returns:
        Daily frame indexed by date, in SI units, gaps left as NaN — not filled.
        Forward-filling an observational record would invent measurements. The
        index is a complete daily sequence, so an absent row becomes an all-NaN
        row rather than a hole.

    Raises:
        KeyError: for any slug other than the two with station data on disk.
    """
    sources = {"huesca": paths.GHCN_HUESCA, "valencia": paths.GHCN_VALENCIA}
    try:
        path = sources[slug]
    except KeyError:
        raise KeyError(
            f"no GHCN station export for {slug!r}; available: {sorted(sources)}"
        ) from None

    raw = pd.read_csv(path, parse_dates=["DATE"])
    frame = pd.DataFrame(index=pd.DatetimeIndex(raw["DATE"], name="date"))

    tmax_f = pd.to_numeric(raw["TMAX"], errors="coerce").to_numpy()
    tmin_f = pd.to_numeric(raw["TMIN"], errors="coerce").to_numpy()
    if "TAVG" in raw.columns:
        tavg_f = pd.to_numeric(raw["TAVG"], errors="coerce").to_numpy()
    else:
        # Huesca reports no TAVG, so derive the midrange in °F first.
        tavg_f = (tmax_f + tmin_f) / 2.0

    frame["tavg"] = (tavg_f - _F_OFFSET) * _F_SCALE
    frame["tmax"] = (tmax_f - _F_OFFSET) * _F_SCALE
    frame["tmin"] = (tmin_f - _F_OFFSET) * _F_SCALE
    frame["prcp"] = pd.to_numeric(raw["PRCP"], errors="coerce").to_numpy() * _INCH_TO_MM

    frame = frame[~frame.index.duplicated(keep="first")].sort_index()
    # Reindex onto a complete daily range so absent dates show up as NaN rows. The
    # alternative, a frame with holes in the index, breaks the daily contract and
    # makes every downstream `shift(1)` silently span the hole.
    full = pd.date_range(frame.index[0], frame.index[-1], freq="D", name="date")
    frame = frame.reindex(full)

    return assert_daily_contract(frame, label=f"ghcn:{slug}")


def ghcn_station(slug: str) -> dict:
    """The station header of a GHCN export: id, name, and coordinates where given.

    Huesca's export carries LATITUDE/LONGITUDE/ELEVATION; Valencia's does not. The
    longitude is returned exactly as the file states it — see `wepr.data.validate`
    for the sign problem in Huesca's, which is resolved there rather than silently
    patched here.
    """
    sources = {"huesca": paths.GHCN_HUESCA, "valencia": paths.GHCN_VALENCIA}
    raw = pd.read_csv(sources[slug], nrows=1)
    head = raw.iloc[0]
    out: dict = {"station": head["STATION"], "name": head["NAME"]}
    for key, column in (("lat", "LATITUDE"), ("lon", "LONGITUDE"), ("elevation_m", "ELEVATION")):
        if column in raw.columns:
            out[key] = float(head[column])
    return out

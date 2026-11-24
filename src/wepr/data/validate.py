"""Data quality gates, and the reanalysis-vs-observation cross-check.

Two separate jobs:

1. **The quality gate.** A completeness and sanity matrix per city, written to
   `reports/data_quality.md`. A city enters the app only if it passes. This is
   what makes a twenty-city claim credible rather than merely large.
2. **The validation exercise.** Compare ERA5 against the independent GHCN
   station records for Huesca and Valencia and report the agreement — bias, MAE,
   correlation, and how the two differ in the tails. Reanalysis that tracks real
   observations closely for temperature and less closely for precipitation is the
   expected result; reporting it is the point.

**Comparing like with like.** Two traps the comparison has to avoid, both of which
would otherwise be read as reanalysis error:

* *Definition.* ERA5's `temperature_2m_mean` is a true 24-hour average. Huesca's
  export has no TAVG, so its mean is the midrange `(TMAX + TMIN) / 2`, which runs
  systematically warm because a diurnal curve spends more of the day near its
  minimum than its maximum. `tmax` and `tmin` are the like-for-like comparisons;
  `tavg` is reported with that caveat attached, not silently alongside them.
* *Location.* The station is not at the city centre. Huesca's export puts it at
  42.0831 N, and the grid cell the app actually serves is ~8 km away. The headline
  figures below use the **deployed** cell, because the question that matters is how
  good the data the app serves is — with the station's own cell reported as a
  sensitivity check, separating grid mismatch from reanalysis error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wepr.cities import CITIES, VALIDATION_SLUGS, get
from wepr.data import load

# Sanity envelope for Iberian daily values. Deliberately generous: the job is to
# catch corrupt data (a sentinel -9999, a unit mix-up, a sign flip), not to clip
# genuine extremes. For scale, the Spanish record high is 47.6 °C and the record
# daily rainfall 817 mm, both well inside these bounds — a 9 km grid cell averages
# extremes down, so anything outside this range is a defect, not weather.
TEMP_MIN_C, TEMP_MAX_C = -25.0, 50.0
PRCP_MAX_MM = 500.0

# A city passes the gate only if every column is at least this complete. ERA5 has no
# gaps by construction, so the realistic failure is a truncated fetch, and the
# threshold is tight enough to catch one.
MAX_MISSING_FRACTION = 0.005

# Shorter than this and a day-of-year climatology has too few years per calendar day
# to be stable, and a trend estimate is not worth reporting. 60 years is comfortably
# inside what the archive offers (76), so this catches a short fetch, not a short
# archive.
MIN_YEARS = 60

# A day counts as wet at or above this depth. 1 mm is the conventional threshold in
# the precipitation-verification literature; it also sidesteps reanalysis "drizzle",
# the well-known tendency of these models to spread a little rain over many dry days.
WET_DAY_MM = 1.0


def _impossible_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Count physically impossible values in a tidy frame."""
    temps = frame[["tavg", "tmax", "tmin"]]
    return {
        "negative_prcp": int((frame["prcp"] < 0).sum()),
        "tmin_above_tmax": int((frame["tmin"] > frame["tmax"]).sum()),
        "temp_out_of_range": int(
            ((temps < TEMP_MIN_C) | (temps > TEMP_MAX_C)).to_numpy().sum()
        ),
        "prcp_out_of_range": int((frame["prcp"] > PRCP_MAX_MM).sum()),
    }


def quality_matrix() -> pd.DataFrame:
    """Build the per-city completeness and sanity matrix.

    Only cities with a processed frame on disk appear; a city in the registry but
    absent from `data/processed/` is reported as `not_fetched` rather than skipped,
    so a partial fetch cannot look like a clean run.

    Returns:
        One row per city: date span, row count, missing fraction per column,
        count of physically impossible values (negative precipitation,
        `tmin > tmax`, temperatures outside a plausible Iberian envelope), and a
        pass/fail verdict.
    """
    available = set(load.available_cities())
    rows: list[dict] = []

    for city in CITIES:
        row: dict = {
            "slug": city.slug,
            "name": city.name,
            "region": city.region,
        }
        if city.slug not in available:
            rows.append({**row, "verdict": "not_fetched", "reasons": "no processed frame"})
            continue

        # A gate that crashes on bad data is not a gate. Report the breakage as a
        # failing row so the rest of the matrix still gets built and the report
        # names what went wrong.
        try:
            frame = load.load_city(city.slug)
        except (load.DataContractError, OSError, ValueError) as exc:
            rows.append({**row, "verdict": "fail", "reasons": f"unreadable: {exc}"})
            continue

        try:
            meta = load.city_metadata(city.slug)
        except (FileNotFoundError, ValueError):
            meta = {}  # the raw payload is gone; provenance is unknown, not fatal
        missing = frame.isna().mean()
        impossible = _impossible_counts(frame)
        years = (frame.index[-1] - frame.index[0]).days / 365.25

        reasons: list[str] = []
        worst_missing = float(missing.max())
        if worst_missing > MAX_MISSING_FRACTION:
            reasons.append(f"{missing.idxmax()} {worst_missing:.2%} missing")
        if years < MIN_YEARS:
            reasons.append(f"only {years:.1f} years")
        for name, count in impossible.items():
            if count:
                reasons.append(f"{count} {name}")

        rows.append(
            {
                **row,
                "grid_elevation_m": meta.get("grid_elevation_m"),
                "start": frame.index[0].date().isoformat(),
                "end": frame.index[-1].date().isoformat(),
                "rows": len(frame),
                "years": round(years, 1),
                **{f"missing_{c}": round(float(missing[c]), 5) for c in load.COLUMNS},
                **impossible,
                "verdict": "pass" if not reasons else "fail",
                "reasons": "; ".join(reasons),
            }
        )

    matrix = pd.DataFrame(rows).set_index("slug")
    # Rows short-circuited above carry no `reasons`; make the column uniformly a
    # string so report rendering never has to special-case NaN.
    if "reasons" in matrix.columns:
        matrix["reasons"] = matrix["reasons"].fillna("")
    return matrix


def passing_slugs() -> list[str]:
    """Slugs that clear the quality gate — the cities the app may serve."""
    matrix = quality_matrix()
    return sorted(matrix.index[matrix["verdict"] == "pass"])


def _agreement(obs: pd.Series, era: pd.Series) -> dict[str, float]:
    """Paired agreement statistics over dates where both series have a value."""
    joined = pd.concat([obs.rename("obs"), era.rename("era")], axis=1, sort=True).dropna()
    if joined.empty:
        return {"n": 0}
    error = joined["era"] - joined["obs"]
    return {
        "n": int(len(joined)),
        "obs_mean": float(joined["obs"].mean()),
        "era_mean": float(joined["era"].mean()),
        "bias": float(error.mean()),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "r": float(joined["obs"].corr(joined["era"])),
        "obs_p01": float(joined["obs"].quantile(0.01)),
        "era_p01": float(joined["era"].quantile(0.01)),
        "obs_p99": float(joined["obs"].quantile(0.99)),
        "era_p99": float(joined["era"].quantile(0.99)),
    }


# Which variables each station record can actually support, and why not the rest.
# Encoded here rather than left to the caller's memory, because the tempting mistake
# is to build a precipitation validation on Valencia's 17%-missing PRCP column.
_USABLE: dict[str, dict[str, str]] = {
    "huesca": {
        "tmax": "",
        "tmin": "",
        "prcp": "station gauge vs ~28 km ERA5 cell; expect weak daily agreement",
        "tavg": "station tavg is the midrange (tmax+tmin)/2, ERA5's is a true "
        "24 h mean — the bias is partly definitional, not error",
    },
    "valencia": {
        "tavg": "",
        "tmax": "35% of station days missing",
        "tmin": "43% of station days missing",
        # prcp deliberately absent: 17% missing is too much to found a
        # precipitation validation on. See `reports/reanalysis_validation.md`.
    },
}


def compare_with_ghcn(slug: str) -> pd.DataFrame:
    """Cross-check ERA5 against the GHCN station record for one city.

    Compares the grid cell the app actually serves — the one at the city's
    registered coordinates — against the station, over the dates both cover.

    Args:
        slug: Either `"huesca"` or `"valencia"` — the cities with station data.

    Returns:
        Per-variable agreement statistics over the overlapping dates only, one row
        per usable variable, with a `caveat` column naming what limits each
        comparison. Variables a station cannot support are omitted rather than
        reported with a quiet footnote.

    Raises:
        KeyError: for a city without a station export.
    """
    if slug not in VALIDATION_SLUGS:
        raise KeyError(
            f"no station record for {slug!r}; available: {list(VALIDATION_SLUGS)}"
        )

    station = load.load_ghcn(slug)
    reanalysis = load.load_city(slug)

    rows = []
    for variable, caveat in _USABLE[slug].items():
        stats = _agreement(station[variable], reanalysis[variable])
        rows.append({"variable": variable, **stats, "caveat": caveat})

    out = pd.DataFrame(rows).set_index("variable")
    out.attrs["city"] = slug
    out.attrs["station"] = load.ghcn_station(slug)
    return out


def wet_day_agreement(slug: str = "huesca") -> dict[str, float]:
    """Does the reanalysis get *whether* it rained right, if not how much?

    Daily precipitation depth is where reanalysis is weakest, so a bare MAE in
    millimetres understates how usable it is. This scores occurrence instead, as a
    2x2 contingency table over a 1 mm threshold, which is the honest framing given
    the app reports rain *probability* as its headline.

    Only Huesca is supported: Valencia's PRCP is 17% missing.

    Returns:
        Hit rate (fraction of observed wet days the reanalysis called wet), false
        alarm ratio, frequency bias, and the dry/wet day counts behind them.
    """
    if slug != "huesca":
        raise KeyError(
            f"precipitation validation is only defensible for 'huesca'; "
            f"{slug!r} has too much missing station PRCP"
        )

    station = load.load_ghcn(slug)["prcp"]
    reanalysis = load.load_city(slug)["prcp"]
    joined = pd.concat(
        [station.rename("obs"), reanalysis.rename("era")], axis=1, sort=True
    ).dropna()

    obs_wet = joined["obs"] >= WET_DAY_MM
    era_wet = joined["era"] >= WET_DAY_MM
    hits = int((obs_wet & era_wet).sum())
    misses = int((obs_wet & ~era_wet).sum())
    false_alarms = int((~obs_wet & era_wet).sum())
    correct_dry = int((~obs_wet & ~era_wet).sum())

    return {
        "n": int(len(joined)),
        "threshold_mm": WET_DAY_MM,
        "obs_wet_days": hits + misses,
        "era_wet_days": hits + false_alarms,
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_dry": correct_dry,
        "hit_rate": hits / (hits + misses) if hits + misses else float("nan"),
        "false_alarm_ratio": false_alarms / (hits + false_alarms)
        if hits + false_alarms
        else float("nan"),
        "frequency_bias": (hits + false_alarms) / (hits + misses)
        if hits + misses
        else float("nan"),
        # Mean depth over all paired days: tests whether the totals balance even
        # when individual days do not.
        "obs_mean_mm": float(joined["obs"].mean()),
        "era_mean_mm": float(joined["era"].mean()),
    }


def station_location_check(slug: str = "huesca", *, session=None) -> pd.DataFrame:
    """Resolve where Huesca's station actually is, and quantify grid mismatch.

    Huesca's export states `LONGITUDE 0.3264` — east of Greenwich, which would put
    the station ~60 km east of the city. That is almost certainly a lost sign: at
    42.0831 N, 0.3264 **W** is Huesca-Pirineos airport, and the archive's grid
    elevation there (544 m) matches the station's stated 541 m to within 3 m, where
    the eastern reading lands on a 791 m cell.

    Rather than assert that, this resolves it empirically — fetch all three
    candidate cells and report which agrees with the station record. It needs the
    network, so it is a reporting tool, not part of the gate.

    Returns:
        One row per candidate location: served grid cell, elevation, and `tmax`
        agreement against the station.
    """
    from wepr.data import fetch  # local import: keeps the module import network-free

    if slug != "huesca":
        raise KeyError("only Huesca's export carries station coordinates")

    station = load.load_ghcn(slug)
    header = load.ghcn_station(slug)
    city = get(slug)
    start = station.index[0].date().isoformat()
    end = station.index[-1].date().isoformat()

    candidates = {
        "station, longitude as written (E)": (header["lat"], header["lon"]),
        "station, longitude as west (W)": (header["lat"], -abs(header["lon"])),
        "city centre, as deployed": (city.lat, city.lon),
    }

    rows = []
    for label, (lat, lon) in candidates.items():
        block = fetch.fetch_point(lat, lon, start, end, session=session)
        era = pd.Series(
            block["daily"]["temperature_2m_max"],
            index=pd.DatetimeIndex(pd.to_datetime(block["daily"]["time"]), name="date"),
            dtype="float64",
        )
        stats = _agreement(station["tmax"], era)
        rows.append(
            {
                "candidate": label,
                "requested_lat": lat,
                "requested_lon": lon,
                "grid_lat": block["grid_latitude"],
                "grid_lon": block["grid_longitude"],
                "grid_elevation_m": block["grid_elevation_m"],
                "station_elevation_m": header["elevation_m"],
                "tmax_bias": stats["bias"],
                "tmax_mae": stats["mae"],
                "tmax_r": stats["r"],
            }
        )
    return pd.DataFrame(rows).set_index("candidate")


def eligible_slugs(
    requested: list[str] | None = None, ignore_gate: bool = False
) -> tuple[list[str], str]:
    """The cities a modelling script may use, and how they were chosen.

    Lives here rather than in a script because both `scripts/train_all.py` and
    `scripts/evaluate_all.py` need exactly the same answer: if they disagreed about
    which cities are eligible, a model could be trained and never evaluated, or
    evaluated and never deployed.

    The default is the gate's verdict — a city whose data cannot support a model
    should not have one. `ignore_gate` exists for a genuinely partial archive: the
    Open-Meteo free tier caps requests per day, so a cold 77-year fetch stops
    part-way and development would otherwise have nothing to run against. Anything
    produced under it is provisional and is labelled so in `reports/metrics.json`.

    Args:
        requested: Limit to these slugs. Unknown or ineligible slugs raise.
        ignore_gate: Accept any city with processed data on disk.

    Returns:
        The eligible slugs, and a one-line description of the basis for reporting.

    Raises:
        KeyError: if a requested slug has no processed data, or fails the gate
            while `ignore_gate` is False.
    """
    on_disk = load.available_cities()

    if ignore_gate:
        eligible = list(on_disk)
        basis = f"{len(eligible)} city/cities on disk (gate ignored)"
    else:
        passing = set(passing_slugs())
        eligible = [s for s in on_disk if s in passing]
        basis = f"{len(eligible)} city/cities passing the data-quality gate"

    if requested:
        unknown = sorted(set(requested) - set(on_disk))
        if unknown:
            raise KeyError(f"no processed data for {unknown}; run scripts/fetch_data.py")
        excluded = sorted(set(requested) - set(eligible))
        if excluded:
            raise KeyError(
                f"{excluded} has processed data but fails the data-quality gate. "
                "See reports/data_quality.md, or pass --ignore-gate to proceed anyway."
            )
        eligible = [s for s in eligible if s in set(requested)]
        basis = f"{len(eligible)} of the requested city/cities"

    return eligible, basis

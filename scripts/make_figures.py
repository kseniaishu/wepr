"""Render the figures used by the app and the README.

    .venv/bin/python scripts/make_figures.py

Writes PNGs to `app/static/figures/`. Uses the Agg backend and never calls
`plt.show()`, so it is safe to run headless in CI.

It also writes `reports/trends.md` and `reports/trends.json`. Those are the trend
figures' own caption data, produced from the same fit in the same pass: computing
them in a second script would mean fitting twice and leaving open the possibility
of the figure and the table disagreeing about the same city's warming rate. The
JSON is what `/trends` serves — Phase 7 gave the trends page the same contract the
map page has, because the rule that decided the map ("a web request never triggers
analysis") decides this the same way.

Implemented in Phase 4; the trends artifact added in Phase 7.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

import matplotlib

# Set before pyplot is imported anywhere, so this is safe on a headless machine.
matplotlib.use("Agg")

import matplotlib.colors  # noqa: E402
import matplotlib.dates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from wepr import paths, spatial, trends  # noqa: E402
from wepr.cities import get  # noqa: E402
from wepr.data import load, validate  # noqa: E402
from wepr.models import seasonal  # noqa: E402

DPI = 160

# One palette for the whole project, so a reader moving between the climatology
# figure and the trend figure does not have to re-learn what a colour means.
INK = "#1b1f23"
MUTED = "#6a737d"
OBSERVED = "#24618f"
FITTED = "#b4421f"
EXTRAPOLATED = "#c77b35"
SCENARIO = "#4b5f3f"
GRID = "#dfe2e5"


def _style(ax) -> None:
    """The shared axis treatment: no top/right spines, soft horizontal grid."""
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, name: str) -> "object":
    path = paths.FIGURES / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# -- the trend figure -----------------------------------------------------------


def figure_trend(slug: str, variable: str = "tavg", through_year: int = 2100):
    """Annual means, the robust fit, and the forward extrapolation.

    Two panels, deliberately not one. The left panel is the city's own record in
    absolute °C. The right panel is the published CMIP6 scenario envelope.

    **Why they are not drawn on a shared axis.** A scenario figure is an anomaly
    against its own baseline period — 1850–1900 for the AR6 global numbers,
    1986–2005 for the Mediterranean ones. Overlaying one on a local record means
    expressing both as anomalies against that same baseline, which requires the
    record to cover it. `trends.baseline_mean` returns NaN when it does not, and
    that is what decides the layout here rather than anyone's judgement: where the
    baseline is covered the envelope is drawn against the extrapolation on one
    anomaly axis; where it is not, the envelope is drawn beside it with both
    baselines printed and a note saying they cannot be read against a common zero.
    Sliding a 1850–1900 anomaly onto a 1950-baselined axis would overstate it by
    most of a degree, which is the size of the effect being discussed.
    """
    frame = load.load_city(slug)
    city = get(slug)
    series = trends.annual_means(frame, variable)
    trend = trends.theil_sen(series.index.to_numpy(), series.to_numpy())
    forward = trends.extrapolate(trend, through_year)

    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [2.6, 1.0]}
    )

    ax.plot(series.index, series.to_numpy(), marker="o", markersize=2.6,
            linewidth=0.9, color=OBSERVED, label=f"annual mean ({len(series)} yr)")
    fitted_years = np.array([trend.first_year, trend.last_year], dtype="float64")
    ax.plot(fitted_years, trend.value_at(fitted_years), color=FITTED, linewidth=2.0,
            label=f"Theil–Sen fit: {trend.slope_per_decade:+.3f} °C/decade")

    ax.fill_between(forward.index, forward["year_lo"], forward["year_hi"],
                    color=EXTRAPOLATED, alpha=0.14,
                    label="± 90% of interannual scatter")
    ax.fill_between(forward.index, forward["rate_lo"], forward["rate_hi"],
                    color=EXTRAPOLATED, alpha=0.30, label="rate 95% CI, widening")
    ax.plot(forward.index, forward["central"], color=EXTRAPOLATED, linewidth=1.6,
            linestyle="--", label="statistical extrapolation — NOT a projection")
    ax.axvline(trend.last_year, color=MUTED, linewidth=0.8, linestyle=":")
    ax.annotate("record ends", (trend.last_year, ax.get_ylim()[0]),
                xytext=(4, 4), textcoords="offset points", fontsize=7, color=MUTED)

    _style(ax)
    # Padded to clear the legend, which sits above the axes because the
    # extrapolation cone fills every corner inside them.
    ax.set_title(f"{city.name} — observed rate, extrapolated", fontsize=11,
                 color=INK, loc="left", pad=44)
    ax.set_ylabel("°C", fontsize=9, color=MUTED)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left", labelcolor=INK,
              bbox_to_anchor=(0.0, 1.005), ncol=2, borderaxespad=0.0)

    caption = trend.describe()
    if not trend.is_climatological:
        caption += (
            f"\nFewer than {trends.MIN_TREND_YEARS} years: this is the record's own "
            "slope, not a climate trend. Do not read the dashed line as warming."
        )
    if np.isfinite(trend.lag1_autocorrelation) and trend.lag1_autocorrelation > 0.2:
        caption += (
            f"\nResidual lag-1 autocorrelation {trend.lag1_autocorrelation:+.2f}: the "
            "stated CI assumes independent years and is therefore too narrow."
        )
    ax.text(0.0, -0.17, caption, transform=ax.transAxes, fontsize=7.5,
            color=MUTED, va="top")

    _scenario_panel(bx, frame, variable)
    fig.suptitle("")
    return _save(fig, f"trend_{slug}.png")


def _scenario_panel(bx, frame: pd.DataFrame, variable: str) -> None:
    """The published envelope, on whatever footing the record can actually support.

    Drawn horizontally: the row labels carry a scenario, a region and a season each,
    and stacked vertically they read without the overlap a rotated x axis gives.
    """
    rows = []
    for region, season in (("mediterranean-land", "JJA"), ("mediterranean-land", "DJF"),
                           ("global", "annual")):
        block = trends.scenarios(region=region, season=season)
        for _, row in block.iterrows():
            rows.append(row)

    # Mediterranean rows first and in colour, because they are the ones that speak
    # to Iberia; the global rows are context and are drawn muted.
    labels = [
        f"{r['scenario']}  ·  {'Med' if r['region'] == 'mediterranean-land' else 'global'}"
        f" {r['season']}"
        for r in rows
    ]
    positions = np.arange(len(rows))[::-1]

    for position, row in zip(positions, rows):
        colour = SCENARIO if row["region"] == "mediterranean-land" else MUTED
        bx.plot([row["low"], row["high"]], [position, position], color=colour,
                linewidth=6.0, solid_capstyle="butt", alpha=0.55)
        if row["median"] is not None and np.isfinite(row["median"]):
            bx.plot([row["median"]], [position], marker="|", markersize=12,
                    color=colour, markeredgewidth=2)

    _style(bx)
    bx.xaxis.grid(True, color=GRID, linewidth=0.6)
    bx.yaxis.grid(False)
    bx.set_yticks(positions)
    bx.set_yticklabels(labels, fontsize=7.5, color=INK)
    bx.set_xlabel("°C above each row's own baseline", fontsize=8, color=MUTED)
    bx.set_xlim(left=0)
    bx.set_title("Published CMIP6 envelope, 2081–2100", fontsize=10, color=INK,
                 loc="left", pad=44)

    # Which baselines the local record could have been put on, decided by the data
    # rather than by anyone's judgement.
    baselines = sorted({tuple(r["baseline"]) for r in rows})
    covered = [b for b in baselines
               if np.isfinite(trends.baseline_mean(frame, b, variable))]
    note = (
        "Separate axes on purpose: each row is an anomaly against its own baseline "
        f"({', '.join(f'{a}–{b}' for a, b in baselines)}), and the local record "
        + ("covers " + ", ".join(f"{a}–{b}" for a, b in covered) if covered
           else "covers none of them")
        + ". Sliding these onto the left panel's axis would misstate them by most "
        "of a degree.\n" + trends.SCENARIO_GAP
    )
    bx.text(0.0, -0.22, note, transform=bx.transAxes, fontsize=6.4, color=MUTED,
            va="top", wrap=True)


# -- the climatology figure -----------------------------------------------------


def figure_climatology(slug: str, variable: str = "tavg"):
    """The seasonal product, drawn as what it is: a distribution, not a line.

    The p10–p90 ribbon is the figure's main object and the mean is the thin line
    inside it, which is the opposite of the usual emphasis and is the point — the
    product returns an interval, so the interval is what the reader should see
    first. Rain probability shares the panel on a second axis because it comes off
    the same pooled slots and is the headline precipitation output.
    """
    model = seasonal.SeasonalClimatology(variable=variable).fit(load.load_city(slug))
    table = model.table()
    city = get(slug)
    # A non-leap reference year, so the x axis is dates rather than slot numbers.
    days = pd.Timestamp(2001, 1, 1) + pd.to_timedelta(table.index - 1, unit="D")

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.fill_between(days, table["p10"], table["p90"], color=OBSERVED, alpha=0.20,
                    label=f"p10–p90 ({int(seasonal.NOMINAL_COVERAGE * 100)}% of days)")
    ax.plot(days, table["mean"], color=OBSERVED, linewidth=1.4, label="detrended mean")

    _style(ax)
    ax.set_ylabel("°C", fontsize=9, color=MUTED)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    ax.set_xlim(days[0], days[-1])

    rx = ax.twinx()
    rx.plot(days, table["rain_probability"], color=SCENARIO, linewidth=1.1,
            alpha=0.85, label=f"P(≥{seasonal.WET_DAY_MM} mm)")
    rx.set_ylim(0, 1)
    rx.set_ylabel("rain probability", fontsize=9, color=MUTED)
    rx.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "left"):
        rx.spines[side].set_visible(False)
    rx.spines["right"].set_color(GRID)

    handles = ax.get_legend_handles_labels()[0] + rx.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + rx.get_legend_handles_labels()[1]
    ax.legend(handles, labels, fontsize=7.5, frameon=False, loc="upper left",
              labelcolor=INK)

    ax.set_title(
        f"{city.name} — climatological expectation by calendar day "
        f"({model.first_year_}–{model.last_year_})",
        fontsize=11, color=INK, loc="left",
    )
    ax.text(
        0.0, -0.16,
        "Not a forecast. The band is the historical spread for that calendar day "
        "after the record's linear trend is removed; a trend-adjusted estimate for "
        "a given future year shifts the whole band bodily up or down by "
        f"{model.trend_.slope_per_decade:+.3f} °C per decade away from "
        f"{model.anchor_year_:.0f}.",
        transform=ax.transAxes, fontsize=7.5, color=MUTED, va="top",
    )
    return _save(fig, f"climatology_{slug}.png")


# -- the skill figure -----------------------------------------------------------


def figure_skill(metrics: dict):
    """Skill against both baselines, per city, from `reports/metrics.json`.

    Read from the metrics file rather than recomputed, so this figure cannot
    disagree with the published numbers — if it is wrong, the file is wrong, and
    there is one thing to fix rather than two.
    """
    records = [
        r for r in metrics.get("models", {}).values()
        if r.get("product") == "nextday" and r.get("target") == "tavg_target"
    ]
    if not records:
        return None

    records.sort(key=lambda r: r["skill_vs_persistence"])
    names = [get(r["model"].split(".")[0]).name for r in records]
    positions = np.arange(len(records))
    height = 0.38

    fig, ax = plt.subplots(figsize=(8.0, max(3.0, 0.34 * len(records) + 1.4)))
    ax.barh(positions + height / 2, [r["skill_vs_persistence"] for r in records],
            height=height, color=FITTED, label="vs persistence (tomorrow = today)")
    ax.barh(positions - height / 2, [r["skill_vs_climatology"] for r in records],
            height=height, color=OBSERVED, label="vs day-of-year climatology")
    ax.axvline(0, color=INK, linewidth=0.9)

    _style(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.yaxis.grid(False)
    ax.set_yticks(positions)
    ax.set_yticklabels(names, fontsize=8, color=INK)
    ax.set_xlabel("MAE skill  (1 − model / baseline)", fontsize=9, color=MUTED)
    ax.set_title("Next-day temperature: skill against both baselines", fontsize=11,
                 color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", labelcolor=INK,
              bbox_to_anchor=(0.0, -0.06), ncol=2, borderaxespad=0.0)

    note = (
        "Zero means no better than the baseline. Persistence is the hard one: "
        "day-to-day temperature autocorrelation in Iberia is high."
    )
    if any(r.get("provisional") for r in records):
        note = "PROVISIONAL — fitted on data that fails the quality gate.\n" + note
    ax.text(0.0, -0.13 - 0.04 * note.count("\n"), note, transform=ax.transAxes,
            fontsize=7.5, color=MUTED, va="top")
    return _save(fig, "skill.png")


def figure_seasonal(metrics: dict):
    """How the seasonal product is actually judged: by its interval, not its mean.

    The product returns a band, so the metric that decides whether it works is
    whether the band contains the right fraction of days. An 80% interval that
    catches 55% of days is mislabelled however small its MAE, and plotting MAE here
    instead would hide exactly that. The trend term's contribution is printed as a
    count rather than drawn, because on a short record it is a flat zero across
    every city and a bar chart of twenty near-zeros reads as a defect in the figure.
    """
    records = [r for r in metrics.get("models", {}).values() if r.get("product") == "seasonal"]
    if not records:
        return None

    records.sort(key=lambda r: r["coverage"])
    names = [get(r["model"].split(".")[0]).name for r in records]
    coverage = np.array([r["coverage"] for r in records])
    nominal = records[0]["nominal_coverage"]
    positions = np.arange(len(records))

    fig, ax = plt.subplots(figsize=(7.4, max(3.0, 0.3 * len(records) + 1.6)))
    ax.hlines(positions, nominal, coverage, color=MUTED, linewidth=1.0)
    ax.scatter(coverage, positions, s=26, color=OBSERVED, zorder=3,
               label="realised coverage")
    ax.axvline(nominal, color=FITTED, linewidth=1.2,
               label=f"nominal {nominal:.0%}")

    _style(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.yaxis.grid(False)
    ax.set_yticks(positions)
    ax.set_yticklabels(names, fontsize=8, color=INK)
    ax.set_xlabel("fraction of held-out days inside the p10–p90 band", fontsize=9,
                  color=MUTED)
    ax.set_title("Seasonal estimate: is the interval the width it claims?",
                 fontsize=11, color=INK, loc="left")
    # Every point sits just short of nominal, so the left of the panel is the one
    # reliably empty corner.
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", labelcolor=INK)
    span = float(nominal - coverage.min())
    ax.set_xlim(coverage.min() - 0.25 * span, nominal + 0.12 * span)

    helped = sum(r["trend_earns_its_place"] for r in records)
    calibrated = sum(r["interval_is_calibrated"] for r in records)
    note = (
        f"Calibrated (within 5 points of nominal) in {calibrated} of {len(records)} "
        f"cities. The trend adjustment improves MAE in {helped} of {len(records)}: "
        "on this record it is noise, and is reported as such rather than dropped."
    )
    if any(r.get("provisional") for r in records):
        note = "PROVISIONAL — fitted on data that fails the quality gate.\n" + note
    ax.text(0.0, -0.12 - 0.05 * note.count("\n"), note, transform=ax.transAxes,
            fontsize=7.5, color=MUTED, va="top")
    return _save(fig, "seasonal.png")


# -- the validation figure ------------------------------------------------------


def figure_validation(slug: str = "huesca"):
    """Reanalysis against the GHCN station record, where the two overlap.

    Returns None when they do not. That is the current state — the station files
    run 1990–2024 and the fetch has only reached the 1950s and 60s — and a figure
    drawn from an empty join would be a blank panel implying a comparison that was
    never made.
    """
    try:
        comparison = validate.compare_with_ghcn(slug)
    except (KeyError, FileNotFoundError):
        return None

    usable = comparison[comparison["n"] > 0]
    if usable.empty:
        return None

    station = load.load_ghcn(slug)
    reanalysis = load.load_city(slug)
    variables = [v for v in ("tmax", "tmin", "tavg") if v in usable.index]
    if not variables:
        return None

    fig, axes = plt.subplots(1, len(variables), figsize=(3.6 * len(variables), 3.8),
                             squeeze=False)
    for ax, variable in zip(axes[0], variables):
        joined = pd.concat(
            [station[variable].rename("obs"), reanalysis[variable].rename("era")],
            axis=1,
        ).dropna()
        ax.scatter(joined["obs"], joined["era"], s=2, alpha=0.12, color=OBSERVED,
                   edgecolors="none")
        limits = [min(joined.min()), max(joined.max())]
        ax.plot(limits, limits, color=INK, linewidth=0.8, linestyle="--")
        _style(ax)
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
        row = usable.loc[variable]
        ax.set_title(
            f"{variable}  bias {row['bias']:+.2f} °C, MAE {row['mae']:.2f}, "
            f"r {row['r']:.3f}",
            fontsize=8.5, color=INK, loc="left",
        )
        ax.set_xlabel("station, °C", fontsize=8, color=MUTED)
        ax.set_ylabel("ERA5-Land, °C", fontsize=8, color=MUTED)

    station_header = comparison.attrs["station"]
    fig.suptitle(
        f"{get(slug).name}: reanalysis vs station {station_header['station']} "
        f"({int(usable['n'].max()):,} paired days)",
        fontsize=11, color=INK, x=0.0, ha="left",
    )
    return _save(fig, f"validation_{slug}.png")


# -- the trend table ------------------------------------------------------------


def figure_map(target: "dt.date", variable: str = "tavg"):
    """The interpolated field over Iberia, with its confidence layers drawn.

    **Why the map is drawn in two alphas rather than one.** Inside the convex hull
    of the twenty cities a cell is interpolated — surrounded by evidence. Outside
    it, every city lies to one side, so the cell is extrapolated. Those are
    different claims, so they are not painted the same: solid inside the hull,
    translucent outside it, and nothing at all beyond `spatial.HULL_MARGIN_KM` or
    off the cities' own landmass.

    Hatching marks where the terrain rises above the highest city in the fit. The
    Pyrenees and the Sierra Nevada rise far above Salamanca, so the surface has no
    evidence about elevation there; `TrendSurface.predict` holds the elevation term
    at its fitted ceiling across those cells rather than carrying the coefficient
    on, and the hatching says where that happened. A reader should be able to see
    exactly which ground is held flat without reading a caption.

    That guard is not cosmetic. Unclamped, the July surface's +2.3 °C/km — a
    continentality artefact, since July temperature has zero correlation with
    elevation across these cities — reached 3,064 m and made the Sierra Nevada and
    the Pyrenees the hottest ground in Iberia. See `wepr.spatial`.

    The city markers stay on top at all times. Twenty points cannot resolve a 9 km
    field, and the single most honest thing the figure can do is let a reader
    count the evidence.

    Args:
        target: The date whose climatological expectation is mapped.
        variable: Which field — `tavg` for temperature, `prcp` for rain
            probability.

    Returns:
        The path written.

    Raises:
        SpatialError: If the elevation grid has not been built.
        ValueError: If no city has a trained seasonal model.
    """
    estimates = seasonal.city_estimates(target)
    if estimates.empty:
        raise ValueError(
            "no city has a trained seasonal model; run scripts/train_all.py"
        )

    if variable == "prcp":
        values = estimates["rain_probability"].to_numpy() * 100.0
        units, label, colours, non_negative = (
            "%", "Rain probability", "YlGnBu", True,
        )
    else:
        values = estimates["expected_c"].to_numpy()
        units, label, colours, non_negative = (
            "°C", "Expected mean temperature", "RdYlBu_r", False,
        )

    grid = spatial.load_elevation()
    field = spatial.interpolate_field(
        estimates["lat"].to_numpy(),
        estimates["lon"].to_numpy(),
        values,
        units=units,
        non_negative=non_negative,
        elevation=grid,
    )

    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    mesh_lon, mesh_lat = np.meshgrid(field.lons, field.lats)
    # Colour limits are set by the *interpolated* region, not the whole field.
    # A handful of extrapolated Pyrenean cells would otherwise stretch the scale
    # and wash out every lowland difference the map exists to show; they clip
    # instead, which the colourbar's arrows declare.
    inside = field.values[field.interpolated]
    inside = inside[np.isfinite(inside)]
    scale_source = inside if inside.size else field.values[field.drawn]
    vmin = float(np.nanmin(scale_source))
    vmax = float(np.nanmax(scale_source))

    # Solid where interpolated, translucent where extrapolated. Two draws of the
    # same array under the same colour limits, so the colours stay comparable.
    for mask, alpha in (
        (field.interpolated, 1.0),
        (field.drawn & ~field.interpolated, 0.32),
    ):
        ax.pcolormesh(
            mesh_lon,
            mesh_lat,
            np.where(mask, field.values, np.nan),
            cmap=colours,
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
            shading="nearest",
            rasterized=True,
        )

    if field.extrapolates_elevation.any():
        ax.contourf(
            mesh_lon,
            mesh_lat,
            field.extrapolates_elevation.astype(float),
            levels=[0.5, 1.5],
            colors="none",
            hatches=["////"],
        )
        ax.contour(
            mesh_lon,
            mesh_lat,
            field.extrapolates_elevation.astype(float),
            levels=[0.5],
            colors=[MUTED],
            linewidths=0.5,
        )

    ax.scatter(
        estimates["lon"],
        estimates["lat"],
        s=26,
        c="white",
        edgecolors=INK,
        linewidths=0.9,
        zorder=5,
    )
    for slug, row in estimates.iterrows():
        ax.annotate(
            row["name"],
            (row["lon"], row["lat"]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=6.5,
            color=INK,
            zorder=6,
        )

    bar = fig.colorbar(
        plt.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax), cmap=colours
        ),
        ax=ax,
        fraction=0.035,
        pad=0.02,
        extend="both",
    )
    bar.set_label(f"{label} ({units})", fontsize=8, color=MUTED)
    bar.ax.tick_params(colors=MUTED, labelsize=7)
    bar.outline.set_visible(False)

    # A degree of longitude is shorter than a degree of latitude at these
    # latitudes, so without this Iberia comes out stretched east-west.
    ax.set_aspect(1.0 / np.cos(np.radians(float(np.mean(field.lats)))))
    ax.set_xlim(field.lons[0], field.lons[-1])
    ax.set_ylim(field.lats[0], field.lats[-1])
    ax.set_xlabel("longitude", fontsize=8, color=MUTED)
    ax.set_ylabel("latitude", fontsize=8, color=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=7)

    surface = field.surface
    surface_note = (
        "plain distance weighting (no surface fitted)"
        if surface is None
        else (
            # The elevation term is reported in the mapped field's own units, not
            # in °C: on the rain map it is percentage points per kilometre.
            f"surface in elevation, latitude and longitude "
            f"(R²={surface.r2:.2f}, {surface.lapse_rate_c_per_km:+.1f} {units}/km "
            f"with elevation, held flat above {surface.elevation_range[1]:.0f} m) "
            f"plus distance-weighted residuals"
        )
    )
    n = len(estimates)
    ax.set_title(
        f"{label}, {target:%-d %B}\n"
        f"climatological expectation from {n} cities · not a forecast",
        fontsize=10,
        color=INK,
        loc="left",
    )
    highest = 0.0 if surface is None else surface.elevation_range[1]
    # Hard-wrapped rather than left to flow: `_save` uses bbox_inches="tight", so
    # a line wider than the axes silently widens the whole figure and pushes the
    # colourbar away from the map.
    caption = "\n".join(
        [
            f"Method: {surface_note}.",
            f"Solid inside the hull of the {n} cities; translucent beyond it is "
            f"extrapolation, and nothing is drawn off their landmass.",
            f"Hatching marks cells above the highest city in the fit ({highest:.0f} m): "
            f"the elevation term is held at its fitted ceiling there, not carried on.",
            "Colour limits come from the interpolated region only, so extrapolated "
            "extremes clip rather than stretching the scale.",
            f"{n} cities cannot resolve a 9 km field: the furthest drawn cell is "
            f"{field.nearest_city_km[field.drawn].max():.0f} km from any of them. "
            f"ERA5 reanalysis, not station measurement.",
        ]
    )
    # Figure coordinates, below the axes, with room reserved for it — in axes
    # coordinates it collided with the longitude label and ran off the width.
    fig.subplots_adjust(bottom=0.2)
    fig.text(0.02, 0.015, caption, fontsize=6.5, color=MUTED, va="bottom", linespacing=1.6)

    written = _save(fig, f"map_{variable}_{target.month:02d}.png")

    # Beside the PNG, what the PNG cannot say for itself. The app reads this
    # instead of rebuilding a field per request, which keeps the rule that a web
    # request never triggers analysis.
    summary = field.summary()
    summary.update(
        {
            "variable": variable,
            "label": label,
            "target": target.isoformat(),
            "figure": f"map_{variable}_{target.month:02d}.png",
            "month": target.month,
            "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "n_cities": int(n),
            # The map inherits the archive's provisional status. The seasonal
            # models behind it are fitted on however many years have been
            # fetched, and below the data-quality gate that is a caveat the page
            # must carry rather than a footnote in a report nobody opens.
            "min_years": int(estimates["n_years"].min()),
            "gate_min_years": validate.MIN_YEARS,
            "provisional": bool(estimates["n_years"].min() < validate.MIN_YEARS),
            "missing_cities": list(estimates.attrs.get("missing", [])),
            "hull_margin_km": spatial.HULL_MARGIN_KM,
            "idw_power": spatial.IDW_POWER,
            "elevation_source": grid.source,
            "cities": [
                {
                    "slug": slug,
                    "name": row["name"],
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "region": row["region"],
                    "value": float(v),
                    "n_years": int(row["n_years"]),
                }
                for (slug, row), v in zip(estimates.iterrows(), values)
            ],
        }
    )
    paths.map_summary(variable, target.month).write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return written


def write_trend_table(table: pd.DataFrame, variable: str):
    """Write `reports/trends.md` from the same fits the figures used."""
    lines = [
        f"# Observed trends in `{variable}`",
        "",
        "Produced by `scripts/make_figures.py`, from the same Theil–Sen fits the",
        "`trend_<city>.png` figures draw. Theil–Sen rather than OLS because a",
        "long daily record contains outlier years; the OLS slope is reported",
        "beside it so the choice of estimator is auditable rather than asserted.",
        "",
        "**These are not climate projections.** Extrapolating a fitted trend is a",
        "statement about the record, not about emissions. See `wepr.trends` and",
        "the scenario panel of each figure for what projection actually requires.",
        "",
    ]

    if table.empty:
        lines += ["No city could be fitted.", ""]
    else:
        short = table[~table["is_climatological"]]
        if not short.empty:
            lines += [
                f"> **{len(short)} of {len(table)} cities have fewer than "
                f"{trends.MIN_TREND_YEARS} complete years on disk.** Their slopes are "
                "the record's own, not a warming rate — a 1950s–60s window sits on the "
                "mid-century plateau, where the real trend is near zero or negative. "
                "Resume `scripts/fetch_data.py` before quoting any of these.",
                "",
            ]
        lines += [
            "| City | Region | °C/decade | 95% CI | OLS | Years | Span | Climatological |",
            "|---|---|---:|---|---:|---:|---|---|",
        ]
        for slug, row in table.sort_values("slope_per_decade", ascending=False).iterrows():
            lines.append(
                f"| {row['name']} | {row['region']} | {row['slope_per_decade']:+.3f} | "
                f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] | "
                f"{row['ols_slope_per_decade']:+.3f} | {int(row['n_years'])} | "
                f"{int(row['first_year'])}–{int(row['last_year'])} | "
                f"{'yes' if row['is_climatological'] else 'NO'} |"
            )
        lines.append("")

    skipped = table.attrs.get("skipped", {})
    if skipped:
        lines += ["## Could not be fitted", ""]
        lines += [f"- `{slug}` — {why}" for slug, why in sorted(skipped.items())]
        lines.append("")

    lines += [
        "## Published projections quoted alongside",
        "",
        "| Scenario | Region | Season | Period | Baseline | °C | Source |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in trends.AR6_SCENARIOS:
        median = "—" if row["median"] is None else f"{row['median']:.1f}"
        lines.append(
            f"| {row['scenario']} | {row['region']} | {row['season']} | "
            f"{row['period'][0]}–{row['period'][1]} | "
            f"{row['baseline'][0]}–{row['baseline'][1]} | "
            f"{median} [{row['low']:.1f}–{row['high']:.1f}] | {row['source']} |"
        )
    lines += ["", f"> {trends.SCENARIO_GAP}", ""]

    path = paths.REPORTS / "trends.md"
    path.write_text("\n".join(lines))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--city", action="append", help="limit to this slug; repeatable")
    parser.add_argument(
        "--map-year",
        type=int,
        default=None,
        help=(
            "year the twelve monthly maps describe (default: next year). The day "
            "of year is not a flag: the map is published for all twelve months, "
            "because that is the variation worth showing"
        ),
    )
    parser.add_argument("--through-year", type=int, default=2100,
                        help="last year the trend extrapolation reaches")
    args = parser.parse_args()

    paths.ensure_dirs()

    slugs = args.city or load.available_cities()
    if not slugs:
        print(
            "nothing to draw: no city has processed data.\n"
            "Run scripts/fetch_data.py first.",
            file=sys.stderr,
        )
        return 1

    unknown = sorted(set(slugs) - set(load.available_cities()))
    if unknown:
        print(f"error: no processed data for {unknown}", file=sys.stderr)
        return 2

    written, failures = [], 0
    for slug in slugs:
        for label, draw in (
            ("trend", lambda s=slug: figure_trend(s, through_year=args.through_year)),
            ("climatology", lambda s=slug: figure_climatology(s)),
        ):
            try:
                written.append(draw())
            except (ValueError, KeyError, FileNotFoundError) as exc:
                print(f"  {slug} {label}: FAILED: {exc}", file=sys.stderr)
                failures += 1

    for slug in slugs:
        if slug in validate.VALIDATION_SLUGS:
            path = figure_validation(slug)
            if path is None:
                print(f"  {slug} validation: skipped, no overlapping dates "
                      "(see reports/reanalysis_validation.md)")
            else:
                written.append(path)

    if paths.metrics_json().exists():
        metrics = json.loads(paths.metrics_json().read_text())
        for label, draw in (("skill", figure_skill), ("seasonal", figure_seasonal)):
            path = draw(metrics)
            if path is None:
                print(f"  {label}: skipped, reports/metrics.json has no "
                      f"{label} records")
            else:
                written.append(path)
    else:
        print("  skill: skipped, no reports/metrics.json — run scripts/evaluate_all.py")

    # The map is a whole-registry figure, so it is skipped when --city narrowed the
    # run: a map drawn from one city is not a map.
    if args.city:
        print("  map: skipped, --city was given and the map needs the whole registry")
    else:
        # Every month, not one date. The seasonal cycle is the variation a reader
        # can see — about 21 °C between January and July across these cities —
        # whereas the target year moves the field by the fitted trend alone,
        # roughly +0.03 °C a year, which is finer than the colour scale resolves.
        # Publishing twelve months lets `/map` default to the *current* one, which
        # is what stops the page quietly showing a date that has already passed.
        map_year = args.map_year or dt.date.today().year + 1
        print(f"  map: {len(paths.MAP_MONTHS)} months x 2 variables for {map_year}",
              flush=True)
        stop = False
        for variable in ("tavg", "prcp"):
            if stop:
                break
            for month in paths.MAP_MONTHS:
                target = dt.date(map_year, month, paths.MAP_DAY_OF_MONTH)
                try:
                    written.append(figure_map(target, variable))
                except spatial.SpatialError as exc:
                    print(f"  map {variable}: skipped, {exc}".replace("\n", " "))
                    stop = True
                    break
                except (ValueError, KeyError, FileNotFoundError) as exc:
                    print(f"  map {variable} {month:02d}: FAILED: {exc}",
                          file=sys.stderr)
                    failures += 1

    table = trends.trend_table(slugs)
    report = write_trend_table(table, table.attrs["variable"])

    # The machine-readable half of the same fits, for `/trends`. Written here and
    # not computed in the request, for the reason the map is: a web request never
    # triggers analysis. Written from `table` and not refitted, so the page and
    # the `trend_<city>.png` beside it cannot disagree.
    summary_path = paths.trend_summary()
    summary_path.write_text(json.dumps(trends.summary(table), indent=2) + "\n")

    for path in written:
        print(f"  {path.relative_to(paths.ROOT)}")
    print(f"\n{len(written)} figure(s) -> {paths.FIGURES.relative_to(paths.ROOT)}")
    print(f"trend table -> {report.relative_to(paths.ROOT)}")
    print(f"trend summary -> {summary_path.relative_to(paths.ROOT)}")
    if failures:
        print(f"{failures} figure(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

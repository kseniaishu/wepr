"""Long-run temperature trends, and their honest extrapolation.

**What this is not.** Extrapolating a trend fitted to 75 years of reanalysis is
not climate projection. Real projection uses CMIP6 model ensembles under emissions
scenarios. A trend line presented as a 2075 forecast would not survive contact
with anyone who has climate exposure, and should not.

**What this is.** Two things plotted side by side:

1. The observed warming rate estimated from the data with proper uncertainty —
   e.g. `+0.31 °C/decade, 95% CI [0.26, 0.37]` — extrapolated forward and
   labelled explicitly as a statistical extrapolation, with bands that widen.
2. The published IPCC AR6 / CMIP6 scenario envelope for Iberia alongside it,
   with a citation.

The page then says: here is what the observed trend implies if it continues
linearly, and here is what climate science actually projects. Showing that the
difference is understood is the skill being demonstrated.

Theil–Sen is the default estimator rather than OLS because it is robust to the
outlier years that a 75-year daily record inevitably contains.

Implemented in Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .data import load

# Annual means are built from daily values, so a year that the record only
# partially covers is seasonally biased — a 1950 that starts in July is a summer
# mean masquerading as an annual one, and at the end of a fitted line a single
# biased point moves the slope more than any interior year. Years below this
# coverage are excluded rather than down-weighted, because an excluded year is
# visible in `n_years` while a down-weighted one is not.
MIN_YEAR_COVERAGE = 0.95

# Below this, "trend" is the wrong word: the WMO climate normal is 30 years, and a
# slope fitted to less is dominated by the record's own interannual swings. The
# estimate is still returned — refusing to compute it would just move the decision
# somewhere less visible — but `is_climatological` is False and every artifact that
# prints it says so.
MIN_TREND_YEARS = 30

# Multiplier turning a standard deviation into a 90% interval for a normal
# variable. Interannual temperature means are close enough to normal for this to
# be the right order; it is a band, not a p-value.
_Z90 = 1.6448536269514722


@dataclass(frozen=True, slots=True)
class Trend:
    """An estimated linear trend with uncertainty.

    Attributes:
        slope_per_decade: °C per decade.
        ci_low: Lower bound of the 95% confidence interval.
        ci_high: Upper bound of the 95% confidence interval.
        estimator: Which estimator produced it, e.g. `"theil-sen"`.
        first_year: First year in the fitted window.
        last_year: Last year in the fitted window.
        n_years: Years used.
        intercept: Fitted value at year 0, in the estimator's own units — carried
            so the line can be drawn and extrapolated without refitting.
        residual_sd: Standard deviation of the annual means about the fitted line.
            This is interannual variability, not estimation error, and it is the
            larger of the two for any horizon a reader cares about. Separating
            them is the difference between "we don't know the rate well" and
            "individual years scatter", which are not the same claim.
        ols_slope_per_decade: The OLS slope on the same data, reported alongside
            rather than instead. Theil–Sen is the default; showing both makes the
            choice of estimator auditable instead of asserted, and a large gap
            between them is itself a finding about the record.
        lag1_autocorrelation: Lag-1 autocorrelation of the residuals. The
            Theil–Sen confidence interval assumes independent observations, so a
            materially positive value here means the stated interval is **too
            narrow**. Surfaced rather than buried because the assumption is the
            weakest part of the estimate.
        is_climatological: Whether `n_years >= MIN_TREND_YEARS`.
    """

    slope_per_decade: float
    ci_low: float
    ci_high: float
    estimator: str
    first_year: int
    last_year: int
    n_years: int
    intercept: float
    residual_sd: float
    ols_slope_per_decade: float
    lag1_autocorrelation: float
    is_climatological: bool

    def value_at(self, year: float | np.ndarray):
        """The fitted line evaluated at a calendar year."""
        return self.intercept + (self.slope_per_decade / 10.0) * np.asarray(year, dtype="float64")

    def describe(self) -> str:
        """The one-line form used in reports and figure captions."""
        caveat = "" if self.is_climatological else f" (only {self.n_years} yr — not a climate trend)"
        return (
            f"{self.slope_per_decade:+.3f} °C/decade, 95% CI "
            f"[{self.ci_low:+.3f}, {self.ci_high:+.3f}], "
            f"{self.first_year}–{self.last_year}{caveat}"
        )


def annual_means(frame: pd.DataFrame, variable: str = "tavg") -> pd.Series:
    """Per-calendar-year means, excluding years the record only partly covers.

    Args:
        frame: Tidy daily frame from `wepr.data.load`.
        variable: Column to average.

    Returns:
        Mean of `variable` per year, indexed by integer year, named `variable`.
        Years with less than `MIN_YEAR_COVERAGE` of their days observed are
        dropped — see that constant for why exclusion beats down-weighting.

    Raises:
        KeyError: if `variable` is not a column.
    """
    if variable not in frame.columns:
        raise KeyError(f"frame lacks {variable!r}; has {list(frame.columns)}")

    index = pd.DatetimeIndex(frame.index)
    year = pd.Index(index.year, name="year")
    values = frame[variable]

    observed = values.groupby(year).count()
    # Days in each calendar year, not days present in the frame: a year truncated
    # by the fetch window and a year full of NaN are the same defect here, and
    # counting against the calendar catches both with one test.
    in_year = pd.Series(
        [366 if pd.Timestamp(y, 1, 1).is_leap_year else 365 for y in observed.index],
        index=observed.index,
    )
    keep = (observed / in_year) >= MIN_YEAR_COVERAGE

    means = values.groupby(year).mean()
    return means[keep].rename(variable)


def theil_sen(years: np.ndarray, values: np.ndarray, alpha: float = 0.95) -> Trend:
    """Fit a robust linear trend to one value per year.

    Separate from `annual_trend` so it can be tested against a series with a known
    slope without any data on disk, and so a caller holding annual means from
    somewhere else (a station record, a scenario run) can use the same estimator.

    The confidence interval is the one `scipy.stats.theilslopes` derives from the
    Kendall-tau distribution, which assumes the observations are independent.
    Annual means are close to independent — the atmosphere does not remember one
    year into the next nearly as strongly as one day into the next — but not
    exactly so, which is why `lag1_autocorrelation` is returned with it.

    Args:
        years: Calendar years, ascending.
        values: One value per year.
        alpha: Confidence level for the slope interval.

    Returns:
        The fitted trend. Slopes are converted to °C/decade, which is the unit the
        literature uses and the only one a reader can compare against anything.

    Raises:
        ValueError: on mismatched lengths, or fewer than three usable years — two
            points define a line with no residual and no interval, which would be
            reported as a trend with zero uncertainty.
    """
    years = np.asarray(years, dtype="float64")
    values = np.asarray(values, dtype="float64")
    if years.shape != values.shape:
        raise ValueError(f"years {years.shape} and values {values.shape} differ in length")

    finite = np.isfinite(years) & np.isfinite(values)
    years, values = years[finite], values[finite]
    if len(years) < 3:
        raise ValueError(
            f"need at least 3 years to fit a trend with an interval, got {len(years)}"
        )

    fit = stats.theilslopes(values, years, alpha=alpha)
    ols = stats.linregress(years, values)

    residuals = values - (fit.intercept + fit.slope * years)
    # Population sd with one degree of freedom spent on the slope and one on the
    # intercept, matching how the residuals were produced.
    residual_sd = float(np.sqrt(np.sum(residuals**2) / max(len(years) - 2, 1)))

    if len(residuals) > 2 and residuals.std() > 0:
        lag1 = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1])
    else:
        lag1 = float("nan")

    return Trend(
        slope_per_decade=float(fit.slope * 10.0),
        ci_low=float(fit.low_slope * 10.0),
        ci_high=float(fit.high_slope * 10.0),
        estimator="theil-sen",
        first_year=int(years[0]),
        last_year=int(years[-1]),
        n_years=len(years),
        intercept=float(fit.intercept),
        residual_sd=residual_sd,
        ols_slope_per_decade=float(ols.slope * 10.0),
        lag1_autocorrelation=lag1,
        is_climatological=len(years) >= MIN_TREND_YEARS,
    )


def annual_trend(slug: str, variable: str = "tavg") -> Trend:
    """Estimate the long-run trend in a city's annual means.

    Args:
        slug: City slug.
        variable: Column to fit, default daily mean temperature.

    Returns:
        The fitted trend with its confidence interval.
    """
    series = annual_means(load.load_city(slug), variable)
    return theil_sen(series.index.to_numpy(), series.to_numpy())


def extrapolate(trend: Trend, through_year: int) -> pd.DataFrame:
    """Project the fitted line forward, with bands that widen away from the data.

    **This is a statistical extrapolation and nothing more.** It answers "what does
    the observed rate imply if it continues linearly", a question about the record;
    it does not answer "what will the climate do", a question about emissions,
    feedbacks and ocean heat uptake that only a GCM ensemble addresses. Every
    artifact that draws this is required to say so, and to put
    `AR6_SCENARIOS` next to it.

    Two bands are returned, not one, because they answer different questions and
    adding them into a single ribbon would make both unreadable:

    * `rate_lo` / `rate_hi` — the slope confidence interval swung about the last
      observed year. Zero width at the anchor and widening linearly: this is
      uncertainty about *the rate*, which compounds with distance.
    * `year_lo` / `year_hi` — the rate band plus 90% of interannual scatter. This
      is the band an individual future year would fall in. It does not shrink to
      nothing at the anchor, because single years scatter about the trend even
      where the trend is known exactly.

    Args:
        trend: A fitted trend.
        through_year: Last year to project to, inclusive.

    Returns:
        One row per year from the record's last year to `through_year`, with the
        central line and both bands. The first row is the anchor, so a plot can
        join the fitted line to the extrapolation without a gap.

    Raises:
        ValueError: if `through_year` precedes the end of the record.
    """
    if through_year < trend.last_year:
        raise ValueError(
            f"through_year {through_year} is before the record ends ({trend.last_year}); "
            "extrapolate forwards, and read the fitted line backwards from `Trend.value_at`"
        )

    years = np.arange(trend.last_year, through_year + 1, dtype="float64")
    elapsed_decades = (years - trend.last_year) / 10.0
    central = trend.value_at(years)

    # Swung about the anchor rather than about year 0: the fitted value at the end
    # of the record is the best-known point on the line, so that is where the
    # forward uncertainty should pivot. Pivoting at the intercept instead would put
    # a spurious ±2 °C of width on the present day.
    rate_half = np.maximum(
        np.abs(trend.ci_high - trend.slope_per_decade),
        np.abs(trend.slope_per_decade - trend.ci_low),
    ) * elapsed_decades
    scatter = _Z90 * trend.residual_sd

    return pd.DataFrame(
        {
            "year": years.astype("int64"),
            "central": central,
            "rate_lo": central - rate_half,
            "rate_hi": central + rate_half,
            "year_lo": central - rate_half - scatter,
            "year_hi": central + rate_half + scatter,
        }
    ).set_index("year")


# --- The published projections, for the other half of the page ----------------
#
# These are quoted, not computed. Each carries its own baseline period, because a
# warming figure without one is meaningless, and the baselines here are *not* the
# same: comparing a number on a 1850–1900 baseline with one on 1986–2005 as though
# they were interchangeable would overstate the first by roughly 0.6 °C.
#
# `median` is None where the source states a range without a central estimate.
# A plotted median that the source does not give would be an invention, so the
# field stays empty and the figure draws the band alone.
AR6_SCENARIOS: tuple[dict, ...] = (
    # IPCC AR6 WGI Summary for Policymakers, Table SPM.1: global surface
    # temperature change, 2081–2100, best estimate and *very likely* range.
    # Global, not Iberian — see `SCENARIO_GAP` below for why that matters and why
    # it is still worth plotting.
    {
        "scenario": "SSP1-1.9", "region": "global", "season": "annual",
        "period": (2081, 2100), "baseline": (1850, 1900),
        "median": 1.4, "low": 1.0, "high": 1.8,
        "source": "IPCC AR6 WGI SPM, Table SPM.1 (very likely range)",
    },
    {
        "scenario": "SSP1-2.6", "region": "global", "season": "annual",
        "period": (2081, 2100), "baseline": (1850, 1900),
        "median": 1.8, "low": 1.3, "high": 2.4,
        "source": "IPCC AR6 WGI SPM, Table SPM.1 (very likely range)",
    },
    {
        "scenario": "SSP2-4.5", "region": "global", "season": "annual",
        "period": (2081, 2100), "baseline": (1850, 1900),
        "median": 2.7, "low": 2.1, "high": 3.5,
        "source": "IPCC AR6 WGI SPM, Table SPM.1 (very likely range)",
    },
    {
        "scenario": "SSP3-7.0", "region": "global", "season": "annual",
        "period": (2081, 2100), "baseline": (1850, 1900),
        "median": 3.6, "low": 2.8, "high": 4.6,
        "source": "IPCC AR6 WGI SPM, Table SPM.1 (very likely range)",
    },
    {
        "scenario": "SSP5-8.5", "region": "global", "season": "annual",
        "period": (2081, 2100), "baseline": (1850, 1900),
        "median": 4.4, "low": 3.3, "high": 5.7,
        "source": "IPCC AR6 WGI SPM, Table SPM.1 (very likely range)",
    },
    # Cos et al. (2022), Earth System Dynamics 13, 321–340,
    # https://doi.org/10.5194/esd-13-321-2022 — the Mediterranean region in the
    # CMIP6 ensemble, 90% inter-model spread, on a 1986–2005 baseline. Seasonal
    # rather than annual, which is what the paper reports.
    {
        "scenario": "SSP1-2.6", "region": "mediterranean-land", "season": "JJA",
        "period": (2081, 2100), "baseline": (1986, 2005),
        "median": 2.3, "low": 1.2, "high": 3.3,
        "source": "Cos et al. 2022, ESD 13:321 (CMIP6, 90% inter-model spread)",
    },
    {
        "scenario": "SSP5-8.5", "region": "mediterranean-land", "season": "JJA",
        "period": (2081, 2100), "baseline": (1986, 2005),
        "median": 7.4, "low": 5.6, "high": 9.1,
        "source": "Cos et al. 2022, ESD 13:321 (CMIP6, 90% inter-model spread)",
    },
    {
        "scenario": "SSP5-8.5", "region": "mediterranean-land", "season": "DJF",
        "period": (2081, 2100), "baseline": (1986, 2005),
        "median": None, "low": 3.3, "high": 6.8,
        "source": "Cos et al. 2022, ESD 13:321 (CMIP6, 90% inter-model spread)",
    },
)

# Stated rather than silently absent. The right overlay for an annual-mean trend
# in a specific Iberian city is the AR6 Interactive Atlas annual-mean change for
# the MED reference region; those values are not in `AR6_SCENARIOS` because they
# could not be retrieved from a citable source at the time of writing. Adding them
# is a data edit above, not a code change. Until then the figures overlay the
# global annual envelope and label it as global — which is informative in its own
# right, since AR6 finds Mediterranean land warming *faster* than the global mean,
# so a city extrapolation landing below the global band is a warning about the
# extrapolation rather than good news about Iberia.
SCENARIO_GAP = (
    "No annual-mean CMIP6 envelope for the AR6 Mediterranean reference region is "
    "encoded; the global annual envelope is shown instead and labelled as global. "
    "Mediterranean land warms faster than the global mean (AR6 WGI Atlas), so the "
    "global band is a floor for Iberia, not an estimate of it."
)


def scenarios(region: str | None = None, season: str = "annual") -> pd.DataFrame:
    """The quoted projections, as a frame, filtered to one region and season.

    Args:
        region: `"global"`, `"mediterranean-land"`, or None for every region.
        season: `"annual"`, `"JJA"` or `"DJF"`.

    Returns:
        One row per scenario, with its baseline and source. Empty if nothing
        matches, which is a legitimate answer — see `SCENARIO_GAP`.
    """
    frame = pd.DataFrame(list(AR6_SCENARIOS))
    frame = frame[frame["season"] == season]
    if region is not None:
        frame = frame[frame["region"] == region]
    return frame.reset_index(drop=True)


def baseline_mean(frame: pd.DataFrame, baseline: tuple[int, int], variable: str = "tavg") -> float:
    """Mean of `variable` over a baseline period, or NaN if the record misses it.

    A scenario figure is an anomaly against its own baseline, so putting one next
    to a local extrapolation means expressing both as anomalies against that same
    baseline — which requires the local record to actually cover it. Returning NaN
    rather than the mean of whatever years happen to be present is the whole point:
    a 1986–2005 anomaly computed from a 1950–1971 record is off by the very warming
    the page is about.

    Args:
        frame: Tidy daily frame.
        baseline: Inclusive `(first_year, last_year)`.
        variable: Column to average.

    Returns:
        The baseline mean, or NaN if any year of the baseline is missing from the
        annual means.
    """
    first, last = baseline
    means = annual_means(frame, variable)
    wanted = set(range(first, last + 1))
    if not wanted.issubset(set(means.index)):
        return float("nan")
    return float(means.loc[first:last].mean())


def trend_table(slugs: list[str] | None = None, variable: str = "tavg") -> pd.DataFrame:
    """Fitted trends for several cities, one row each, for the report and the page.

    Args:
        slugs: Cities to fit. Defaults to every city with processed data.
        variable: Column to fit.

    Returns:
        One row per city that could be fitted, indexed by slug, carrying the slope,
        its interval, both diagnostics and `is_climatological`. A city whose record
        is too short to fit at all is omitted and named in `frame.attrs["skipped"]`,
        so it is visible rather than silently missing. `frame.attrs["fits"]` holds
        the `Trend` objects the rows were flattened from, which is what lets
        `summary` extrapolate without refitting.
    """
    from .cities import get

    slugs = slugs if slugs is not None else load.available_cities()
    rows, skipped, fits = [], {}, {}

    for slug in slugs:
        try:
            trend = annual_trend(slug, variable)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            skipped[slug] = str(exc)
            continue
        fits[slug] = trend
        rows.append(
            {
                "slug": slug,
                "name": get(slug).name,
                "region": get(slug).region,
                "slope_per_decade": trend.slope_per_decade,
                "ci_low": trend.ci_low,
                "ci_high": trend.ci_high,
                "ols_slope_per_decade": trend.ols_slope_per_decade,
                "residual_sd": trend.residual_sd,
                "lag1_autocorrelation": trend.lag1_autocorrelation,
                "first_year": trend.first_year,
                "last_year": trend.last_year,
                "n_years": trend.n_years,
                "is_climatological": trend.is_climatological,
            }
        )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.set_index("slug")
    frame.attrs["skipped"] = skipped
    frame.attrs["variable"] = variable
    # The fitted objects themselves, alongside the flattened rows. `summary`
    # below needs the intercept and the residual sd together to extrapolate the
    # line, and a second `annual_trend` call per city to recover them would be a
    # second fit — which is how a table and the figure beside it come to disagree
    # about the same city's warming rate.
    frame.attrs["fits"] = fits
    return frame


# --- The publishable artifact ---------------------------------------------------
#
# The climate-trends page faced the same choice the map faced: interpolate a field
# (or fit twenty trends) inside the request, or serve what a script already
# computed. The project's rule — a web request never triggers analysis — settles it
# the same way, so `scripts/make_figures.py` writes this beside the figures it
# draws and `/trends` serves it. The alternative, a trend table held from startup,
# would satisfy the letter of the rule but would fit the slopes a second time,
# independently of the `trend_<city>.png` the page shows next to them. That is
# exactly the divergence that put `reports/trends.md` in the figure script, and it
# would be worse here: a figure and a number disagreeing about the same city's
# warming rate on the same page.

# The years the page quotes the extrapolation at. 2050 is within a career; 2100 is
# where the AR6 scenario period ends, so the two halves of the page can be read
# against each other at one shared year.
DEFAULT_HORIZONS: tuple[int, ...] = (2050, 2100)


def summary(
    table: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> dict:
    """Everything the climate-trends page shows, as JSON-ready data.

    Built from `trend_table`'s own fits — see `frame.attrs["fits"]` — so the
    numbers on the page and the numbers in the figures beside it come from one
    estimation.

    **The two halves are kept structurally apart,** because conflating them is the
    error the page exists to demonstrate an understanding of. `cities[].extrapolation`
    is a statistical statement about *this record*: what the fitted rate implies if
    it continues linearly. `scenarios` is a quoted CMIP6 result, with its source and
    its own baseline. They are different kinds of claim and they are not merged into
    one series, not even where a shared axis would look tidier.

    **`baselines` is why they are not drawn against a common zero.** A scenario
    figure is an anomaly against its own baseline period, so overlaying it on a
    local record requires the record to cover that period. Each entry names the
    cities whose record actually does. On the current fetch none do, and a page
    that silently slid a 1850–1900 anomaly onto a 1950-baselined axis would
    overstate it by most of a degree — the size of the effect under discussion.

    Args:
        table: The frame `trend_table` returned, with its `attrs`.
        horizons: Calendar years to quote the extrapolation at.

    Returns:
        A JSON-serialisable dict. Floats are rounded to the precision the page
        prints, so the artifact does not imply more than it shows.

    Raises:
        KeyError: if `table` did not come from `trend_table` (no `attrs["fits"]`).
            Raised rather than refitting, because a silent refit here is the
            divergence this function exists to prevent.
    """
    import datetime as dt

    if "fits" not in table.attrs:
        raise KeyError(
            "table.attrs['fits'] is missing; pass the frame trends.trend_table "
            "returned rather than a reconstructed one"
        )

    fits: dict[str, Trend] = table.attrs["fits"]
    variable = table.attrs.get("variable", "tavg")
    horizons = tuple(sorted(horizons))

    cities = []
    for slug, row in table.sort_values("slope_per_decade", ascending=False).iterrows():
        fit = fits[slug]
        forward = extrapolate(fit, max(horizons[-1], fit.last_year))
        spans_zero = bool(fit.ci_low <= 0.0 <= fit.ci_high)
        cities.append(
            {
                "slug": slug,
                "name": row["name"],
                "region": row["region"],
                "figure": f"trend_{slug}.png",
                "slope_per_decade": round(fit.slope_per_decade, 3),
                "ci_low": round(fit.ci_low, 3),
                "ci_high": round(fit.ci_high, 3),
                "ci_spans_zero": spans_zero,
                "ols_slope_per_decade": round(fit.ols_slope_per_decade, 3),
                "residual_sd": round(fit.residual_sd, 3),
                "lag1_autocorrelation": (
                    round(fit.lag1_autocorrelation, 3)
                    if np.isfinite(fit.lag1_autocorrelation)
                    else None
                ),
                # A materially positive lag-1 autocorrelation means the stated CI
                # assumes more independent years than the record contains, so the
                # interval is too narrow. Flagged as data rather than left for a
                # reader to infer from the coefficient.
                "ci_is_too_narrow": bool(
                    np.isfinite(fit.lag1_autocorrelation)
                    and fit.lag1_autocorrelation > 0.2
                ),
                "first_year": int(fit.first_year),
                "last_year": int(fit.last_year),
                "n_years": int(fit.n_years),
                "is_climatological": bool(fit.is_climatological),
                # The same test `wepr.serve.TrendTerm.is_sound` applies, minus its
                # distance condition: a 2100 extrapolation is always carried far
                # past its record, so that part is reported per horizon instead of
                # collapsed into one flag here.
                "is_sound": bool(fit.is_climatological and not spans_zero),
                "extrapolation": [
                    {
                        "year": int(year),
                        "years_beyond_record": int(year) - int(fit.last_year),
                        "central": round(float(forward.loc[year, "central"]), 2),
                        "rate_lo": round(float(forward.loc[year, "rate_lo"]), 2),
                        "rate_hi": round(float(forward.loc[year, "rate_hi"]), 2),
                        "year_lo": round(float(forward.loc[year, "year_lo"]), 2),
                        "year_hi": round(float(forward.loc[year, "year_hi"]), 2),
                    }
                    for year in horizons
                    if year in forward.index
                ],
            }
        )

    # Which scenario baselines any city's record actually covers. Loaded here
    # rather than carried on `Trend`, because coverage is a property of the daily
    # record and a trend knows only the years it was fitted on — which, with a gap
    # year dropped by `MIN_YEAR_COVERAGE`, is not the same thing.
    baselines = []
    for baseline in sorted({tuple(s["baseline"]) for s in AR6_SCENARIOS}):
        covered = []
        for slug in fits:
            try:
                mean = baseline_mean(load.load_city(slug), baseline, variable)
            except (FileNotFoundError, KeyError):
                continue
            if np.isfinite(mean):
                covered.append(slug)
        baselines.append(
            {
                "period": [int(baseline[0]), int(baseline[1])],
                "covered_by": sorted(covered),
                "n_covered": len(covered),
            }
        )

    short = table[~table["is_climatological"]] if not table.empty else table
    years = [int(f.n_years) for f in fits.values()]

    return {
        "variable": variable,
        "units": "°C" if variable in {"tavg", "tmax", "tmin"} else "mm",
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "estimator": "theil-sen",
        "min_trend_years": MIN_TREND_YEARS,
        "min_year_coverage": MIN_YEAR_COVERAGE,
        "horizons": [int(y) for y in horizons],
        "n_cities": int(len(table)),
        "n_climatological": int(len(table) - len(short)) if not table.empty else 0,
        # The page's headline caveat. True whenever any city's slope is fitted to
        # less than a climate normal, which on this fetch is all of them.
        "provisional": bool(len(short) > 0),
        "record_years": {"min": min(years), "max": max(years)} if years else None,
        "cities": cities,
        "skipped": dict(table.attrs.get("skipped", {})),
        "baselines": baselines,
        "scenarios": [
            {
                **scenario,
                "period": [int(scenario["period"][0]), int(scenario["period"][1])],
                "baseline": [int(scenario["baseline"][0]), int(scenario["baseline"][1])],
            }
            for scenario in AR6_SCENARIOS
        ],
        "scenario_gap": SCENARIO_GAP,
    }

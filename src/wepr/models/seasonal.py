"""The seasonal estimate — explicitly not a forecast.

For a date in 2031 there are no recent observations to condition on, so there is
nothing for a supervised model to use. The honest output is the climatological
expectation for that day of year, adjusted for the observed warming trend, with
an interval wide enough to reflect that it describes a *distribution of plausible
days*, not a prediction of one.

This replaces Prophet, which previously served these numbers while never being
evaluated. The replacement is about thirty lines of arithmetic whose every term
can be named, and — unlike Prophet here — it is measured against the baselines
like everything else.

What the UI must convey, and what this model therefore returns: a central
expectation, an interval, and the historical range for that day of year. A single
number to two decimal places for a date seven years out is a lie told by
formatting.

**The three terms, in full.** For a target date in year `Y` at calendar slot `d`:

    expected = slot_mean[d] + slope_per_year * (Y - anchor_year)

`slot_mean[d]` is the mean of every observation within `WINDOW` days of slot `d`,
*after* the record's linear trend has been removed; `anchor_year` is the mean year
of the record, so the slot mean is an estimate centred there and the second term
carries it forward. Detrending before pooling matters twice over: it stops the
trend being double-counted, and — the part that is easy to miss — it keeps the
interval honest. Percentiles taken from a warming record include the warming as
*spread*, so a p10–p90 band built from raw values is wider than any single year's
weather ever was, and widens further the longer the record gets.

**What is not modelled, and why.** There is no per-slot trend: 76 observations of
one calendar day cannot support 365 independent slopes, and fitting them would
produce a sawtooth of noise across the year. There is no precipitation trend
either — reanalysis precipitation is the project's weaker field (see CLAUDE.md),
and a trend fitted to it would be the least defensible number on the page.

Implemented in Phase 4.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .. import trends
from ..data import load
from . import baselines
from .nextday import WET_DAY_MM

# Half-width of the day-of-year pooling window, in days. Matches
# `baselines.CLIMATOLOGY_WINDOW` deliberately: the seasonal product and the
# climatology baseline it is scored against must smooth identically, or the
# comparison measures the window rather than the model.
WINDOW = baselines.CLIMATOLOGY_WINDOW

# The interval the model reports. 10th-90th rather than 95%: with ~75 years per
# slot even after pooling, a 95% interval is estimated from the two or three most
# extreme days in the window and moves with whichever freak year is in the record.
LOW_PERCENTILE, HIGH_PERCENTILE = 10.0, 90.0

# Nominal coverage of that interval, used to score calibration. An interval whose
# realised coverage is 0.55 is not an 80% interval, whatever it is labelled.
NOMINAL_COVERAGE = (HIGH_PERCENTILE - LOW_PERCENTILE) / 100.0

# Minimum years in the record before a seasonal model will fit at all. Below this
# the slot percentiles are estimated from a handful of days and the trend term is
# meaningless; see `trends.MIN_TREND_YEARS` for the stronger threshold at which
# the trend becomes a climate statement rather than an arithmetic one.
MIN_FIT_YEARS = 10


@dataclass(frozen=True, slots=True)
class SeasonalEstimate:
    """A climatological expectation for one future date.

    Attributes:
        expected_c: Trend-adjusted climatological mean temperature, °C.
        p10_c: 10th percentile of historical values for that day of year.
        p90_c: 90th percentile of historical values for that day of year.
        rain_probability: Historical fraction of wet days for that day of year.
        trend_adjustment_c: How much of `expected_c` comes from the fitted trend
            rather than the raw historical mean. Surfaced so the trend's
            contribution is visible rather than baked in silently.
        n_years: Years of record behind the estimate.
    """

    expected_c: float
    p10_c: float
    p90_c: float
    rain_probability: float
    trend_adjustment_c: float
    n_years: int


class SeasonalClimatology:
    """Climatology plus trend, fitted to one city's record and persistable.

    A plain object rather than a scikit-learn estimator: it is not cross-validated
    over hyperparameters and never passes through `clone`, so inheriting the
    estimator API would buy nothing and imply a contract it does not keep. It does
    pickle, which is what `wepr.models.registry` needs.

    Attributes are set by `fit` and are deliberately all plain arrays — the whole
    fitted state is 365 numbers per statistic plus a slope, which is why this model
    can be read and defended in a way Prophet could not.
    """

    def __init__(self, window: int = WINDOW, variable: str = "tavg") -> None:
        self.window = window
        self.variable = variable

    # -- fitting ------------------------------------------------------------
    def fit(self, frame: pd.DataFrame) -> SeasonalClimatology:
        """Fit the slot statistics and the trend from a tidy daily frame.

        Args:
            frame: Tidy daily frame from `wepr.data.load`.

        Returns:
            self, fitted.

        Raises:
            ValueError: if the record is shorter than `MIN_FIT_YEARS` usable years.
        """
        index = pd.DatetimeIndex(frame.index)
        year = np.asarray(index.year)
        slot = baselines._calendar_slot(index)
        values = frame[self.variable].to_numpy(dtype="float64")

        annual = trends.annual_means(frame, self.variable)
        if len(annual) < MIN_FIT_YEARS:
            raise ValueError(
                f"{len(annual)} complete year(s) of {self.variable}, need "
                f"{MIN_FIT_YEARS} to fit a seasonal model"
            )

        self.trend_ = trends.theil_sen(annual.index.to_numpy(), annual.to_numpy())
        self.slope_per_year_ = self.trend_.slope_per_decade / 10.0
        # The mean of the *complete* years, so the anchor is not dragged by a
        # partial year at either end of the fetch window.
        self.anchor_year_ = float(np.mean(annual.index.to_numpy(dtype="float64")))
        self.first_year_ = int(annual.index.min())
        self.last_year_ = int(annual.index.max())

        detrended = values - self.slope_per_year_ * (year - self.anchor_year_)
        wet = np.where(np.isnan(frame["prcp"].to_numpy(dtype="float64")), np.nan,
                       (frame["prcp"].to_numpy(dtype="float64") >= WET_DAY_MM).astype("float64"))

        self.slots_ = np.arange(1, 366)
        self.mean_ = np.full(365, np.nan)
        self.sd_ = np.full(365, np.nan)
        self.p_low_ = np.full(365, np.nan)
        self.p_high_ = np.full(365, np.nan)
        self.rain_probability_ = np.full(365, np.nan)
        self.n_observations_ = np.zeros(365, dtype="int64")
        self.n_years_ = np.zeros(365, dtype="int64")

        order = np.argsort(slot, kind="stable")
        slot_sorted = slot[order]
        bounds = np.searchsorted(slot_sorted, self.slots_, side="left")
        ends = np.searchsorted(slot_sorted, self.slots_, side="right")
        per_slot = [order[bounds[i]:ends[i]] for i in range(365)]

        for i in range(365):
            # Circular neighbourhood: the window for 2 January has to reach back
            # into late December, and a clipped window would compute the turn of
            # the year from half the samples.
            neighbours = [(i + k) % 365 for k in range(-self.window, self.window + 1)]
            rows = np.concatenate([per_slot[j] for j in neighbours]) if neighbours else np.array([], int)
            if rows.size == 0:
                continue
            pooled = detrended[rows]
            finite = pooled[np.isfinite(pooled)]
            if finite.size:
                self.mean_[i] = finite.mean()
                self.sd_[i] = finite.std(ddof=1) if finite.size > 1 else np.nan
                self.p_low_[i], self.p_high_[i] = np.percentile(
                    finite, [LOW_PERCENTILE, HIGH_PERCENTILE]
                )
                self.n_observations_[i] = finite.size
                self.n_years_[i] = len(np.unique(year[rows][np.isfinite(pooled)]))
            wet_pooled = wet[rows]
            wet_finite = wet_pooled[np.isfinite(wet_pooled)]
            if wet_finite.size:
                self.rain_probability_[i] = wet_finite.mean()

        return self

    @property
    def fitted(self) -> bool:
        """Whether `fit` has run. Checked by every prediction path."""
        return hasattr(self, "mean_")

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("SeasonalClimatology.fit has not been called")

    # -- prediction ---------------------------------------------------------
    def trend_adjustment(self, target_year: int | np.ndarray):
        """The trend term alone: how far the expectation sits from the slot mean."""
        self._require_fitted()
        return self.slope_per_year_ * (np.asarray(target_year, dtype="float64") - self.anchor_year_)

    def predict_one(self, target: date) -> SeasonalEstimate:
        """The estimate for a single date.

        Args:
            target: Any date, past or future. A past date inside the record gets
                the same treatment as a future one — the model is climatological,
                so it never conditions on nearby observations and there is nothing
                to leak.

        Returns:
            The estimate with its interval.
        """
        self._require_fitted()
        stamp = pd.Timestamp(target)
        i = int(baselines._calendar_slot(pd.DatetimeIndex([stamp]))[0]) - 1
        adjustment = float(self.trend_adjustment(stamp.year))
        return SeasonalEstimate(
            expected_c=float(self.mean_[i] + adjustment),
            p10_c=float(self.p_low_[i] + adjustment),
            p90_c=float(self.p_high_[i] + adjustment),
            rain_probability=float(self.rain_probability_[i]),
            trend_adjustment_c=adjustment,
            n_years=int(self.n_years_[i]),
        )

    def predict_frame(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Estimates for many dates at once — the vectorised form of `predict_one`.

        Used by the backtest, where `predict_one` per day over sixty held-out years
        would be a python loop over twenty thousand dates.
        """
        self._require_fitted()
        index = pd.DatetimeIndex(dates)
        i = baselines._calendar_slot(index) - 1
        adjustment = self.trend_adjustment(np.asarray(index.year))
        return pd.DataFrame(
            {
                "expected": self.mean_[i] + adjustment,
                "p_low": self.p_low_[i] + adjustment,
                "p_high": self.p_high_[i] + adjustment,
                # The untrended slot mean, kept alongside so the backtest can score
                # the trend term's contribution directly instead of inferring it.
                "no_trend": self.mean_[i],
                "rain_probability": self.rain_probability_[i],
                "trend_adjustment": adjustment,
            },
            index=index,
        )

    def table(self) -> pd.DataFrame:
        """The fitted state as one row per calendar slot, for figures and the page.

        365 rows, not 366. 29 February shares slot 60 with 1 March, following
        `wepr.models.baselines._calendar_slot`: giving it a slot of its own would
        found it on a quarter of the samples, and the two dates are
        climatologically indistinguishable.
        """
        self._require_fitted()
        reference = pd.Timestamp(2001, 1, 1) + pd.to_timedelta(self.slots_ - 1, unit="D")
        return pd.DataFrame(
            {
                "slot": self.slots_,
                "date_label": reference.strftime("%d %b"),
                "mean": self.mean_,
                "sd": self.sd_,
                f"p{int(LOW_PERCENTILE)}": self.p_low_,
                f"p{int(HIGH_PERCENTILE)}": self.p_high_,
                "rain_probability": self.rain_probability_,
                "n_observations": self.n_observations_,
                "n_years": self.n_years_,
            }
        ).set_index("slot")


# -- module-level convenience, for scripts and the CLI --------------------------
#
# These load from disk on every call. That is correct for a script and wrong for a
# web request: the app loads each model once at startup and calls `predict_one` on
# the object it is holding, as `wepr.models.registry` describes. No caching is done
# here on purpose — a cache here would quietly serve a stale model after retraining,
# and the fix for slowness is to hold the object, not to memoise the loader.


def fit(slug: str, variable: str = "tavg") -> SeasonalClimatology:
    """Fit a seasonal model to one city's processed record."""
    return SeasonalClimatology(variable=variable).fit(load.load_city(slug))


def estimate(slug: str, target: date) -> SeasonalEstimate:
    """Climatological expectation for a city on a future date.

    Args:
        slug: City slug.
        target: The date to describe. May be arbitrarily far out; the estimate
            does not get worse with distance in the way a forecast would, because
            it was never conditioned on recent observations in the first place.

    Returns:
        The estimate, with its interval.
    """
    from . import registry

    key = registry.model_key(slug, "seasonal", "tavg")
    try:
        model = registry.load(key)
    except FileNotFoundError:
        # Falling back to a fit here would mean a request could trigger training,
        # which is the one thing the app must never do. Scripts get a clear
        # instruction instead.
        raise
    return model.predict_one(target)


def city_estimates(target: date, slugs: Sequence[str] | None = None) -> pd.DataFrame:
    """Every city's climatological expectation for one date, as a frame.

    The map's value source. It exists here rather than in `wepr.spatial` so the
    spatial code stays model-agnostic — it interpolates numbers and knows nothing
    about where they came from — and so the figure script and the Flask app do not
    each grow their own copy of this loop.

    Cities whose seasonal model has not been trained are **omitted**, not filled
    with a placeholder. A map drawn from fifteen cities is a weaker map than one
    drawn from twenty, and `missing` on the result says which, but it is still
    honest; inventing the other five would not be.

    Args:
        target: The date to describe.
        slugs: Limit to these cities; defaults to the whole registry.

    Returns:
        One row per city with a trained model, indexed by slug, carrying `name`,
        `lat`, `lon`, `region`, `expected_c`, `p10_c`, `p90_c`,
        `rain_probability`, `trend_adjustment_c` and `n_years`. The frame carries
        the slugs it could not load in ``frame.attrs["missing"]``.
    """
    from wepr.cities import CITIES, get

    wanted = [get(s) for s in slugs] if slugs is not None else list(CITIES)
    rows = []
    missing: list[str] = []
    for city in wanted:
        try:
            value = estimate(city.slug, target)
        except FileNotFoundError:
            missing.append(city.slug)
            continue
        rows.append(
            {
                "slug": city.slug,
                "name": city.name,
                "lat": city.lat,
                "lon": city.lon,
                "region": city.region,
                "expected_c": value.expected_c,
                "p10_c": value.p10_c,
                "p90_c": value.p90_c,
                "rain_probability": value.rain_probability,
                "trend_adjustment_c": value.trend_adjustment_c,
                "n_years": value.n_years,
            }
        )

    columns = [
        "slug", "name", "lat", "lon", "region", "expected_c", "p10_c", "p90_c",
        "rain_probability", "trend_adjustment_c", "n_years",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty:
        frame = frame.set_index("slug")
    frame.attrs["missing"] = missing
    frame.attrs["target"] = target.isoformat()
    return frame


def city_estimates_for_month(
    year: int, month: int, slugs: Sequence[str] | None = None
) -> pd.DataFrame:
    """Every city's climatological norm for a whole calendar month.

    **Why the map averages a month rather than describing one day.** The map is a
    monthly norm and should be computed as one. Taking a single mid-month day gets
    almost exactly the same answer — across these cities the 15th sits within
    0.05 °C of the month mean, because a month's climatology is close to linear
    and the midpoint of a line is its average — but "almost exactly" by a
    coincidence of the calendar is not a thing to publish. More importantly the
    *label* on a single day is wrong: within November the expectation falls about
    4.9 °C from the 1st to the 30th in Madrid, so a reader who takes a date label
    literally and looks at the end of the month is some 2.5 °C out.

    Averaging the days removes both problems at once: the number is a monthly norm
    by construction, and the label can honestly say "November" because that is what
    was computed.

    The interval columns are averaged too. That makes `p10_c`/`p90_c` the mean of
    the daily 10th and 90th percentiles across the month, which is narrower than
    the percentiles of the pooled month — the latter would include the month's own
    seasonal drift as if it were uncertainty about a given day. The former is the
    right quantity for "a typical day in November"; it is not a claim about the
    spread of the monthly mean.

    Args:
        year: Target year, which the trend term is carried to.
        month: Calendar month, 1-12.
        slugs: Limit to these cities; defaults to the whole registry.

    Returns:
        The same shape as `city_estimates`, averaged over the month. The frame
        carries the slugs it could not load in ``frame.attrs["missing"]``, the
        month in ``frame.attrs["month"]`` and the days averaged in
        ``frame.attrs["n_days"]``.

    Raises:
        ValueError: If `month` is not in 1-12.
    """
    import calendar

    if not 1 <= month <= 12:
        raise ValueError(f"month must be 1-12, got {month}")

    n_days = calendar.monthrange(year, month)[1]
    daily = [city_estimates(date(year, month, d), slugs) for d in range(1, n_days + 1)]

    # Every day sees the same registry, so a city missing on one day is missing on
    # all of them; take the first frame's answer rather than reconciling twenty.
    averaged = sum(f[_MONTH_MEAN_COLUMNS] for f in daily) / len(daily)
    frame = daily[0].drop(columns=_MONTH_MEAN_COLUMNS).join(averaged)

    frame.attrs["missing"] = daily[0].attrs["missing"]
    frame.attrs["month"] = month
    frame.attrs["year"] = year
    frame.attrs["n_days"] = n_days
    # Kept so existing consumers that print a target still have one; it names the
    # month's midpoint, which is the representative day, not a claim about it.
    frame.attrs["target"] = date(year, month, 15).isoformat()
    return frame


# Averaged across the month; everything else is per-city metadata that does not
# vary by day and is carried through from the first day's frame.
_MONTH_MEAN_COLUMNS = [
    "expected_c",
    "p10_c",
    "p90_c",
    "rain_probability",
    "trend_adjustment_c",
]


def day_of_year_distribution(slug: str, variable: str = "tavg") -> pd.DataFrame:
    """Per-day-of-year summary statistics over the whole record.

    Args:
        slug: City slug.
        variable: Column to summarise.

    Returns:
        365 rows — one per calendar slot, with 29 February folded onto 1 March;
        see `SeasonalClimatology.table` for why that is 365 and not 366. Each row
        carries the detrended mean, standard deviation and percentiles, plus the
        wet-day fraction and the sample sizes behind them.
    """
    return SeasonalClimatology(variable=variable).fit(load.load_city(slug)).table()


# -- evaluation ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SeasonalScore:
    """Backtest metrics for the seasonal product.

    A separate dataclass from `wepr.evaluate.Score` rather than a reuse of it,
    because the two products are scored on different questions and forcing them
    into one shape would make the field names lie:

    * `coverage` has no analogue in the next-day score and is the metric that
      matters most here. The product's output is an interval; an interval whose
      realised coverage is 0.55 is not the 80% interval it is labelled as, however
      good its central MAE.
    * `skill_vs_climatology` means something different. For the next-day model the
      climatology baseline is a rival model; here it is *this same model with the
      trend term set to zero*, so the number isolates exactly one question — does
      the trend adjustment earn its place, or is it noise dressed as physics?
    * `skill_vs_last_year` replaces skill against persistence. "Tomorrow equals
      today" is undefined at a seasonal horizon; the honest naive analogue is
      "this date next year will be like this date last year", and that is a real
      competitor, because it carries the seasonal cycle for free.
    * `brier_skill_vs_base_rate` scores the rain probability against the
       city's unconditional wet-day rate *in that fold's training window* rather
       than against a day-of-year climatology. Against a day-of-year climatology
       the skill would be ~0 by construction, since this model's rain probability
       *is* one; against the base rate it asks whether the seasonal cycle in
       rainfall frequency is real.
    """

    model: str
    target: str
    mae: float
    rmse: float
    coverage: float
    nominal_coverage: float
    skill_vs_climatology: float
    skill_vs_last_year: float
    n_folds: int
    n_observations: int
    brier: float | None = None
    brier_skill_vs_base_rate: float | None = None


def _skill(model_error: float, baseline_error: float) -> float:
    if not np.isfinite(baseline_error) or baseline_error == 0.0:
        return float("nan")
    return float(1.0 - model_error / baseline_error)


def backtest(
    frame: pd.DataFrame,
    variable: str = "tavg",
    min_train_years: int = MIN_FIT_YEARS,
    window: int = WINDOW,
) -> tuple[pd.DataFrame, SeasonalScore]:
    """Walk forward a year at a time, refitting the climatology on prior years only.

    **Why the fold is a whole year and not a 90-day block.** The next-day backtest
    advances in `evaluate.DEFAULT_STEP` days because its model's information decays
    in days. This model's information is the seasonal cycle, so a fold shorter than
    a year would score it on one season at a time and a fold boundary mid-summer
    would train on a July the test block then asks about. One year per fold, fitted
    strictly on earlier years, is the only split that keeps both halves whole.

    The trend is refitted inside each fold along with everything else. Fitting it
    once on the full record and reusing it would leak the test years' warming into
    every earlier fold — the subtlest leak available here, and the reason
    `SeasonalClimatology.fit` takes a frame rather than a slug.

    Args:
        frame: Tidy daily frame from `wepr.data.load`.
        variable: Column to model.
        min_train_years: Complete years required before the first fold is scored.
        window: Pooling half-width, passed through to the model.

    Returns:
        Per-date predictions over every held-out year, and the pooled score.

    Raises:
        ValueError: if the record cannot supply a training window and a test year.
    """
    index = pd.DatetimeIndex(frame.index)
    annual = trends.annual_means(frame, variable)
    usable_years = list(annual.index)

    if len(usable_years) <= min_train_years:
        raise ValueError(
            f"{len(usable_years)} complete year(s) of {variable}, which is not more "
            f"than the {min_train_years}-year training window — nothing could be "
            "tested. Fetch more history or lower min_train_years."
        )

    year = np.asarray(index.year)
    observed = frame[variable]
    wet_threshold_hits = (frame["prcp"] >= WET_DAY_MM).astype("float64")

    blocks: list[pd.DataFrame] = []
    for fold, test_year in enumerate(usable_years[min_train_years:]):
        train = frame.loc[index.year < test_year]
        try:
            model = SeasonalClimatology(window=window, variable=variable).fit(train)
        except ValueError:
            # A fold whose training window has too few *complete* years is skipped
            # rather than scored on a degenerate fit. It happens at the start of a
            # record whose first year is partial.
            continue

        test_index = index[year == test_year]
        predicted = model.predict_frame(test_index)
        block = predicted.copy()
        block["actual"] = observed.loc[test_index].to_numpy()
        block["wet_actual"] = wet_threshold_hits.loc[test_index].to_numpy()
        # The naive seasonal competitor: the same calendar date one year earlier.
        # Taken from the observed series, which at this point in the walk is
        # entirely inside the training window.
        block["last_year"] = observed.reindex(test_index - pd.DateOffset(years=1)).to_numpy()
        # The Brier reference, taken from the training window rather than from the
        # whole record. Pooling it over every year would let the test years' own
        # wet-day frequency into the baseline — a small leak, and one that flatters
        # the *baseline* rather than the model, but the rule does not have an
        # exemption for leaks that happen to be conservative.
        block["base_rate"] = float(wet_threshold_hits.loc[train.index].mean())
        block["fold"] = fold
        block["test_year"] = test_year
        blocks.append(block)

    if not blocks:
        raise ValueError("no fold had enough complete training years to fit")

    predictions = pd.concat(blocks)
    scored = predictions.dropna(subset=["actual", "expected", "no_trend", "last_year"])

    actual = scored["actual"].to_numpy(dtype="float64")
    mae = float(np.mean(np.abs(actual - scored["expected"].to_numpy(dtype="float64"))))
    rmse = float(np.sqrt(np.mean((actual - scored["expected"].to_numpy(dtype="float64")) ** 2)))
    inside = (actual >= scored["p_low"].to_numpy()) & (actual <= scored["p_high"].to_numpy())

    brier = brier_skill = None
    wet = scored.dropna(subset=["rain_probability", "wet_actual"])
    if not wet.empty:
        wet_actual = wet["wet_actual"].to_numpy(dtype="float64")
        brier = float(np.mean((wet["rain_probability"].to_numpy(dtype="float64") - wet_actual) ** 2))
        base_rate = wet["base_rate"].to_numpy(dtype="float64")
        brier_skill = _skill(brier, float(np.mean((base_rate - wet_actual) ** 2)))

    score = SeasonalScore(
        model="SeasonalClimatology",
        target=variable,
        mae=mae,
        rmse=rmse,
        coverage=float(inside.mean()),
        nominal_coverage=NOMINAL_COVERAGE,
        skill_vs_climatology=_skill(
            mae, float(np.mean(np.abs(actual - scored["no_trend"].to_numpy(dtype="float64"))))
        ),
        skill_vs_last_year=_skill(
            mae, float(np.mean(np.abs(actual - scored["last_year"].to_numpy(dtype="float64"))))
        ),
        n_folds=int(scored["fold"].nunique()),
        n_observations=len(scored),
        brier=brier,
        brier_skill_vs_base_rate=brier_skill,
    )
    return predictions, score

"""The serving layer: fitted models held in memory, predictions read off them.

Phase 6. This module is what makes the app serve numbers instead of `503`, and
it is deliberately the only place where a trained model and an HTTP request meet.
`app/` imports `ModelStore` and calls it; it never loads a model, builds a feature
or computes a statistic itself.

**Everything happens once, at startup.** `ModelStore.load()` reads every artifact
in `models/`, reads `reports/metrics.json`, and builds each city's inference
feature row. A request then does a dictionary lookup and a `predict` call on an
object already in memory. The rule is not about speed — the next-day pipelines are
microseconds — it is that a web request must never trigger analysis, because
analysis that happens per request is analysis nobody has evaluated.

**A model with no metrics entry is not served.** This is where the project's
central rule stops being a convention and becomes a refusal. `registry.model_key`
names both the file in `models/` and the entry in `reports/metrics.json`; if the
two disagree, the store records the model as unserveable and the endpoint returns
`503` naming `scripts/evaluate_all.py`. The original project served Prophet while
reporting Ridge and RandomForest accuracy. Here that configuration cannot start.
The check goes one step further than key presence and compares the loaded object's
class against the `deployed_estimator` the evaluation recorded, so swapping the
estimator without re-evaluating is caught too.

**The forecast is one day past the archive edge, and says so.** There is no
"tomorrow" available here. ERA5 is a reanalysis with roughly `ARCHIVE_LATENCY_DAYS`
of latency, so even a complete fetch leaves the last observation several days
behind the present, and the honest description of the next-day product is "one day
ahead of the last observation in the record". Every result therefore carries
`as_of` (the observations it conditions on), `valid_for` (the day it describes) and
`lag_days` (how far `valid_for` sits behind today), and the UI is required to show
them. On the current fetch, which ends in 1971, `valid_for` is a date in 1972: the
model is making a genuine one-day-ahead prediction from information strictly prior
to it, which is exactly what it was evaluated on, but it is not news about the
weather and must never be presented as such.

**Baselines travel with the prediction.** Every forecast carries what persistence
and the day-of-year climatology would have said for the same date. The project's
rule is that a model is always reported against both baselines; shipping those
numbers in the response rather than only in `reports/metrics.json` means the page
cannot quietly show the model alone.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import features, paths
from .cities import CITIES, City, get
from .data import load
from .models import registry
from .models.nextday import WET_DAY_MM
from .models.seasonal import SeasonalEstimate

# Roughly how far behind the present the ERA5 archive runs. Open-Meteo serves
# `era5_seamless` to about five days before today; a little slack is allowed so a
# fetch run at the weekend is not reported as stale. Used only to classify a
# forecast as near-real-time or historical — nothing is computed from it.
ARCHIVE_LATENCY_DAYS = 10

# The furthest year a seasonal estimate will be produced for. The estimate's
# trend term is linear in the target year, so nothing stops the arithmetic from
# running to 2500 and returning a number with a straight face. 2100 is the horizon
# the AR6 scenarios quoted in `wepr.trends` are stated to, so it is the furthest
# out this project can put a published envelope next to its own extrapolation —
# past it, the estimate would be the only number on the page and the least
# defensible one. Requests beyond it are refused rather than answered.
MAX_SEASONAL_YEAR = 2100

# The product/target pairs the store serves, matching `scripts/train_all.py`.
NEXTDAY_TARGETS: tuple[str, ...] = ("tavg", "prcp")
SEASONAL_TARGET = "tavg"


class NotServeable(RuntimeError):
    """A prediction was asked for that this store cannot honestly produce.

    Carries the remediation command, because every reason a model is missing is
    fixed by running something: `scripts/fetch_data.py`, `scripts/train_all.py`
    or `scripts/evaluate_all.py`. The API turns this into a `503` whose body is
    the message.
    """

    def __init__(self, message: str, detail: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


# -- what the store reports about itself ---------------------------------------


@dataclass(frozen=True, slots=True)
class Accuracy:
    """One model's measured accuracy, read from `reports/metrics.json`.

    Not recomputed here. These numbers come from the walk-forward backtest in
    `scripts/evaluate_all.py` and are looked up by the same key that names the
    artifact on disk, so the figure shown beside a prediction is the figure
    measured for the model that produced it.

    Fields absent for a product are `None` rather than zero: the seasonal product
    has `coverage` and no `skill_vs_persistence`, the next-day product the
    reverse, and a zero in either slot would read as a measurement.

    Attributes:
        mae: Mean absolute error over the held-out folds, in the target's units.
        rmse: Root mean squared error over the same folds.
        n_folds: Walk-forward folds behind the figures.
        n_observations: Held-out days scored.
        provisional: True when the metrics rest on a record that fails the
            data-quality gate. On the current fetch this is true everywhere.
        estimator: Class name of the estimator that was evaluated.
        skill_vs_climatology: 1 - MAE/MAE_climatology. For the seasonal product
            the "climatology" baseline is this same model with its trend term
            zeroed, which is a different question — see `SeasonalScore`.
        skill_vs_persistence: 1 - MAE/MAE_persistence. Next-day only.
        skill_vs_last_year: Seasonal only; the naive seasonal competitor.
        coverage: Seasonal only; realised coverage of the reported interval.
        nominal_coverage: Seasonal only; what that interval is labelled as.
        brier: Brier score of the rain probability, where one is produced.
        beats_both_baselines: Next-day only. False is published, not hidden.
    """

    mae: float
    rmse: float
    n_folds: int
    n_observations: int
    provisional: bool
    estimator: str
    skill_vs_climatology: float | None = None
    skill_vs_persistence: float | None = None
    skill_vs_last_year: float | None = None
    coverage: float | None = None
    nominal_coverage: float | None = None
    brier: float | None = None
    beats_both_baselines: bool | None = None

    @classmethod
    def from_metrics(cls, entry: dict) -> Accuracy:
        """Build from one `reports/metrics.json` model entry."""
        return cls(
            mae=float(entry["mae"]),
            rmse=float(entry["rmse"]),
            n_folds=int(entry["n_folds"]),
            n_observations=int(entry["n_observations"]),
            provisional=bool(entry.get("provisional", False)),
            estimator=str(entry.get("estimator", "?")),
            skill_vs_climatology=_optional_float(entry.get("skill_vs_climatology")),
            skill_vs_persistence=_optional_float(entry.get("skill_vs_persistence")),
            skill_vs_last_year=_optional_float(entry.get("skill_vs_last_year")),
            coverage=_optional_float(entry.get("coverage")),
            nominal_coverage=_optional_float(entry.get("nominal_coverage")),
            brier=_optional_float(entry.get("brier")),
            beats_both_baselines=entry.get("beats_both_baselines"),
        )


def _optional_float(value) -> float | None:
    """Coerce to float, mapping `None` and NaN alike to `None`.

    A NaN reaching JSON serialises as the literal `NaN`, which is not valid JSON
    and which `json.loads` in a browser rejects. A skill score is legitimately NaN
    when its baseline error is zero, so this case is real rather than defensive.
    """
    if value is None:
        return None
    number = float(value)
    return None if not np.isfinite(number) else number


@dataclass(frozen=True, slots=True)
class Baselines:
    """What the two required baselines say for the same date.

    Carried on the forecast itself so a page cannot show the model's number
    without them. `persistence_c` is literally the last observed mean temperature;
    `climatology_c` is the seasonal model's trend-adjusted slot mean for
    `valid_for`, which is the same climatology the backtest scored against.

    Attributes:
        persistence_c: "Tomorrow equals today", in °C. The one to beat.
        climatology_c: Day-of-year norm for `valid_for`, in °C, or `None` if the
            city has no seasonal model.
    """

    persistence_c: float
    climatology_c: float | None


@dataclass(frozen=True, slots=True)
class RainForecast:
    """The two-stage precipitation prediction, reported as two statistics.

    Never multiplied together. `probability` is the headline because it is the
    part that is well estimated; `expected_mm_if_wet` is an expectation
    *conditional on rain* and is meaningless without the probability beside it.
    Multiplying them and printing the product as millimetres is the original
    project's error and is not available from this shape.

    Attributes:
        probability: P(precipitation >= `wet_threshold_mm`) on `valid_for`.
        expected_mm_if_wet: E[mm | wet], Duan-corrected.
        wet_threshold_mm: The depth that counts as a wet day.
    """

    probability: float
    expected_mm_if_wet: float
    wet_threshold_mm: float = WET_DAY_MM


@dataclass(frozen=True, slots=True)
class NextDayForecast:
    """A one-day-ahead prediction, with everything needed to read it honestly.

    Attributes:
        slug: City slug.
        name: Display name.
        as_of: Date of the last observation the prediction conditions on. This is
            the archive edge, not today.
        valid_for: The date described — always `as_of` plus one day.
        lag_days: Days between `valid_for` and today. Zero or negative means the
            forecast is about the future.
        is_near_real_time: Whether `lag_days` is within `ARCHIVE_LATENCY_DAYS`.
            False means the archive is far enough behind that this is a historical
            one-day-ahead prediction rather than a weather forecast, which is the
            case on an incomplete fetch.
        observed: The observations at `as_of`, so a reader can see what the
            prediction was conditioned on.
        tavg_c: Predicted daily mean temperature for `valid_for`, °C.
        rain: The precipitation prediction, as two separate statistics.
        baselines: What persistence and climatology say for the same date.
        tavg_accuracy: Measured accuracy of the deployed temperature model.
        rain_accuracy: Measured accuracy of the deployed precipitation model.
        provisional: True if either model's metrics are provisional.
    """

    slug: str
    name: str
    as_of: dt.date
    valid_for: dt.date
    lag_days: int
    is_near_real_time: bool
    observed: dict[str, float]
    tavg_c: float
    rain: RainForecast
    baselines: Baselines
    tavg_accuracy: Accuracy | None
    rain_accuracy: Accuracy | None
    provisional: bool

    def to_json(self) -> dict:
        """The API response shape. Dates as ISO strings, nothing NaN."""
        return {
            "slug": self.slug,
            "name": self.name,
            "as_of": self.as_of.isoformat(),
            "valid_for": self.valid_for.isoformat(),
            "lag_days": self.lag_days,
            "is_near_real_time": self.is_near_real_time,
            "product": "next-day forecast",
            "observed": self.observed,
            "temperature": {
                "expected_c": self.tavg_c,
                "units": "°C",
                "accuracy": asdict(self.tavg_accuracy) if self.tavg_accuracy else None,
            },
            "rain": {
                "probability": self.rain.probability,
                "expected_mm_if_wet": self.rain.expected_mm_if_wet,
                "wet_threshold_mm": self.rain.wet_threshold_mm,
                "accuracy": asdict(self.rain_accuracy) if self.rain_accuracy else None,
            },
            "baselines": asdict(self.baselines),
            "provisional": self.provisional,
        }


@dataclass(frozen=True, slots=True)
class TrendTerm:
    """The trend half of a seasonal estimate, broken out so it can be judged.

    **Why this is a separate object with a warning on it.** The seasonal
    expectation is a slot mean plus `slope x (target_year - anchor_year)`, and
    nothing in that arithmetic knows how far it is being carried. On the 22-year
    1950–1971 fetch this was built against, the anchor was 1960 and a 2027 target
    was 66 years out, so a slope of -0.3 °C/decade — whose 95% confidence interval
    comfortably spanned zero — moved Seville's estimate by nearly 2 °C. That was
    the largest single term on the page and it was measured to be noise: the
    backtest's `skill_vs_climatology` scores this model against itself with the
    trend zeroed, and the answer was negative for every city.

    On the longer record fetched since (1978–2025, anchor 2002, 26 years out
    against 48 of record) the same test passes for every city and `is_sound` is
    true throughout. The flag is kept, not retired: what changed is the record,
    not the arithmetic, and a differently-windowed refetch would trip it again.

    So the store reports the term, its confidence interval, how far it is being
    extrapolated, and the estimate without it. The model is not changed here —
    it is the Phase 4 product and `reports/metrics.json` is keyed to it — but a
    reader is given everything needed to see what the trend contributes and to
    discount it. `is_sound` is the flag the UI keys its warning off.

    Attributes:
        adjustment_c: The term's contribution to the expectation, in °C.
        slope_per_decade: The fitted Theil-Sen slope.
        ci_low: Lower bound of its 95% confidence interval.
        ci_high: Upper bound of the same.
        anchor_year: The year the slot mean is centred on.
        extrapolation_years: `target_year - anchor_year`. The distance the slope
            is being carried, which is what turns a small rate into a large term.
        record_years: Years of record the slope was fitted to.
        is_climatological: Whether the record is long enough
            (`trends.MIN_TREND_YEARS`) for the slope to be a climate statement.
        ci_spans_zero: Whether the confidence interval includes no warming at all.
        is_sound: True only if the slope is climatological, its interval excludes
            zero, and it is not being carried further than the record is long.
            False means the UI must show `expected_without_trend_c` beside the
            headline rather than only the headline.
    """

    adjustment_c: float
    slope_per_decade: float
    ci_low: float
    ci_high: float
    anchor_year: float
    extrapolation_years: float
    record_years: int
    is_climatological: bool
    ci_spans_zero: bool
    is_sound: bool

    @classmethod
    def from_model(cls, model, target_year: int, adjustment_c: float) -> TrendTerm:
        """Read the term off a fitted `SeasonalClimatology`."""
        trend = model.trend_
        extrapolation = float(target_year) - float(model.anchor_year_)
        spans_zero = bool(trend.ci_low <= 0.0 <= trend.ci_high)
        return cls(
            adjustment_c=float(adjustment_c),
            slope_per_decade=float(trend.slope_per_decade),
            ci_low=float(trend.ci_low),
            ci_high=float(trend.ci_high),
            anchor_year=float(model.anchor_year_),
            extrapolation_years=extrapolation,
            record_years=int(trend.n_years),
            is_climatological=bool(trend.is_climatological),
            ci_spans_zero=spans_zero,
            # Carrying a slope further than the window it was fitted to is the
            # condition that turns an unremarkable rate into the dominant term,
            # so it disqualifies the estimate on its own.
            is_sound=bool(
                trend.is_climatological
                and not spans_zero
                and abs(extrapolation) <= trend.n_years
            ),
        )


@dataclass(frozen=True, slots=True)
class SeasonalOutlook:
    """A climatological expectation for an arbitrary date — not a forecast.

    The wrapper exists so the API response can carry the warning and the record
    length alongside the numbers. `SeasonalEstimate` is the model's output;
    everything else here is the context that stops it being misread.

    Attributes:
        slug: City slug.
        name: Display name.
        target: The date described.
        estimate: The model's output, with its interval.
        accuracy: Measured accuracy, including interval coverage.
        record: First and last year of the record behind it.
        trend: The trend term broken out, with the test of whether it is sound.
        provisional: Whether the metrics rest on a record below the gate.
    """

    slug: str
    name: str
    target: dt.date
    estimate: SeasonalEstimate
    accuracy: Accuracy | None
    record: tuple[int, int]
    trend: TrendTerm
    provisional: bool

    @property
    def expected_without_trend_c(self) -> float:
        """The slot mean alone — the estimate with the trend term removed.

        Shown beside the headline whenever `trend.is_sound` is false, so a reader
        can see how much of the number is climatology and how much is a slope
        extrapolated past what it can support.
        """
        return self.estimate.expected_c - self.estimate.trend_adjustment_c

    def to_json(self) -> dict:
        """The API response shape.

        `is_forecast` is `False` and present in every response on purpose. A
        consumer reading only this payload has to be told, in the payload, that
        the interval is a distribution of plausible days and not a prediction of
        one.
        """
        return {
            "slug": self.slug,
            "name": self.name,
            "target": self.target.isoformat(),
            "product": "seasonal estimate",
            "is_forecast": False,
            "method": (
                "day-of-year climatology from detrended observations, plus the "
                "fitted warming trend carried to the target year"
            ),
            "temperature": {
                "expected_c": self.estimate.expected_c,
                "p10_c": self.estimate.p10_c,
                "p90_c": self.estimate.p90_c,
                "interval": "10th-90th percentile of historical values",
                "trend_adjustment_c": self.estimate.trend_adjustment_c,
                "units": "°C",
            },
            "rain_probability": self.estimate.rain_probability,
            "n_years": self.estimate.n_years,
            "record": {"from": self.record[0], "to": self.record[1]},
            "trend": asdict(self.trend)
            | {"expected_without_trend_c": self.expected_without_trend_c},
            "accuracy": asdict(self.accuracy) if self.accuracy else None,
            "provisional": self.provisional,
        }


@dataclass
class CityState:
    """Everything the store holds for one city, after a successful load.

    `row` is a single row — the inference features at `as_of` — rather than the
    whole engineered frame. Twenty frames of 8,000 rows by 38 columns is only a
    few megabytes, but holding them would invite a later route to recompute
    something from them per request, and the point of this layer is that the only
    thing in memory is what a prediction needs.
    """

    city: City
    as_of: pd.Timestamp
    row: pd.DataFrame
    predictors: list[str]
    observed: dict[str, float]
    record: tuple[int, int]
    models: dict[str, object] = field(default_factory=dict)


# -- the store ------------------------------------------------------------------


class ModelStore:
    """Every fitted model the app serves, loaded once.

    Built by `load()` at application startup and then read-only. Nothing here
    fits, fetches or writes; a `ModelStore` that cannot answer a question raises
    `NotServeable` with the command that would fix it.

    **Construction never raises.** A missing artifact, a stale pickle, an
    unevaluated model and an empty `data/processed/` are all normal states of a
    fresh clone, and an app that refuses to boot because of one is less useful
    than an app that boots and explains. Every failure is recorded in `problems`
    and surfaced by `status()` and by the `503` bodies.
    """

    def __init__(self) -> None:
        self.cities: dict[str, CityState] = {}
        self.metrics: dict[str, dict] = {}
        self.metrics_generated_at: str | None = None
        self.metrics_gate: dict = {}
        self.problems: list[dict] = []
        self.loaded_at: dt.datetime | None = None

    # -- loading ---------------------------------------------------------------

    @classmethod
    def not_loaded(cls, reason: str) -> ModelStore:
        """An empty store that knows why it is empty.

        The factory builds one of these when `LOAD_MODELS` is false. Without it
        every endpoint would answer "no diagnosis recorded", which is the one
        unhelpful `503` this layer is supposed to make impossible: a refusal that
        does not name its cause is no better than a placeholder number.
        """
        store = cls()
        store._note("*", "*", reason, "create the app with LOAD_MODELS enabled")
        store.loaded_at = dt.datetime.now(dt.timezone.utc)
        return store

    @classmethod
    def load(cls, slugs: list[str] | None = None) -> ModelStore:
        """Load every available model and inference row.

        Args:
            slugs: Limit to these cities. Defaults to every city in the registry
                that has processed data on disk.

        Returns:
            The store. Check `store.ready` before trusting it to answer anything.
        """
        store = cls()
        store._read_metrics()

        wanted = (
            [get(s).slug for s in slugs]
            if slugs is not None
            else [c.slug for c in CITIES if load.processed_path(c.slug).exists()]
        )
        for slug in wanted:
            store._load_city(slug)
        store.loaded_at = dt.datetime.now(dt.timezone.utc)
        return store

    def _note(self, slug: str, key: str, reason: str, remedy: str) -> None:
        self.problems.append(
            {"slug": slug, "key": key, "reason": reason, "remedy": remedy}
        )

    def _read_metrics(self) -> None:
        """Read `reports/metrics.json`, or record why it could not be read.

        A missing or corrupt metrics file is not a crash. It means nothing can be
        served, because an unevaluated model is exactly what this project refuses
        to deploy, and the recorded problem names the script that writes it.
        """
        path = paths.metrics_json()
        if not path.exists():
            self._note(
                "*",
                "*",
                f"no metrics at {path}",
                ".venv/bin/python scripts/evaluate_all.py",
            )
            return
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            self._note(
                "*",
                "*",
                f"metrics at {path} is not valid JSON ({exc})",
                ".venv/bin/python scripts/evaluate_all.py",
            )
            return
        self.metrics = payload.get("models", {})
        self.metrics_generated_at = payload.get("generated_at")
        self.metrics_gate = payload.get("gate", {})

    def _load_city(self, slug: str) -> None:
        """Load one city's models and build its inference row."""
        city = get(slug)
        try:
            frame = load.load_city(slug)
        except (FileNotFoundError, load.DataContractError) as exc:
            self._note(slug, "*", str(exc), ".venv/bin/python scripts/fetch_data.py")
            return

        try:
            state = self._build_state(city, frame)
        except ValueError as exc:
            self._note(slug, "*", str(exc), ".venv/bin/python scripts/fetch_data.py")
            return

        for target in NEXTDAY_TARGETS:
            self._attach(state, "nextday", target)
        self._attach(state, "seasonal", SEASONAL_TARGET)

        if state.models:
            self.cities[slug] = state

    def _build_state(self, city: City, frame: pd.DataFrame) -> CityState:
        """Engineer the features and keep the last row that is complete.

        Features first, then the predictor list — in that order, always. Reversing
        it is the original project's bug, and it would be a quieter failure here
        than it was there: the pipelines carry `feature_names_in_`, so a short
        list raises, but a list in the wrong *order* would not.

        **`as_of` is the last row with no missing predictor, which is not always
        the last row of the record.** A 30-day rolling mean is NaN wherever the
        preceding month has a gap, so a hole near the archive edge legitimately
        pushes the forecast back a day or more. That is reported rather than
        patched: filling it would mean conditioning the forecast on a number
        nobody observed.
        """
        engineered = features.build_features(frame)
        predictors = features.feature_columns(engineered)
        complete = engineered.dropna(subset=predictors)
        if complete.empty:
            raise ValueError(
                f"{city.slug}: no row has a complete set of {len(predictors)} "
                "predictors — the record is shorter than the longest feature window"
            )

        as_of = complete.index[-1]
        observed = {
            variable: _optional_float(frame.loc[as_of, variable])
            for variable in load.COLUMNS
        }
        years = pd.DatetimeIndex(frame.index).year
        return CityState(
            city=city,
            as_of=as_of,
            row=complete.loc[[as_of], predictors],
            predictors=predictors,
            observed=observed,
            record=(int(years.min()), int(years.max())),
        )

    def _attach(self, state: CityState, product: str, target: str) -> None:
        """Load one model if it exists, was evaluated, and is the thing evaluated.

        Three gates, in order, each of which records a problem instead of raising:

        1. the artifact exists — otherwise `train_all.py`;
        2. `reports/metrics.json` has an entry under the same key — otherwise
           `evaluate_all.py`, because an unevaluated model is not deployed here;
        3. the loaded object's class matches the `deployed_estimator` recorded
           with that entry — otherwise the artifact was retrained with a different
           estimator and the published accuracy no longer describes it.

        The third gate is the one that would have caught the original project. A
        key check alone passes if someone swaps Ridge for Prophet under the same
        filename; a class check does not.
        """
        key = registry.model_key(state.city.slug, product, target)
        try:
            model = registry.load(key)
        except FileNotFoundError:
            self._note(
                state.city.slug,
                key,
                "not trained",
                f".venv/bin/python scripts/train_all.py --city {state.city.slug}",
            )
            return
        except registry.ModelVersionError as exc:
            self._note(
                state.city.slug,
                key,
                str(exc),
                ".venv/bin/python scripts/train_all.py",
            )
            return

        entry = self.metrics.get(key)
        if entry is None:
            self._note(
                state.city.slug,
                key,
                "trained but never evaluated — no entry in reports/metrics.json, "
                "so there is no measured accuracy to publish with its predictions",
                ".venv/bin/python scripts/evaluate_all.py",
            )
            return

        evaluated = entry.get("deployed_estimator") or entry.get("estimator")
        deployed = type(model).__name__
        if evaluated and evaluated != deployed:
            self._note(
                state.city.slug,
                key,
                f"the artifact on disk is a {deployed}, but the published metrics "
                f"were measured on a {evaluated}. The evaluated model must be the "
                "deployed model.",
                ".venv/bin/python scripts/train_all.py && "
                ".venv/bin/python scripts/evaluate_all.py",
            )
            return

        state.models[f"{product}.{target}"] = model

    # -- what the store can do -------------------------------------------------

    @property
    def ready(self) -> bool:
        """Whether any city can be served at all."""
        return bool(self.cities)

    def served_slugs(self) -> list[str]:
        """Cities with a complete next-day pair and a seasonal model, sorted.

        A city with only some of its models is held back from this list but still
        answerable individually, so a partial training run degrades per-product
        rather than per-city.
        """
        needed = {f"nextday.{t}" for t in NEXTDAY_TARGETS} | {
            f"seasonal.{SEASONAL_TARGET}"
        }
        return sorted(s for s, c in self.cities.items() if needed <= set(c.models))

    def n_models_loaded(self) -> int:
        """How many fitted models are held in memory."""
        return sum(len(c.models) for c in self.cities.values())

    def accuracy(self, slug: str, product: str, target: str) -> Accuracy | None:
        """The measured accuracy of one deployed model, or None if unserved."""
        entry = self.metrics.get(registry.model_key(slug, product, target))
        return Accuracy.from_metrics(entry) if entry else None

    def _state(self, slug: str) -> CityState:
        get(slug)  # raises KeyError for an unknown slug, before any disk talk
        state = self.cities.get(slug)
        if state is None:
            raise NotServeable(
                f"no models are loaded for {slug!r}",
                self._remedy_for(slug),
            )
        return state

    def _model(self, slug: str, product: str, target: str):
        state = self._state(slug)
        model = state.models.get(f"{product}.{target}")
        if model is None:
            key = registry.model_key(slug, product, target)
            raise NotServeable(
                f"{key} is not being served", self._remedy_for(slug, key)
            )
        return model

    def _remedy_for(self, slug: str, key: str | None = None) -> str:
        """The recorded reason and fix for a slug or key, joined for a 503 body."""
        matches = [
            p
            for p in self.problems
            if p["slug"] in (slug, "*") and (key is None or p["key"] in (key, "*"))
        ]
        if not matches:
            return "no diagnosis recorded; check models/ and reports/metrics.json"
        return "; ".join(f"{p['reason']} — fix: {p['remedy']}" for p in matches)

    # -- the two products ------------------------------------------------------

    def forecast(self, slug: str, today: dt.date | None = None) -> NextDayForecast:
        """The one-day-ahead prediction for one city.

        Args:
            slug: City slug.
            today: Override for the current date, so the staleness classification
                is testable without freezing the clock.

        Returns:
            The forecast, with its baselines and the deployed models' measured
            accuracy.

        Raises:
            KeyError: unknown slug.
            NotServeable: the city's next-day models are not loaded.
        """
        state = self._state(slug)
        temperature = self._model(slug, "nextday", "tavg")
        rain_model = self._model(slug, "nextday", "prcp")

        # The pipelines were fitted on a DataFrame, so they carry
        # `feature_names_in_` and will raise if this frame's columns differ in
        # name or order. That check is the reason `row` is kept as a one-row
        # DataFrame rather than flattened to an array, where a reordering would
        # pass silently and return numbers computed from the wrong predictors.
        X = state.row
        tavg_c = float(np.asarray(temperature.predict(X)).ravel()[0])
        rain = rain_model.predict_rain(X)[0]

        valid_for = (state.as_of + pd.Timedelta(days=1)).date()
        reference = today or dt.date.today()
        lag_days = (reference - valid_for).days

        climatology = None
        seasonal_model = state.models.get(f"seasonal.{SEASONAL_TARGET}")
        if seasonal_model is not None:
            climatology = float(seasonal_model.predict_one(valid_for).expected_c)

        tavg_accuracy = self.accuracy(slug, "nextday", "tavg")
        rain_accuracy = self.accuracy(slug, "nextday", "prcp")

        return NextDayForecast(
            slug=slug,
            name=state.city.name,
            as_of=state.as_of.date(),
            valid_for=valid_for,
            lag_days=lag_days,
            is_near_real_time=lag_days <= ARCHIVE_LATENCY_DAYS,
            observed=state.observed,
            tavg_c=tavg_c,
            rain=RainForecast(
                probability=float(rain.probability),
                expected_mm_if_wet=float(rain.expected_mm_if_wet),
            ),
            baselines=Baselines(
                # Persistence is "tomorrow equals today" — literally the last
                # observed mean, not a model output. At a one-day horizon this is
                # a genuinely strong competitor, which is why it is shown rather
                # than only cited.
                persistence_c=float(state.observed["tavg"]),
                climatology_c=climatology,
            ),
            tavg_accuracy=tavg_accuracy,
            rain_accuracy=rain_accuracy,
            provisional=any(
                a.provisional for a in (tavg_accuracy, rain_accuracy) if a
            ),
        )

    def seasonal(self, slug: str, target: dt.date) -> SeasonalOutlook:
        """The climatological expectation for one city on an arbitrary date.

        Args:
            slug: City slug.
            target: Any date, past or future. Distance does not degrade this
                estimate the way it would a forecast, because the estimate was
                never conditioned on recent observations in the first place.

        Returns:
            The estimate with its interval and the record behind it.

        Raises:
            KeyError: unknown slug.
            ValueError: `target` is beyond `MAX_SEASONAL_YEAR`.
            NotServeable: the city's seasonal model is not loaded.
        """
        if target.year > MAX_SEASONAL_YEAR:
            raise ValueError(
                f"{target.isoformat()} is beyond {MAX_SEASONAL_YEAR}. The trend "
                "term is linear in the target year, so this model will return a "
                "number for any date; past 2100 there is no published scenario "
                "envelope to show it against, and an unaccompanied extrapolation "
                "that far out is not a defensible estimate."
            )
        model = self._model(slug, "seasonal", SEASONAL_TARGET)
        state = self._state(slug)
        accuracy = self.accuracy(slug, "seasonal", SEASONAL_TARGET)
        estimate = model.predict_one(target)
        return SeasonalOutlook(
            slug=slug,
            name=state.city.name,
            target=target,
            estimate=estimate,
            accuracy=accuracy,
            record=state.record,
            trend=TrendTerm.from_model(
                model, target.year, estimate.trend_adjustment_c
            ),
            provisional=bool(accuracy and accuracy.provisional),
        )

    # -- introspection ---------------------------------------------------------

    def status(self) -> dict:
        """What is loaded, what is not, and why — the body of `/api/status`.

        Exposed as an endpoint rather than printed at startup because the thing a
        reader most needs to know about this app is which of its claims are
        currently backed by a measured model, and a log line scrolls away.
        """
        as_of = sorted({c.as_of.date().isoformat() for c in self.cities.values()})
        return {
            "ready": self.ready,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "cities_served": self.served_slugs(),
            "n_models": self.n_models_loaded(),
            "archive_edge": {
                "as_of": as_of[-1] if as_of else None,
                "latency_allowance_days": ARCHIVE_LATENCY_DAYS,
            },
            "metrics": {
                "generated_at": self.metrics_generated_at,
                "n_entries": len(self.metrics),
                "gate_enforced": self.metrics_gate.get("enforced"),
                "note": self.metrics_gate.get("note"),
                # How many measured figures are below the data-quality gate, so a
                # page can decide whether to caveat them. Counted rather than
                # inferred from the wording of `note`: the note is prose written by
                # the evaluation script and sniffing it for "PROVISIONAL" would put
                # a presentation decision at the mercy of a sentence someone
                # reworded. On a complete record this is 0 and the caveat does not
                # render; on a truncated one it is the number of figures affected.
                "n_provisional": sum(
                    1 for entry in self.metrics.values() if entry.get("provisional")
                ),
            },
            "problems": self.problems,
        }

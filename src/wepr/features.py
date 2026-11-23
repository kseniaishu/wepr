"""Feature engineering, and the guard that keeps it honest.

**The hard rule.** A feature is legal only if its value is knowable strictly
before the target timestamp. Rolling windows, expanding means and shifts all have
to be checked against that, because the failure is silent: a leaked feature does
not raise, it just produces an accuracy figure that cannot be reproduced in
deployment.

`assert_causal` below turns that rule into an executable test, so it is enforced
rather than merely intended.

**The framing, stated once.** Row `t` holds day `t`'s observations; the target at
row `t` is day `t+1`'s value. So a predictor at row `t` is legal if and only if it
uses data from days up to and including `t` — today's observations included, since
the forecast is made at the end of today. That is why nothing here applies a
defensive `shift(1)` before a rolling window: a window *ending today* is already
legal, and shifting it would throw away the single most informative observation.
A window ending tomorrow would not be, and `assert_causal` is what proves the
difference.

Targets are built separately by `add_targets`, not by `build_features`. They are
the one thing that is *supposed* to look forward, so mixing them into the predictor
builder would make the causality check meaningless — it would flag the target as a
leak, and the obvious fix would be to stop checking.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

TARGETS: tuple[str, ...] = ("tavg_target", "prcp_target")

# The source variables every feature is derived from.
BASE_VARIABLES: tuple[str, ...] = ("tavg", "tmax", "tmin", "prcp")

# How far back each family of features reaches. Kept as constants rather than
# inlined so the feature set is visible in one place and can be changed in one.
LAGS: tuple[int, ...] = (1,)
ROLL_WINDOWS: tuple[int, ...] = (3, 7, 30)
ANOMALY_WINDOW: int = 7
WET_WINDOWS: tuple[int, ...] = (7, 30)

# A day counts as wet at or above this depth, matching `wepr.data.validate`.
WET_DAY_MM: float = 1.0

# Minimum prior occurrences of a calendar day before its climatology is reported.
# Below this the mean is too noisy to be worth a number, and a NaN that gets dropped
# is more honest than an average of two years pretending to be a norm.
MIN_CLIMATOLOGY_YEARS: int = 5


def _calendar_key(index: pd.DatetimeIndex) -> pd.Index:
    """Group key identifying a calendar day, as `MMDD`.

    Not `day_of_year`, which drifts by one across leap years: 1 March is day 60 in
    1951 and day 61 in 1952, so a day-of-year climatology silently averages 29
    February with 1 March. `MMDD` pins each feature to an actual calendar date.
    """
    return pd.Index(index.month * 100 + index.day, name="mmdd")


def _prior_year_climatology(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Mean of this calendar day's value over *strictly earlier* years.

    The statistic is the expanding mean inside each calendar-day group, which is
    the genuinely leak-free way to express "the historical norm for this date": at
    row `t` it averages only the occurrences that precede it. `shift(1)` within the
    group excludes the current year's own value, so the feature is a norm the
    current observation can be compared against rather than one it is already part
    of.

    **Written out rather than expressed as `transform(lambda ...)`.** The readable
    one-liner is

        grouped.transform(lambda s: s.shift(1).expanding(min_periods=N).mean())

    and it calls back into python once per calendar day per variable — 366 groups
    x 4 variables x 20 cities. Profiling the serving layer's startup put 95% of it
    here, 0.44 s per city, which is 9 s to boot the app and 9 s again on every
    reload during development. The form below is the same statistic computed with
    three vectorised group operations.

    The two forms are asserted equal in `tests/test_features.py`, on real city
    records rather than on a toy frame, because an optimisation to a feature
    definition is an optimisation to what every model was fitted on: a divergence
    here would silently invalidate `reports/metrics.json` rather than fail.

    The subtlety the equivalence turns on is the NaN handling. `expanding().mean()`
    averages the non-missing observations seen so far and counts only those toward
    `min_periods`, so it returns a value even at a row whose own shifted input is
    NaN. A plain `cumsum()` propagates NaN into that position instead, which is why
    the sum below is taken over zero-filled values and the divisor is a separate
    cumulative count of the non-missing ones.
    """
    key = _calendar_key(index)
    # Prior years only: the shift happens inside the calendar-day group, so it
    # steps back one *year*, not one day.
    shifted = series.groupby(key, sort=False).shift(1)
    total = shifted.fillna(0.0).groupby(key, sort=False).cumsum()
    count = shifted.notna().astype("int64").groupby(key, sort=False).cumsum()
    mean = total / count
    return mean.where(count >= MIN_CLIMATOLOGY_YEARS)


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Build the predictor matrix for the next-day models.

    Every feature is strictly backward-looking — see the module docstring for what
    that means given this framing — and `assert_causal` is the executable proof.

    The families, and why each is here:

    * **Today's observations**, unchanged. Available at forecast time and, at a
      one-day horizon, the strongest single predictor there is. Persistence is the
      baseline to beat precisely because of this.
    * **Lags** (`LAGS`). Yesterday's value, giving the model a direction of change.
    * **Rolling means** (`ROLL_WINDOWS`), each window *ending today*. Recent
      regime rather than a single day's noise.
    * **Anomalies** (`ANOMALY_WINDOW`): today minus its own rolling mean. This
      replaces the *ratio* the original code used, for two reasons. °C has an
      arbitrary zero, so a ratio of temperatures is not a meaningful quantity, and
      it explodes whenever the denominator passes through 0 °C — on this record the
      7-day mean comes within 0.05 °C of zero, giving ratios beyond ±30. For
      precipitation the denominator is *exactly* zero on thousands of days,
      yielding `inf`, which `fillna(0)` does not catch. A difference is stable and
      reads directly as "warmer, or wetter, than the recent norm".
    * **Prior-year climatology**. The norm for this calendar date from earlier
      years only, so the model can work with a departure from normal rather than
      having to learn the seasonal cycle from scratch.
    * **Diurnal range** (`tmax - tmin`), a serviceable proxy for cloud cover.
    * **Wet-day counts** (`WET_WINDOWS`). Precipitation is zero-inflated, so how
      *often* it rained recently carries information a mean of mostly-zeros blurs.
    * **Cyclical calendar encodings**, `sin`/`cos` of the day of year. The only
      features drawn from the calendar rather than the data, and legal for that
      reason — the calendar is knowable arbitrarily far ahead. They are cyclical
      because a raw day-of-year has a discontinuity at the year boundary, where
      31 December and 1 January are a year apart numerically and a day apart in
      fact.

    Args:
        raw: Tidy daily frame from `wepr.data.load`.

    Returns:
        Frame with the engineered feature columns added. Early rows carry NaN where
        a window or a climatology has insufficient history; they are left in place
        for the caller to drop deliberately, since silently trimming rows is how
        the original code manufactured an apparent improvement.

    Raises:
        KeyError: if `raw` lacks a base variable.
    """
    missing = [v for v in BASE_VARIABLES if v not in raw.columns]
    if missing:
        raise KeyError(f"raw frame lacks {missing}; expected {list(BASE_VARIABLES)}")

    out = raw.copy()
    index = out.index
    new: dict[str, pd.Series] = {}

    for var in BASE_VARIABLES:
        series = out[var]

        for lag in LAGS:
            new[f"{var}_lag{lag}"] = series.shift(lag)

        for window in ROLL_WINDOWS:
            # min_periods == window: a partial window is a different statistic, and
            # letting it through would make the first rows quietly incomparable to
            # the rest.
            new[f"{var}_roll{window}"] = series.rolling(window, min_periods=window).mean()

        new[f"{var}_anom{ANOMALY_WINDOW}"] = (
            series - series.rolling(ANOMALY_WINDOW, min_periods=ANOMALY_WINDOW).mean()
        )

        new[f"{var}_clim"] = _prior_year_climatology(series, index)

    new["trange"] = out["tmax"] - out["tmin"]
    new["trange_roll7"] = new["trange"].rolling(7, min_periods=7).mean()

    wet = (out["prcp"] >= WET_DAY_MM).astype("float64")
    for window in WET_WINDOWS:
        new[f"prcp_wet{window}"] = wet.rolling(window, min_periods=window).sum()

    # 365.25 rather than 365: over a 76-year record an integer period accumulates
    # about three weeks of phase error against the actual seasons.
    angle = 2 * np.pi * index.day_of_year.to_numpy() / 365.25
    new["doy_sin"] = pd.Series(np.sin(angle), index=index)
    new["doy_cos"] = pd.Series(np.cos(angle), index=index)

    return pd.concat([out, pd.DataFrame(new, index=index)], axis=1)


def add_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Add tomorrow's temperature and precipitation as supervised targets.

    Kept out of `build_features` deliberately: a target looks one day forward by
    definition, so building it alongside the predictors would make `assert_causal`
    report it as a leak and invite someone to weaken the check.

    The final row's targets are NaN, because tomorrow has not happened. That NaN is
    left in place. The original code called `ffill()` after `shift(-1)`, which
    copied the last real observation into the final row and invented a label —
    training on a fabricated target, then reporting the resulting accuracy.

    Args:
        frame: Any frame carrying `tavg` and `prcp`.

    Returns:
        A copy with `tavg_target` and `prcp_target` added.
    """
    out = frame.copy()
    out["tavg_target"] = out["tavg"].shift(-1)
    out["prcp_target"] = out["prcp"].shift(-1)
    return out


def labelled(frame: pd.DataFrame, predictors: Sequence[str] | None = None) -> pd.DataFrame:
    """Rows usable for supervised fitting: complete predictors and a real target.

    Dropping incomplete rows is unavoidable — a 30-day window has no value on day
    one — but it has to be done *deliberately and once*, with the predictor list
    computed after feature engineering. The original code's reported "improvement"
    from adding rolling features was entirely an artifact of the 14-row trim those
    features caused, because the predictor list had been captured beforehand and
    the new columns were never actually fed to the model.

    Args:
        frame: An engineered frame that has been through `add_targets`.
        predictors: Columns required to be complete. Defaults to
            `feature_columns(frame)` — computed here, after engineering, which is
            the whole point.

    Returns:
        The subset of rows with no missing predictor and no missing target.
    """
    columns = list(predictors) if predictors is not None else feature_columns(frame)
    targets = [t for t in TARGETS if t in frame.columns]
    return frame.dropna(subset=columns + targets)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """The predictor columns of an engineered frame.

    Always call this **after** `build_features`, never before. Capturing the
    predictor list before engineering features was the central bug in the original
    code: the rolling columns were created but never fed to the model, and the
    reported "improvement" came from a 14-row trim rather than from the features.

    Args:
        frame: An engineered frame.

    Returns:
        Column names to use as predictors, excluding targets and metadata.
    """
    excluded = set(TARGETS) | {"name", "station", "city", "slug"}
    return [c for c in frame.columns if c not in excluded]


def assert_causal(
    build: Callable[[pd.DataFrame], pd.DataFrame],
    raw: pd.DataFrame,
    cut_points: Sequence[int] | None = None,
    atol: float = 1e-10,
) -> None:
    """Fail if any feature built by `build` depends on data after its own row.

    The test rests on one observation: if a feature is strictly backward-looking,
    then deleting the future cannot change the past. So build the features on the
    full series, build them again on a truncated prefix, and compare the rows the
    two runs share. Any column that disagrees has seen the future.

    This catches the mistakes that are easy to make and hard to spot — a
    `rolling(...)` without `.shift(1)`, a `center=True` window, a target created
    with `shift(-1)` then forward-filled, a scaler fitted on the whole series, a
    `groupby().transform("mean")` where an `expanding().mean()` was meant.

    **Limitation, stated plainly.** The method truncates the *tail*, so it detects
    only leaks whose source lies after the cut point. A fill that reaches backwards
    across a gap entirely before every cut — `bfill()` on row 0, say — is real
    leakage this will not catch. It is a strong check, not a proof, so it is used
    alongside reading the feature definitions rather than instead of it.

    Args:
        build: The feature-building function under test.
        raw: A tidy daily frame to test against.
        cut_points: Row counts to truncate at. Defaults to three cuts spread
            across the middle of the series.
        atol: Absolute tolerance for float comparison.

    Raises:
        AssertionError: naming every column that leaked, and the first date at
            which it differed.
    """
    full = build(raw)

    n = len(raw)
    if cut_points is None:
        cut_points = [int(n * f) for f in (0.5, 0.7, 0.9)]

    leaks: dict[str, pd.Timestamp] = {}

    for cut in cut_points:
        if not 0 < cut < n:
            raise ValueError(f"cut point {cut} outside 1..{n - 1}")

        truncated = build(raw.iloc[:cut])
        shared_index = full.index.intersection(truncated.index)
        shared_cols = full.columns.intersection(truncated.columns)

        for col in shared_cols:
            a = full.loc[shared_index, col]
            b = truncated.loc[shared_index, col]

            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                # NaN in the same position on both sides is agreement, not a leak.
                differs = ~(
                    ((a - b).abs() <= atol) | (a.isna() & b.isna())
                )
            else:
                differs = (a != b) & ~(a.isna() & b.isna())

            if differs.any():
                first = differs.idxmax()
                # Keep the earliest divergence seen across all cut points.
                if col not in leaks or first < leaks[col]:
                    leaks[col] = first

    if leaks:
        lines = "\n".join(
            f"  - {col}: first differs at {date.date()}"
            for col, date in sorted(leaks.items())
        )
        raise AssertionError(
            "future information detected in "
            f"{len(leaks)} feature column(s) — these change when later data is "
            f"removed, so they cannot be computed at prediction time:\n{lines}"
        )

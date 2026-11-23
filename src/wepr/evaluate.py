"""Walk-forward backtesting and skill scores.

The evaluation shape is inherited from the original project and was the right
instinct: an expanding training window advanced in fixed steps, so the model is
only ever fitted on data preceding the fold it is scored on.

Two rules this module exists to enforce:

- **Every model is reported against baselines** — day-of-year climatology and
  persistence (tomorrow = today). A model that beats neither is reported as such,
  not quietly dropped. Persistence is a genuinely strong baseline at one-day
  range; beating it is the whole task.
- **The evaluated model is the deployed model.** Metrics are written to
  `reports/metrics.json` keyed by the same identifier the app loads from
  `models/`, so publishing a number for a model the app does not serve becomes a
  key mismatch rather than an unnoticed lie.

Implemented in Phase 3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone

from . import features
from .models import baselines

# Fold geometry. 10 years of initial training, then 90-day test blocks.
DEFAULT_START = 3650
DEFAULT_STEP = 90


@dataclass(frozen=True, slots=True)
class Score:
    """Metrics for one model on one target.

    `skill_vs_*` is the fractional improvement in MAE over that baseline:
    `1 - mae_model / mae_baseline`. Positive means better, zero means no better
    than the baseline, negative means worse. Reporting skill rather than raw MAE
    is what makes a number interpretable — "MAE 1.8 °C" means nothing on its own.

    `brier` and `brier_skill_vs_climatology` are populated only for a model with
    an occurrence stage, and are the metrics that actually matter there: the app's
    headline precipitation output is a probability, so scoring only the expected
    depth would leave the published number unmeasured.
    """

    model: str
    target: str
    mae: float
    rmse: float
    skill_vs_climatology: float
    skill_vs_persistence: float
    n_folds: int
    n_observations: int
    brier: float | None = None
    brier_skill_vs_climatology: float | None = None
    # Per-fold spread against persistence, the binding baseline. Pooled skill can
    # be comfortably positive while individual folds lose, and "beats both
    # baselines" as a bare boolean hides that. These are what turn a claim of
    # skill into a defensible one.
    folds_beating_climatology: int | None = None
    folds_beating_persistence: int | None = None
    skill_vs_persistence_fold_sd: float | None = None


def base_variable(target: str) -> str:
    """The observed column a target is built from: `tavg_target` -> `tavg`."""
    if not target.endswith("_target"):
        raise ValueError(
            f"target {target!r} does not end in '_target'; baselines cannot be "
            "aligned without knowing which observed column it came from"
        )
    return target[: -len("_target")]


def _assert_contiguous_daily(index: pd.DatetimeIndex) -> None:
    """Fail if the index has a gap, which would silently corrupt every shift.

    Baseline alignment below is a `shift(-1)`, which means "the next row" and not
    "the next day". On a contiguous daily index those coincide; across a hole they
    do not, and persistence would quietly be compared against a value two or more
    days away. `wepr.data.load` reindexes onto a complete daily range precisely so
    this holds, so a failure here means the frame did not come through it.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("backtest needs a DatetimeIndex")
    if len(index) < 2:
        return
    gaps = index.to_series().diff().iloc[1:]
    if not (gaps == pd.Timedelta(days=1)).all():
        first = gaps[gaps != pd.Timedelta(days=1)].index[0]
        raise ValueError(
            f"index is not contiguous daily — first gap before {first.date()}; "
            "load the frame through wepr.data.load, which reindexes onto a "
            "complete daily range"
        )


def baseline_predictions(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """Climatology and persistence, aligned to `target` rather than to the index.

    **The alignment, because this is where an off-by-one would silently inflate
    every skill score.** Row `t` holds day `t`'s observations and `target[t]` is
    day `t+1`'s value. The baselines in `models.baselines` predict the value *at*
    their own index date, so the prediction of `target[t]` is the baseline's value
    at row `t+1` — one `shift(-1)`.

    Applied to persistence that reduces to today's observation, which is exactly
    what "tomorrow equals today" should mean. Applied to climatology it gives the
    norm for tomorrow's calendar date, computed from earlier years only, so both
    stay knowable at forecast time.
    """
    observed = frame[base_variable(target)]
    # Persistence is written as the observation itself rather than as
    # `baselines.persistence(observed).shift(-1)`. The two are the same quantity —
    # the shifts cancel — but composing them would strand a NaN on the final row,
    # where `shift(-1)` has nothing to pull from. The equivalence is asserted in
    # `tests/test_evaluate.py` so this stays a stated identity, not an assumption.
    return pd.DataFrame(
        {
            "climatology": baselines.climatology(observed).shift(-1),
            "persistence": observed,
        },
        index=frame.index,
    )


def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def _skill(model_mae: float, baseline_mae: float) -> float:
    """Fractional MAE improvement over a baseline, or NaN if the baseline is perfect."""
    if not np.isfinite(baseline_mae) or baseline_mae == 0.0:
        return float("nan")
    return float(1.0 - model_mae / baseline_mae)


def backtest(
    frame: pd.DataFrame,
    model,
    predictors: list[str],
    target: str,
    start: int = DEFAULT_START,
    step: int = DEFAULT_STEP,
) -> tuple[pd.DataFrame, Score]:
    """Walk forward through the series, refitting on each expanding window.

    The model is cloned and refitted per fold; any scaler or imputer must be
    fitted inside the fold, which means passing a `Pipeline` rather than a bare
    estimator.

    Metrics are pooled over all predictions rather than averaged over folds. The
    original code averaged per-fold MAEs, which weights a short final fold the
    same as a full one.

    Args:
        frame: Engineered frame.
        model: Any scikit-learn estimator or pipeline.
        predictors: Predictor columns, from `features.feature_columns`.
        target: Target column name.
        start: Rows in the first training window.
        step: Test block length.

    Returns:
        Per-date predictions, and the pooled score.
    """
    if step < 1:
        raise ValueError(f"step must be at least 1, got {step}")
    if target not in frame.columns:
        raise KeyError(f"frame lacks {target!r}; call features.add_targets first")

    _assert_contiguous_daily(pd.DatetimeIndex(frame.index))

    # Baselines are derived on the full contiguous frame, before any rows are
    # dropped, because the alignment shift means "the next row".
    reference = baseline_predictions(frame, target)
    observed = frame[base_variable(target)]

    # Both the model and the baselines are scored on exactly the same rows. If the
    # row sets differed, `1 - mae_model / mae_baseline` would be a ratio between
    # two different problems and the skill number would be meaningless.
    usable = features.labelled(frame, predictors).dropna(subset=[target])
    usable = usable.loc[reference.loc[usable.index].notna().all(axis=1)]

    if len(usable) <= start:
        raise ValueError(
            f"only {len(usable)} usable rows, which is not more than the "
            f"{start}-row initial training window — nothing could be tested. "
            "Fetch more history or lower `start`."
        )

    X = usable[predictors]
    y = usable[target]

    # The occurrence stage is scored probabilistically where the model has one,
    # against a day-of-year wet-day frequency from prior years. Expected depth
    # alone would leave the app's headline precipitation number unmeasured.
    probabilistic = hasattr(model, "predict_proba_wet")
    if probabilistic:
        threshold = getattr(model, "wet_threshold_mm", 0.1)
        # `.where(notna)` matters: `NaN >= 0.1` is False, so without it every
        # missing day would enter the day-of-year wet-day frequency as an observed
        # dry day and bias the Brier reference — and therefore the reported skill —
        # upward. ERA5 is gap-free today, but a partial fetch is this project's
        # normal state, so the guard is not hypothetical.
        wet_observed = (observed >= threshold).astype("float64").where(observed.notna())
        wet_reference = baselines.climatology(wet_observed).shift(-1)
        wet_actual = (y >= threshold).astype("float64")

    frames: list[pd.DataFrame] = []
    n_folds = 0

    for split in range(start, len(usable), step):
        test = slice(split, min(split + step, len(usable)))
        X_test = X.iloc[test]
        if X_test.empty:
            continue

        fitted = clone(model).fit(X.iloc[:split], y.iloc[:split])
        block = pd.DataFrame(
            {
                "actual": y.iloc[test],
                "predicted": np.asarray(fitted.predict(X_test), dtype="float64"),
                "fold": n_folds,
            },
            index=X_test.index,
        )
        if probabilistic:
            block["probability"] = np.asarray(
                fitted.predict_proba_wet(X_test), dtype="float64"
            )
            block["wet_actual"] = wet_actual.iloc[test]
        frames.append(block)
        n_folds += 1

    predictions = pd.concat(frames)
    predictions = predictions.join(reference.loc[predictions.index])

    actual = predictions["actual"].to_numpy(dtype="float64")
    model_mae = _mae(actual, predictions["predicted"].to_numpy(dtype="float64"))

    brier = brier_skill = None
    if probabilistic:
        wet = predictions["wet_actual"].to_numpy(dtype="float64")
        brier = float(np.mean((predictions["probability"].to_numpy() - wet) ** 2))
        climatological = wet_reference.loc[predictions.index].to_numpy(dtype="float64")
        reference_brier = float(np.mean((climatological - wet) ** 2))
        brier_skill = _skill(brier, reference_brier)

    # Per-fold skill, computed on each fold's own rows.
    by_fold = predictions.groupby("fold")
    fold_skill_persistence = by_fold.apply(
        lambda g: _skill(
            _mae(g["actual"].to_numpy(), g["predicted"].to_numpy()),
            _mae(g["actual"].to_numpy(), g["persistence"].to_numpy()),
        ),
        include_groups=False,
    )
    fold_skill_climatology = by_fold.apply(
        lambda g: _skill(
            _mae(g["actual"].to_numpy(), g["predicted"].to_numpy()),
            _mae(g["actual"].to_numpy(), g["climatology"].to_numpy()),
        ),
        include_groups=False,
    )

    score = Score(
        model=type(model).__name__,
        target=target,
        mae=model_mae,
        rmse=_rmse(actual, predictions["predicted"].to_numpy(dtype="float64")),
        skill_vs_climatology=_skill(
            model_mae, _mae(actual, predictions["climatology"].to_numpy(dtype="float64"))
        ),
        skill_vs_persistence=_skill(
            model_mae, _mae(actual, predictions["persistence"].to_numpy(dtype="float64"))
        ),
        n_folds=n_folds,
        n_observations=len(predictions),
        brier=brier,
        brier_skill_vs_climatology=brier_skill,
        folds_beating_climatology=int((fold_skill_climatology > 0).sum()),
        folds_beating_persistence=int((fold_skill_persistence > 0).sum()),
        skill_vs_persistence_fold_sd=float(fold_skill_persistence.std()),
    )
    return predictions, score

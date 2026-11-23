"""The next-day forecast — the only genuine forecast in the project.

Predicts tomorrow from observations available today. Supervised learning is
appropriate here precisely because the conditioning information exists.

**Temperature.** Ridge regression on the lagged and rolling features, inside a
`Pipeline` so the scaler is fitted per fold and never on the full series.

**Precipitation.** A two-stage model, carried over from the original project where
it was the best idea in the code: a classifier for *whether* it rains, and a
regressor for *how much*, fitted on wet days only. Daily rainfall is
zero-inflated — most days are dry — and a single regressor on that target
predicts a small positive number every day, which is wrong on both the dry days
and the wet ones.

Two corrections to the original:

- It multiplied probability by amount and reported the product as a millimetre
  figure. That product is neither a probability nor an expected depth for a wet
  day, and reads as false precision. The app surfaces **rain probability** as the
  headline, with the conditional expected amount as a secondary figure.
- Precipitation cannot be negative. Predictions are clamped at zero.

Implemented in Phase 3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .. import features

# A day counts as wet above this threshold, in mm. 0.1 mm is the conventional
# cutoff; below it, reanalysis "drizzle" is numerical noise rather than rain.
WET_DAY_MM = 0.1

# Ridge penalty grid, searched inside the training fold. Chosen by generalised
# cross-validation on the training rows only, so the choice never sees the test
# block and the fold stays clean.
ALPHAS: tuple[float, ...] = tuple(np.logspace(-2, 3, 11))

# Below this many wet days the conditional amount stage has nothing to learn, and
# its mean is reported instead of a fitted line. Dry-climate cities such as Murcia
# make this a real case rather than a defensive branch.
MIN_WET_DAYS = 30


@dataclass(frozen=True, slots=True)
class RainPrediction:
    """A precipitation prediction, reported honestly.

    Attributes:
        probability: P(rain tomorrow), in [0, 1].
        expected_mm_if_wet: Expected depth conditional on it raining at all.
            Meaningless on its own — only read it alongside the probability.
    """

    probability: float
    expected_mm_if_wet: float


def temperature_pipeline() -> Pipeline:
    """An unfitted next-day temperature pipeline.

    Returned unfitted so the backtest can clone and refit it per fold, and so the
    estimator that gets evaluated is constructed by the same function that builds
    the one shipped to `models/` — the mismatch this project exists to avoid.

    `StandardScaler` is a pipeline step rather than a preprocessing call because
    fitting it on the full series before splitting leaks the test block's mean and
    variance into training. `RidgeCV` picks its penalty by generalised
    cross-validation on whatever rows it is handed, which inside a fold is the
    training rows alone.
    """
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", RidgeCV(alphas=list(ALPHAS))),
        ]
    )


class TwoStageRain(RegressorMixin, BaseEstimator):
    """Occurrence classifier x conditional amount, for a zero-inflated target.

    The shape is the original project's best idea. Most days are dry, so a single
    regressor on daily millimetres minimises its loss by predicting a small
    positive number every day — which is wrong on the dry days by that amount and
    wrong on the wet days by almost everything. Splitting the question into "will
    it rain" and "given that it rains, how much" gives two well-posed problems.

    **Two depth statistics, each used where it is correct.** `predict` returns
    `P(wet) x median[mm | wet]` for the backtest, because MAE is minimised by the
    conditional median. `predict_rain` reports `P(wet)` alongside the Duan-corrected
    conditional *mean*, because "expected mm if it rains" is a mean and a
    log-scale fit inverted with `expm1` alone is not one — it runs about a third
    low. Conflating the two is how a model ends up either penalised for being
    unbiased or displaying a figure that quietly isn't the expectation it claims.

    `predict` is for scoring, never for display, and `predict_rain` is what the app
    calls: the original code multiplied the two stages together and printed the
    product as millimetres, which is the one thing this class must not be used to
    do.
    """

    def __init__(
        self,
        wet_threshold_mm: float = WET_DAY_MM,
        alphas: tuple[float, ...] = ALPHAS,
        min_wet_days: int = MIN_WET_DAYS,
    ) -> None:
        self.wet_threshold_mm = wet_threshold_mm
        self.alphas = alphas
        self.min_wet_days = min_wet_days

    @staticmethod
    def _rows(X, mask):
        """Boolean row selection that works for a DataFrame or an array."""
        return X[mask] if hasattr(X, "iloc") else np.asarray(X)[mask]

    def fit(self, X, y) -> TwoStageRain:
        # X is deliberately *not* coerced to an array. Keeping the DataFrame lets
        # the inner pipelines record `feature_names_in_`, so a frame whose columns
        # are reordered at inference raises instead of silently returning numbers
        # computed from the wrong predictors — Phase 4 builds that frame in a
        # separate code path, which makes this a live trap rather than a
        # theoretical one.
        y = np.asarray(y, dtype="float64")
        wet = y >= self.wet_threshold_mm

        # A fold with no variation in the label cannot support a classifier, and
        # scikit-learn raises rather than guessing. The honest fallback is the
        # observed base rate, which is 0 or 1 exactly when that is all the data
        # shows.
        self.base_rate_ = float(wet.mean()) if wet.size else 0.0
        if wet.all() or not wet.any():
            self.occurrence_ = None
        else:
            self.occurrence_ = Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("logit", LogisticRegression(max_iter=1000)),
                ]
            ).fit(X, wet)

        wet_amounts = y[wet]
        self.mean_wet_mm_ = float(wet_amounts.mean()) if wet_amounts.size else 0.0
        self.median_wet_mm_ = float(np.median(wet_amounts)) if wet_amounts.size else 0.0
        if wet_amounts.size >= self.min_wet_days:
            # log1p because wet-day depths are strongly right-skewed: a linear
            # model on raw mm is dominated by a handful of storm days, and its
            # predictions go negative on dry-looking ones. The inverse is applied
            # in `predict_amount_if_wet`.
            X_wet = self._rows(X, wet)
            self.amount_ = Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("ridge", RidgeCV(alphas=list(self.alphas))),
                ]
            ).fit(X_wet, np.log1p(wet_amounts))
            # Duan smearing. Back-transforming a log-scale fit with expm1 alone
            # returns roughly a conditional *median*, not a mean, because
            # exp(E[log y]) != E[y] — on this data it understates wet-day depth by
            # about 35%. The estimator is y = exp(Xb + e) - 1, so
            # E[y|X] = exp(Xb) * E[exp(e)] - 1, and E[exp(e)] is estimated
            # non-parametrically from the training residuals. Fitted inside the
            # fold like everything else, so it carries no test information.
            residuals = np.log1p(wet_amounts) - self.amount_.predict(X_wet)
            self.smearing_ = float(np.mean(np.exp(residuals)))
        else:
            self.amount_ = None
            self.smearing_ = 1.0

        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba_wet(self, X) -> np.ndarray:
        """P(rain), one value per row, in [0, 1]."""
        if self.occurrence_ is None:
            return np.full(X.shape[0], self.base_rate_, dtype="float64")
        return self.occurrence_.predict_proba(X)[:, 1]

    def predict_median_if_wet(self, X) -> np.ndarray:
        """Median mm given rain — the depth statistic MAE rewards.

        Inverting the log-scale fit with `expm1` alone returns approximately the
        conditional *median*, because `exp(E[log y]) != E[y]`. That is the right
        statistic to score under MAE, which is minimised by the median rather than
        the mean, so it is what `predict` feeds the backtest.
        """
        if self.amount_ is None:
            return np.full(X.shape[0], self.median_wet_mm_, dtype="float64")
        return np.clip(np.expm1(self.amount_.predict(X)), 0.0, None)

    def predict_mean_if_wet(self, X) -> np.ndarray:
        """E[mm | wet] — the conditional expectation, Duan-corrected.

        This is the figure the app shows, so it has to be an expectation in fact
        and not only in name. The uncorrected `expm1` form understates wet-day
        depth by roughly a third on this data.
        """
        if self.amount_ is None:
            return np.full(X.shape[0], self.mean_wet_mm_, dtype="float64")
        predicted = np.exp(self.amount_.predict(X)) * self.smearing_ - 1.0
        return np.clip(predicted, 0.0, None)

    def predict(self, X) -> np.ndarray:
        """`P(wet) x median[mm | wet]`, clamped at zero — for scoring, not display.

        Deliberately built from the median rather than the mean. The backtest
        compares this against climatology and persistence under MAE, and MAE is
        minimised by the conditional median; scoring a mean-based prediction would
        penalise the model for being unbiased. The mean is the right answer to a
        different question — "how much, if it rains" — and `predict_rain` is where
        that one is answered.
        """
        return np.clip(
            self.predict_proba_wet(X) * self.predict_median_if_wet(X), 0.0, None
        )

    def predict_rain(self, X) -> list[RainPrediction]:
        """The two stages reported separately, which is how the app must show them.

        Always a list, one entry per row, so the caller's handling does not change
        shape with the size of the input.
        """
        probabilities = self.predict_proba_wet(X)
        amounts = self.predict_mean_if_wet(X)
        return [
            RainPrediction(probability=float(p), expected_mm_if_wet=float(a))
            for p, a in zip(probabilities, amounts, strict=True)
        ]


def precipitation_model() -> TwoStageRain:
    """An unfitted two-stage precipitation model, for cloning in the backtest."""
    return TwoStageRain()


def _design(frame: pd.DataFrame, predictors: list[str], target: str):
    """Labelled rows as `(X, y)`, dropping incomplete predictors and targets once.

    `features.labelled` is what does the dropping, with the predictor list passed
    in explicitly — the list must have been computed by `feature_columns` *after*
    `build_features`, which is the bug the original code shipped.
    """
    if target not in frame.columns:
        raise KeyError(
            f"frame lacks {target!r}; call features.add_targets before fitting"
        )
    usable = features.labelled(frame, predictors).dropna(subset=[target])
    if usable.empty:
        raise ValueError(
            f"no rows with complete predictors and a {target!r} value — "
            "the frame is shorter than the longest feature window"
        )
    return usable[predictors], usable[target]


def fit_temperature(frame: pd.DataFrame, predictors: list[str]):
    """Fit the next-day temperature model.

    Args:
        frame: Engineered frame.
        predictors: Predictor columns, from `features.feature_columns`.

    Returns:
        A fitted `Pipeline`.
    """
    X, y = _design(frame, predictors, "tavg_target")
    return temperature_pipeline().fit(X, y)


def fit_precipitation(frame: pd.DataFrame, predictors: list[str]):
    """Fit the two-stage next-day precipitation model.

    Args:
        frame: Engineered frame.
        predictors: Predictor columns, from `features.feature_columns`.

    Returns:
        A fitted two-stage model exposing `predict_rain`.
    """
    X, y = _design(frame, predictors, "prcp_target")
    return precipitation_model().fit(X, y)

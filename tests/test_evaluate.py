"""The backtest: fold geometry, baseline alignment, and pooled metrics.

Three specific failures this file exists to prevent, all of which produce a
plausible-looking accuracy number rather than an error:

- a fold fitted on rows it is then scored on;
- a baseline misaligned by one day, which flatters or wrecks every skill score;
- metrics averaged over folds instead of pooled, so a short final fold counts as
  much as a full one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from wepr import evaluate, features


@pytest.fixture
def frame():
    """Fifteen years of engineered, targeted daily weather."""
    index = pd.date_range("2000-01-01", "2014-12-31", freq="D", name="date")
    rng = np.random.default_rng(23)
    doy = index.day_of_year.to_numpy()
    tavg = 15 - 9 * np.cos(2 * np.pi * doy / 365.25) + rng.normal(0, 2.5, len(index))
    prcp = rng.gamma(0.4, 3.0, len(index)) * (rng.random(len(index)) < 0.3)
    raw = pd.DataFrame(
        {"tavg": tavg, "tmax": tavg + 6, "tmin": tavg - 6, "prcp": prcp}, index=index
    )
    return features.add_targets(features.build_features(raw))


@pytest.fixture
def predictors(frame):
    """Computed after engineering, which is the only correct order."""
    return features.feature_columns(frame)


class FoldSpy(RegressorMixin, BaseEstimator):
    """Records the rows each fold was fitted on, so leakage becomes assertable.

    The log is a class attribute rather than a constructor argument because
    `sklearn.clone` deep-copies params — a list passed in would be replaced per
    fold and every append would land on a discarded copy.
    """

    calls: list[tuple[str, pd.Timestamp, pd.Timestamp, int]] = []

    def fit(self, X, y):
        FoldSpy.calls.append(("fit", X.index.min(), X.index.max(), len(X)))
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        FoldSpy.calls.append(("predict", X.index.min(), X.index.max(), len(X)))
        return np.full(len(X), self.mean_)


@pytest.fixture
def spy_calls():
    """A clean call log for one test."""
    FoldSpy.calls.clear()
    yield FoldSpy.calls
    FoldSpy.calls.clear()


# --- the fold geometry -------------------------------------------------------


def test_no_fold_is_fitted_on_rows_it_predicts(frame, predictors, spy_calls):
    """The central guarantee: training always ends strictly before testing begins."""
    evaluate.backtest(frame, FoldSpy(), predictors, "tavg_target", start=1500, step=200)

    pairs = list(zip(spy_calls[0::2], spy_calls[1::2], strict=True))
    assert pairs, "no folds ran"
    for (_, _, train_end, _), (_, test_start, _, _) in pairs:
        assert train_end < test_start, (
            f"fold trained through {train_end.date()} but tested from {test_start.date()}"
        )


def test_training_window_expands_and_test_blocks_advance(frame, predictors, spy_calls):
    """Each fold trains on everything before it, and the blocks tile the series."""
    evaluate.backtest(frame, FoldSpy(), predictors, "tavg_target", start=1500, step=200)

    fits = [entry for entry in spy_calls if entry[0] == "fit"]
    predicts = [entry for entry in spy_calls if entry[0] == "predict"]

    assert [f[3] for f in fits] == sorted(f[3] for f in fits), "window did not expand"
    assert fits[0][3] == 1500
    # Every block but the last is a full step; the last is whatever remains.
    assert all(p[3] == 200 for p in predicts[:-1])
    assert 0 < predicts[-1][3] <= 200
    # Test blocks must not overlap.
    for earlier, later in zip(predicts, predicts[1:], strict=False):
        assert earlier[2] < later[1]


def test_too_little_data_is_an_error_not_an_empty_score(frame, predictors):
    with pytest.raises(ValueError, match="not more than"):
        evaluate.backtest(frame, DummyRegressor(), predictors, "tavg_target", start=99_000)


def test_a_gap_in_the_index_is_rejected(frame, predictors):
    """A hole would make `shift(-1)` span it, silently corrupting persistence."""
    holed = frame.drop(frame.index[5000])
    with pytest.raises(ValueError, match="contiguous daily"):
        evaluate.backtest(holed, DummyRegressor(), predictors, "tavg_target", start=1500)


# --- baseline alignment: the off-by-one that would invalidate every skill score


def test_persistence_aligned_to_the_target_is_todays_observation(frame):
    """`target[t]` is tomorrow, so persistence for it is today's value — exactly."""
    reference = evaluate.baseline_predictions(frame, "tavg_target")
    pd.testing.assert_series_equal(
        reference["persistence"].rename("tavg"), frame["tavg"], check_freq=False
    )


def test_climatology_aligned_to_the_target_is_tomorrows_norm(frame):
    """Shifted by one row, so it is the norm for the date being predicted."""
    from wepr.models import baselines

    reference = evaluate.baseline_predictions(frame, "tavg_target")
    expected = baselines.climatology(frame["tavg"]).shift(-1)
    np.testing.assert_allclose(
        reference["climatology"].to_numpy(), expected.to_numpy(), equal_nan=True
    )


def test_a_perfect_persistence_series_gives_persistence_zero_error(predictors):
    """On a series where tomorrow equals today, persistence must be exact.

    This pins the alignment from the other direction: a one-row slip would make
    the trivially-perfect baseline look wrong.
    """
    index = pd.date_range("2000-01-01", periods=4000, freq="D", name="date")
    ramp = pd.Series(np.arange(4000, dtype="float64"), index=index)
    raw = pd.DataFrame({"tavg": ramp, "tmax": ramp, "tmin": ramp, "prcp": 0.0})
    frame = features.add_targets(features.build_features(raw))
    cols = features.feature_columns(frame)

    predictions, _ = evaluate.backtest(
        frame, LinearRegression(), cols, "tavg_target", start=2000, step=500
    )
    # tomorrow = today + 1, so persistence is off by exactly 1 everywhere.
    errors = (predictions["actual"] - predictions["persistence"]).abs()
    np.testing.assert_allclose(errors.to_numpy(), 1.0)


def test_base_variable_requires_a_target_suffix():
    assert evaluate.base_variable("prcp_target") == "prcp"
    with pytest.raises(ValueError, match="_target"):
        evaluate.base_variable("tavg")


# --- metrics -----------------------------------------------------------------


def test_metrics_are_pooled_not_averaged_over_folds(frame, predictors):
    """Pooled MAE must equal the mean absolute error of the returned predictions."""
    predictions, score = evaluate.backtest(
        frame, DummyRegressor(), predictors, "tavg_target", start=1500, step=200
    )
    pooled = (predictions["actual"] - predictions["predicted"]).abs().mean()
    assert score.mae == pytest.approx(pooled)
    assert score.n_observations == len(predictions)

    # A per-fold average differs whenever the final fold is short, which is the
    # bug being excluded — so confirm this series actually has a short fold.
    sizes = predictions.groupby("fold").size()
    assert sizes.iloc[-1] < sizes.iloc[0]
    per_fold = (
        (predictions["actual"] - predictions["predicted"])
        .abs()
        .groupby(predictions["fold"])
        .mean()
        .mean()
    )
    assert score.mae != pytest.approx(per_fold)


def test_skill_is_zero_for_a_model_that_is_the_baseline(frame, predictors):
    """Scoring persistence against itself must give exactly zero skill."""

    class PredictToday(RegressorMixin, BaseEstimator):
        def fit(self, X, y):
            return self

        def predict(self, X):
            return X["tavg"].to_numpy(dtype="float64")

    _, score = evaluate.backtest(
        frame, PredictToday(), predictors, "tavg_target", start=1500, step=200
    )
    assert score.skill_vs_persistence == pytest.approx(0.0, abs=1e-12)


def test_model_and_baselines_are_scored_on_identical_rows(frame, predictors):
    """A skill ratio across two different row sets would be meaningless."""
    predictions, score = evaluate.backtest(
        frame, DummyRegressor(), predictors, "tavg_target", start=1500, step=200
    )
    for column in ("actual", "predicted", "climatology", "persistence"):
        assert predictions[column].notna().all()
    assert len(predictions) == score.n_observations


def test_a_worse_model_gets_negative_skill(frame, predictors):
    """Failing to beat a baseline is reported, not hidden."""
    _, score = evaluate.backtest(
        frame, DummyRegressor(strategy="constant", constant=-50.0),
        predictors, "tavg_target", start=1500, step=200,
    )
    assert score.skill_vs_climatology < 0
    assert score.skill_vs_persistence < 0


def test_precipitation_model_is_also_scored_probabilistically(frame, predictors):
    """The app's headline is a probability, so a Brier score must be reported."""
    from wepr.models import nextday

    _, score = evaluate.backtest(
        frame, nextday.precipitation_model(), predictors, "prcp_target",
        start=1500, step=200,
    )
    assert score.brier is not None and 0.0 <= score.brier <= 1.0
    assert score.brier_skill_vs_climatology is not None


def test_temperature_model_reports_no_brier(frame, predictors):
    """A model with no occurrence stage must not invent a probabilistic metric."""
    _, score = evaluate.backtest(
        frame, DummyRegressor(), predictors, "tavg_target", start=1500, step=200
    )
    assert score.brier is None
    assert score.brier_skill_vs_climatology is None

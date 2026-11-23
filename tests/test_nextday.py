"""The next-day models, and the two corrections to the original project.

The original multiplied rain probability by amount and printed the product as a
millimetre figure, and let a regressor return negative rainfall. Both are
structural here rather than matters of care: the clamp is in `predict`, and the
two stages are only reachable separately through `predict_rain`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from wepr import features
from wepr.models import nextday


@pytest.fixture
def frame():
    """Fifteen years of engineered, targeted daily weather.

    Wet and dry days follow a two-state Markov chain rather than independent
    draws. That matters: real rainfall is strongly autocorrelated — wet spells
    cluster — which is both why persistence is a baseline worth beating and the
    only reason an occurrence classifier has anything to learn. On i.i.d. rain the
    honest result is no skill, which makes it the wrong fixture for testing that
    the model finds signal when signal exists.
    """
    index = pd.date_range("2000-01-01", "2014-12-31", freq="D", name="date")
    rng = np.random.default_rng(31)
    doy = index.day_of_year.to_numpy()
    tavg = 15 - 9 * np.cos(2 * np.pi * doy / 365.25) + rng.normal(0, 2.5, len(index))

    # P(wet | wet yesterday) = 0.65, P(wet | dry yesterday) = 0.12, giving a
    # stationary wet fraction near 0.25 and spells long enough to be learnable.
    wet = np.zeros(len(index), dtype=bool)
    draws = rng.random(len(index))
    for i in range(1, len(index)):
        wet[i] = draws[i] < (0.65 if wet[i - 1] else 0.12)
    # Shape 2.0 rather than a near-zero shape so almost every wet day clears the
    # 0.1 mm cutoff; otherwise a third of the wet days label as dry and the
    # occurrence target is mostly threshold noise.
    prcp = rng.gamma(2.0, 2.0, len(index)) * wet

    raw = pd.DataFrame(
        {"tavg": tavg, "tmax": tavg + 6, "tmin": tavg - 6, "prcp": prcp}, index=index
    )
    return features.add_targets(features.build_features(raw))


@pytest.fixture
def predictors(frame):
    return features.feature_columns(frame)


@pytest.fixture
def design(frame, predictors):
    usable = features.labelled(frame, predictors)
    return usable[predictors], usable["prcp_target"]


# --- temperature -------------------------------------------------------------


def test_temperature_model_is_a_pipeline_with_the_scaler_inside(frame, predictors):
    """A bare estimator would force the scaler to be fitted before splitting.

    The hard rule is that scalers are fitted inside the training fold. Returning a
    `Pipeline` is what makes that automatic under `clone`-and-refit.
    """
    model = nextday.temperature_pipeline()
    assert isinstance(model, Pipeline)
    assert isinstance(model.named_steps["scale"], StandardScaler)


def test_temperature_model_fits_and_predicts_plausibly(frame, predictors):
    fitted = nextday.fit_temperature(frame, predictors)
    usable = features.labelled(frame, predictors)
    predicted = fitted.predict(usable[predictors])
    assert predicted.shape == (len(usable),)
    # In-sample, so this is a smoke test rather than an accuracy claim.
    assert np.abs(predicted - usable["tavg_target"]).mean() < 3.0


def test_fitting_without_targets_is_a_clear_error(frame, predictors):
    bare = frame.drop(columns=["tavg_target", "prcp_target"])
    with pytest.raises(KeyError, match="add_targets"):
        nextday.fit_temperature(bare, predictors)


def test_fitting_too_short_a_series_is_a_clear_error(predictors):
    index = pd.date_range("2000-01-01", periods=20, freq="D", name="date")
    short = features.add_targets(
        features.build_features(
            pd.DataFrame(
                {"tavg": 10.0, "tmax": 16.0, "tmin": 4.0, "prcp": 0.0}, index=index
            )
        )
    )
    with pytest.raises(ValueError, match="no rows with complete predictors"):
        nextday.fit_temperature(short, features.feature_columns(short))


# --- precipitation: the hard rule -------------------------------------------


def test_predicted_precipitation_is_never_negative(design):
    """Precipitation cannot be negative. This is a rule, not a preference."""
    X, y = design
    fitted = nextday.precipitation_model().fit(X, y)
    assert (fitted.predict(X) >= 0).all()
    assert (fitted.predict_mean_if_wet(X) >= 0).all()
    assert (fitted.predict_median_if_wet(X) >= 0).all()


def test_probability_stays_in_the_unit_interval(design):
    X, y = design
    fitted = nextday.precipitation_model().fit(X, y)
    probability = fitted.predict_proba_wet(X)
    assert ((probability >= 0) & (probability <= 1)).all()


def test_predict_rain_reports_the_two_stages_separately(design):
    """The app's contract: a probability and a conditional depth, never a product."""
    X, y = design
    fitted = nextday.precipitation_model().fit(X, y)
    predictions = fitted.predict_rain(X.head(5))

    assert len(predictions) == 5
    assert all(isinstance(p, nextday.RainPrediction) for p in predictions)
    for p in predictions:
        assert 0.0 <= p.probability <= 1.0
        assert p.expected_mm_if_wet >= 0.0


def test_predict_rain_always_returns_a_list(design):
    """Including for a single row, so the caller's handling never changes shape."""
    X, y = design
    fitted = nextday.precipitation_model().fit(X, y)
    assert len(fitted.predict_rain(X.head(1))) == 1


def test_predict_is_built_from_the_median_because_mae_rewards_medians(design):
    """`predict` is the scoring quantity: P(wet) x median, not P(wet) x mean."""
    X, y = design
    fitted = nextday.precipitation_model().fit(X, y)
    np.testing.assert_allclose(
        fitted.predict(X),
        fitted.predict_proba_wet(X) * fitted.predict_median_if_wet(X),
        rtol=1e-9,
    )


def test_the_displayed_amount_is_a_mean_not_a_median(design):
    """Right-skewed depths put the mean above the median; `predict_rain` shows the mean.

    Inverting a log-scale fit with `expm1` alone yields roughly a median. Reporting
    that as "expected mm if wet" would understate depth by about a third, so the
    Duan smearing factor is applied — and must actually move the number.
    """
    X, y = design
    fitted = nextday.precipitation_model().fit(X, y)
    assert fitted.smearing_ > 1.0
    assert (fitted.predict_mean_if_wet(X) > fitted.predict_median_if_wet(X)).all()
    # predict_rain must expose the mean, not the median.
    reported = np.array([p.expected_mm_if_wet for p in fitted.predict_rain(X.head(20))])
    np.testing.assert_allclose(reported, fitted.predict_mean_if_wet(X.head(20)))


def test_the_duan_correction_makes_the_expected_depth_roughly_unbiased(design):
    """The whole point of the correction: E[mm|wet] must match observed wet-day mean."""
    X, y = design
    fitted = nextday.precipitation_model().fit(X, y)
    wet = y >= nextday.WET_DAY_MM
    observed = y[wet].mean()
    corrected = fitted.predict_mean_if_wet(X[wet.to_numpy()]).mean()
    uncorrected = fitted.predict_median_if_wet(X[wet.to_numpy()]).mean()
    assert abs(corrected - observed) < abs(uncorrected - observed)
    assert corrected == pytest.approx(observed, rel=0.25)


def test_occurrence_stage_carries_real_signal(design):
    """P(rain) must be higher on days that were actually wet, or the stage is noise."""
    X, y = design
    fitted = nextday.precipitation_model().fit(X, y)
    probability = pd.Series(fitted.predict_proba_wet(X), index=X.index)
    wet = y >= nextday.WET_DAY_MM
    assert probability[wet].mean() > probability[~wet].mean() + 0.1


# --- degenerate folds, which a dry-climate city makes real --------------------


def test_an_all_dry_training_fold_does_not_crash(design):
    """Murcia-like folds exist; scikit-learn raises on a single-class label."""
    X, _ = design
    fitted = nextday.precipitation_model().fit(X, np.zeros(len(X)))
    assert fitted.occurrence_ is None
    np.testing.assert_allclose(fitted.predict_proba_wet(X), 0.0)
    np.testing.assert_allclose(fitted.predict(X), 0.0)


def test_an_all_wet_training_fold_does_not_crash(design):
    X, _ = design
    fitted = nextday.precipitation_model().fit(X, np.full(len(X), 5.0))
    assert fitted.occurrence_ is None
    np.testing.assert_allclose(fitted.predict_proba_wet(X), 1.0)
    np.testing.assert_allclose(fitted.predict_mean_if_wet(X), 5.0)


def test_too_few_wet_days_falls_back_to_their_mean(design):
    """Below `MIN_WET_DAYS` the amount stage reports a mean, not a fitted line."""
    X, _ = design
    sparse = np.zeros(len(X))
    sparse[:5] = 4.0
    fitted = nextday.precipitation_model().fit(X, sparse)
    assert fitted.amount_ is None
    np.testing.assert_allclose(fitted.predict_mean_if_wet(X), 4.0)
    np.testing.assert_allclose(fitted.predict_median_if_wet(X), 4.0)


def test_the_two_stage_model_is_clonable(design):
    """The backtest clones per fold, so this is a precondition for evaluation."""
    model = nextday.precipitation_model()
    copy = clone(model)
    assert copy.get_params() == model.get_params()
    assert not hasattr(copy, "occurrence_")


def test_wet_threshold_is_respected(design):
    """The threshold defines the occurrence label, so changing it must matter."""
    X, _ = design
    amounts = np.where(np.arange(len(X)) % 2 == 0, 0.05, 0.0)
    # 0.05 mm is below the default 0.1 mm cutoff: all days are dry.
    fitted = nextday.precipitation_model().fit(X, amounts)
    assert fitted.base_rate_ == 0.0
    # Lower the threshold and half the days become wet.
    permissive = nextday.TwoStageRain(wet_threshold_mm=0.01).fit(X, amounts)
    assert permissive.base_rate_ == pytest.approx(0.5, abs=0.01)


def test_a_reordered_predictor_frame_raises(design):
    """The precipitation model must not silently predict from the wrong columns.

    `fit` keeps the DataFrame rather than coercing to an array so the inner
    pipelines record `feature_names_in_`. Phase 4 builds its inference frame in a
    separate code path, which is exactly where a column-order slip would happen.
    """
    X, y = design
    fitted = nextday.precipitation_model().fit(X, y)
    assert len(fitted.feature_names_in_) == X.shape[1]
    with pytest.raises(ValueError, match="feature names"):
        fitted.predict(X[list(reversed(list(X.columns)))])

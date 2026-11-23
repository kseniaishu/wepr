"""Persisting models, and the identifier that keeps metrics honest.

The registry's real job is not serialisation. It is that `model_key` names both
the file in `models/` and the entry in `reports/metrics.json`, so publishing a
number for a model the app does not serve becomes a missing key instead of a
quiet lie — the original project reported Ridge and RandomForest accuracy while
serving Prophet.
"""

from __future__ import annotations

import numpy as np
import pytest
import sklearn
from sklearn.dummy import DummyRegressor

from wepr.models import registry


@pytest.fixture
def fitted():
    return DummyRegressor(strategy="constant", constant=7.0).fit(
        np.zeros((5, 2)), np.full(5, 7.0)
    )


# --- the identifier ----------------------------------------------------------


def test_model_key_is_the_documented_shape():
    assert registry.model_key("huesca", "nextday", "tavg") == "huesca.nextday.tavg"
    assert registry.model_key("a-coruna", "seasonal", "prcp") == "a-coruna.seasonal.prcp"


@pytest.mark.parametrize(
    "product,target",
    [("forecast", "tavg"), ("nextday", "temp"), ("", "tavg"), ("nextday", "")],
)
def test_model_key_rejects_unknown_products_and_targets(product, target):
    """A typo must fail loudly here rather than become an orphan key later."""
    with pytest.raises(ValueError):
        registry.model_key("huesca", product, target)


def test_the_key_names_both_the_file_and_the_metric(isolated_paths, fitted):
    """One identifier, both sides — the whole point of the module."""
    key = registry.model_key("huesca", "nextday", "tavg")
    path = registry.save(fitted, key)
    assert path.name == f"{key}.joblib"
    assert registry.available_keys() == [key]


# --- round trip --------------------------------------------------------------


def test_saved_model_loads_and_predicts_identically(isolated_paths, fitted):
    key = registry.model_key("huesca", "nextday", "tavg")
    registry.save(fitted, key)
    loaded = registry.load(key)
    X = np.zeros((3, 2))
    np.testing.assert_allclose(loaded.predict(X), fitted.predict(X))


def test_metadata_records_the_provenance(isolated_paths, fitted):
    key = registry.model_key("huesca", "nextday", "prcp")
    registry.save(fitted, key)
    meta = registry.metadata(key)
    assert meta["key"] == key
    assert meta["sklearn_version"] == sklearn.__version__
    assert "saved_at" in meta and "joblib_version" in meta
    # The estimator itself is deliberately not returned.
    assert "model" not in meta


def test_available_keys_is_empty_before_training(isolated_paths):
    assert registry.available_keys() == []


# --- failure modes, each with a route forward --------------------------------


def test_loading_an_untrained_model_names_the_command_that_would_fix_it(isolated_paths):
    """An untrained app must explain itself, not return a placeholder."""
    with pytest.raises(FileNotFoundError, match=r"train_all\.py --city huesca"):
        registry.load("huesca.nextday.tavg")


def test_a_model_from_another_sklearn_is_rejected(isolated_paths, fitted):
    """joblib artifacts are not portable across versions, so this is checked.

    The envelope is written directly with a foreign version string rather than by
    monkeypatching `sklearn.__version__` around `save`. `monkeypatch` is a single
    function-scoped instance shared with the `isolated_paths` fixture, so an
    `undo()` here would also revert the path redirection and send `load` at the
    real `models/` directory.
    """
    import joblib

    key = registry.model_key("huesca", "nextday", "tavg")
    joblib.dump(
        {
            "envelope_version": registry.ENVELOPE_VERSION,
            "key": key,
            "model": fitted,
            "sklearn_version": "0.0.1-not-this-one",
        },
        registry.model_path(key),
    )

    with pytest.raises(registry.ModelVersionError, match="not portable"):
        registry.load(key)


def test_a_file_that_is_not_an_envelope_is_rejected(isolated_paths, fitted):
    """A bare pickled estimator predates the registry and carries no provenance."""
    import joblib

    key = registry.model_key("huesca", "nextday", "tavg")
    joblib.dump(fitted, registry.model_path(key))
    with pytest.raises(registry.ModelVersionError, match="not a wepr model envelope"):
        registry.load(key)


def test_an_envelope_from_a_future_layout_is_rejected(isolated_paths, fitted):
    import joblib

    key = registry.model_key("huesca", "nextday", "tavg")
    joblib.dump(
        {"envelope_version": 99, "model": fitted, "sklearn_version": sklearn.__version__},
        registry.model_path(key),
    )
    with pytest.raises(registry.ModelVersionError, match="envelope version"):
        registry.load(key)


def test_paths_are_resolved_at_call_time(isolated_paths, fitted):
    """Capturing `paths.MODELS` at import would write into the real models/ dir."""
    assert isolated_paths in registry.model_path("huesca.nextday.tavg").parents

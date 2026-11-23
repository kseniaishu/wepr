"""Training, persisting and loading the fitted models.

Models are persisted with `joblib` to `models/` and loaded at app startup. The
app never trains — a web request must not trigger a fit, and an import must not
either.

**The identifier is the contract.** `model_key()` builds the name used for both
the file in `models/` and the metrics entry in `reports/metrics.json`. Because
both sides are keyed the same way, publishing a metric for a model the app does
not serve shows up as a missing key instead of going unnoticed. That mismatch —
reporting Ridge and RandomForest accuracy while serving Prophet — was the central
scientific error in the original project, and keying both sides identically is
what structurally prevents it from recurring.

`joblib` artifacts are not portable across scikit-learn versions, so the version
is recorded with each model and checked on load.

Implemented in Phase 3.
"""

from __future__ import annotations

import datetime as dt
import platform
from pathlib import Path

import joblib
import sklearn

from .. import paths

PRODUCTS: frozenset[str] = frozenset({"nextday", "seasonal"})
TARGETS: frozenset[str] = frozenset({"tavg", "prcp"})

# Bumped if the envelope's own layout changes, so an old artifact is rejected with
# an explanation rather than unpacked into the wrong shape.
ENVELOPE_VERSION = 1


class ModelVersionError(RuntimeError):
    """A persisted model was built with a different scikit-learn than this one."""


def model_key(slug: str, product: str, target: str) -> str:
    """Build the canonical identifier for one fitted model.

    Args:
        slug: City slug.
        product: Either `"nextday"` or `"seasonal"`.
        target: Either `"tavg"` or `"prcp"`.

    Returns:
        An identifier such as `"huesca.nextday.tavg"`, used as both the model
        filename stem and the metrics key.
    """
    if product not in PRODUCTS:
        raise ValueError(f"unknown product {product!r}")
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}")
    return f"{slug}.{product}.{target}"


def model_path(key: str) -> Path:
    """Where the artifact for `key` lives. Resolved at call time, not at import.

    `paths.MODELS` is read on every call rather than captured in a module-level
    constant so the test suite's `isolated_paths` fixture can redirect it — a
    constant bound at import would write into the real `models/` directory.
    """
    return paths.MODELS / f"{key}.joblib"


def save(model, key: str) -> Path:
    """Persist a fitted model, recording the library versions it was built with.

    Args:
        model: Fitted estimator.
        key: Identifier from `model_key`.

    Returns:
        Path written.
    """
    path = model_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "envelope_version": ENVELOPE_VERSION,
            "key": key,
            "model": model,
            "sklearn_version": sklearn.__version__,
            "joblib_version": joblib.__version__,
            "python_version": platform.python_version(),
            "saved_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        },
        path,
    )
    return path


def load(key: str):
    """Load a persisted model, verifying it was built with this scikit-learn.

    Args:
        key: Identifier from `model_key`.

    Returns:
        The fitted estimator.

    Raises:
        FileNotFoundError: if the model has not been trained yet, with the
            `scripts/train_all.py` invocation that would produce it.
    """
    path = model_path(key)
    if not path.exists():
        slug = key.split(".")[0]
        raise FileNotFoundError(
            f"no trained model for {key!r} at {path}. Train it with:\n"
            f"    .venv/bin/python scripts/train_all.py --city {slug}"
        )

    envelope = joblib.load(path)
    if not isinstance(envelope, dict) or "model" not in envelope:
        raise ModelVersionError(
            f"{path} is not a wepr model envelope — it predates the registry. "
            "Delete it and re-run scripts/train_all.py."
        )
    if envelope.get("envelope_version") != ENVELOPE_VERSION:
        raise ModelVersionError(
            f"{path} uses envelope version {envelope.get('envelope_version')!r}, "
            f"this build expects {ENVELOPE_VERSION}. Re-run scripts/train_all.py."
        )

    # An unpickled estimator built by a different scikit-learn can load without
    # error and then behave incorrectly, so this is checked rather than trusted.
    built_with = envelope.get("sklearn_version")
    if built_with != sklearn.__version__:
        raise ModelVersionError(
            f"{key} was fitted with scikit-learn {built_with}, this environment "
            f"has {sklearn.__version__}. joblib artifacts are not portable across "
            "versions. Re-run scripts/train_all.py."
        )
    return envelope["model"]


def metadata(key: str) -> dict:
    """The provenance recorded with a persisted model, without unpickling its estimator.

    Reads the whole envelope — `joblib` has no partial read — but returns only the
    scalar fields, so a caller wanting to report *when* a model was trained does
    not have to hold the estimator.
    """
    path = model_path(key)
    if not path.exists():
        raise FileNotFoundError(f"no trained model for {key!r} at {path}")
    envelope = joblib.load(path)
    return {k: v for k, v in envelope.items() if k != "model"}


def available_keys() -> list[str]:
    """Keys that currently have an artifact on disk, sorted."""
    if not paths.MODELS.exists():
        return []
    return sorted(p.stem for p in paths.MODELS.glob("*.joblib"))

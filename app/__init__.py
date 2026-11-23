"""Flask application factory.

Two structural rules, both carried over from the project assessment:

- **No analysis logic lives here.** This package handles HTTP: routing, request
  validation, rendering, error handling. Anything that computes a number imports
  it from `wepr`.
- **Nothing trains at import or on request.** Models are loaded from `models/`
  once at startup. A web request must never trigger a fit.

The factory pattern is used so tests can build an app with test configuration
without a module-level global, and so there is no import-time side effect.

**Phase 6 put a `wepr.serve.ModelStore` behind the second rule.** The factory
loads every trained model, reads `reports/metrics.json` and builds each city's
inference feature row, once, here. A request is then a dictionary lookup and a
`predict` call. Startup costs about a second for twenty cities and sixty models.

**Startup does not fail when the models are absent.** An empty `models/` is the
normal state of a fresh clone, and a stale pickle is the normal state of a
`pip install -U scikit-learn`. In both cases the app boots, `MODELS_LOADED` is
false, and the endpoints return `503` naming the script to run — which is more
useful than a traceback at boot, and is the same contract `/map` already keeps.
`/api/status` reports exactly what is loaded and what is not.
"""

from __future__ import annotations

from flask import Flask, current_app

from wepr.serve import ModelStore


def create_app(config: dict | None = None) -> Flask:
    """Build and configure the Flask application.

    Args:
        config: Optional configuration overrides, applied before the models are
            loaded so that `LOAD_MODELS=False` is honoured. Tests use that to
            exercise the untrained-app contract without emptying `models/`.

    Returns:
        The configured application, with its model store attached.
    """
    app = Flask(__name__)

    app.config.from_mapping(
        JSON_SORT_KEYS=False,
        # Set truthfully below, from whether anything actually loaded.
        MODELS_LOADED=False,
        # The escape hatch for tests and for a boot that must not touch `models/`.
        LOAD_MODELS=True,
    )
    if config:
        app.config.from_mapping(config)

    store = (
        ModelStore.load()
        if app.config["LOAD_MODELS"]
        else ModelStore.not_loaded("model loading is disabled for this app")
    )
    app.extensions["model_store"] = store
    app.config["MODELS_LOADED"] = store.ready

    from app.api import bp as api_bp
    from app.views import bp as views_bp

    app.register_blueprint(views_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    return app


def model_store() -> ModelStore:
    """The store held by the current application.

    A function rather than a module-level global: a global would be shared between
    the apps a test session creates and would survive past the request context it
    belongs to.
    """
    return current_app.extensions["model_store"]

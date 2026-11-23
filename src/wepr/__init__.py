"""wepr — temperature and precipitation estimates for Iberian cities.

Two distinct products, which must never be conflated:

1. **Next-day forecast** — predicts tomorrow from observations available today.
   A genuine forecast; supervised models are appropriate.
2. **Seasonal estimate** — for an arbitrary future date there are no recent
   observations to condition on, so the output is a climatological expectation
   ("the historical norm for that day of year", optionally trend-adjusted).
   It is *not* a forecast and the UI must never imply it is.

Nothing in this package trains, fetches or reads data at import time. Importing
`wepr` must stay cheap enough for the Flask app to do it at startup.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]

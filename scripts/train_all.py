"""Train and persist every model the app serves.

    .venv/bin/python scripts/train_all.py [--city huesca]

Writes fitted models to `models/`, keyed by `wepr.models.registry.model_key`, so
the app loads exactly what was evaluated.

Implemented in Phase 3.
"""

from __future__ import annotations

import argparse
import sys

from wepr import features, paths
from wepr.data import load, validate
from wepr.models import nextday, registry, seasonal

# The products this script fits, as (target column, registry target, factory).
# `scripts/evaluate_all.py` backtests the same factories, which is the mechanism
# behind "the evaluated model is the deployed model": both scripts construct the
# estimator by calling the same function rather than each spelling one out.
PRODUCTS = (
    ("tavg_target", "tavg", nextday.fit_temperature),
    ("prcp_target", "prcp", nextday.fit_precipitation),
)

# The seasonal product is a single artifact per city, not one per target: the
# fitted `SeasonalClimatology` carries both the temperature expectation and the
# day-of-year rain probability, because both are read off the same pooled slots.
# Splitting it into `.seasonal.tavg` and `.seasonal.prcp` would persist the same
# object twice and let the two copies drift apart.
SEASONAL_TARGET = "tavg"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--city", action="append", help="limit to this slug; repeatable")
    parser.add_argument(
        "--ignore-gate",
        action="store_true",
        help="train cities that fail the data-quality gate (development only)",
    )
    args = parser.parse_args()

    paths.ensure_dirs()

    try:
        slugs, basis = validate.eligible_slugs(args.city, args.ignore_gate)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not slugs:
        print(
            "nothing to train: no city has processed data that passes the gate.\n"
            "Run scripts/fetch_data.py, then scripts/validate_data.py, and see "
            "reports/data_quality.md for what each city is short of.",
            file=sys.stderr,
        )
        return 1

    print(f"training {basis}", flush=True)
    if args.ignore_gate:
        print(
            "  ! --ignore-gate: these models rest on data the gate rejected. "
            "Metrics from them are provisional.",
            flush=True,
        )

    failures = 0
    for slug in slugs:
        raw = load.load_city(slug)
        # Features first, then the predictor list. In that order, always: capturing
        # the list before engineering is what made the original project's rolling
        # features silently unused.
        frame = features.add_targets(features.build_features(raw))
        predictors = features.feature_columns(frame)

        span = f"{raw.index[0].date()}..{raw.index[-1].date()}"

        # Seasonal first, and from `raw` rather than `frame`: it is a
        # climatological model, so it takes the observations directly and has no
        # use for the lagged predictors the next-day models need.
        seasonal_key = registry.model_key(slug, "seasonal", SEASONAL_TARGET)
        try:
            model = seasonal.SeasonalClimatology(variable=SEASONAL_TARGET).fit(raw)
        except (ValueError, KeyError) as exc:
            print(f"  {seasonal_key:34s} FAILED: {exc}", file=sys.stderr)
            failures += 1
        else:
            registry.save(model, seasonal_key)
            trend = model.trend_
            print(
                f"  {seasonal_key:34s} {model.n_observations_.max():>6,} obs/slot  "
                f"{span}  trend {trend.slope_per_decade:+.3f} °C/dec"
                + ("" if trend.is_climatological else f" (only {trend.n_years} yr)"),
                flush=True,
            )

        for target, registry_target, fit in PRODUCTS:
            key = registry.model_key(slug, "nextday", registry_target)
            try:
                model = fit(frame, predictors)
            except (ValueError, KeyError) as exc:
                print(f"  {key:34s} FAILED: {exc}", file=sys.stderr)
                failures += 1
                continue
            registry.save(model, key)
            print(
                f"  {key:34s} {len(features.labelled(frame, predictors)):>6,} rows  "
                f"{span}  {len(predictors)} predictors",
                flush=True,
            )

    print(f"models -> {paths.MODELS}")
    print(
        "next: .venv/bin/python scripts/evaluate_all.py"
        + (" --ignore-gate" if args.ignore_gate else "")
    )
    if failures:
        print(f"{failures} model(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

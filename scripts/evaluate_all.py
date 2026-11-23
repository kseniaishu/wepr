"""Backtest every model against both baselines and write the metrics.

    .venv/bin/python scripts/evaluate_all.py

Writes `reports/metrics.json`, keyed identically to the files in `models/`, so
the README and the UI can quote verified numbers rather than remembered ones.
Every model is reported against day-of-year climatology and persistence,
including the ones that fail to beat them.

Implemented in Phase 3.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import sys

import sklearn

from wepr import evaluate, features, paths
from wepr.data import load, validate
from wepr.models import nextday, registry, seasonal

# The same factories `scripts/train_all.py` fits. Backtesting a different
# estimator than the one deployed is the precise failure this project was rebuilt
# to eliminate, so both scripts call the same constructor rather than each
# describing the model independently.
PRODUCTS = (
    ("tavg_target", "tavg", nextday.temperature_pipeline),
    ("prcp_target", "prcp", nextday.precipitation_model),
)

# The seasonal product's own backtest, which is a different shape and reported in
# different fields — see `wepr.models.seasonal.SeasonalScore` for why it is not
# forced into `evaluate.Score`. Its fold is one whole year, so it takes no `step`.
SEASONAL_TARGET = "tavg"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--city", action="append", help="limit to this slug; repeatable")
    parser.add_argument(
        "--ignore-gate",
        action="store_true",
        help="evaluate cities that fail the data-quality gate (development only)",
    )
    parser.add_argument("--start", type=int, default=evaluate.DEFAULT_START,
                        help="rows in the first training window")
    parser.add_argument("--step", type=int, default=evaluate.DEFAULT_STEP,
                        help="test block length")
    args = parser.parse_args()

    paths.ensure_dirs()

    try:
        slugs, basis = validate.eligible_slugs(args.city, args.ignore_gate)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not slugs:
        print("nothing to evaluate; see scripts/train_all.py", file=sys.stderr)
        return 1

    print(f"backtesting {basis}, start={args.start} step={args.step}", flush=True)
    if args.ignore_gate:
        print(
            "  ! --ignore-gate: provisional numbers from data the gate rejected. "
            "Do not publish them as the project's accuracy.",
            flush=True,
        )
    header = f"  {'key':34s} {'MAE':>8s} {'RMSE':>8s} {'vs clim':>8s} {'vs naive':>8s}  verdict"
    print(header)

    metrics: dict[str, dict] = {}
    failures = 0

    for slug in slugs:
        raw = load.load_city(slug)
        frame = features.add_targets(features.build_features(raw))
        # After build_features, never before.
        predictors = features.feature_columns(frame)

        # Seasonal first, scored on the raw observations: it is climatological, so
        # it never sees the engineered predictors.
        seasonal_key = registry.model_key(slug, "seasonal", SEASONAL_TARGET)
        try:
            _, seasonal_score = seasonal.backtest(raw, variable=SEASONAL_TARGET)
        except (ValueError, KeyError) as exc:
            print(f"  {seasonal_key:34s} FAILED: {exc}", file=sys.stderr)
            failures += 1
        else:
            # The trend term is the only thing this model adds over a plain
            # day-of-year climatology, so `skill_vs_climatology <= 0` means the
            # trend is not earning its place on this record. That is published as
            # it stands; a negative number here is a finding, not a failure.
            # Same deployed-vs-evaluated check the next-day branch runs: the
            # backtest above fits a `SeasonalClimatology` fold by fold, so the
            # artifact in models/ must be one too.
            try:
                deployed = type(registry.load(seasonal_key)).__name__
            except FileNotFoundError:
                deployed = None
            except registry.ModelVersionError as exc:
                deployed = f"UNLOADABLE: {exc}"

            record = dataclasses.asdict(dataclasses.replace(seasonal_score, model=seasonal_key))
            record["product"] = "seasonal"
            record["estimator"] = "SeasonalClimatology"
            record["estimator_steps"] = None
            record["deployed_estimator"] = deployed
            record["provisional"] = args.ignore_gate
            record["wet_day_threshold_mm"] = seasonal.WET_DAY_MM
            record["baselines"] = {
                "climatology": "this model with the trend term zeroed",
                "last_year": "same calendar date one year earlier",
            }
            record["trend_earns_its_place"] = seasonal_score.skill_vs_climatology > 0
            record["interval_is_calibrated"] = (
                abs(seasonal_score.coverage - seasonal_score.nominal_coverage) <= 0.05
            )
            record["data_span"] = {
                "from": str(raw.index[0].date()),
                "to": str(raw.index[-1].date()),
                "years": round(len(raw) / 365.25, 1),
            }
            metrics[seasonal_key] = record
            verdict = (
                "trend helps" if record["trend_earns_its_place"] else "TREND ADDS NOTHING"
            )
            print(
                f"  {seasonal_key:34s} {seasonal_score.mae:8.3f} {seasonal_score.rmse:8.3f} "
                f"{seasonal_score.skill_vs_climatology:+8.3f} "
                f"{seasonal_score.skill_vs_last_year:+8.3f}"
                f"  {verdict}, {seasonal_score.coverage:.0%} inside the "
                f"{seasonal_score.nominal_coverage:.0%} interval",
                flush=True,
            )

        for target, registry_target, factory in PRODUCTS:
            key = registry.model_key(slug, "nextday", registry_target)
            estimator = factory()
            try:
                _, score = evaluate.backtest(
                    frame, estimator, predictors, target,
                    start=args.start, step=args.step,
                )
            except (ValueError, KeyError) as exc:
                print(f"  {key:34s} FAILED: {exc}", file=sys.stderr)
                failures += 1
                continue

            beats_both = (
                score.skill_vs_climatology > 0 and score.skill_vs_persistence > 0
            )
            # The class actually sitting in models/ — not the class we just
            # backtested. Key equality alone would not have caught the original
            # project's error (metrics from Ridge, predictions from Prophet); a
            # class comparison does.
            try:
                deployed = type(registry.load(key)).__name__
            except FileNotFoundError:
                deployed = None
            except registry.ModelVersionError as exc:
                deployed = f"UNLOADABLE: {exc}"

            record = dataclasses.asdict(dataclasses.replace(score, model=key))
            record["product"] = "nextday"
            record["estimator"] = type(estimator).__name__
            record["estimator_steps"] = (
                [type(s).__name__ for _, s in estimator.steps]
                if hasattr(estimator, "steps")
                else None
            )
            record["deployed_estimator"] = deployed
            record["beats_both_baselines"] = beats_both
            # The provisional marker lives on the record, not only in the gate
            # block beside it. A README generator reading
            # metrics["models"][key]["mae"] must not be able to pick up a
            # clean-looking number with the caveat left behind.
            record["provisional"] = args.ignore_gate
            # Which event the probability describes. Without this the Brier score
            # is unreadable, and it differs from the 1.0 mm wet day used by the
            # station cross-check in wepr.data.validate.
            record["wet_threshold_mm"] = (
                getattr(estimator, "wet_threshold_mm", None)
                if target == "prcp_target"
                else None
            )
            record["data_span"] = {
                "from": str(raw.index[0].date()),
                "to": str(raw.index[-1].date()),
                "years": round(len(raw) / 365.25, 1),
            }
            record["n_predictors"] = len(predictors)
            metrics[key] = record

            # A model that beats neither baseline is printed as such. Reporting it
            # is the point: a quietly dropped result is how an honest pipeline
            # turns into a flattering one.
            verdict = (
                "beats both" if beats_both
                else "beats climatology only" if score.skill_vs_climatology > 0
                else "beats persistence only" if score.skill_vs_persistence > 0
                else "BEATS NEITHER"
            )
            print(
                f"  {key:34s} {score.mae:8.3f} {score.rmse:8.3f} "
                f"{score.skill_vs_climatology:+8.3f} {score.skill_vs_persistence:+8.3f}"
                f"  {verdict}",
                flush=True,
            )

    # The structural check this project exists to keep: every published metric must
    # correspond to a model the app can actually load. A mismatch is a loud warning
    # rather than a silent discrepancy, which is what let the original code report
    # Ridge accuracy while serving Prophet.
    on_disk = set(registry.available_keys())
    unbacked = sorted(set(metrics) - on_disk)
    untested = sorted(on_disk - set(metrics))
    mismatched = sorted(
        key
        for key, record in metrics.items()
        if record["deployed_estimator"] is not None
        and record["deployed_estimator"] != record["estimator"]
    )

    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "fold_geometry": {"start": args.start, "step": args.step},
        "baselines": {
            "nextday": [
                "day-of-year climatology (prior years only)",
                "persistence (tomorrow = today)",
            ],
            # Persistence is undefined at a seasonal horizon, so the naive
            # competitor there is last year's value on the same calendar date.
            "seasonal": [
                "day-of-year climatology with the trend term zeroed",
                "same calendar date one year earlier",
            ],
        },
        "gate": {
            "enforced": not args.ignore_gate,
            "passing_cities": validate.passing_slugs(),
            "note": (
                "PROVISIONAL — these metrics come from cities that fail the "
                "data-quality gate in reports/data_quality.md and must not be "
                "published as the project's accuracy."
                if args.ignore_gate else
                "Every city evaluated here passes the data-quality gate."
            ),
        },
        "sklearn_version": sklearn.__version__,
        "consistency": {
            "metrics_without_a_model": unbacked,
            "models_without_a_metric": untested,
            "estimator_mismatches": mismatched,
        },
        "models": metrics,
    }
    paths.metrics_json().write_text(json.dumps(payload, indent=2) + "\n")

    print(f"\nmetrics -> {paths.metrics_json()}  ({len(metrics)} model(s))")
    if untested:
        # Not fatal: evaluating one city deliberately leaves the others' models in
        # place. Reported so a partial rewrite of metrics.json is visible.
        print(f"  ! {len(untested)} model(s) in models/ have no metric: {untested[:4]}",
              file=sys.stderr)

    # These two *are* fatal. A published metric with no loadable model behind it,
    # or one describing a different estimator than the app would serve, is the
    # exact failure this project was rebuilt to make impossible — so it exits
    # non-zero rather than printing a warning nobody reads in CI.
    if unbacked:
        print(
            f"error: {len(unbacked)} metric(s) have no model in models/: "
            f"{unbacked[:4]} — run scripts/train_all.py first",
            file=sys.stderr,
        )
    if mismatched:
        print(
            f"error: {len(mismatched)} metric(s) describe a different estimator "
            f"than the one in models/: {mismatched[:4]}",
            file=sys.stderr,
        )
    if failures:
        print(f"{failures} backtest(s) failed", file=sys.stderr)
    return 1 if (failures or unbacked or mismatched) else 0


if __name__ == "__main__":
    raise SystemExit(main())

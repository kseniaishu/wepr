"""JSON endpoints consumed by the map and the city pages.

Kept separate from `views` because these have a different contract: a page may be
redesigned freely, but a JSON response shape is depended upon by the frontend and
is covered by a contract test in `tests/`.

**Phase 6 made the prediction endpoints live.** They read the `wepr.serve`
`ModelStore` the factory loaded at startup; nothing here fits, loads or computes.
The `503` path did not go away — it is now returned for the specific model that
is missing, with the specific command that would produce it, instead of being
hard-coded for all of them.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict

from flask import Blueprint, jsonify, request

from app import model_store
from wepr import __version__
from wepr.cities import CITIES, get
from wepr.serve import MAX_SEASONAL_YEAR, NotServeable

bp = Blueprint("api", __name__)


def _unserveable(exc: NotServeable):
    """Turn a store refusal into a `503` carrying its remediation.

    `503` rather than `404`: the resource is real and the app simply cannot answer
    for it yet, which is a server-state problem and not a bad URL. The body always
    names a command, because every reason the store refuses is fixed by running
    one.
    """
    return (
        jsonify(
            error=exc.message,
            detail=exc.detail,
            status_url="/api/status",
        ),
        503,
    )


@bp.get("/health")
def health():
    """Liveness and version, for deployment checks.

    Deliberately independent of whether the models loaded: a health check that
    fails on an untrained app would make "the app is up" and "the app has models"
    the same signal, and they are different operational facts. `/api/status` is
    where the second one is answered.
    """
    return jsonify(status="ok", version=__version__)


@bp.get("/status")
def status():
    """What is loaded, what is not, and the command that fixes each gap.

    The most useful endpoint in the app for a reader deciding how much to trust a
    number on a page: it reports the archive edge the forecasts are issued from,
    when the metrics were generated, whether the data-quality gate is being
    enforced, and every model that was refused along with why.
    """
    return jsonify(model_store().status())


@bp.get("/cities")
def list_cities():
    """Every city the app covers, and whether each is currently served."""
    served = set(model_store().served_slugs())
    return jsonify(
        cities=[asdict(c) for c in CITIES],
        served=sorted(served),
    )


@bp.get("/cities/<slug>")
def city_detail(slug: str):
    """One city's registry entry."""
    try:
        city = get(slug)
    except KeyError:
        return jsonify(error=f"unknown city {slug!r}"), 404
    return jsonify(asdict(city))


@bp.get("/map/<variable>")
@bp.get("/map/<variable>/<int:month>")
def map_summary(variable: str, month: int | None = None):
    """What the published map of `variable` actually shows, for one month.

    Carries the method, the coverage and the city values behind the figure, so a
    reader can check the map's claims rather than trusting its caption. Returns
    503 until `scripts/make_figures.py` has run, never a placeholder field.

    With no month the current one is served, matching what `/map` opens on, and
    `available_months` says which others exist.
    """
    from wepr import paths

    from app.views import (
        MAP_VARIABLES,
        default_map_month,
        load_map_summary,
        map_is_stale,
        published_map_months,
    )

    if variable not in MAP_VARIABLES:
        return jsonify(
            error=f"unknown map variable {variable!r}",
            available=list(MAP_VARIABLES),
        ), 404

    if month is None:
        month = default_map_month(variable)
    elif month not in paths.MAP_MONTHS:
        return jsonify(
            error=f"month {month} is out of range",
            available_months=list(paths.MAP_MONTHS),
        ), 404

    summary = load_map_summary(variable, month)
    if summary is None:
        return (
            jsonify(
                error=f"map for month {month:02d} not yet generated",
                detail="run scripts/make_figures.py to populate reports/ and "
                "app/static/figures/",
                available_months=published_map_months(variable),
            ),
            503,
        )
    # Reported rather than hidden: the target is fixed when the figure is drawn,
    # so a reader deserves to know when it has slipped into the past.
    return jsonify(
        {
            **summary,
            "available_months": published_map_months(variable),
            "stale": map_is_stale(summary),
        }
    )


@bp.get("/trends")
def trend_summary():
    """The fitted warming trends and the scenario envelope quoted beside them.

    Carries every number the `/trends` page prints: each city's Theil-Sen slope
    with its 95% interval, the OLS slope beside it so the choice of estimator is
    auditable, the residual diagnostics, the forward extrapolation at each horizon,
    and the published CMIP6 rows with their own baselines.

    The extrapolation and the scenarios are separate keys and are never merged
    into one series. A consumer that plots them on a shared axis is misreporting
    both: they are anomalies against different baselines, and `baselines` records
    which of those baselines the local record actually covers — none, on an
    incomplete fetch.
    """
    from app.views import load_trend_summary

    summary = load_trend_summary()
    if summary is None:
        return (
            jsonify(
                error="trends not yet generated",
                detail="run scripts/make_figures.py to populate reports/trends.json",
            ),
            503,
        )
    return jsonify(summary)


@bp.get("/forecast/<slug>")
def forecast(slug: str):
    """Next-day forecast for one city, from the models held since startup.

    **Not "tomorrow".** The response carries `as_of`, `valid_for` and `lag_days`
    because the forecast is one day past the edge of the ERA5 archive, which runs
    several days behind the present even when fully fetched. A consumer that
    renders `temperature.expected_c` without `valid_for` is misreporting it, so
    both are required fields rather than optional context.

    The two baselines and the deployed models' measured accuracy travel in the
    same response, so the number cannot be shown alone.
    """
    try:
        get(slug)
    except KeyError:
        return jsonify(error=f"unknown city {slug!r}"), 404

    try:
        return jsonify(model_store().forecast(slug).to_json())
    except NotServeable as exc:
        return _unserveable(exc)


@bp.get("/seasonal/<slug>")
def seasonal(slug: str):
    """Climatological expectation for one city on an arbitrary date.

    Query parameters:
        date: ISO `YYYY-MM-DD`. Defaults to one year from today, which is far
            enough out that no reader could mistake the answer for a forecast.

    This is **not** a forecast and the response says so in a field. For a date in
    2031 there are no observations to condition on, so the output is the
    historical norm for that day of year, trend-adjusted, with the interval the
    norm was estimated from.
    """
    try:
        get(slug)
    except KeyError:
        return jsonify(error=f"unknown city {slug!r}"), 404

    raw = request.args.get("date")
    if raw is None:
        target = dt.date.today() + dt.timedelta(days=365)
    else:
        try:
            target = dt.date.fromisoformat(raw)
        except ValueError:
            return (
                jsonify(
                    error=f"date {raw!r} is not an ISO YYYY-MM-DD date",
                    example="/api/seasonal/madrid?date=2031-07-15",
                ),
                400,
            )

    try:
        return jsonify(model_store().seasonal(slug, target).to_json())
    except ValueError as exc:
        # The horizon limit, which is a scientific judgement made in `wepr.serve`
        # rather than an HTTP concern: past 2100 there is no published scenario
        # envelope to show an extrapolation against.
        return jsonify(error=str(exc), max_year=MAX_SEASONAL_YEAR), 400
    except NotServeable as exc:
        return _unserveable(exc)

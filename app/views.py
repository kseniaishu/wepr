"""HTML pages.

Three pages: the map, a per-city page, and the climate-trends page. All three are
built; Phase 7 added the trends page and the visual design.

**Two of the three serve a generated artifact, and the third is live.** The rule
is that a web request never triggers analysis, and it is satisfied two different
ways here. `/map` and `/trends` read JSON a script wrote; `/city/<slug>` reads
models the factory loaded at startup. The deciding question each time was whether
the thing a request needs can be *held* — a fitted model can, an 11,664-cell
interpolated field and a set of twenty refitted slopes should not be.

**Why the map page serves an artifact rather than a live field.** Interpolating
twenty cities onto an 11,664-cell grid is fast, but it needs every city's seasonal
model, and the project's rule is that a web request never triggers analysis and
models are held from startup rather than loaded per request. So
`scripts/make_figures.py` renders the map and writes what it shows to
`reports/map_<variable>.json`, and this page serves both.

**The city page is live, and it is the reason Phase 6 exists.** It reads the
`wepr.serve.ModelStore` the factory loaded at startup, so a request is a lookup
and a `predict`. The map could become live the same way; it has not been changed
because its artifact also depends on the elevation grid, which is a separate
fetch.

**What this page is required to say, and the reason the requirement exists.** The
two products are different claims and the page must not blur them:

- the next-day forecast is a real forecast, but it is issued from the archive
  edge, which is days behind the present at best and decades behind it on an
  incomplete fetch — so `as_of` and `valid_for` are shown, never "tomorrow";
- the seasonal estimate is a climatological expectation and is labelled as not a
  forecast, with an interval rather than a bare number.

Conflating those two was the central scientific error in the original project, so
`tests/test_api.py` asserts that both statements are present in the rendered HTML
rather than trusting the template to keep them.
"""

from __future__ import annotations

import datetime as dt
import json

from flask import Blueprint, abort, render_template, request

from app import model_store
from wepr import __version__, paths
from wepr.cities import CITIES, get, regions
from wepr.serve import MAX_SEASONAL_YEAR, NotServeable

bp = Blueprint("views", __name__)

# The fields the map is published for, in the order the page offers them.
MAP_VARIABLES = ("tavg", "prcp")

# How far ahead the city page's seasonal estimate looks when the reader has not
# asked for a date. A year out is far enough that nobody could read the answer as
# a forecast, and near enough that the day of year is recognisable weather.
DEFAULT_SEASONAL_HORIZON_DAYS = 365


def load_map_summary(variable: str) -> dict | None:
    """What the map figure shows, or None if it has not been generated.

    Returns None rather than raising so a page can render an explanation. An
    untrained or un-generated app says so; it never shows a placeholder number.
    """
    path = paths.map_summary(variable)
    if not path.exists():
        return None
    try:
        summary = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    figure = paths.FIGURES / summary.get("figure", "")
    # A summary without its PNG is a half-written artifact; treat it as absent
    # rather than rendering a broken image tag.
    return summary if figure.exists() else None


def load_trend_summary() -> dict | None:
    """The fitted trends `scripts/make_figures.py` published, or None.

    Same contract as `load_map_summary` and the same reason for it: the trends are
    a generated artifact, so a request serves what a script computed rather than
    fitting twenty Theil-Sen slopes in the request path.

    One deliberate difference — this does **not** require the `trend_<city>.png`
    figures to exist. The map is a picture and a summary of that picture, so a
    summary without its PNG is half-written; the trends page is a table of numbers
    that stands on its own, and each city's figure is shown only where
    `figure_if_present` finds it.
    """
    path = paths.trend_summary()
    if not path.exists():
        return None
    try:
        summary = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    # A summary with no cities in it is an empty artifact, not a page: the script
    # writes one when no city could be fitted, and there is nothing to draw.
    return summary if summary.get("cities") else None


def figure_if_present(name: str) -> str | None:
    """A figure's filename if `scripts/make_figures.py` produced it, else None.

    The city page shows the climatology and trend panels when they exist and
    simply omits them when they do not, rather than rendering a broken image. The
    figures are generated artifacts and `app/static/figures/` is gitignored, so
    absent is the normal state of a fresh clone.
    """
    return name if (paths.FIGURES / name).exists() else None


@bp.get("/")
def index():
    """Landing page: the cities, and which of them are currently served."""
    store = model_store()
    return render_template(
        "index.html",
        cities=CITIES,
        regions=regions(),
        version=__version__,
        served=set(store.served_slugs()),
        status=store.status(),
        map_available=any(load_map_summary(v) is not None for v in MAP_VARIABLES),
        trends_available=load_trend_summary() is not None,
    )


@bp.get("/city/<slug>")
def city_page(slug: str):
    """One city's two products, side by side and labelled as different claims.

    Returns 503 when the city's models are not loaded, with the command that
    would train them — the same contract the map page keeps. A partially trained
    city renders the half that works: the seasonal estimate does not need the
    next-day models and vice versa, so one missing artifact does not blank the
    page.
    """
    try:
        city = get(slug)
    except KeyError:
        abort(404)

    store = model_store()

    forecast = unserveable = None
    try:
        forecast = store.forecast(slug)
    except NotServeable as exc:
        unserveable = exc

    raw_date = request.args.get("date", "")
    date_error = None
    target = dt.date.today() + dt.timedelta(days=DEFAULT_SEASONAL_HORIZON_DAYS)
    if raw_date:
        try:
            target = dt.date.fromisoformat(raw_date)
        except ValueError:
            date_error = f"{raw_date!r} is not a date in YYYY-MM-DD form."

    seasonal = None
    if date_error is None:
        try:
            seasonal = store.seasonal(slug, target)
        except ValueError as exc:
            date_error = str(exc)
        except NotServeable as exc:
            unserveable = unserveable or exc

    return (
        render_template(
            "city.html",
            city=city,
            forecast=forecast,
            seasonal=seasonal,
            target=target,
            date_error=date_error,
            max_year=MAX_SEASONAL_YEAR,
            unserveable=unserveable,
            climatology_figure=figure_if_present(f"climatology_{slug}.png"),
            trend_figure=figure_if_present(f"trend_{slug}.png"),
            version=__version__,
        ),
        200 if (forecast or seasonal) else 503,
    )


@bp.get("/map")
@bp.get("/map/<variable>")
def map_page(variable: str = "tavg"):
    """The interpolated field over Iberia, with its confidence layers.

    Returns 503 when the map has not been generated, with the command that
    generates it — the same contract as an untrained model endpoint.
    """
    if variable not in MAP_VARIABLES:
        abort(404)
    summary = load_map_summary(variable)
    return (
        render_template(
            "map.html",
            variable=variable,
            variables=MAP_VARIABLES,
            summary=summary,
            version=__version__,
        ),
        200 if summary else 503,
    )


@bp.get("/trends")
def trends_page():
    """The observed warming rate and the published scenario envelope, side by side.

    The distinction between those two is the whole point of the page, so they are
    rendered as two sections that never share an axis or a table. The left-hand
    claim is statistical: this is what the fitted rate implies if it continues
    linearly, with bands that widen. The right-hand claim is quoted: this is what
    CMIP6 projects under named scenarios, with its source and its own baseline.

    Returns 503 when the artifact has not been generated, with the command that
    generates it — the same contract as `/map` and an untrained model endpoint.
    """
    summary = load_trend_summary()
    figures = {}
    if summary:
        figures = {
            city["slug"]: figure_if_present(city["figure"])
            for city in summary["cities"]
        }
    return (
        render_template(
            "trends.html",
            summary=summary,
            figures=figures,
            version=__version__,
        ),
        200 if summary else 503,
    )

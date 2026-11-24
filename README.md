# wepr

Temperature and precipitation estimates for twenty cities across Spain and Portugal,
served by a Flask app from models trained on 76 years of ECMWF reanalysis.

> **Status: complete and serving.** Every page is built and live — the cities, a
> per-city page, the map, and the climate-trends page. The archive record runs
> **1950-01-01 to 2026-11-15**: 28,078 days, 76.9 years, all twenty cities, 0.00%
> missing on every column. That clears this project's own 60-year data-quality gate,
> so the accuracy figures below are quoted from `reports/metrics.json` rather than
> marked provisional, and the UI's provisional callouts no longer fire. The app loads
> 60 trained models at startup; `/api/status` reports `ready: true` with no problems.

This began as a UK Extended Project Qualification artefact covering two cities. It has
been rebuilt as a portfolio project: twenty cities, a reanalysis archive in place of two
station CSVs, the two products separated, every figure measured against baselines, and
the station CSVs repurposed as an independent validation set.

## What it does

Two deliberately separate products, because conflating them is the easiest way to be
wrong in public:

**Next-day forecast.** Predicts tomorrow from observations available today. A genuine
forecast, evaluated by walk-forward backtesting against two baselines — day-of-year
climatology and persistence ("tomorrow = today"). Persistence is hard to beat at
one-day range, and the margin over it is the figure that matters; it is reported
below and it is single-digit.

**Seasonal estimate.** For a date years out there are no recent observations to
condition on, so there is nothing for a forecast model to use. The honest output is the
climatological expectation for that day of year, trend-adjusted, with an interval. It is
labelled as such and never presented as a forecast.

**Climate trends.** A third page shows the warming rate measured in the record with
its confidence interval, extrapolated forward with bands that widen, beside the
published IPCC AR6 / CMIP6 scenario envelope with its citation — because a trend
extrapolation is not a climate projection, and showing the difference is the point.
Across the twenty cities the measured rate runs **+0.143 °C/decade (Lisbon) to
+0.350 (Huesca)**, and all twenty intervals exclude zero.
The two are separate keys in the artifact and separate panels on the page; they never
share an axis or a series. The page also prints which scenario baseline periods the
local record actually covers, since that is what decides whether a common zero even
exists — and on this record that is one of the two: all twenty cities cover 1986–2005,
none covers 1850–1900, which is the baseline AR6's headline ranges are quoted against.
No fetch can fix that, because ERA5 itself begins in 1950.

One result from building this is worth stating plainly, because it is the kind of thing
a short record hides. An earlier fetch covered 1978–2025 and produced rates of
+0.22 to +0.56 °C/decade. Restoring 1950–1977 — the mid-century plateau — cut the same
estimator's rates to +0.14 to +0.35. The warming is unambiguous either way; the *rate*
was overstated by roughly two thirds by a window that happened to start at the bottom
of a trough. Nothing about the model changed between those two answers, only the span
of record handed to it.

## Measured accuracy

Every figure here is read from `reports/metrics.json`, which is written by
`scripts/evaluate_all.py` and keyed to the model files the app actually serves. The
evaluation is walk-forward: expanding-window folds starting at 3,650 days and stepping
90 days, so each prediction is made from a model fitted only on dates before it.

| Product | Metric | Range over 20 cities | Beats both baselines |
|---|---|---|---|
| Next-day temperature | MAE | **0.77 – 1.35 °C** (median 1.04) | 20 / 20 |
| Next-day temperature | skill vs. day-of-year climatology | +42.6% – +54.5% | |
| Next-day temperature | skill vs. persistence | **+5.5% – +11.3%** (median +8.9%) | |
| Next-day rain | Brier score | 0.135 – 0.217 | 20 / 20 |
| Next-day rain | Brier skill vs. climatology | +9.3% – +29.1% | |
| Seasonal temperature | MAE | 1.41 – 2.57 °C | |
| Seasonal temperature | 80% interval coverage | 78.1% – 80.1% (nominal 80%) | |

Three things in that table are worth reading carefully rather than skimming:

**The honest headline is the persistence column, not the climatology column.** Beating
a day-of-year climatology by half is easy — it has no idea what today looked like.
Beating "tomorrow = today" by 5–11% is the actual claim, and at one-day range that is
a reasonable result rather than an impressive one. The next-day model is a
`StandardScaler → RidgeCV` pipeline over 34 causal predictors; it beats both baselines
on all twenty cities, it beats climatology in all 252 folds of all twenty, and it beats
persistence in 227–245 of those 252 — so the margin is consistent across folds, not an
average rescued by a good run.

**Rain is reported as a probability, never as millimetres.** Precipitation is
zero-inflated and comes from the coarser ERA5 grid, so the headline is an occurrence
probability with a Brier score attached. A two-stage model is the right shape for the
target, but the product of an occurrence probability and a conditional amount is not a
millimetre figure and is not presented as one.

**The seasonal product is barely better than its own climatology, and that is the
finding.** Skill over a day-of-year climatology with the trend term zeroed is
+0.5% – +4.2% — positive on all twenty cities, consistently, and small. What it buys
is the trend term, which is sound on all twenty on this record; the interval
calibration is the more useful number, and 80% intervals covering 78.1–80.1% of
outcomes means the stated uncertainty is close to real.

## The map

Twenty cities interpolated into a continuous field over Iberia — a picture that
implies far more spatial resolution than twenty points support, so the map is built
to show its own limits rather than describe them in a caption.

It does not weight raw values by distance. A linear surface in **elevation,
latitude and longitude** is fitted first — across these cities that predicts annual
mean temperature to a leave-one-out MAE of **0.45 °C**, against 1.95 °C for the
city mean alone, at a physically sensible lapse rate of about **−4.4 °C/km** — and
only the residuals are distance-weighted. Interpolating raw values would paint the
Ebro valley's temperature onto the 2,500 m Pyrenean cells above it. The design was
chosen by cross-validation over the twenty cities, not by in-sample fit.

What the map shows about itself:

- **solid** inside the convex hull of the cities, where a cell sits between
  observations; **translucent** outside it, where every city is on one side and the
  value is an extrapolation;
- **hatched** above the highest city the surface was fitted on (776 m) — the
  Pyrenees and the Sierra Nevada — where the elevation term is *held at its fitted
  ceiling* rather than carried further. The cities reach 776 m and the terrain
  reaches 3,064 m, so that term is the one covariate asked to reach kilometres
  beyond its data. Left unclamped it was adding +5.3 °C to the highest cells and
  painting the Sierra Nevada as the hottest ground in Iberia in July, because
  across these cities July temperature has zero correlation with elevation and the
  fitted coefficient is really continentality in disguise;
- **city markers on top**, always, so the sparsity of the evidence is visible;
- **nothing at all** over the sea, beyond 95 km from the city hull, or off the
  cities' own landmass.

That last rule is doing real work. North Africa is excluded by **land
connectivity**, not by distance: on the 0.1° grid Tarifa is 90.9 km outside the
city hull and Tetouan is 89.0 km, so no distance threshold separates them — but the
Strait of Gibraltar is water, so a connected-component filter does.

The land mask and the elevation covariate are the same artifact,
`data/grid/elevation.npz`, fetched from the same digital elevation model the
archive reports for the cell it served (Madrid comes back as 666 m either way), so
the map's terrain and the data's terrain cannot disagree. It is committed, because
terrain is static reference data and a clone should render the map without a
network call. Its one known failure: the endpoint clamps at zero instead of
returning bathymetry, so land at or below sea level — the Guadalquivir marshes, the
Albufera — reads as sea and goes unpainted.

## How it serves

The Flask factory loads every trained model from `models/`, reads
`reports/metrics.json`, and builds each city's inference feature row **once, at
startup** — about a second for twenty cities and sixty models. A request is then a
dictionary lookup and a `predict` call. A web request never triggers analysis,
because analysis that happens per request is analysis nobody has evaluated.

Three refusals are built into that load, and each returns `503` naming the command
that fixes it rather than a placeholder number:

- a model with **no entry in `reports/metrics.json`** is not served at all. The
  key that names the file on disk also names the metrics entry, so an unevaluated
  model cannot reach a user;
- a model whose **class no longer matches** the `deployed_estimator` the metrics
  were measured on is not served either. A key check alone would pass if someone
  swapped the estimator under the same filename — which is the original project's
  failure exactly, where the published MAE came from Ridge and RandomForest while
  the app served an unevaluated Prophet;
- a **stale `joblib` artifact** from a different scikit-learn is rejected on load.

None of these stop the app from booting. An empty `models/` is the normal state of
a fresh clone, so the app starts, says what is missing and keeps serving the pages
that do not need a model.

### The forecast is dated, never called "tomorrow"

ERA5 is a reanalysis and runs several days behind the present even when fully
fetched, so the next-day product is "one day ahead of the last observation in the
record", not "tomorrow". Every response carries `as_of`, `valid_for` and
`lag_days`, and the UI shows all three. On the complete fetch the archive edge is
2026-11-15 and the forecast describes 2026-11-16 — about a week behind the present,
which is the archive's own publication lag rather than a gap in the download. It is a
genuine one-day-ahead prediction from information strictly prior to its date, which is
exactly what the backtest measured, but it is not news about today's weather, and the
page says so in those words.

Both baselines travel with every prediction: what persistence and the day-of-year
climatology say for the same date are in the JSON and on the page, so a model
number cannot be shown alone.

### Endpoints

```
GET /                      the cities, and which are currently served
GET /city/<slug>           both products, labelled as different claims
GET /map[/<variable>]      the interpolated field and what it shows about itself
GET /trends                the measured warming rate beside the published envelope
GET /api/status            what loaded, what did not, and the command that fixes it
GET /api/forecast/<slug>   next-day forecast, with baselines and measured accuracy
GET /api/seasonal/<slug>?date=YYYY-MM-DD
                           climatological expectation, with its interval
GET /api/cities[/<slug>]   the registry
GET /api/trends            fitted slopes, their intervals, and the quoted scenarios
GET /api/health            liveness and version
```

## Data

**ERA5-Land and ERA5**, via the [Open-Meteo archive API](https://open-meteo.com/) —
any location from 1950 to near-present, in °C and mm, gap-free.

Temperature is **ERA5-Land** at 0.1° (~9 km). Precipitation is **ERA5** at 0.25°
(~28 km), because Open-Meteo's `era5_land` model returns `precipitation_sum` as null
for every date in the archive; its `era5_seamless` model serves ERA5-Land's temperature
verbatim and fills precipitation from ERA5. The model is pinned explicitly rather than
left to the API's default, so a refetch is reproducible.

Both are **reanalysis**, not direct observation: a physical model with observations
assimilated into it. That is a reasonable choice here — complete, consistent and
spatially uniform, where station records are patchy and schema-divergent — but it is not
the same as measurement, and reanalysis precipitation is less reliable than its
temperature, particularly for extremes and in mountainous terrain. Here it is also the
coarser of the two fields, which is a further reason the app reports rain *probability*
rather than a millimetre figure.

So the original NOAA GHCN-Daily station exports for Huesca and Valencia are kept in
`data/validation/` as an **independent check**: the reanalysis is compared against real
station observations for those two cities, and the agreement is reported rather than
assumed.

## Setup

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install -e .
.venv/bin/python -m pytest
.venv/bin/python run.py          # http://localhost:8000
```

The editable install is required: the package lives under `src/`, so `import wepr` only
resolves once it is installed.

## Layout

```
src/wepr/        modelling library — data, features, models, evaluation, trends,
                 maps, and the serving layer the app reads
app/             Flask app: routes, templates, static. HTTP only, no analysis
scripts/         artifact producers: fetch_data, fetch_elevation, train_all,
                 evaluate_all, make_figures
data/validation/ original GHCN station exports, used to validate the reanalysis
data/grid/       elevation grid — the map's land mask and its elevation covariate
tests/           leakage guards, registry invariants, serving refusals, API contract
reports/         metrics.json, the data-quality gate, and the artifacts the map
                 and trends pages serve (map_<variable>.json, trends.json)
```

Three structural rules: nothing trains at import time, `app/` never computes a number
it could import from `wepr`, and **a web request never triggers analysis**. The third
is satisfied two ways: `/city/<slug>` reads models the factory loaded at startup,
while `/map` and `/trends` serve JSON a script wrote — and return `503` naming the
script when it has not run, never a placeholder number.

## Licence

MIT — see [LICENSE](LICENSE).

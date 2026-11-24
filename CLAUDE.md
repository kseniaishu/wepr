# wepr — Iberian temperature & precipitation estimates

A Flask web app serving temperature and precipitation estimates for twenty cities in
Spain and Portugal, from models trained on the ERA5-Land reanalysis archive.
Originally a UK EPQ artefact covering two cities; being rebuilt as a portfolio project.

## Running it

```bash
.venv/bin/python run.py     # serves on http://wepr.localhost:8000
```

Port **8000**, not 5000 — macOS Monterey's AirPlay Receiver can claim 5000.
Local hostname is `wepr.localhost` (see `docs/local-domain.md`).

The entry point is `run.py`, **not** `app.py`: the application package is `app/`, and a
module and package of the same name shadow each other confusingly.

Install: `pip install -r requirements-dev.txt && pip install -e .`
The editable install is what puts `wepr` on the path — the package lives under `src/`,
so without it `import wepr` fails. Tests: `.venv/bin/python -m pytest`.

## Ground truth about the data

**Primary source: the Open-Meteo archive API, model `era5_seamless`.** Any lat/lon,
1950 to roughly a week before present, **already in °C and mm**, no gaps, no API key.
Multi-city requests batch into one call, subject to a ~16,000 location-day cap per
request (`wepr.data.fetch.MAX_LOCATION_DAYS`), so a full fetch is ~40 windowed calls
and *will* be rate-limited partway — the client waits out the 429 window rather than
backing off in seconds.

**Temperature is ERA5-Land (0.1°, ~9 km); precipitation is ERA5 (0.25°, ~28 km).**
Phase 1 probed the API directly and found that `models=era5_land` returns
`precipitation_sum` as **null for every date from 1950 to the present** — Open-Meteo
does not serve that field for the land model, so requesting ERA5-Land alone would mean
no precipitation at all. `era5_seamless` returns ERA5-Land's temperature verbatim
(identical values on the same 0.1° cell, verified against `era5_land`) and fills
precipitation from ERA5. Pin the model explicitly; the API's default picks a different,
undocumented blend on a finer grid and is not reproducible.

This makes precipitation both the less reliable field *and* the coarser one, which
sharpens rather than softens the caveat below.

> **There are no unit conversions in the primary pipeline.** If you find yourself
> writing `(x - 32) * 5/9` or `x * 2.54` against ERA5-Land data, something is wrong.

**The caveat that must appear in the README and the UI.** ERA5-Land is *reanalysis*, not
direct observation: a physical model with observations assimilated into it. For this use
it is defensible and arguably preferable — complete, consistent, spatially uniform — but
it is **not station measurement**, and reanalysis precipitation is materially less
reliable than its temperature, especially for extremes and where relief varies steeply
inside a grid cell (which is why Granada and Huesca are flagged `Altitude` in the
registry — a label meaning **broken terrain, not high ground**: on the elevations the
archive actually served, Salamanca, Valladolid, Bragança and Madrid all sit higher than
Huesca, but on the flat meseta, so they are `Interior`).

### The GHCN CSVs are now the validation set, not the input

`data/validation/weather_huesca.csv` and `weather_vlc.csv` are the original NOAA
GHCN-Daily station exports. They are **no longer modelling inputs**. Their job is to
cross-check the reanalysis against real observations for the two cities where genuine
station data is in hand, and to report the agreement. *"I validated the reanalysis
against independent station records"* is a far stronger claim than either source alone.

These two files **are** in °F and inches, and are the one place conversion happens:

```python
celsius = (fahrenheit - 32) * 5 / 9
mm      = inches * 25.4          # note: mm, not cm — the project is SI throughout
```

They have **different schemas** and must not be treated as interchangeable:

| File | Station | Range | Usable |
|---|---|---|---|
| `weather_huesca.csv` | SPE00120161 | 1990-01-01 → 2024-03-31 | PRCP, TMAX, TMIN; <0.1% missing |
| `weather_vlc.csv` | SPM00008284 | 1990-01-01 → 2024-03-01 | TAVG only |

Valencia missingness: TMIN 43.3%, TMAX 35.1%, PRCP 16.5%, SNWD 99.9%. So Valencia
supports a **temperature** comparison only — do not build a precipitation validation on
it. Huesca has no TAVG column, so derive it as `(TMAX + TMIN) / 2` **before** the °F→°C
conversion. Leave gaps as `NaN`; forward-filling an observational record invents
measurements.

`weather_madrid.csv` and `huesca.csv` were referenced by the retired legacy scripts and
**do not exist**. Madrid is now in scope, but as a reanalysis city like any other.

**Two traps in the comparison, both of which otherwise read as reanalysis error.**
*Definition*: ERA5's `temperature_2m_mean` is a true 24-hour average, while the Huesca
midrange `(TMAX + TMIN) / 2` runs systematically warm — a diurnal curve spends more of
the day near its minimum. So `tmax`/`tmin` are the like-for-like comparisons and `tavg`
carries that caveat. *Location*: Huesca's export states `LONGITUDE 0.3264`, east of
Greenwich, which would put the station ~60 km east of the city. The sign is lost in the
export — at 42.0831 N, 0.3264 **W** is Huesca-Pirineos airport, and the archive's grid
elevation there (544 m) matches the station's stated 541 m, where the eastern reading
lands on a 791 m cell. `wepr.data.validate.station_location_check` resolves this
empirically rather than asserting it.

Also note the Huesca missingness figure depends on the definition: <0.1% of the **rows
present** in the file, but 3.4% of the **dates in its span**, since 427 dates are absent
from the export entirely. `load_ghcn` reindexes onto a complete daily range so those
become visible NaN rows rather than holes in the index — a hole would make every
downstream `shift(1)` silently span it.

## The two products — keep them distinct

Conflating these was the central scientific error in the original code.

1. **Next-day forecast.** Predicts tomorrow from observations available today.
   A genuine forecast. Supervised models are appropriate.
2. **Seasonal estimate.** For an arbitrary future date (2031, 2047...) there are no
   recent observations to condition on, so the output is a climatological expectation —
   roughly "the historical norm for that day of year", trend-adjusted. It is **not** a
   forecast and the UI must never imply it is. Return an interval, never a bare number
   to two decimal places.

## The climate-trends page

The long-range page shows **two things side by side**, and the distinction between them
is the skill being demonstrated:

1. The observed warming rate estimated from the record with proper uncertainty
   (e.g. `+0.31 °C/decade, 95% CI [0.26, 0.37]`), extrapolated forward, labelled
   explicitly as a **statistical extrapolation**, with bands that widen.
2. The published **IPCC AR6 / CMIP6 scenario envelope** for Iberia alongside it, cited.

Extrapolating a trend fitted to 75 years of reanalysis is **not** climate projection —
real projection uses CMIP6 ensembles under emissions scenarios. Never present a fitted
trend line as a 2075 forecast. Theil–Sen is the default estimator over OLS, being robust
to the outlier years a long daily record inevitably contains.

**Built in Phase 7, and it serves an artifact like the map does.** The rule that a web
request never triggers analysis decided this the same way it decided `/map`:
`scripts/make_figures.py` writes `reports/trends.json` from the *same* fits that draw
`trend_<city>.png`, and `/trends` and `/api/trends` serve it, returning 503 with the
command to run when it is absent. A startup-held trend table would have obeyed the
letter of the rule while fitting every slope a second time, independently of the
figures shown beside it — which is the divergence that put `reports/trends.md` in the
figure script in the first place.

The two claims are kept apart **structurally**, not just in wording: `cities[].extrapolation`
and `scenarios` are separate keys that are never merged into one series, and the page
renders them as two panels that share no axis. `summary()["baselines"]` records which
scenario baseline periods the local record actually covers, and on the complete record
the answer is **one of the two**: 1986–2005 is covered by all twenty cities, 1850–1900
by none — and 1850–1900 is the baseline AR6's headline ranges are quoted against. That
is the reason the two panels *cannot* be drawn against a common zero, rather than a
stylistic preference. Sliding an 1850–1900 anomaly onto a 1950-baselined axis would
overstate it by most of a degree, which is the size of the effect under discussion.
Note that the full 1950–2026 fetch did not fix this and no fetch can: ERA5 itself
begins in 1950, so the pre-industrial baseline is outside the archive, not outside
the download.

The page's chart is a forest plot of the twenty slopes with their 95% intervals against
a zero line. One series, so one hue and no legend; the emphatic colour is reserved for a
slope whose interval clears zero. On the 22-year record *none* did, and the flat field of
one colour was the finding rather than a rendering choice; on the complete 76-year record
**all twenty do**, and the template says which case it is rather than hard-coding either
— so the figure reports the record it was given instead of a conclusion chosen in
advance. Keep that branch for the same reason `TrendTerm.is_sound` is kept.

## The map page

The map shows a continuous field interpolated from the twenty cities, which is a
picture that *implies* far more resolution than twenty points support. Four things
answer that, and all four are in `wepr.spatial`:

- **Residuals are interpolated, not values.** A linear surface in elevation,
  latitude and longitude is fitted first, then only its residuals are
  distance-weighted. Chosen by leave-one-out CV over the twenty cities (annual
  mean LOO MAE 1.95 °C → 0.45 °C; lapse rate ≈ −4.4 °C/km), not by in-sample R².
  Plain weighting of raw values paints the Ebro valley onto the Pyrenean ridges.
- **Solid inside the city hull, translucent outside it.** Inside is
  interpolation; outside every city is on one side, which is extrapolation. They
  are different claims and are not drawn the same.
- **The elevation term is clamped to its fitted support, and hatching marks
  where.** The cities span 0–776 m; the grid reaches 3,064 m. Above the highest
  city `TrendSurface.predict` holds elevation at the ceiling rather than carrying
  the coefficient on, so those cells are surface-at-776 m plus residuals.
  **This was found by finally drawing the map, and it matters most in July.**
  July temperature has *zero* correlation with elevation across these twenty
  cities (+0.00) — Iberian summer heat is continental — yet the fit still assigns
  +2.3 °C/km, because once lat and lon take the north–south gradient the leftover
  elevation spread tracks continentality (Atlantic cities at sea level and
  20.2 °C, Interior at 475 m and 25.4 °C). Unclamped, that reached 3,064 m, added
  +5.3 °C, and made the Sierra Nevada and the Pyrenees the hottest ground in
  Iberia on a July map — the very failure the residual design exists to prevent,
  with the sign reversed. The annual fit is fine (−3.81 °C/km) and January is
  correctly signed (−7.72), so it is the summer map that exposes it.
  `spatial.py` previously claimed elevation "contributes almost nothing in July,
  but it costs nothing either". In-sample, true; out-of-sample, on exactly the
  terrain drawn, it cost five degrees. Three tests in `tests/test_spatial.py`
  keep the guard executable, including one that reproduces the confound on the
  real city geometry.
- **City markers always on top**, so a reader can count the evidence.

`HULL_MARGIN_KM` is 95 km, chosen by measuring named places, and the trade is
stated in the module: Figueres is lost, a sliver of France near Bayonne is gained.
**Morocco is excluded by land connectivity, not distance** — on the 0.1° grid
Tarifa is 90.9 km outside the hull and Tetouan 89.0 km, so no threshold separates
them; the Strait is sea, so a connected-component filter does. `tests/test_spatial.py`
keeps both facts executable.

### The elevation grid

`data/grid/elevation.npz`, built once by `scripts/fetch_elevation.py` from
Open-Meteo's elevation endpoint. It is the **land mask** and the **elevation
covariate** at once, and it is the same DEM the archive reports for the cell it
served (Madrid: 666.0 m both ways), so the map's terrain and the data's terrain
cannot disagree.

- **It is committed**, unlike everything else under `data/` — terrain is static
  reference data and the file is small, so a clone renders the map offline. Same
  rationale as `data/validation/`. The `.partial.npz` is not committed.
- **Sea reads as exactly 0.0**, because the endpoint clamps rather than returning
  bathymetry. So land at or below sea level is indistinguishable from water: the
  Guadalquivir marshes and the Albufera go unpainted. The error direction is
  toward *not* painting, which is the right way round.
- **Only drawable cells are fetched** (8,823 of 11,664 at the 95 km margin), and
  **the fetch resumes** — it shares Open-Meteo's quotas with the archive client and
  will hit them. The binding one is a *location* budget, not a request count: the
  wall lands at 5,000 cached points regardless of how they are batched.

### The map page serves an artifact, not a live field

`scripts/make_figures.py` renders `map_<variable>.png` and writes what it shows to
`reports/map_<variable>.json`; `/map` and `/api/map/<variable>` serve both, and
return **503 with the commands to run** when they are absent. This is not a
placeholder design — it is the "a web request never triggers analysis" rule. When
Phase 6 holds the models at startup the page can go live without changing its
contract.

## Hard rules

- **No future information as a predictor.** A feature is legal only if its value is
  knowable strictly before the target timestamp. Rolling windows, expanding means and
  shifts must all be checked. `wepr.features.assert_causal` enforces this executably —
  use it on every new feature builder. It has a stated limitation (it truncates the tail,
  so it cannot see a leak sourced before every cut point); read the definitions too.
- **Fit on train only.** Scalers, imputers and encoders are fitted inside the training
  fold, never on the full series before splitting. In practice: pass a `Pipeline` to the
  backtest, not a bare estimator.
- **Every model is reported against baselines.** Always include (a) day-of-year
  climatology and (b) persistence (tomorrow = today). A model that does not beat both is
  reported as such, not quietly dropped. Persistence is genuinely strong at one-day
  range; beating it is the whole task.
- **The evaluated model must be the deployed model.** `wepr.models.registry.model_key`
  keys both the file in `models/` and the entry in `reports/metrics.json`, so a mismatch
  surfaces as a missing key. In the original code the MAE figures came from
  Ridge/RandomForest while the app served Prophet, whose accuracy was never measured.
- **Recompute feature lists after engineering features.** Call
  `features.feature_columns(frame)` *after* `build_features`. The original bug:
  `predictors` was captured first, so the rolling features were never used and the
  reported "improvement" was an artifact of a 14-row trim.
- **Precipitation cannot be negative.** Clamp at zero, and report rain *probability* as
  the headline rather than a spuriously precise millimetre figure. The two-stage model
  (occurrence classifier × conditional amount) is the right shape for a zero-inflated
  target; just don't multiply the two and call the product millimetres.
- **Prophet is not a dependency and should not be reintroduced.** It served the app's
  predictions while never being evaluated. The seasonal product is now an explicit
  climatology-plus-trend model that can be measured. It also pulled a Stan toolchain that
  compiled for minutes and was the likeliest install failure for anyone cloning the repo.

## Code conventions

- Reusable logic lives in `src/wepr/`; scripts that produce artifacts live in `scripts/`.
  **No analysis logic at module import time** — the Flask app must not train on import,
  and `tests/test_imports.py` enforces it.
- `app/` handles HTTP only: routing, validation, rendering, errors. Anything that computes
  a number imports it from `wepr`. The dependency runs one way — `wepr` never imports
  `app`.
- No `plt.show()` in anything the app imports. Set the `Agg` backend and write figures to
  `app/static/figures/`.
- Trained models are persisted with `joblib` to `models/` and loaded **once at startup**,
  never per request.
- Metrics are written to `reports/metrics.json` so the README quotes verified numbers
  instead of remembered ones.
- Dependencies are pinned exactly in `requirements.txt`; dev extras in
  `requirements-dev.txt`.
- An untrained app returns `503` with an explanation. It never returns a placeholder
  number.

## Repo hygiene

Never commit: `.venv/` (700 MB), `__pycache__/`, `.DS_Store`, `*.icloud` stubs, generated
artifacts under `data/raw/`, `data/processed/`, `models/`, `reports/`,
`app/static/figures/`, or `13-Formulario_autoriz_de_regreso.pdf` — an unrelated personal
document that happens to sit in this folder. All are in `.gitignore`; keep them there.

`data/validation/` **is** committed — the two station CSVs are small and are the
reproducible basis of the validation claim. So is `data/grid/elevation.npz`, for the
same reason; its `.partial.npz` is not.

The earthquake project (`seismic.py`, `seismic.html`, `cleaned_data.csv`) was moved to
`../seismic-project/` in Phase 0. It is a separate piece of work; don't pull it back in.

## The serving layer

`wepr.serve.ModelStore` is the only place a trained model and an HTTP request meet.
The factory builds one at startup; `app/` calls it and never loads a model, builds
a feature or computes a statistic itself. Everything — artifacts, metrics, and each
city's inference feature row — is loaded once, because analysis in the request path
is analysis nobody has evaluated.

**Three refusals, each returning 503 with the command that fixes it.** A model is
not served if it has no entry in `reports/metrics.json`, if its class no longer
matches the `deployed_estimator` that entry records, or if its `joblib` artifact
was built by a different scikit-learn. The second is the one that would have caught
the original project: a key check alone passes when someone swaps the estimator
under the same filename. **None of them stop the app booting** — an empty `models/`
is the normal state of a fresh clone, so the store records the problem and
`/api/status` reports it.

**The forecast is dated, never called "tomorrow".** It is one day past the archive
edge, which is several days behind the present even on a complete fetch and decades
behind it on this one. `as_of`, `valid_for` and `lag_days` are required fields, not
optional context, and the city page states in words that a stale forecast is not
today's weather. Both baselines ship inside the response so a page cannot show the
model's number alone.

**`features._prior_year_climatology` is written out rather than as a
`transform(lambda ...)`.** The lambda was 95% of startup — 0.44 s per city, 9 s to
boot. The vectorised form is the same statistic; `tests/test_features.py` keeps the
readable form as the specification and asserts the two are identical — on a
synthetic record, on one with gaps (including 29 February, whose calendar key is
shared with nothing), and on every real city record present. An optimisation to a
feature definition is an optimisation to what every model was fitted on, so a
divergence here would not fail, it would silently invalidate
`reports/metrics.json`.

### The seasonal trend term — the same machinery, three different verdicts

The seasonal expectation is a slot mean plus `slope x (target_year - anchor_year)`,
and nothing in that arithmetic knows how far it is being carried. **The term has now
been checked on three different records and reached a different verdict on each. That
history is the point and is worth keeping: the model never changed, the record did.**

*On the first 1950–1971 fetch* the anchor was 1960, so a 2027 target was 66 years
out: Seville's estimate was dragged −1.96 °C by a slope whose 95% CI spanned zero,
every city's CI spanned zero, and `skill_vs_climatology` was negative everywhere.
The term was noise being extrapolated three times the length of the record.

*On the 1978–2025 window fetched 2026-11-22* the anchor was 2002, a 2027 target 26
years out against 48 years of record, and every slope came back positive with a CI
excluding zero — but steep, +0.221 °C/decade (Lisbon) to +0.559 (Granada), because
a window starting in 1978 begins *after* the mid-century plateau and sits on the
steepest part of the century.

*On the complete 1950–2026-11-15 record fetched 2026-11-23* — 76.9 years, the whole
archive — every slope is still positive with a CI excluding zero, but the rates are
**materially lower**: +0.143 °C/decade (Lisbon) to +0.350 (Huesca), roughly 60% of
the 48-year figures. Restoring 1950–1977 put the plateau back in the fit and flattened
it. This is the single most defensible result the longer fetch bought: the warming is
unambiguous on every city, *and* the shorter window overstated its rate by about
two thirds. The anchor is now 1987.5, so a 2027 target is 39.5 years out against 76
years of record — the first time the extrapolation distance has been shorter than the
record it was fitted on. `TrendTerm.is_sound` is true for all twenty and the trend
contributes +0.56 °C (Lisbon) to +1.38 °C (Huesca).

The backtest agrees but modestly: seasonal `skill_vs_climatology` is positive for
20 of 20 cities, by 0.5–4.2%, and `trend_earns_its_place` is true for all twenty.
Small, consistent and in the right direction is the honest description.

The model was never changed through any of the three verdicts — it is the Phase 4
product and `reports/metrics.json` is keyed to it. What does the work is
`serve.TrendTerm`, which breaks the term out with its CI, its extrapolation distance
and an `is_sound` flag, and the city page shows the estimate *without* the trend
beside the headline whenever that flag is false. Keep that path: it is not dead code
now that the flag is true for every city, it is the guard that made the short-record
answer honest, and a refetch of a narrower window would trip it again.

## Project status

Phases 0–7 are complete: foundation, data layer, features, models and evaluation,
the seasonal/trends products, the map, the serving layer, and the visual design with
the climate-trends page. Every route is built — `/`, `/city/<slug>`, `/map`,
`/trends`, and `/api/{status,health,cities,forecast,seasonal,map,trends}`.

**The design rule, for anything added later.** The palette is not chosen in the CSS;
it is `scripts/make_figures.py`'s palette, token for token, because the pages and the
PNGs appear on the same screen and a reader must not have to re-learn what a colour
means between them. Blue is the observed record, rust the fit, amber "extrapolated,
distrust it" — which is why amber also carries every provisional callout — and green
the published scenario envelope. Contrast was measured rather than eyeballed: every
token clears 4.5:1 as text or 3:1 as a mark in both modes, and where the amber had to
carry words it gets a darker step (`--caution-ink`) instead of being used at 3.2:1.
Dark mode is a second set of steps from the same hues, never an inversion, and no
colour has its only definition inside a media query.

**What is a free choice, and what the choices were.** Only the *surface* tokens —
grounds, rules, shadows, type. The page sits on a fixed gradient layer painted by
`body::before` (not `background-attachment: fixed`, which stutters on iOS, and not a
`body` background, which compresses to an invisible band on a 6,000 px page). Type is
Source Serif 4 for headings and headline figures against Inter for running text — the
journal-paper pairing, with real fallback stacks so an offline clone degrades to
Georgia rather than to nothing. Sections are a small number of large `.panel` blocks
with a 2 px coloured top edge keyed to their kind (`--data` measured, `--caution`
expectation, `--scenario` quoted), not a grid of equal tiles; the opening block sits
directly on the gradient, because a page where everything is boxed has no hierarchy.

One measured consequence worth keeping: the gradient's foot is darker than its head,
which dropped `--muted` to 4.13:1 there. It is `#616971` rather than `#6a737d` for
that reason alone. Re-measure against **both** ends of the ramp if the grounds change.

`run.py` honours `PORT`, defaulting to 8000, so a second copy can run alongside the
first without editing it.

**The elevation grid is built, and `/map` is live.** `scripts/fetch_elevation.py`
completed on 2026-11-24 after three days of quota refusals, and it completed *because*
it resumes: the fetch was cut off twice by Open-Meteo's hourly cap and picked up from
`elevation.partial.npz` each time. The cap is a **location** budget, not a request
count — the wall landed at exactly 5,000 cached points, 50 requests of 100 coordinates
— which is worth knowing before anyone tunes `MAX_ELEVATION_POINTS` expecting fewer,
larger requests to help. `data/grid/elevation.npz` is committed, so a clone renders
the map with no network call at all.

The spatial code is therefore no longer verified only against the synthetic terrain in
`tests/conftest.py`: the real coastline is drawn, and the connectivity filter's
exclusion of Morocco is **observed rather than argued** — the rendered map stops at the
Strait with no African cells painted. The synthetic fixtures stay: they are the fast,
deterministic check, and `tests/test_spatial.py` keeps both facts executable.

Drawing it also bought the project its sharpest bug, which no test had caught because
no test could: the elevation term was being extrapolated 2.3 km past its support and
made the mountains the hottest ground in Iberia in July. See the map-page section
above. The general lesson is worth keeping — the figure was the test. A covariate that
is harmless in-sample across twenty low-lying cities is not harmless when evaluated on
terrain none of them occupy.

**The archive fetch is complete. The record is 1950-01-01 to 2026-11-15** — 28,078
days, 76.9 years, every one of the twenty cities, 0.00% missing on every column. The
whole pipeline was re-run in order behind it (refetch, data-quality, retrain,
re-evaluate, then figures and `reports/trends.json`), so models, metrics, figures and
artifacts are consistent with each other; `/api/status` reports 60 models, `ready:
true` and no problems.

**Both of the caveats the earlier notes carried are now discharged, and neither was
discharged by weakening the thing that raised it.**

- **The data-quality gate passes.** `validate.MIN_YEARS` is still 60 and the record
  is 76.9, so every city passes on the gate as originally written — the threshold was
  never moved to fit the data. `reports/metrics.json` has `provisional: false` on all
  60 entries and the UI's provisional callouts no longer render. The templates keep
  the branch that draws them, because it is the thing that would speak up again on a
  partial refetch.
- **1950–1977 is present**, so the fit no longer starts after the mid-century plateau.
  This is what cut the measured rates from +0.22…+0.56 °C/decade to +0.14…+0.35, and
  it is the finding the trends page now reports: the warming is unambiguous on all
  twenty cities, and the shorter window overstated its rate by roughly two thirds.
  The rates remain consistent with AR6 having Mediterranean land warm faster than the
  global mean, and the page still labels the extrapolation as statistical, not
  projection.

**Accuracy figures are therefore publishable for the first time**, and the README
quotes them from `reports/metrics.json` rather than from memory: next-day temperature
MAE 0.77–1.35 °C (median 1.04), beating day-of-year climatology by 43–55% and
persistence by 5.5–11.3%, on 20 of 20 cities against both baselines; next-day
precipitation Brier 0.135–0.217, 9.3–29.1% better than climatology, 20 of 20 against
both; seasonal MAE 1.41–2.57 °C with 80% intervals covering 78.1–80.1% of outcomes.
Persistence remains the real opponent and the margin over it is single-digit, which
is the number to quote honestly rather than the 50% over climatology.

`docs/legacy-inventory.md` records what the four retired scripts contained: the ideas
worth reusing and the bugs that must not return. Read it before reimplementing anything
that sounds familiar.

## Audience

The author is a second-year Data Science student building this for internship
applications, and needs to be able to defend every technical decision in an interview.
When you make a non-obvious choice, say why in one line. Prefer a small number of
well-understood models over an ensemble nobody can explain.

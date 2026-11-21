# wepr — Weather Prediction for Huesca & Valencia

A Flask web app that serves temperature and precipitation estimates for two Spanish
cities from models trained on historical station data. Originally an EPQ artefact,
now being rebuilt as a portfolio project.

## Running it

```bash
.venv/bin/python app.py     # serves on http://wepr.localhost:8000
```

Port **8000**, not 5000 — macOS Monterey's AirPlay Receiver can claim 5000.
Local hostname is `wepr.localhost` (see `docs/local-domain.md`).

## Ground truth about the data

Both CSVs are **NOAA GHCN-Daily** station exports in **°F and inches**. The existing
conversions are correct and must not be "fixed":

```python
celsius = (fahrenheit - 32) * 5 / 9
cm      = inches * 2.54
```

The two files have **different schemas** — code must not assume they are interchangeable:

| File | Station | Range | Columns | Usable |
|---|---|---|---|---|
| `weather_huesca.csv` | SPE00120161 | 1990-01-01 → 2024-03-31 | PRCP, TMAX, TMIN (+lat/lon/elev) | all; <0.1% missing |
| `weather_vlc.csv` | SPM00008284 | 1990-01-01 → 2024-03-01 | PRCP, SNWD, TAVG, TMAX, TMIN | **TAVG only** |

Valencia missingness: TMIN 43.3%, TMAX 35.1%, PRCP 16.5%, SNWD 99.9%.

Consequences:
- **Huesca is the primary city.** It has no TAVG, so derive it as `(TMAX + TMIN) / 2`
  *before* the °F→°C conversion.
- **Valencia is temperature-only.** Its PRCP is too sparse to model honestly. Do not
  ship a Valencia precipitation number.
- `weather_madrid.csv` and `huesca.csv` are referenced by the legacy scripts but **do not
  exist**. Madrid is out of scope.

`cleaned_data.csv`, `seismic.py` and `seismic.html` are from an unrelated earthquake
project. Leave them alone unless explicitly asked.

## The two products — keep them distinct

Conflating these is the central scientific error in the original code.

1. **Next-day forecast.** Predicts tomorrow from observations available today.
   A genuine forecast. Supervised models are appropriate.
2. **Seasonal estimate.** For an arbitrary future date (2027, 2031...), there are no
   recent observations to condition on, so the output is a climatological expectation —
   roughly "the historical norm for that day of year." It is **not** a forecast and the
   UI must never imply it is.

## Hard rules

- **No future information as a predictor.** A feature is only legal if its value is
  knowable strictly before the target timestamp. Rolling windows, expanding means and
  shifts must all be checked for this.
- **Fit on train only.** Scalers, imputers and encoders are fitted inside the training
  fold, never on the full series before splitting.
- **Every model is reported against baselines.** Always include (a) day-of-year
  historical mean and (b) persistence (tomorrow = today). A model that does not beat
  both is reported as such, not quietly dropped.
- **The evaluated model must be the deployed model.** Never publish a metric for one
  model while the app serves another. In the original code the MAE figures came from
  Ridge/RandomForest while the app served Prophet, whose accuracy was never measured.
- **Recompute feature lists after engineering features.** The original bug: `predictors`
  was captured before the rolling features were created, so they were never used and the
  reported "improvement" was an artifact of a 14-row trim.
- **Precipitation cannot be negative.** Clamp at zero, and prefer reporting rain
  probability over a spuriously precise millimetre figure.

## Code conventions

- Reusable logic lives in modules under `src/`; scripts that produce artifacts live in
  `scripts/`. No analysis logic at module import time — the Flask app must not train on
  import.
- No `plt.show()` in anything the app imports. Figures are written to `static/figures/`.
- Trained models are persisted with `joblib` to `models/` and loaded at startup.
- Metrics are written to `reports/metrics.json` so the README can quote verified numbers
  instead of remembered ones.
- Dependencies are pinned in `requirements.txt`.

## Repo hygiene

Never commit: `.venv/` (700 MB), `__pycache__/`, `.DS_Store`, `*.icloud` stubs, or
`13-Formulario_autoriz_de_regreso.pdf` — an unrelated personal document that happens to
sit in this folder. All are in `.gitignore`; keep them there.

## Audience

The author is a second-year Data Science student building this for internship
applications, and needs to be able to defend every technical decision in an interview.
When you make a non-obvious choice, say why in one line. Prefer a small number of
well-understood models over an ensemble nobody can explain.

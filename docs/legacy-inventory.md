# Legacy script inventory

Phase 0 retired four scripts from the project root. They are recoverable from git
history at commit `773aed2` and earlier. This file records **what was in them**, so the
rebuild can reuse the good ideas and avoid repeating the bugs.

| Script | Read | Status |
|---|---|---|
| `test.py` | `weather_madrid.csv` (missing) | retired — superseded |
| `test2.py` | `weather_madrid.csv` (missing) | retired — superseded |
| `weather_prediction.py` | `weather_madrid.csv` (missing) | retired — superseded |
| `huesca.py` | `huesca.csv` (missing) | retired — superseded |
| `app.py` | imported `weather_prediction` | retired — replaced by `app/` factory + `run.py` |

All four were near-duplicates of one another: the same load → target → backtest →
Prophet pipeline, diverging only in which models were fitted and which plots were drawn.
None of them ran, because every data file they referenced had been renamed or removed.

## Ideas worth carrying forward

**1. Walk-forward backtest.** An expanding training window advanced in fixed steps:

```python
for i in range(start, weather.shape[0], step):   # start=3650, step=90
    train = weather.iloc[:i, :]
    test  = weather.iloc[i:(i + step), :]
```

This is the right evaluation shape for a time series — train only on the past, never
shuffle. It becomes `src/wepr/evaluate.py`. What it lacked: baselines to compare against,
and any aggregation other than a mean of per-fold MAEs (which silently weights a short
final fold the same as a full one).

**2. Rolling and seasonal features.** Rolling means at 3- and 14-day horizons, the ratio
of each day to its rolling mean, and expanding per-month / per-day-of-year means. The
expanding mean is the genuinely clever part: at each row it averages only that calendar
day's *prior* years, so it is leakage-free by construction. These become
`src/wepr/features.py`, each with an explicit check that the value is knowable strictly
before the target timestamp.

**3. Two-step precipitation model.** A classifier for *whether* it rains times a
regressor for *how much*, fitted on wet days only:

```python
rain_prob   = rf_rain_occurrence.predict_proba(X)[:, 1]
rain_amount = rf_rain_amount.predict(X)
return rain_prob * rain_amount
```

This is the standard way to handle a zero-inflated target and it is a good instinct.
Carried into `src/wepr/models/nextday.py` — with the rain *probability* surfaced in the
UI, since that is the honest output, rather than a product that is neither a probability
nor an expected depth.

**4. Feature-importance reporting.** Worth keeping as a figure in `reports/`.

## Bugs that must not return

- **`predictors` computed before feature engineering.** The list was captured once, then
  rolling and seasonal columns were created afterwards, so the "updated" MAE was fitted on
  the *same* features as the first run. The apparent improvement came from
  `weather.iloc[max(rolling_horizons):, :]` trimming 14 rows, not from the new features.
  Recompute feature lists after engineering.
- **Evaluated model ≠ served model.** MAE was printed for Ridge/RandomForest while
  `get_predictions()` served Prophet, whose accuracy was never measured.
- **Next-day model used for arbitrary future dates.** `get_predictions('2031-06-21')`
  called a model trained to map today's observations to tomorrow's. For a date years out
  there are no observations to condition on; the honest output is climatology.
- **`weather.ffill()` after `shift(-1)`.** Forward-filling the target column copies the
  last real observation into the final row, inventing a label.
- **`pct_diff` divides by a rolling mean** that can be zero for precipitation, yielding
  `inf`, which `fillna(0)` does not catch.
- **No negative-precipitation clamp.** Ridge happily predicts negative rainfall.
- **`render_template('result.html')`** — the file was `results.html`. The route raised
  `TemplateNotFound` on every submission.

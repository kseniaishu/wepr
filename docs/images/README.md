# README images

Images the top-level `README.md` embeds. Two kinds, maintained differently.

## Figures — copied by the script

`map_july.png`, `map_january.png`, `skill.png`, `validation_huesca.png`,
`trend_madrid.png`, `seasonal.png`.

`app/static/figures/` is gitignored, because the figures are generated artifacts
and `scripts/make_figures.py` reproduces them from the archive. But GitHub cannot
render an image that is not in the repository, so the handful the README embeds
are copied here and committed. Along with `data/grid/elevation.npz` and the
station CSVs in `data/validation/`, these are the only generated files under
version control, and they exist for the README rather than for the app.

**They are copied by the script, not by hand.** `make_figures.py` refreshes them
on every run from the `README_IMAGES` mapping near the top of the file. A list of
`cp` commands in a readme rots the first time someone regenerates the figures and
forgets to run it, and a README showing a figure that no longer matches the
numbers printed beside it is worse than a README showing no figure at all.

To change the set, edit `README_IMAGES` and re-run:

```bash
.venv/bin/python scripts/make_figures.py
```

The script skips any figure the current run did not produce, so a `--city` run, or
a map skipped for a missing elevation grid, leaves the existing copies alone
rather than deleting them or failing.

## Screenshots — captured by hand

`ui-home.png`, `ui-city.png`, `ui-map.png`, `ui-trends.png`: the running app at
1280 px wide, dark scheme.

These cannot be script-maintained, so they are the one thing here that can drift
from what the app actually looks like. Re-capture them after any visual change.
Each shows a specific claim and is worth keeping on-message:

| File | What it has to show |
|---|---|
| `ui-home.png` | the two products kept apart, and the data-quality gate line |
| `ui-city.png` | a dated forecast with both baselines beside it |
| `ui-map.png` | the month switcher, and the note on why there is no year control |
| `ui-trends.png` | the forest plot, measured rates separate from the scenarios |

To re-capture, run the app and take a full-page screenshot of each route at
1280 px:

```bash
.venv/bin/python run.py    # then / , /city/seville , /map , /trends
```

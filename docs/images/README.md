# README assets

Images the top-level `README.md` embeds. Two kinds.

## Committed copies of generated figures

`scripts/make_figures.py` writes its PNGs to `app/static/figures/`, which is
gitignored — they are generated artifacts and are reproduced by the script. But
GitHub cannot render an image that is not in the repository, so the handful the
README embeds are copied here and committed.

Refresh them after regenerating:

```bash
cp app/static/figures/skill.png             docs/images/skill.png
cp app/static/figures/seasonal.png          docs/images/seasonal.png
cp app/static/figures/validation_huesca.png docs/images/validation_huesca.png
cp app/static/figures/trend_madrid.png      docs/images/trend_madrid.png
cp app/static/figures/map_tavg_07.png       docs/images/map_july.png
cp app/static/figures/map_tavg_01.png       docs/images/map_january.png
```

## Screenshots of the running app

`ui-home.png`, `ui-city.png`, `ui-trends.png` — full-page captures at 1280 px wide,
in the site's dark theme.

Taken with headless Chrome against a local server, using a throwaway profile so the
real one is untouched. Start the app, then for each page:

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless=new --disable-gpu --hide-scrollbars \
  --user-data-dir="$(mktemp -d)" \
  --window-size=1280,2353 \
  --screenshot=docs/images/ui-home.png \
  http://localhost:8000/
```

The window height is the page's `document.documentElement.scrollHeight`, so the
capture is the whole page rather than a viewport crop. `/trends` is ~7,300 px tall,
which is unusable in a README, so it is deliberately captured at 1800 px — the top
of the page through the forest plot.

Chrome does not exit on its own after `--screenshot` here; kill it once the file is
written.

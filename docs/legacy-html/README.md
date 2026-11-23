# The original EPQ artefact, as it was

This directory is an archive, not code. Nothing here is imported, served or
tested; it is kept so the rebuild can be compared against what it replaced.

| File | What it was |
|---|---|
| `index.html`, `weather.html` | the original static site |
| `static/styles.css` | its stylesheet |
| `*.jpg`, `*.jpeg` | its images |
| `flask-index.html.jinja` | `templates/index.html` from the original Flask app |
| `flask-results.html.jinja` | `templates/results.html` from the same |

The two Jinja templates are suffixed `.jinja` and *not* left in
`app/templates/`, because a template sitting in the live app's template
directory looks like a template the live app renders. It did not, and it
cannot: it refers to a `weather_prediction` module that Phase 0 retired. The
same reasoning moved the images and the stylesheet out of `app/static/`, where
Flask would have served them to the public from a URL no page links to.

`index.html` references `sun.jpeg`, `flowers_bottom.jpg` and
`static/styles.css` as siblings, which is why they are laid out this way here —
the archive renders if you open it, rather than being a set of files that once
rendered somewhere else.

What the retired Python scripts contained — the ideas worth reusing and the
bugs that must not return — is in [`../legacy-inventory.md`](../legacy-inventory.md).

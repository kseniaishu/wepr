"""The city registry — the single source of truth for what the app covers.

Twenty cities chosen for climatic spread rather than population: Atlantic and
Mediterranean, coastal and interior, flat terrain and broken. That spread is the
point. A model that works in Seville and A Coruña alike has been tested against
genuinely different climates, which is a stronger claim than twenty variations
on the same one.

**The `Altitude` region means broken terrain, not high ground.** It is the one
label whose name could mislead, so: the criterion is how much the terrain *varies*
inside a grid cell, not how high the city sits. A ~9 km cell reports one elevation
for everything inside it, so where relief is steep that single number represents
nothing very well, and the reanalysis inherits the error. Granada sits in a basin
under the Sierra Nevada (3,479 m) and Huesca at the Pyrenean foothills, so both are
flagged `Altitude` — even though, on the elevations the archive actually served,
Salamanca (821 m), Valladolid (698 m), Bragança (682 m) and Madrid (666 m) are all
*higher* than Huesca (471 m). Those four sit on the flat meseta, where one
elevation per cell is a fair summary, so they are `Interior`. Sorting by altitude
would group the wrong cities together and say nothing about where the data is
weakest.

Elevation is deliberately **not** hard-coded. Open-Meteo returns the elevation of
the grid cell it actually served, which is the number that matters for
interpreting the data, and in broken terrain it can differ from the city's official
altitude by tens of metres or more — Granada's cell came back at 697 m against a
city centre nearer 740 m. `wepr.data.fetch` records it alongside the
series.

A city appearing here is a *candidate*. It only enters the app after passing the
data-quality gate in `reports/data_quality.md`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class City:
    """A location the app can serve.

    Attributes:
        slug: URL-safe identifier, stable forever — it appears in API paths and
            in model filenames, so renaming one invalidates cached artifacts.
        name: Display name.
        country: ISO 3166-1 alpha-2.
        lat: Latitude, decimal degrees north.
        lon: Longitude, decimal degrees east (negative for west).
        region: Coarse climate grouping, used for figure grouping and for
            reporting skill by regime rather than only in aggregate. `Altitude`
            denotes broken terrain rather than high ground — see the module
            docstring, which is where that distinction is argued.
    """

    slug: str
    name: str
    country: str
    lat: float
    lon: float
    region: str


CITIES: tuple[City, ...] = (
    # --- Atlantic (north and northwest): mild, wet, maritime -----------------
    City("a-coruna", "A Coruña", "ES", 43.3623, -8.4115, "Atlantic"),
    City("oviedo", "Oviedo", "ES", 43.3619, -5.8494, "Atlantic"),
    City("bilbao", "Bilbao", "ES", 43.2630, -2.9350, "Atlantic"),
    City("porto", "Porto", "PT", 41.1579, -8.6291, "Atlantic"),
    City("lisbon", "Lisbon", "PT", 38.7223, -9.1393, "Atlantic"),
    # --- Mediterranean coast: hot dry summers, mild wet winters -------------
    City("barcelona", "Barcelona", "ES", 41.3874, 2.1686, "Mediterranean"),
    City("valencia", "Valencia", "ES", 39.4699, -0.3763, "Mediterranean"),
    City("murcia", "Murcia", "ES", 37.9922, -1.1307, "Mediterranean"),
    City("malaga", "Málaga", "ES", 36.7213, -4.4214, "Mediterranean"),
    City("faro", "Faro", "PT", 37.0194, -7.9304, "Mediterranean"),
    City("palma", "Palma", "ES", 39.5696, 2.6502, "Mediterranean"),
    # --- Interior plateau: continental, large annual range ------------------
    City("madrid", "Madrid", "ES", 40.4168, -3.7038, "Interior"),
    City("zaragoza", "Zaragoza", "ES", 41.6488, -0.8891, "Interior"),
    City("valladolid", "Valladolid", "ES", 41.6523, -4.7245, "Interior"),
    City("salamanca", "Salamanca", "ES", 40.9701, -5.6635, "Interior"),
    City("badajoz", "Badajoz", "ES", 38.8794, -6.9707, "Interior"),
    City("seville", "Seville", "ES", 37.3891, -5.9845, "Interior"),
    City("braganca", "Bragança", "PT", 41.8061, -6.7567, "Interior"),
    # --- Broken terrain: steep relief inside the grid cell, so the hardest
    # --- cases for reanalysis. Not simply the highest cities — see the module
    # --- docstring; four `Interior` cities above sit higher than Huesca. -----
    City("granada", "Granada", "ES", 37.1773, -3.5986, "Altitude"),
    City("huesca", "Huesca", "ES", 42.1362, -0.4087, "Altitude"),
)

BY_SLUG: dict[str, City] = {c.slug: c for c in CITIES}

# The two cities with independent GHCN station records on disk, so reanalysis can
# be checked against real observations. See `wepr.data.validate`.
VALIDATION_SLUGS: tuple[str, ...] = ("huesca", "valencia")


def get(slug: str) -> City:
    """Look up a city by slug, raising a clear error for an unknown one.

    Raises:
        KeyError: if `slug` is not a registered city.
    """
    try:
        return BY_SLUG[slug]
    except KeyError:
        known = ", ".join(sorted(BY_SLUG))
        raise KeyError(f"unknown city {slug!r}; known slugs: {known}") from None


def regions() -> tuple[str, ...]:
    """Distinct regions, in first-appearance order."""
    seen: dict[str, None] = {}
    for c in CITIES:
        seen.setdefault(c.region, None)
    return tuple(seen)

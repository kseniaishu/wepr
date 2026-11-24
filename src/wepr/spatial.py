"""Interpolation of city values onto a map grid.

The map page shows a continuous field rather than twenty dots, which means
interpolating between cities and clipping the result to the coastline so no
colour is painted over the sea.

**The honesty constraint.** Interpolating twenty points across Iberia produces a
smooth picture that implies far more spatial resolution than twenty cities
support, particularly across the Pyrenees and the Cantabrian range where the real
field changes over kilometres. Three things answer that here, and all three are
returned by `interpolate_field` so the page can render them rather than having to
trust a caption:

1. The city markers stay visible on top, so a reader can see how sparse the
   evidence is.
2. `nearest_city_km` gives the distance from every cell to the closest city, so
   confidence can be shaded by it.
3. `interpolated` marks the cells that lie inside the convex hull of the cities.
   Outside the hull nothing is being interpolated — it is extrapolation, and the
   page shows it faded rather than solid.

**Why residuals are interpolated, not values.** Across these twenty cities a
linear surface in elevation, latitude and longitude predicts annual mean
temperature to a leave-one-out MAE of 0.45 °C, against 1.95 °C for the city mean
alone — elevation and position explain most of the between-city spread, and the
fitted lapse rate (about -4.4 °C/km) is physically sensible. So the field is built
as that surface plus inverse-distance-weighted *residuals*. Plain
distance-weighting of raw values would paint the Ebro valley's temperature onto
2,500 m Pyrenean cells, which is not a resolution problem but a physics one.

The surface design — elevation, latitude and longitude, no interactions — was
chosen by leave-one-out cross-validation across the twenty cities rather than by
in-sample fit, and it wins or ties for every field tested (annual mean, July mean,
January mean, annual precipitation). One design serves every field instead of four
special cases.

**Elevation is the one covariate that needs a guard, and July is why.** Its fitted
coefficient is sound on the annual mean (-3.81 °C/km, correlation -0.65 with the
city values) and steeper but correctly signed in January (-7.72). In July it is
*positive*, +2.3 °C/km, against a raw correlation with elevation of exactly +0.00 —
Iberian summer heat is continental, not elevation-controlled, so once lat and lon
absorb the north-south gradient the leftover elevation spread tracks continentality
instead (Atlantic cities at sea level and 20.2 °C, Interior ones at 475 m and
25.4 °C).

An earlier note here said elevation "contributes almost nothing in July, but it
costs nothing either". The first half was right and the second was wrong, and only
drawing the map could show it: the cities span 0-776 m while the grid reaches
3,064 m, so that coefficient was carried 2.3 km beyond its support and added
+5.3 °C to the highest cells — painting the Sierra Nevada and the Pyrenees as the
hottest ground in Iberia on a July map. In-sample it cost nothing; out-of-sample,
on exactly the terrain the map draws, it cost five degrees. `TrendSurface.predict`
therefore clips elevation to the range it was fitted on. See its docstring.

**What the land mask can and cannot do.** It comes from Open-Meteo's elevation
endpoint, which is the same digital elevation model the archive reports for the
cell it served — a city fetch and a bare elevation lookup at Madrid both return
666.0 m — so the map's terrain and the data's terrain cannot disagree. But the
endpoint clamps at zero instead of returning bathymetry, so land at or below sea
level is indistinguishable from water: the Guadalquivir marshes and the Valencian
Albufera read as sea and go unpainted. That is a handful of cells on the lowest
coastal flats, and it errs toward not painting, which is the right direction for a
map that must not overclaim.

Implemented in Phase 5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wepr import paths

# The grid. 0.1° is not a free choice: it is ERA5-Land's own cell size, and the
# archive serves cell centres on exact multiples of 0.1 (Madrid came back at
# 40.400, -3.700). Interpolating onto a finer grid would invent structure the
# source does not have; onto a coarser one would throw away a coastline.
RESOLUTION = 0.1

# Iberia plus the Balearics, wide enough to hold every city's cell with room for
# the coastline: Cabo da Roca at -9.5, Menorca at 4.3, Tarifa at 36.0, the
# Cantabrian coast at 43.8.
BOUNDS = (36.0, 44.0, -9.9, 4.4)

# How far beyond the convex hull of the cities a cell may sit and still be drawn
# at all (faded, never solid).
#
# Distance alone cannot decide this, and the measurement that proves it is worth
# keeping: on the 0.1° grid, Tarifa in Spain is 90.9 km outside the hull while
# Tetouan in Morocco is 89.0 km. The Strait of Gibraltar is 14 km wide, so no
# hull-distance threshold separates southern Spain from north Africa. Morocco is
# therefore excluded by **land connectivity** instead — see `connected_to_points` —
# and this margin only has to keep France out.
#
# At 95 km the trade is explicit: all of mainland Iberia is drawn except the
# Figueres corner (117 km); France is excluded except a sliver around Bayonne
# (77 km), which is nearer to Bilbao than much of interior Spain is to any city,
# so interpolating it is not the overreach it looks like. Menorca is excluded both
# by distance (131 km) and by connectivity, which is consistent: no city is on it.
HULL_MARGIN_KM = 95.0

# Inverse-distance weighting exponent. Two is the conventional Shepard value, and
# the choice matters much less here than it would on raw values, because what is
# being weighted is a residual with a standard deviation under a degree.
IDW_POWER = 2.0

EARTH_RADIUS_KM = 6371.0088


class SpatialError(RuntimeError):
    """Raised when a grid or a surface cannot be built from what was given."""


@dataclass(frozen=True, slots=True)
class ElevationGrid:
    """The cached terrain the map is drawn on.

    Attributes:
        lats: Grid latitudes, ascending, length ``n_lat``.
        lons: Grid longitudes, ascending, length ``n_lon``.
        elevation: Metres above sea level, shape ``(n_lat, n_lon)``. Sea is 0.0.
        elevation: Metres above sea level, shape ``(n_lat, n_lon)``. Sea is 0.0,
            and **NaN means never fetched** — the artifact deliberately covers only
            the region the map can draw, so cells far outside the city hull carry
            no terrain at all. `land_mask` treats NaN as not-land, which is right:
            an unknown cell must not be painted.
        resolution: Grid spacing in degrees.
        margin_km: How far beyond the city hull the artifact actually has terrain
            for. `interpolate_field` refuses to draw further out than this rather
            than silently rendering a coastline that was never fetched.
        source: Where the values came from, carried so an artifact can say so.
    """

    lats: np.ndarray
    lons: np.ndarray
    elevation: np.ndarray
    resolution: float
    margin_km: float
    source: str

    @property
    def land(self) -> np.ndarray:
        """Boolean land mask — see the module docstring on its one failure mode."""
        return land_mask(self.elevation)


@dataclass(frozen=True, slots=True)
class TrendSurface:
    """A linear surface in elevation, latitude and longitude.

    Attributes:
        coefficients: ``[intercept, per_metre, per_degree_lat, per_degree_lon]``.
        r2: In-sample coefficient of determination. Reported, but the design was
            not chosen on it — see `fit_surface`.
        residual_sd: Standard deviation of the city residuals, in the field's own
            units. This is the magnitude the distance-weighting step is left to
            explain, and the honest measure of how much of the map is physics.
        elevation_range: The ``(min, max)`` elevation the surface was fitted on.
            Cells outside it are extrapolating the lapse rate, which
            `interpolate_field` records rather than hides.
        n_cities: How many cities the fit used.
    """

    coefficients: np.ndarray
    r2: float
    residual_sd: float
    elevation_range: tuple[float, float]
    n_cities: int

    @property
    def lapse_rate_c_per_km(self) -> float:
        """The fitted elevation term per kilometre, the readable form."""
        return float(self.coefficients[1] * 1000.0)

    @property
    def per_degree_lat(self) -> float:
        return float(self.coefficients[2])

    @property
    def per_degree_lon(self) -> float:
        return float(self.coefficients[3])

    def predict(
        self, elevation: np.ndarray, lats: np.ndarray, lons: np.ndarray
    ) -> np.ndarray:
        """Evaluate the surface, with elevation clipped to its fitted support.

        **Elevation is clamped to `elevation_range`; latitude and longitude are
        not.** The cities span 0-776 m but the grid reaches 3,064 m, so the
        elevation term is the one covariate asked to reach far outside the data
        that set it — 2.3 km past the highest city, against a few tenths of a
        degree for lat and lon, which the hull margin bounds anyway.

        Without the clamp the July map is visibly wrong, and wrong in the exact
        way the residual design exists to prevent. Across these twenty cities
        July temperature has **zero** correlation with elevation (+0.00): Iberian
        summer heat is continental, not elevation-controlled. The fit still
        assigns elevation +2.3 °C/km, because once lat and lon absorb the
        north-south gradient the residual elevation spread aligns with
        continentality — the Atlantic cities sit at sea level and 20.2 °C, the
        Interior ones average 475 m and 25.4 °C. Carried to 3,064 m that
        coefficient added +5.3 °C and painted the Sierra Nevada and the Pyrenees
        as the hottest ground in Iberia in July.

        Clamping says the honest thing instead: above the highest city the fit
        has no evidence about elevation, so the surface stops varying with it and
        the distance-weighted residuals carry the cell alone. `Field.
        extrapolates_elevation` already marks exactly these cells, and the map
        already hatches them — that hatching now means "held flat here" rather
        than "extrapolated here", which is the stronger claim to be able to make.

        Nothing inside the fitted range changes, so the annual-mean result the
        design was chosen on (-3.81 °C/km, leave-one-out MAE 0.45 °C) is
        untouched, and city residuals are unaffected because every city is
        within the range by construction.
        """
        c = self.coefficients
        lo, hi = self.elevation_range
        supported = np.clip(elevation, lo, hi)
        return c[0] + c[1] * supported + c[2] * lats + c[3] * lons


@dataclass(frozen=True, slots=True)
class Field:
    """An interpolated field and everything needed to render it honestly.

    Attributes:
        lats: Grid latitudes, ascending.
        lons: Grid longitudes, ascending.
        values: The field, shape ``(n_lat, n_lon)``, NaN wherever nothing should
            be drawn — sea, and land too far from any city to justify a colour.
        interpolated: True for cells inside the convex hull of the cities. These
            are interpolated; the remaining non-NaN cells are extrapolated and the
            page draws them faded.
        nearest_city_km: Great-circle distance from each cell to the closest city.
        extrapolates_elevation: True where the cell's elevation lies outside the
            range the surface was fitted on, so the lapse rate is being applied
            beyond the evidence. True across the high Pyrenees and Sierra Nevada.
        surface: The fitted trend surface, or None if there were too few cities
            and the field is plain distance-weighting of raw values.
        units: The field's units, carried through for the colour-bar label.
        clamped_at_zero: Whether negatives were clamped, as precipitation requires.
    """

    lats: np.ndarray
    lons: np.ndarray
    values: np.ndarray
    interpolated: np.ndarray
    nearest_city_km: np.ndarray
    extrapolates_elevation: np.ndarray
    surface: TrendSurface | None
    units: str
    clamped_at_zero: bool

    @property
    def drawn(self) -> np.ndarray:
        """Cells carrying a value at all, interpolated or extrapolated."""
        return np.isfinite(self.values)

    def summary(self) -> dict:
        """A JSON-safe digest, for the API and for figure captions."""
        drawn = self.drawn
        s = self.surface
        return {
            "resolution_deg": (
                round(float(self.lons[1] - self.lons[0]), 4) if len(self.lons) > 1 else None
            ),
            "units": self.units,
            "cells_drawn": int(drawn.sum()),
            "cells_interpolated": int((self.interpolated & drawn).sum()),
            "cells_extrapolated": int((~self.interpolated & drawn).sum()),
            "value_min": float(np.nanmin(self.values)) if drawn.any() else None,
            "value_max": float(np.nanmax(self.values)) if drawn.any() else None,
            "max_distance_to_city_km": (
                round(float(self.nearest_city_km[drawn].max()), 1) if drawn.any() else None
            ),
            "clamped_at_zero": self.clamped_at_zero,
            "surface": None
            if s is None
            else {
                "lapse_rate_c_per_km": round(s.lapse_rate_c_per_km, 2),
                "per_degree_lat": round(s.per_degree_lat, 3),
                "per_degree_lon": round(s.per_degree_lon, 3),
                "r2": round(s.r2, 3),
                "residual_sd": round(s.residual_sd, 3),
                "fitted_elevation_range_m": [
                    round(s.elevation_range[0], 1),
                    round(s.elevation_range[1], 1),
                ],
                "n_cities": s.n_cities,
            },
        }


# -- the grid -------------------------------------------------------------------


def grid_axes(
    resolution: float = RESOLUTION,
    bounds: tuple[float, float, float, float] = BOUNDS,
) -> tuple[np.ndarray, np.ndarray]:
    """The map's latitude and longitude axes.

    Snapped to exact multiples of `resolution` so the grid lands on ERA5-Land's
    own cell centres instead of halfway between them.

    Args:
        resolution: Spacing in degrees.
        bounds: ``(lat_min, lat_max, lon_min, lon_max)``.

    Returns:
        Ascending latitudes and longitudes.

    Raises:
        SpatialError: If the resolution is not positive or the bounds are empty.
    """
    if resolution <= 0:
        raise SpatialError(f"resolution must be positive, got {resolution}")
    lat_min, lat_max, lon_min, lon_max = bounds
    if lat_max <= lat_min or lon_max <= lon_min:
        raise SpatialError(f"empty bounds {bounds}")

    def axis(lo: float, hi: float) -> np.ndarray:
        # Build on integer steps, so 0.1 spacing does not drift by floating point
        # over 140 cells and leave the last one off-grid.
        first = int(np.ceil(round(lo / resolution, 6)))
        last = int(np.floor(round(hi / resolution, 6)))
        return np.arange(first, last + 1) * resolution

    return axis(lat_min, lat_max), axis(lon_min, lon_max)


def haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Great-circle distance in kilometres between broadcastable coordinates.

    Great-circle rather than planar degrees because Iberia spans eight degrees of
    latitude: a degree of longitude is 89 km at Tarifa and 80 km at Bilbao, so
    weighting by degree distance would quietly stretch the north.
    """
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def land_mask(elevation: np.ndarray) -> np.ndarray:
    """Land where elevation is above zero.

    The endpoint clamps at zero rather than returning bathymetry, so this cannot
    distinguish water from land at or below sea level — the Guadalquivir marshes
    and the Albufera read as sea. See the module docstring; the error direction is
    deliberately toward leaving cells unpainted.

    NaN — a cell the artifact never fetched — is not land either, for the same
    reason: ``NaN > 0`` is False, so an unknown cell goes unpainted rather than
    being guessed at.
    """
    with np.errstate(invalid="ignore"):
        return np.asarray(elevation, dtype=float) > 0.0


def load_elevation(path=None) -> ElevationGrid:
    """Load the cached elevation grid written by `scripts/fetch_elevation.py`.

    Args:
        path: Override for the artifact location.

    Returns:
        The grid.

    Raises:
        SpatialError: If the artifact is missing, with the command that builds it.
    """
    path = path or paths.ELEVATION_GRID
    if not path.exists():
        raise SpatialError(
            f"no elevation grid at {path}. Build it once with:\n"
            f"    .venv/bin/python scripts/fetch_elevation.py"
        )
    with np.load(path, allow_pickle=False) as z:
        lats = z["lats"]
        lons = z["lons"]
        elevation = z["elevation"]
        resolution = float(z["resolution"])
        margin_km = float(z["margin_km"])
        source = str(z["source"])
    if elevation.shape != (len(lats), len(lons)):
        raise SpatialError(
            f"elevation grid at {path} is {elevation.shape}, expected "
            f"{(len(lats), len(lons))} — rebuild it"
        )
    return ElevationGrid(lats, lons, elevation, resolution, margin_km, source)


# -- the trend surface ----------------------------------------------------------


def fit_surface(
    elevation: np.ndarray, lats: np.ndarray, lons: np.ndarray, values: np.ndarray
) -> TrendSurface:
    """Least-squares fit of ``value ~ elevation + lat + lon``.

    The design is fixed rather than selected per field, and it was chosen by
    leave-one-out cross-validation over the twenty cities, not by in-sample
    R² — with twenty points and four parameters, in-sample fit would reward
    complexity the record cannot support. On the annual mean it reaches a
    leave-one-out MAE of 0.45 °C against 1.95 °C for the city mean alone.

    Args:
        elevation: Metres per city.
        lats: Latitudes per city.
        lons: Longitudes per city.
        values: The field value per city.

    Returns:
        The fitted surface.

    Raises:
        SpatialError: If fewer than five finite cities are given — four
            parameters need more than four points to mean anything — or if the
            design is rank-deficient.
    """
    elevation, lats, lons, values = (
        np.asarray(v, dtype=float).ravel() for v in (elevation, lats, lons, values)
    )
    if not (len(elevation) == len(lats) == len(lons) == len(values)):
        raise SpatialError("elevation, lats, lons and values must be the same length")

    ok = np.isfinite(elevation) & np.isfinite(lats) & np.isfinite(lons) & np.isfinite(values)
    if ok.sum() < 5:
        raise SpatialError(
            f"need at least 5 cities with finite values to fit a 4-parameter "
            f"surface, got {int(ok.sum())}"
        )

    design = np.column_stack([np.ones(ok.sum()), elevation[ok], lats[ok], lons[ok]])
    target = values[ok]
    coefficients, _residues, rank, _sv = np.linalg.lstsq(design, target, rcond=None)
    if rank < design.shape[1]:
        raise SpatialError(
            f"city coordinates are degenerate (design rank {rank} < 4); a surface "
            f"in elevation, latitude and longitude is not identifiable from them"
        )

    fitted = design @ coefficients
    residuals = target - fitted
    spread = float(((target - target.mean()) ** 2).sum())
    r2 = 1.0 - float((residuals**2).sum()) / spread if spread > 0 else 0.0
    return TrendSurface(
        coefficients=coefficients,
        r2=r2,
        residual_sd=float(residuals.std(ddof=0)),
        elevation_range=(float(elevation[ok].min()), float(elevation[ok].max())),
        n_cities=int(ok.sum()),
    )


# -- distance weighting ---------------------------------------------------------


def idw(
    lats: np.ndarray,
    lons: np.ndarray,
    values: np.ndarray,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    power: float = IDW_POWER,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-distance-weighted field, plus distance to the nearest point.

    Shepard weighting, chosen over a spline because it is bounded by the observed
    range: it cannot produce a cell warmer than the warmest city, which a
    thin-plate spline across twenty sparse points very much can.

    Args:
        lats: Point latitudes.
        lons: Point longitudes.
        values: Value per point. NaN entries are dropped.
        grid_lats: Target latitudes, ascending.
        grid_lons: Target longitudes, ascending.
        power: Shepard exponent.

    Returns:
        The weighted field and the nearest-point distance in km, both shaped
        ``(len(grid_lats), len(grid_lons))``.

    Raises:
        SpatialError: If no point carries a finite value.
    """
    lats, lons, values = (np.asarray(v, dtype=float).ravel() for v in (lats, lons, values))
    ok = np.isfinite(lats) & np.isfinite(lons) & np.isfinite(values)
    if not ok.any():
        raise SpatialError("no point carries a finite value to interpolate from")
    lats, lons, values = lats[ok], lons[ok], values[ok]

    mesh_lat, mesh_lon = np.meshgrid(grid_lats, grid_lons, indexing="ij")
    # (n_points, n_lat, n_lon)
    distance = haversine_km(
        lats[:, None, None], lons[:, None, None], mesh_lat[None], mesh_lon[None]
    )
    nearest = distance.min(axis=0)

    # A cell sitting exactly on a city would divide by zero. Those cells take the
    # city's own value, which is what any sane interpolator should do there.
    coincident = distance <= 0.0
    safe = np.where(coincident, 1.0, distance)
    weight = safe ** (-power)
    weight = np.where(coincident.any(axis=0)[None], np.where(coincident, 1.0, 0.0), weight)

    field = (weight * values[:, None, None]).sum(axis=0) / weight.sum(axis=0)
    return field, nearest


def hull_distance_km(
    lats: np.ndarray, lons: np.ndarray, grid_lats: np.ndarray, grid_lons: np.ndarray
) -> np.ndarray:
    """Distance in km from each cell to the convex hull of the points, 0 inside.

    Inside the hull a value is a weighted average of points that surround the
    cell, which is interpolation. Outside it every point lies to one side, which
    is extrapolation — a different claim, and the map draws it differently.
    """
    from scipy.spatial import ConvexHull, Delaunay

    points = np.column_stack([np.asarray(lons, float).ravel(), np.asarray(lats, float).ravel()])
    mesh_lat, mesh_lon = np.meshgrid(grid_lats, grid_lons, indexing="ij")

    if len(points) < 3:
        # No hull exists; treat every cell as outside so nothing is called
        # interpolated on the strength of two points.
        return np.full(mesh_lat.shape, np.inf)

    inside = Delaunay(points).find_simplex(np.column_stack([mesh_lon.ravel(), mesh_lat.ravel()]))
    inside = (inside >= 0).reshape(mesh_lat.shape)

    vertices = points[ConvexHull(points).vertices]
    best = np.full(mesh_lat.shape, np.inf)
    for i in range(len(vertices)):
        a = vertices[i]
        b = vertices[(i + 1) % len(vertices)]
        edge = b - a
        denominator = float(edge @ edge)
        if denominator == 0.0:
            continue
        t = ((mesh_lon - a[0]) * edge[0] + (mesh_lat - a[1]) * edge[1]) / denominator
        t = np.clip(t, 0.0, 1.0)
        # Nearest point on this hull edge, then a true great-circle distance to it,
        # so the margin means the same thing in Galicia as in Murcia.
        best = np.minimum(
            best, haversine_km(mesh_lat, mesh_lon, a[1] + t * edge[1], a[0] + t * edge[0])
        )

    return np.where(inside, 0.0, best)


def connected_to_points(
    mask: np.ndarray,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    """Restrict a mask to the connected regions that contain at least one point.

    This is what keeps the map on the cities' own landmass. A hull margin wide
    enough to include Tarifa also includes Tetouan — the Strait is only 14 km
    across — so distance cannot exclude north Africa. Connectivity can: the Strait
    is water, so Morocco is a different component from the peninsula and is
    dropped regardless of how close it is.

    It cuts the right way elsewhere too. Mallorca is kept because Palma is on it;
    Menorca is dropped because no city is. France is *not* excluded by this —
    the Pyrenees are land, so France is the same component as Spain — which is
    what `HULL_MARGIN_KM` is left to handle.

    Args:
        mask: Candidate cells, shape ``(len(grid_lats), len(grid_lons))``.
        grid_lats: Grid latitudes, ascending.
        grid_lons: Grid longitudes, ascending.
        lats: Seed point latitudes.
        lons: Seed point longitudes.

    Returns:
        `mask` with every component holding no seed point removed.
    """
    from scipy import ndimage

    labels, count = ndimage.label(np.asarray(mask, dtype=bool))
    if count == 0:
        return np.zeros_like(labels, dtype=bool)

    seeds = sample(labels, grid_lats, grid_lons, lats, lons)
    keep = {int(v) for v in np.atleast_1d(seeds) if int(v) > 0}
    if not keep:
        # Every seed landed on a cell the mask excludes — a city on a sea cell, or
        # outside the margin. Returning the mask unchanged would silently restore
        # north Africa, so return nothing and let the caller notice an empty map.
        return np.zeros_like(labels, dtype=bool)
    return np.isin(labels, list(keep))


# -- the public entry points ----------------------------------------------------


def interpolate_field(
    lats: np.ndarray,
    lons: np.ndarray,
    values: np.ndarray,
    *,
    resolution: float = RESOLUTION,
    units: str = "",
    non_negative: bool = False,
    margin_km: float = HULL_MARGIN_KM,
    power: float = IDW_POWER,
    elevation: ElevationGrid | None = None,
) -> Field:
    """Interpolate city values onto the map grid, with its honesty layers.

    The field is a fitted surface in elevation, latitude and longitude plus
    inverse-distance-weighted residuals, masked to land, and masked again beyond
    `margin_km` outside the convex hull of the cities. See the module docstring
    for why each of those steps is there.

    Args:
        lats: City latitudes.
        lons: City longitudes.
        values: Value per city; NaN entries are dropped.
        resolution: Grid spacing in degrees. Must match the cached elevation grid.
        units: Units label, carried into `Field.units`.
        non_negative: Clamp the result at zero. Required for precipitation, which
            cannot be negative and which a residual fit can otherwise push below it.
        margin_km: How far outside the city hull a cell may still be drawn.
        power: Shepard exponent for the residual weighting.
        elevation: Pre-loaded grid, mostly so the app can load it once at startup
            and the tests can supply a synthetic one.

    Returns:
        The field and its masks.

    Raises:
        SpatialError: If the elevation grid is missing, does not match
            `resolution`, or does not reach as far out as `margin_km`.
    """
    grid = elevation or load_elevation()
    if not np.isclose(grid.resolution, resolution):
        raise SpatialError(
            f"elevation grid is {grid.resolution}° but {resolution}° was asked for; "
            f"rebuild the grid or pass the matching resolution"
        )
    if margin_km > grid.margin_km:
        raise SpatialError(
            f"asked to draw {margin_km:.0f} km beyond the city hull but the cached "
            f"elevation grid only covers {grid.margin_km:.0f} km. Rebuild it with "
            f"scripts/fetch_elevation.py --margin {margin_km:.0f}, or draw no "
            f"further out than it reaches."
        )
    grid_lats, grid_lons = grid.lats, grid.lons
    mesh_lat, mesh_lon = np.meshgrid(grid_lats, grid_lons, indexing="ij")

    lats = np.asarray(lats, dtype=float).ravel()
    lons = np.asarray(lons, dtype=float).ravel()
    values = np.asarray(values, dtype=float).ravel()

    # The surface needs each city's own elevation, read off the same grid the map
    # is drawn on rather than from the city record, so the residual a city
    # contributes is measured against the terrain the map will add back.
    city_elevation = sample(grid.elevation, grid_lats, grid_lons, lats, lons)

    surface: TrendSurface | None
    try:
        surface = fit_surface(city_elevation, lats, lons, values)
    except SpatialError:
        # Too few cities, or degenerate coordinates. Fall back to weighting raw
        # values and say so by leaving `surface` as None, rather than silently
        # serving a different method under the same name.
        surface = None

    if surface is None:
        field, nearest = idw(lats, lons, values, grid_lats, grid_lons, power=power)
        extrapolates_elevation = np.zeros(field.shape, dtype=bool)
    else:
        residual = values - surface.predict(city_elevation, lats, lons)
        residual_field, nearest = idw(lats, lons, residual, grid_lats, grid_lons, power=power)
        field = surface.predict(grid.elevation, mesh_lat, mesh_lon) + residual_field
        low, high = surface.elevation_range
        extrapolates_elevation = (grid.elevation < low) | (grid.elevation > high)

    clamped = False
    if non_negative:
        below = field < 0.0
        clamped = bool(below.any())
        field = np.where(below, 0.0, field)

    hull = hull_distance_km(lats, lons, grid_lats, grid_lons)
    drawable = connected_to_points(
        grid.land & (hull <= margin_km), grid_lats, grid_lons, lats, lons
    )
    field = np.where(drawable, field, np.nan)

    # A cell whose centre is marginally outside the hull but which *contains* a
    # city is not extrapolation: the hull boundary passes through it. Half a cell
    # diagonal is that tolerance, and without it the nine hull-vertex cities —
    # A Coruña, Bilbao, Palma and the rest — would each have their own cell drawn
    # faded, which would be absurd.
    half_cell_km = resolution * 111.32 * np.sqrt(2) / 2

    return Field(
        lats=grid_lats,
        lons=grid_lons,
        values=field,
        interpolated=drawable & (hull <= half_cell_km),
        nearest_city_km=nearest,
        extrapolates_elevation=extrapolates_elevation & drawable,
        surface=surface,
        units=units,
        clamped_at_zero=clamped,
    )


def sample(
    gridded: np.ndarray,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    """Read a gridded array at scattered coordinates, nearest cell.

    Nearest rather than bilinear on purpose: the elevation grid *is* a cell-mean
    terrain, and smoothing it between cells would report a Pyrenean valley floor
    as something between the valley and the ridge, which is neither.
    """
    lat_index = np.abs(np.asarray(grid_lats, float)[:, None] - np.asarray(lats, float)).argmin(axis=0)
    lon_index = np.abs(np.asarray(grid_lons, float)[:, None] - np.asarray(lons, float)).argmin(axis=0)
    return np.asarray(gridded)[lat_index, lon_index]


def interpolate(
    lats: np.ndarray,
    lons: np.ndarray,
    values: np.ndarray,
    resolution: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate scattered city values onto a regular grid.

    The plain form of `interpolate_field`, kept because this is the signature the
    rest of the project was written against. Callers that need the confidence
    layers — the hull mask, the distance field — should use `interpolate_field`.

    Args:
        lats: City latitudes.
        lons: City longitudes.
        values: Value per city.
        resolution: Grid spacing in degrees.

    Returns:
        Grid latitudes, grid longitudes, and the interpolated field with
        off-land cells set to NaN.
    """
    field = interpolate_field(lats, lons, values, resolution=resolution)
    return field.lats, field.lons, field.values

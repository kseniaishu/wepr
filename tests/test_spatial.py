"""The map's interpolation, its masks and its honesty layers.

Nothing here touches the network. The committed elevation artifact is a fetched
grid, and a test that required it would fail on a clone that had not built it yet,
so these run against the synthetic terrain in `conftest.py`. What is being asserted
is the machinery: that the surface recovers coefficients it was given, that
distance weighting cannot overshoot, that the sea is never painted, and that a
cell outside the city hull is never called interpolated.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import synthetic_elevation_grid
from wepr import spatial
from wepr.cities import CITIES

CITY_LATS = np.array([c.lat for c in CITIES])
CITY_LONS = np.array([c.lon for c in CITIES])


# -- the grid -------------------------------------------------------------------


def test_grid_axes_land_on_exact_multiples_of_the_resolution():
    """ERA5-Land's cell centres sit on multiples of 0.1, so the grid must too."""
    lats, lons = spatial.grid_axes(0.1)
    for axis in (lats, lons):
        assert np.allclose(axis / 0.1, np.round(axis / 0.1)), "axis drifted off-grid"
    assert np.allclose(np.diff(lats), 0.1)
    assert np.allclose(np.diff(lons), 0.1)


def test_grid_axes_stays_inside_its_bounds():
    lat_min, lat_max, lon_min, lon_max = spatial.BOUNDS
    lats, lons = spatial.grid_axes()
    assert lats[0] >= lat_min and lats[-1] <= lat_max
    assert lons[0] >= lon_min and lons[-1] <= lon_max


def test_grid_axes_covers_every_city():
    """A city outside the grid would be interpolated from but never drawn."""
    lats, lons = spatial.grid_axes()
    for city in CITIES:
        assert lats[0] <= city.lat <= lats[-1], city.slug
        assert lons[0] <= city.lon <= lons[-1], city.slug


@pytest.mark.parametrize("resolution", [0.0, -0.1])
def test_grid_axes_rejects_a_non_positive_resolution(resolution):
    with pytest.raises(spatial.SpatialError, match="positive"):
        spatial.grid_axes(resolution)


def test_grid_axes_rejects_empty_bounds():
    with pytest.raises(spatial.SpatialError, match="empty bounds"):
        spatial.grid_axes(0.1, (44.0, 36.0, -9.9, 4.4))


# -- distance -------------------------------------------------------------------


def test_haversine_matches_known_separations():
    """Madrid-Barcelona is about 505 km; a degree of latitude about 111 km."""
    assert spatial.haversine_km(40.4168, -3.7038, 41.3874, 2.1686) == pytest.approx(505, abs=10)
    assert spatial.haversine_km(40.0, 0.0, 41.0, 0.0) == pytest.approx(111.2, abs=0.5)
    assert spatial.haversine_km(40.0, -3.0, 40.0, -3.0) == 0.0


def test_haversine_shrinks_a_degree_of_longitude_with_latitude():
    """The reason planar degrees are not used: the north would be stretched."""
    south = spatial.haversine_km(36.0, 0.0, 36.0, 1.0)
    north = spatial.haversine_km(43.0, 0.0, 43.0, 1.0)
    assert north < south
    assert south == pytest.approx(90, abs=2)
    assert north == pytest.approx(81, abs=2)


# -- the land mask --------------------------------------------------------------


def test_land_mask_treats_zero_and_nan_as_not_land():
    """Sea reads as 0.0 from the endpoint; NaN means never fetched."""
    elevation = np.array([[666.0, 0.0, np.nan, -5.0, 0.1]])
    assert spatial.land_mask(elevation).tolist() == [[True, False, False, False, True]]


def test_land_mask_does_not_warn_on_nan():
    """A NaN comparison must stay silent, not emit a numpy RuntimeWarning."""
    with np.errstate(all="raise"):
        spatial.land_mask(np.array([np.nan, 1.0]))


# -- the trend surface ----------------------------------------------------------


def test_fit_surface_recovers_a_plane_it_was_given_exactly():
    rng = np.random.default_rng(0)
    elevation = rng.uniform(0, 900, 30)
    lats = rng.uniform(36, 44, 30)
    lons = rng.uniform(-9, 3, 30)
    values = 20.0 - 0.0044 * elevation - 0.7 * lats + 0.1 * lons

    surface = spatial.fit_surface(elevation, lats, lons, values)

    assert surface.lapse_rate_c_per_km == pytest.approx(-4.4, abs=1e-6)
    assert surface.per_degree_lat == pytest.approx(-0.7, abs=1e-9)
    assert surface.per_degree_lon == pytest.approx(0.1, abs=1e-9)
    assert surface.r2 == pytest.approx(1.0)
    assert surface.residual_sd == pytest.approx(0.0, abs=1e-9)
    assert surface.n_cities == 30


def test_fit_surface_records_the_elevation_range_it_was_fitted_on():
    """Applying the lapse rate above this range is extrapolation, so it is kept."""
    surface = spatial.fit_surface(
        [10.0, 200.0, 400.0, 600.0, 820.0, 50.0],
        [36.0, 38.0, 40.0, 42.0, 43.0, 37.0],
        [-9.0, -6.0, -3.0, 0.0, 2.0, -8.0],
        [18.0, 16.0, 14.0, 12.0, 11.0, 17.0],
    )
    assert surface.elevation_range == (10.0, 820.0)


def test_fit_surface_ignores_cities_with_a_missing_value():
    surface = spatial.fit_surface(
        [10.0, 200.0, 400.0, 600.0, 820.0, 50.0],
        [36.0, 38.0, 40.0, 42.0, 43.0, 37.0],
        [-9.0, -6.0, -3.0, 0.0, 2.0, -8.0],
        [18.0, 16.0, 14.0, 12.0, 11.0, np.nan],
    )
    assert surface.n_cities == 5


def test_fit_surface_needs_more_points_than_parameters():
    with pytest.raises(spatial.SpatialError, match="at least 5"):
        spatial.fit_surface([0, 1, 2, 3], [36, 37, 38, 39], [-9, -8, -7, -6], [1, 2, 3, 4])


def test_fit_surface_rejects_degenerate_coordinates():
    """Cities on one line cannot identify a surface in latitude and longitude."""
    n = 8
    lats = np.linspace(36, 43, n)
    with pytest.raises(spatial.SpatialError, match="degenerate"):
        spatial.fit_surface(np.zeros(n), lats, np.zeros(n), np.arange(n, dtype=float))


def test_fit_surface_rejects_mismatched_lengths():
    with pytest.raises(spatial.SpatialError, match="same length"):
        spatial.fit_surface([0, 1], [36, 37, 38], [-9, -8, -7], [1, 2, 3])


def test_fit_surface_on_the_real_cities_finds_a_physical_lapse_rate():
    """The claim the map rests on: position explains most of the city spread.

    Values here are the real annual means recorded in the registry's own cities,
    hard-coded so this test does not depend on how far the archive fetch has got.
    """
    elevation = np.array(
        [59, 191, 25, 24, 682, 22, 697, 471, 66, 666, 22, 41, 240, 32, 88, 821, 18, 17, 698, 228],
        dtype=float,
    )
    lats = np.array(
        [43.3, 38.9, 41.4, 43.3, 41.8, 37.0, 37.2, 42.1, 38.7, 40.4,
         36.7, 38.0, 43.4, 39.6, 41.2, 41.0, 37.4, 39.5, 41.7, 41.6]
    )
    lons = np.array(
        [-8.4, -7.0, 2.1, -2.9, -6.8, -7.9, -3.6, -0.4, -9.1, -3.7,
         -4.4, -1.1, -5.8, 2.7, -8.6, -5.7, -6.0, -0.4, -4.7, -0.9]
    )
    values = np.array(
        [13.3, 16.5, 15.0, 13.6, 11.7, 17.5, 14.9, 12.4, 16.0, 13.2,
         17.5, 17.4, 11.6, 17.6, 14.4, 11.6, 18.1, 16.9, 11.9, 14.6]
    )

    surface = spatial.fit_surface(elevation, lats, lons, values)

    # A plausible surface lapse rate, shallower than the free-air 6.5 °C/km.
    assert -7.0 < surface.lapse_rate_c_per_km < -3.0
    assert surface.per_degree_lat < 0.0, "Iberia must cool northward"
    assert surface.r2 > 0.9
    # The whole justification for interpolating residuals rather than values: what
    # distance weighting is left to explain is small next to the city spread.
    assert surface.residual_sd < 0.25 * values.std()


# -- distance weighting ---------------------------------------------------------


def test_idw_returns_a_point_value_exactly_at_that_point():
    lats = np.array([40.0, 41.0, 42.0])
    lons = np.array([-3.0, -2.0, -1.0])
    values = np.array([10.0, 20.0, 30.0])
    field, nearest = spatial.idw(lats, lons, values, lats, lons)
    for i in range(3):
        assert field[i, i] == pytest.approx(values[i])
        assert nearest[i, i] == pytest.approx(0.0, abs=1e-9)


def test_idw_never_leaves_the_range_of_its_inputs():
    """The reason IDW is used over a spline: it cannot overshoot."""
    rng = np.random.default_rng(3)
    values = rng.uniform(5, 25, len(CITIES))
    grid_lats, grid_lons = spatial.grid_axes(0.5)
    field, _ = spatial.idw(CITY_LATS, CITY_LONS, values, grid_lats, grid_lons)
    assert field.min() >= values.min() - 1e-9
    assert field.max() <= values.max() + 1e-9


def test_idw_reproduces_a_constant_field():
    grid_lats, grid_lons = spatial.grid_axes(0.5)
    field, _ = spatial.idw(CITY_LATS, CITY_LONS, np.full(len(CITIES), 7.0), grid_lats, grid_lons)
    assert np.allclose(field, 7.0)


def test_idw_nearest_distance_is_zero_at_a_city_and_positive_away_from_one():
    grid_lats, grid_lons = spatial.grid_axes(0.1)
    _, nearest = spatial.idw(
        CITY_LATS, CITY_LONS, np.arange(len(CITIES), dtype=float), grid_lats, grid_lons
    )
    assert nearest.min() == pytest.approx(0.0, abs=6.0)
    assert nearest.max() > 100.0


def test_idw_drops_missing_values_rather_than_poisoning_the_field():
    values = np.arange(len(CITIES), dtype=float)
    values[3] = np.nan
    grid_lats, grid_lons = spatial.grid_axes(0.5)
    field, _ = spatial.idw(CITY_LATS, CITY_LONS, values, grid_lats, grid_lons)
    assert np.isfinite(field).all()


def test_idw_requires_at_least_one_finite_value():
    with pytest.raises(spatial.SpatialError, match="no point carries a finite value"):
        spatial.idw([40.0], [-3.0], [np.nan], np.array([40.0]), np.array([-3.0]))


# -- the hull -------------------------------------------------------------------


def test_hull_distance_is_zero_at_every_city_coordinate():
    """Measured at the city itself, every city is on or inside its own hull."""
    hull = spatial.hull_distance_km(CITY_LATS, CITY_LONS, CITY_LATS, CITY_LONS)
    assert np.diag(hull).max() == pytest.approx(0.0, abs=1e-9)


def test_city_grid_cells_sit_within_half_a_cell_of_the_hull():
    """Why `interpolated` carries a half-cell tolerance.

    Nine cities are hull *vertices*, so snapping them to the nearest cell centre
    can land just outside the hull — up to 5.0 km for Bilbao. That is less than
    half a cell diagonal (7.9 km at 0.1°), which is the tolerance the field uses,
    and this test is what keeps the two facts tied together.
    """
    grid_lats, grid_lons = spatial.grid_axes(0.1)
    hull = spatial.hull_distance_km(CITY_LATS, CITY_LONS, grid_lats, grid_lons)
    half_cell_km = 0.1 * 111.32 * np.sqrt(2) / 2
    worst = 0.0
    for city in CITIES:
        i = int(np.abs(grid_lats - city.lat).argmin())
        j = int(np.abs(grid_lons - city.lon).argmin())
        worst = max(worst, float(hull[i, j]))
    assert 0.0 < worst <= half_cell_km, f"worst city cell is {worst:.1f} km outside"


def test_hull_distance_separates_interior_from_far_field():
    """Madrid is interior; the mid-Atlantic is a long way outside."""
    grid_lats, grid_lons = spatial.grid_axes(0.1)
    hull = spatial.hull_distance_km(CITY_LATS, CITY_LONS, grid_lats, grid_lons)

    def at(lat, lon):
        return hull[int(np.abs(grid_lats - lat).argmin()), int(np.abs(grid_lons - lon).argmin())]

    assert at(40.4, -3.7) == 0.0
    # The far southeast Mediterranean corner, well beyond any city.
    assert at(38.0, 4.4) > 200.0


def test_hull_margin_is_the_documented_compromise():
    """The margin was chosen by measuring named places; keep those numbers true.

    `spatial.HULL_MARGIN_KM` is justified in the module docstring by exactly these
    cases, including the ones it gets wrong. If the city registry changes the hull
    moves, and this test is what notices.
    """
    grid_lats, grid_lons = spatial.grid_axes(0.1)
    hull = spatial.hull_distance_km(CITY_LATS, CITY_LONS, grid_lats, grid_lons)

    def at(lat, lon):
        return hull[int(np.abs(grid_lats - lat).argmin()), int(np.abs(grid_lons - lon).argmin())]

    margin = spatial.HULL_MARGIN_KM
    # Iberian land the margin is meant to reach, Tarifa at the southern tip
    # included — that is what moved the margin from 90 km to 95 km.
    for name, lat, lon in [
        ("Tarifa", 36.0, -5.6),
        ("Cabo de São Vicente", 37.0, -9.0),
        ("Fisterra", 42.9, -9.3),
        ("Girona", 42.0, 2.8),
        ("Gijón", 43.5, -5.7),
    ]:
        assert at(lat, lon) <= margin, f"{name} should be drawable"
    # French land the margin is meant to exclude. France is land-connected to
    # Spain across the Pyrenees, so the margin is the only thing holding it back.
    for name, lat, lon in [
        ("Pau", 43.3, -0.4),
        ("Perpignan", 42.7, 2.9),
        ("Toulouse", 43.6, 1.4),
    ]:
        assert at(lat, lon) > margin, f"{name} should not be drawable"


def test_distance_cannot_separate_southern_spain_from_morocco():
    """The measurement behind `connected_to_points`, kept as an executable fact.

    Tarifa and Tetouan are within a few kilometres of each other in hull
    distance, so *no* margin admits one and excludes the other. If this ever
    stops being true the connectivity step could be dropped — and this test is
    what would say so.
    """
    grid_lats, grid_lons = spatial.grid_axes(0.1)
    hull = spatial.hull_distance_km(CITY_LATS, CITY_LONS, grid_lats, grid_lons)

    def at(lat, lon):
        return hull[int(np.abs(grid_lats - lat).argmin()), int(np.abs(grid_lons - lon).argmin())]

    tarifa = at(36.0, -5.6)
    tetouan = at(35.6, -5.4)
    assert abs(tarifa - tetouan) < 15.0, (
        f"Tarifa {tarifa:.1f} km and Tetouan {tetouan:.1f} km are now far enough "
        f"apart that a margin alone might separate them"
    )


def test_hull_distance_calls_nothing_interpolated_from_two_points():
    """Two points have no hull, so no cell may claim to be surrounded by them."""
    grid_lats, grid_lons = spatial.grid_axes(1.0)
    hull = spatial.hull_distance_km([40.0, 41.0], [-3.0, -2.0], grid_lats, grid_lons)
    assert np.isinf(hull).all()


# -- the field ------------------------------------------------------------------


def _field(**kwargs):
    grid = kwargs.pop("elevation", None) or synthetic_elevation_grid()
    values = kwargs.pop("values", None)
    if values is None:
        values = 20.0 - 0.3 * (CITY_LATS - 36.0)
    return spatial.interpolate_field(
        CITY_LATS, CITY_LONS, values, units="°C", elevation=grid, **kwargs
    )


def test_field_never_paints_the_sea():
    grid = synthetic_elevation_grid()
    field = _field(elevation=grid)
    assert np.isnan(field.values[~grid.land]).all(), "colour was painted over water"


def test_field_never_paints_beyond_the_hull_margin():
    grid = synthetic_elevation_grid()
    field = _field(elevation=grid)
    hull = spatial.hull_distance_km(CITY_LATS, CITY_LONS, grid.lats, grid.lons)
    assert np.isnan(field.values[hull > spatial.HULL_MARGIN_KM]).all()


def test_field_never_paints_a_cell_it_has_no_terrain_for():
    """NaN elevation means never fetched, and an unknown cell must stay blank."""
    grid = synthetic_elevation_grid(cover=False)
    field = _field(elevation=grid)
    assert np.isnan(field.values[~np.isfinite(grid.elevation)]).all()


def test_interpolated_cells_are_a_subset_of_drawn_cells():
    field = _field()
    assert not (field.interpolated & ~field.drawn).any()
    assert field.interpolated.any(), "nothing was interpolated at all"
    assert (field.drawn & ~field.interpolated).any(), "nothing was flagged extrapolated"


def test_every_city_cell_is_interpolated_not_extrapolated():
    """A city must sit in the solid part of the map, never the faded part."""
    field = _field()
    for city in CITIES:
        i = int(np.abs(field.lats - city.lat).argmin())
        j = int(np.abs(field.lons - city.lon).argmin())
        assert field.interpolated[i, j], f"{city.slug} landed outside the hull"


def test_surface_plus_residual_reconstructs_a_city_value_exactly():
    """The correctness check on the two-stage construction.

    Evaluated at the city's own coordinate, surface plus distance-weighted
    residual must sum back to the observation, because that is where the residual
    was measured and the weighting is exact at its own points. (At the *cell*
    centre it need not: a 0.1° cell centre is up to 8 km from the city, with its
    own elevation, so the field there legitimately differs — see the next test.)
    """
    grid = synthetic_elevation_grid()
    rng = np.random.default_rng(11)
    values = 15.0 + rng.normal(0, 3.0, len(CITIES))

    city_elevation = spatial.sample(grid.elevation, grid.lats, grid.lons, CITY_LATS, CITY_LONS)
    surface = spatial.fit_surface(city_elevation, CITY_LATS, CITY_LONS, values)
    residual = values - surface.predict(city_elevation, CITY_LATS, CITY_LONS)
    weighted, _ = spatial.idw(CITY_LATS, CITY_LONS, residual, CITY_LATS, CITY_LONS)

    rebuilt = surface.predict(city_elevation, CITY_LATS, CITY_LONS) + np.diag(weighted)
    assert np.allclose(rebuilt, values, atol=1e-9)


def test_field_lands_close_to_each_city_value_at_that_citys_cell():
    """The cell is not the city, but it must not be far off either."""
    grid = synthetic_elevation_grid()
    rng = np.random.default_rng(11)
    values = 15.0 + rng.normal(0, 3.0, len(CITIES))
    field = _field(elevation=grid, values=values)
    for n, city in enumerate(CITIES):
        i = int(np.abs(field.lats - city.lat).argmin())
        j = int(np.abs(field.lons - city.lon).argmin())
        if np.isfinite(field.values[i, j]):
            assert field.values[i, j] == pytest.approx(values[n], abs=2.0), city.slug


def test_field_records_the_fitted_surface():
    field = _field()
    assert field.surface is not None
    assert field.surface.n_cities == len(CITIES)
    assert field.units == "°C"


def test_field_flags_elevation_extrapolation_above_the_fitted_range():
    """The synthetic ridge reaches 2,800 m; no city is above 900 m."""
    grid = synthetic_elevation_grid()
    field = _field(elevation=grid)
    assert field.extrapolates_elevation.any()
    high = grid.elevation > field.surface.elevation_range[1]
    assert (field.extrapolates_elevation & field.drawn).sum() <= high.sum()
    # And it is only ever claimed for cells actually drawn.
    assert not (field.extrapolates_elevation & ~field.drawn).any()


def test_field_clamps_precipitation_at_zero():
    """A residual fit can push a dry city below zero; millimetres cannot be."""
    grid = synthetic_elevation_grid()
    values = np.array([0.0] * (len(CITIES) - 1) + [40.0])
    field = spatial.interpolate_field(
        CITY_LATS, CITY_LONS, values, units="mm", non_negative=True, elevation=grid
    )
    drawn = field.drawn
    assert (field.values[drawn] >= 0.0).all()
    assert field.clamped_at_zero is True


def test_field_does_not_claim_to_have_clamped_when_it_did_not():
    field = _field()
    assert field.clamped_at_zero is False


def test_field_falls_back_to_raw_weighting_when_a_surface_is_unidentifiable():
    """Three cities cannot fit four parameters, so the method changes — and says so."""
    grid = synthetic_elevation_grid()
    field = spatial.interpolate_field(
        CITY_LATS[:3], CITY_LONS[:3], np.array([10.0, 12.0, 14.0]), elevation=grid
    )
    assert field.surface is None
    assert not field.extrapolates_elevation.any()


def test_field_refuses_a_margin_the_cached_grid_does_not_cover():
    grid = synthetic_elevation_grid(margin_km=50.0)
    with pytest.raises(spatial.SpatialError, match="only covers 50 km"):
        _field(elevation=grid, margin_km=90.0)


def test_field_refuses_a_resolution_that_is_not_the_grids():
    grid = synthetic_elevation_grid(resolution=0.1)
    with pytest.raises(spatial.SpatialError, match="rebuild the grid"):
        _field(elevation=grid, resolution=0.25)


def test_field_summary_is_json_safe_and_self_describing():
    import json

    field = _field()
    summary = field.summary()
    json.dumps(summary)  # must not raise
    assert summary["units"] == "°C"
    assert summary["cells_drawn"] == int(field.drawn.sum())
    assert summary["cells_interpolated"] + summary["cells_extrapolated"] == summary["cells_drawn"]
    assert summary["surface"]["n_cities"] == len(CITIES)
    assert summary["resolution_deg"] == pytest.approx(0.1)


# -- the documented plain signature ---------------------------------------------


def test_interpolate_returns_the_documented_triple(monkeypatch):
    """`interpolate` is the signature the rest of the project was written against."""
    grid = synthetic_elevation_grid()
    monkeypatch.setattr(spatial, "load_elevation", lambda *a, **k: grid)
    lats, lons, values = spatial.interpolate(
        CITY_LATS, CITY_LONS, 20.0 - 0.3 * (CITY_LATS - 36.0)
    )
    assert lats.shape == grid.lats.shape
    assert lons.shape == grid.lons.shape
    assert values.shape == (len(lats), len(lons))
    assert np.isnan(values[~grid.land]).all()


# -- loading the artifact -------------------------------------------------------


def test_load_elevation_names_the_script_that_builds_it(tmp_path):
    missing = tmp_path / "nope.npz"
    with pytest.raises(spatial.SpatialError, match="scripts/fetch_elevation.py"):
        spatial.load_elevation(missing)


def test_load_elevation_rejects_a_grid_whose_shape_disagrees_with_its_axes(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez_compressed(
        path,
        lats=np.arange(5.0),
        lons=np.arange(7.0),
        elevation=np.zeros((5, 6)),
        resolution=np.array(0.1),
        margin_km=np.array(90.0),
        source=np.array("test"),
    )
    with pytest.raises(spatial.SpatialError, match="rebuild it"):
        spatial.load_elevation(path)


def test_load_elevation_round_trips_what_the_script_writes(tmp_path):
    path = tmp_path / "grid.npz"
    grid = synthetic_elevation_grid()
    np.savez_compressed(
        path,
        lats=grid.lats,
        lons=grid.lons,
        elevation=grid.elevation,
        resolution=np.array(grid.resolution),
        margin_km=np.array(grid.margin_km),
        source=np.array("round trip"),
    )
    loaded = spatial.load_elevation(path)
    assert np.allclose(loaded.lats, grid.lats)
    assert loaded.resolution == pytest.approx(0.1)
    assert loaded.margin_km == pytest.approx(spatial.HULL_MARGIN_KM)
    assert loaded.source == "round trip"
    assert np.array_equal(loaded.land, grid.land)


# -- sampling -------------------------------------------------------------------


def test_sample_reads_the_nearest_cell():
    gridded = np.arange(12.0).reshape(3, 4)
    lats = np.array([40.0, 41.0, 42.0])
    lons = np.array([-3.0, -2.0, -1.0, 0.0])
    got = spatial.sample(gridded, lats, lons, [40.04, 41.96, 42.0], [-2.96, -1.1, 0.0])
    assert got.tolist() == [0.0, 10.0, 11.0]


# -- land connectivity ----------------------------------------------------------


def test_connectivity_drops_a_sea_separated_landmass():
    """The mechanism that excludes Morocco, on a grid small enough to read.

    Two land blocks separated by a column of sea; only the left one holds a seed.
    """
    mask = np.zeros((5, 7), dtype=bool)
    mask[1:4, 0:3] = True  # the "peninsula"
    mask[1:4, 4:7] = True  # across the "strait"
    lats = np.arange(5.0)
    lons = np.arange(7.0)

    kept = spatial.connected_to_points(mask, lats, lons, [2.0], [1.0])

    assert kept[1:4, 0:3].all(), "the seeded landmass was dropped"
    assert not kept[1:4, 4:7].any(), "a sea-separated landmass was kept"


def test_connectivity_keeps_every_landmass_that_holds_a_seed():
    mask = np.zeros((5, 7), dtype=bool)
    mask[1:4, 0:3] = True
    mask[1:4, 4:7] = True
    lats = np.arange(5.0)
    lons = np.arange(7.0)
    kept = spatial.connected_to_points(mask, lats, lons, [2.0, 2.0], [1.0, 5.0])
    assert np.array_equal(kept, mask)


def test_connectivity_returns_nothing_when_no_seed_lands_on_the_mask():
    """Returning the mask unchanged here would silently restore north Africa."""
    mask = np.zeros((5, 7), dtype=bool)
    mask[1:4, 4:7] = True
    kept = spatial.connected_to_points(mask, np.arange(5.0), np.arange(7.0), [2.0], [1.0])
    assert not kept.any()


def test_connectivity_handles_an_entirely_empty_mask():
    kept = spatial.connected_to_points(
        np.zeros((4, 4), dtype=bool), np.arange(4.0), np.arange(4.0), [1.0], [1.0]
    )
    assert kept.shape == (4, 4)
    assert not kept.any()


def test_field_drops_an_island_with_no_city_on_it():
    """Menorca has no city; a field must not paint it from Mallorca's values."""
    grid = synthetic_elevation_grid()
    elevation = grid.elevation.copy()
    # An isolated land block in the far southeast, ringed by sea, no city on it.
    i = slice(len(grid.lats) - 3, len(grid.lats))
    j = slice(len(grid.lons) - 3, len(grid.lons))
    elevation[i, j] = 300.0
    island = spatial.ElevationGrid(
        grid.lats, grid.lons, elevation, grid.resolution, grid.margin_km, "island test"
    )
    field = _field(elevation=island)
    assert np.isnan(field.values[i, j]).all()

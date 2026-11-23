"""Build the map's elevation grid, once.

    .venv/bin/python scripts/fetch_elevation.py [--margin 95] [--force]

Writes `data/grid/elevation.npz`: ground elevation in metres on a 0.1° grid over
Iberia and the Balearics, from Open-Meteo's elevation endpoint.

**Why this is committed, unlike everything else under `data/`.** Terrain is static
reference data — it does not get refetched the way a daily series does — and the
artifact is tens of kilobytes. Committing it means a clone can render the map with
no network access at all, which is the same reason `data/validation/` is committed.

The grid does two jobs at once, which is why it is worth the requests:

- it is the **land mask**, so no colour is painted over the sea;
- it is the **elevation covariate** in the map's trend surface, so the field is
  not interpolated flat across the Pyrenees.

It is also the same digital elevation model the archive reports for the cell it
served — a Madrid city fetch and a bare lookup at the same coordinate both return
666.0 m — so the map's terrain and the data's terrain cannot disagree.

**Only drawable cells are fetched.** `wepr.spatial` will not paint a cell more than
`--margin` kilometres outside the convex hull of the cities, so fetching terrain
there would be buying data to throw away: at the default margin that is 8,823 cells
rather than 11,664, a quarter fewer requests. Cells outside the margin stay NaN in
the artifact, which `land_mask` treats as not-land.

**The fetch resumes.** Open-Meteo's daily request cap is shared with the archive
client and is genuinely reachable here — the first attempt at this grid died at
2,800 of 11,664 points. Every chunk that arrives is written to
`data/grid/elevation.partial.npz` as it lands, so re-running tomorrow continues
where this stopped instead of starting over. That is the same bargain
`scripts/fetch_data.py` makes with its per-year cache.

Implemented in Phase 5.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys

import numpy as np
import requests

from wepr import paths, spatial
from wepr.cities import CITIES
from wepr.data import fetch


def partial_path():
    """Where part-fetched progress is parked between runs."""
    return paths.ELEVATION_GRID.with_suffix(".partial.npz")


def _load_partial(lats, lons, resolution: float, margin_km: float):
    """Previous progress, if it describes the same grid. Otherwise nothing.

    A partial built for a different resolution or a smaller margin cannot be
    extended, so it is discarded rather than merged into a grid it does not match.
    """
    path = partial_path()
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            if (
                not np.isclose(float(z["resolution"]), resolution)
                or float(z["margin_km"]) != margin_km
                or z["lats"].shape != lats.shape
                or z["lons"].shape != lons.shape
                or not np.allclose(z["lats"], lats)
                or not np.allclose(z["lons"], lons)
            ):
                return None
            return z["elevation"].copy()
    except (OSError, KeyError, ValueError):
        return None


def _save_partial(elevation, lats, lons, resolution: float, margin_km: float) -> None:
    np.savez_compressed(
        partial_path(),
        lats=lats,
        lons=lons,
        elevation=elevation,
        resolution=np.array(resolution),
        margin_km=np.array(margin_km),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--resolution",
        type=float,
        default=spatial.RESOLUTION,
        help="grid spacing in degrees (default 0.1, ERA5-Land's own cell size)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=spatial.HULL_MARGIN_KM,
        help="fetch cells within this many km of the city hull (default 95)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild even though the artifact already exists",
    )
    args = parser.parse_args()

    paths.ensure_dirs()
    destination = paths.ELEVATION_GRID

    if destination.exists() and not args.force:
        grid = spatial.load_elevation(destination)
        print(
            f"{destination} already holds a {len(grid.lats)}x{len(grid.lons)} grid at "
            f"{grid.resolution}° out to {grid.margin_km:.0f} km beyond the city hull.\n"
            f"Terrain does not change; pass --force to rebuild."
        )
        return 0

    try:
        lats, lons = spatial.grid_axes(args.resolution)
    except spatial.SpatialError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    mesh_lat, mesh_lon = np.meshgrid(lats, lons, indexing="ij")

    # Which cells the map could ever draw. Purely geometric, so this costs nothing
    # and is known before a single request goes out.
    hull = spatial.hull_distance_km(
        np.array([c.lat for c in CITIES]),
        np.array([c.lon for c in CITIES]),
        lats,
        lons,
    )
    needed = hull <= args.margin

    elevation = _load_partial(lats, lons, args.resolution, args.margin)
    resumed = elevation is not None
    if not resumed:
        elevation = np.full(mesh_lat.shape, np.nan)

    outstanding = needed & ~np.isfinite(elevation)
    already = int((needed & np.isfinite(elevation)).sum())
    chunks = -(-int(outstanding.sum()) // fetch.MAX_ELEVATION_POINTS)

    print(
        f"{len(lats)}x{len(lons)} = {mesh_lat.size} cells over {spatial.BOUNDS} "
        f"at {args.resolution}°\n"
        f"{int(needed.sum())} within {args.margin:.0f} km of the city hull "
        f"({100 * needed.mean():.0f}%); the rest are never drawn and not fetched",
        flush=True,
    )
    if resumed:
        print(f"resuming: {already} points already cached in {partial_path().name}", flush=True)
    if not outstanding.any():
        print("every needed cell is already cached", flush=True)
    else:
        print(
            f"{int(outstanding.sum())} points still to fetch, "
            f"{chunks} requests at {fetch.MAX_ELEVATION_POINTS} each",
            flush=True,
        )

    index = np.flatnonzero(outstanding.ravel())
    flat_lat = mesh_lat.ravel()
    flat_lon = mesh_lon.ravel()
    flat_elevation = elevation.ravel()

    session = requests.Session()
    failure: fetch.FetchError | None = None
    try:
        for start in range(0, len(index), fetch.MAX_ELEVATION_POINTS):
            take = index[start : start + fetch.MAX_ELEVATION_POINTS]
            try:
                values = fetch.fetch_elevation(
                    flat_lat[take], flat_lon[take], session=session
                )
            except fetch.FetchError as exc:
                failure = exc
                break
            flat_elevation[take] = values
            done = start + len(take)
            # Written every chunk, not at the end: the point of the partial is to
            # survive the quota wall that killed the first attempt at this grid.
            _save_partial(
                flat_elevation.reshape(mesh_lat.shape),
                lats,
                lons,
                args.resolution,
                args.margin,
            )
            print(f"  · {already + done}/{int(needed.sum())} cached", flush=True)
    finally:
        session.close()

    elevation = flat_elevation.reshape(mesh_lat.shape)
    remaining = int((needed & ~np.isfinite(elevation)).sum())

    if remaining:
        print(
            f"\nstopped with {remaining} of {int(needed.sum())} needed cells "
            f"outstanding.\nProgress is in {partial_path()} — re-run this script to "
            f"resume.",
            file=sys.stderr,
        )
        if failure is not None:
            print(f"error: {failure}", file=sys.stderr)
        return 1

    land = spatial.land_mask(elevation)
    np.savez_compressed(
        destination,
        lats=lats,
        lons=lons,
        elevation=elevation,
        resolution=np.array(args.resolution),
        margin_km=np.array(args.margin),
        source=np.array(
            f"Open-Meteo elevation endpoint ({fetch.ELEVATION_URL}), "
            f"{int(needed.sum())} cells within {args.margin:.0f} km of the city "
            f"hull, fetched {dt.datetime.now(dt.UTC).date().isoformat()}"
        ),
    )
    partial_path().unlink(missing_ok=True)

    print(
        f"\nwrote {destination} ({destination.stat().st_size / 1024:.0f} kB)\n"
        f"  fetched      {int(needed.sum()):6d} cells\n"
        f"  land         {int(land.sum()):6d}\n"
        f"  sea          {int((needed & ~land).sum()):6d}\n"
        f"  not fetched  {int((~needed).sum()):6d} (outside the drawable margin, NaN)\n"
        f"  elevation    {np.nanmin(elevation[land]):.0f} - {np.nanmax(elevation[land]):.0f} m\n"
        f"\nThe endpoint clamps at zero, so land at or below sea level reads as sea "
        f"(the Guadalquivir marshes, the Albufera). That is stated in wepr.spatial."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

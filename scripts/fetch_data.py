"""Fetch the ERA5 daily archive for every registered city.

    .venv/bin/python scripts/fetch_data.py [--start 1950-01-01] [--city huesca]

Writes raw responses to `data/raw/` and tidy frames to `data/processed/`. The
archive is immutable for past dates, so cached files are reused unless
`--refresh` is passed.
"""

from __future__ import annotations

import argparse
import json
import sys

from wepr import paths
from wepr.cities import CITIES, get
from wepr.data import fetch, load


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default="1950-01-01", help="first date, YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="last date; defaults to the archive edge")
    parser.add_argument("--city", action="append", help="limit to this slug; repeatable")
    parser.add_argument("--refresh", action="store_true", help="re-fetch cached dates")
    parser.add_argument(
        "--assemble-only",
        action="store_true",
        help="skip the network; build frames from the years already cached",
    )
    args = parser.parse_args()

    paths.ensure_dirs()

    try:
        cities = [get(slug) for slug in args.city] if args.city else list(CITIES)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    end = args.end or fetch.archive_edge()
    if args.assemble_only:
        print(f"assembling {len(cities)} city/cities from cached years", flush=True)
    else:
        print(
            f"fetching {len(cities)} city/cities, {args.start} .. {end}, "
            f"model={fetch.MODEL}{' (refresh)' if args.refresh else ''}",
            flush=True,
        )

    def log(message: str) -> None:
        print(f"  · {message}", flush=True)

    try:
        if args.assemble_only:
            raw_files = fetch.assemble_cached(cities, log=log)
        else:
            raw_files = fetch.fetch_cities(
                cities, args.start, end, refresh=args.refresh, log=log
            )
    except fetch.FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        # A quota failure still leaves cached years behind, so salvage them rather
        # than discarding a partial fetch the user has already paid for.
        print("assembling the years that did arrive...", file=sys.stderr)
        raw_files = fetch.assemble_cached(cities, log=log)
        if not raw_files:
            return 1

    failures = 0
    for city in cities:
        path = raw_files.get(city.slug)
        if path is None:
            print(f"  {city.slug:12s} NOT FETCHED", file=sys.stderr)
            failures += 1
            continue

        try:
            frame = load.tidy(json.loads(path.read_text()))
        except load.DataContractError as exc:
            print(f"  {city.slug:12s} CONTRACT FAILURE: {exc}", file=sys.stderr)
            failures += 1
            continue
        load.write_processed(city.slug, frame)
        meta = load.city_metadata(city.slug)
        print(
            f"  {city.slug:12s} {len(frame):>6,} rows  "
            f"{frame.index[0].date()} .. {frame.index[-1].date()}  "
            f"grid ({meta['grid_latitude']:.2f}, {meta['grid_longitude']:.2f}) "
            f"{meta['grid_elevation_m']:.0f} m",
            flush=True,
        )

    print(f"raw -> {paths.DATA_RAW}\nprocessed -> {paths.DATA_PROCESSED}")
    if failures:
        print(f"{failures} city/cities failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

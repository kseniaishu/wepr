"""The tidy daily contract, and the one place unit conversion is allowed."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from tests.conftest import daily_payload, tidy_frame
from wepr.data import load


# --- the archive path: a rename, never an arithmetic conversion ---------------


def test_tidy_produces_the_contracted_shape():
    frame = tidy_frame(days=31)
    assert list(frame.columns) == list(load.COLUMNS)
    assert frame.index.name == "date"
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert frame.index.is_monotonic_increasing
    assert len(frame) == 31


def test_tidy_does_not_convert_units():
    """The archive already serves °C and mm, so values must pass through intact."""
    payload = daily_payload(days=3)
    frame = load.tidy(payload)
    assert frame["tavg"].tolist() == payload["daily"]["temperature_2m_mean"][:3]
    assert frame["prcp"].tolist() == payload["daily"]["precipitation_sum"][:3]


def test_tidy_drops_trailing_nulls_past_the_archive_edge():
    """A window reaching past the edge returns null padding, not a data gap."""
    frame = load.tidy(daily_payload(start="2020-01-01", days=5, trailing_nulls=4))
    assert len(frame) == 5
    assert frame.index[-1] == pd.Timestamp("2020-01-05")


def test_tidy_keeps_interior_gaps_as_nan_rather_than_filling_them():
    """Interior holes are a defect for the quality gate to report, not to patch."""
    payload = daily_payload(days=6)
    payload["daily"]["temperature_2m_mean"][2] = None
    frame = load.tidy(payload)
    assert frame["tavg"].isna().sum() == 1
    assert len(frame) == 6


def test_tidy_tolerates_overlapping_windows():
    """Two windows sharing a date must not break the daily index."""
    payload = daily_payload(days=4)
    for key, values in payload["daily"].items():
        values.append(values[-1])
    payload["daily"]["time"][-1] = payload["daily"]["time"][-2]
    assert len(load.tidy(payload)) == 4


def test_tidy_rejects_a_payload_with_no_daily_block():
    with pytest.raises(load.DataContractError, match="daily.time"):
        load.tidy({"slug": "x", "daily": {}})


def test_tidy_rejects_a_payload_missing_a_variable():
    payload = daily_payload(days=3)
    del payload["daily"]["precipitation_sum"]
    with pytest.raises(load.DataContractError, match="prcp"):
        load.tidy(payload)


# --- the contract check itself ------------------------------------------------


def test_contract_rejects_a_date_gap():
    frame = tidy_frame(days=10).drop(index=pd.Timestamp("2020-01-05"))
    with pytest.raises(load.DataContractError, match="missing date"):
        load.assert_daily_contract(frame)


def test_contract_rejects_an_unsorted_index():
    frame = tidy_frame(days=5).iloc[::-1]
    with pytest.raises(load.DataContractError, match="not sorted"):
        load.assert_daily_contract(frame)


def test_contract_rejects_a_non_datetime_index():
    frame = tidy_frame(days=5).reset_index(drop=True)
    frame.index.name = "date"
    with pytest.raises(load.DataContractError, match="not DatetimeIndex"):
        load.assert_daily_contract(frame)


def test_contract_rejects_a_misnamed_index():
    frame = tidy_frame(days=5)
    frame.index.name = "time"
    with pytest.raises(load.DataContractError, match="not 'date'"):
        load.assert_daily_contract(frame)


# --- the processed cache round-trip -------------------------------------------


def test_processed_round_trip_preserves_the_contract(isolated_paths):
    original = tidy_frame(days=40)
    load.write_processed("testville", original)
    restored = load.load_city("testville")
    assert restored.index.equals(original.index)
    pd.testing.assert_frame_equal(restored, original, atol=1e-3)


def test_load_city_names_the_fix_when_nothing_is_cached(isolated_paths):
    with pytest.raises(FileNotFoundError, match="fetch_data.py"):
        load.load_city("madrid")


def test_available_cities_lists_what_is_on_disk(isolated_paths):
    assert load.available_cities() == []
    load.write_processed("zaragoza", tidy_frame(days=5))
    load.write_processed("porto", tidy_frame(days=5))
    assert load.available_cities() == ["porto", "zaragoza"]


# --- the GHCN station exports: the only unit conversion in the codebase -------


def test_ghcn_rejects_a_city_without_station_data():
    with pytest.raises(KeyError, match="no GHCN station export"):
        load.load_ghcn("madrid")


@pytest.mark.parametrize("slug", ["huesca", "valencia"])
def test_ghcn_satisfies_the_daily_contract(slug):
    frame = load.load_ghcn(slug)
    load.assert_daily_contract(frame, label=slug)
    assert list(frame.columns) == list(load.COLUMNS)


@pytest.mark.parametrize("slug", ["huesca", "valencia"])
def test_ghcn_values_are_plausible_in_si_units(slug):
    """A failed conversion shows up as Fahrenheit-shaped temperatures."""
    frame = load.load_ghcn(slug)
    assert -25 < frame["tmax"].min() and frame["tmax"].max() < 50
    assert frame["prcp"].min() >= 0
    assert frame["prcp"].max() < 500


def test_huesca_tavg_is_the_midrange_of_tmax_and_tmin():
    """Huesca has no TAVG column; it is derived. Check the derivation held."""
    frame = load.load_ghcn("huesca").dropna(subset=["tmax", "tmin", "tavg"])
    expected = (frame["tmax"] + frame["tmin"]) / 2
    pd.testing.assert_series_equal(frame["tavg"], expected, check_names=False, atol=1e-9)


def test_valencia_tavg_is_reported_not_derived():
    """Valencia *has* TAVG, so it must not be overwritten by the midrange."""
    frame = load.load_ghcn("valencia").dropna(subset=["tmax", "tmin", "tavg"])
    midrange = (frame["tmax"] + frame["tmin"]) / 2
    assert not frame["tavg"].equals(midrange)


def test_ghcn_gaps_are_left_as_nan():
    """Forward-filling an observational record would invent measurements."""
    frame = load.load_ghcn("valencia")
    assert frame["tmin"].isna().sum() > 0
    # An absent date becomes an all-NaN row, never a hole in the index.
    assert frame.index.freqstr in {"D", None}
    assert (frame.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()


def test_ghcn_missingness_matches_the_documented_figures():
    """Guards against a silent re-export of the CSVs with different coverage."""
    valencia = load.load_ghcn("valencia")
    present = valencia.dropna(how="all")
    assert 0.40 < present["tmin"].isna().mean() < 0.46
    assert 0.33 < present["tmax"].isna().mean() < 0.38
    assert 0.14 < present["prcp"].isna().mean() < 0.19


def test_ghcn_station_headers():
    huesca = load.ghcn_station("huesca")
    assert huesca["station"] == "SPE00120161"
    assert huesca["elevation_m"] == pytest.approx(541.0)
    # Valencia's export carries no coordinates, so none are invented.
    assert "lat" not in load.ghcn_station("valencia")

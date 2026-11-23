"""The quality gate, and the shape of the reanalysis-vs-station comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import tidy_frame
from wepr.data import load, validate


def _long_frame(years: int = 70, slug: str = "madrid") -> pd.DataFrame:
    """A clean frame long enough to clear the gate's minimum-record rule."""
    index = pd.date_range("1950-01-01", periods=int(years * 365.25), freq="D", name="date")
    doy = index.day_of_year.to_numpy()
    seasonal = 15.0 - 8.0 * np.cos(2 * np.pi * doy / 365.25)
    return pd.DataFrame(
        {
            "tavg": seasonal,
            "tmax": seasonal + 6.0,
            "tmin": seasonal - 6.0,
            "prcp": np.where(doy % 7 == 0, 4.0, 0.0),
        },
        index=index,
    )


@pytest.fixture
def staged(isolated_paths, write_raw):
    """Put a clean frame on disk for one city, as a completed fetch would."""

    def _stage(slug: str, frame: pd.DataFrame, elevation: float = 650.0) -> None:
        load.write_processed(slug, frame)
        write_raw(
            slug,
            {
                "slug": slug,
                "grid_latitude": 40.0,
                "grid_longitude": -3.0,
                "grid_elevation_m": elevation,
                "model": "era5_seamless",
                "daily": {},
            },
        )

    return _stage


# --- the gate -----------------------------------------------------------------


def test_a_clean_long_frame_passes(staged):
    staged("madrid", _long_frame())
    row = validate.quality_matrix().loc["madrid"]
    assert row["verdict"] == "pass"
    assert row["reasons"] == ""
    assert row["years"] >= validate.MIN_YEARS


def test_an_unfetched_city_is_reported_not_skipped(staged):
    """A partial fetch must not be able to look like a clean run."""
    staged("madrid", _long_frame())
    matrix = validate.quality_matrix()
    assert len(matrix) == 20
    assert matrix.loc["porto", "verdict"] == "not_fetched"
    assert "madrid" in validate.passing_slugs()
    assert "porto" not in validate.passing_slugs()


def test_a_short_record_fails(staged):
    staged("madrid", _long_frame(years=10))
    row = validate.quality_matrix().loc["madrid"]
    assert row["verdict"] == "fail"
    assert "only 10" in row["reasons"]


def test_excess_missingness_fails(staged):
    frame = _long_frame()
    frame.iloc[:500, frame.columns.get_loc("prcp")] = np.nan
    staged("madrid", frame)
    row = validate.quality_matrix().loc["madrid"]
    assert row["verdict"] == "fail"
    assert "prcp" in row["reasons"]
    assert row["missing_prcp"] > 0


def test_negative_precipitation_fails():
    frame = tidy_frame(days=10)
    frame.iloc[3, frame.columns.get_loc("prcp")] = -0.4
    assert validate._impossible_counts(frame)["negative_prcp"] == 1


def test_tmin_above_tmax_fails():
    frame = tidy_frame(days=10)
    frame.iloc[2, frame.columns.get_loc("tmin")] = 99.0
    counts = validate._impossible_counts(frame)
    assert counts["tmin_above_tmax"] == 1


def test_a_fahrenheit_mixup_is_caught_by_the_sanity_envelope():
    """The failure mode the envelope exists for: data arriving in the wrong unit."""
    frame = tidy_frame(days=10)
    frame[["tavg", "tmax", "tmin"]] = frame[["tavg", "tmax", "tmin"]] * 9 / 5 + 32
    assert validate._impossible_counts(frame)["temp_out_of_range"] > 0


def test_a_sentinel_value_is_caught():
    frame = tidy_frame(days=10)
    frame.iloc[5, frame.columns.get_loc("tavg")] = -9999.0
    assert validate._impossible_counts(frame)["temp_out_of_range"] == 1


def test_genuine_iberian_extremes_are_not_flagged():
    """The envelope must catch defects without clipping real weather."""
    frame = tidy_frame(days=10)
    frame.iloc[0, frame.columns.get_loc("tmax")] = 47.6   # Spanish record high
    frame.iloc[1, frame.columns.get_loc("tmin")] = -24.0  # Pyrenean cold
    frame.iloc[2, frame.columns.get_loc("prcp")] = 300.0  # a severe gota fría
    counts = validate._impossible_counts(frame)
    assert counts["temp_out_of_range"] == 0
    assert counts["prcp_out_of_range"] == 0


def test_the_matrix_records_the_served_elevation(staged):
    """Granada and Huesca are flagged Altitude precisely because this differs."""
    staged("granada", _long_frame(), elevation=1023.0)
    assert validate.quality_matrix().loc["granada", "grid_elevation_m"] == 1023.0


# --- the agreement statistics -------------------------------------------------


def test_agreement_on_identical_series_is_perfect():
    series = pd.Series(
        [1.0, 2.0, 3.0, 4.0], index=pd.date_range("2020-01-01", periods=4, name="date")
    )
    stats = validate._agreement(series, series)
    assert stats["bias"] == 0.0
    assert stats["mae"] == 0.0
    assert stats["r"] == pytest.approx(1.0)
    assert stats["n"] == 4


def test_agreement_uses_only_dates_both_series_cover():
    index = pd.date_range("2020-01-01", periods=5, name="date")
    obs = pd.Series([1.0, np.nan, 3.0, 4.0, np.nan], index=index)
    era = pd.Series([2.0, 2.0, 4.0, np.nan, 5.0], index=index)
    stats = validate._agreement(obs, era)
    assert stats["n"] == 2
    assert stats["bias"] == pytest.approx(1.0)


def test_agreement_reports_a_constant_offset_as_bias_not_error():
    index = pd.date_range("2020-01-01", periods=50, name="date")
    obs = pd.Series(np.linspace(0, 20, 50), index=index)
    stats = validate._agreement(obs, obs + 1.5)
    assert stats["bias"] == pytest.approx(1.5)
    assert stats["r"] == pytest.approx(1.0)


def test_agreement_on_no_overlap_returns_nothing_rather_than_nan_soup():
    a = pd.Series([1.0], index=pd.date_range("2020-01-01", periods=1, name="date"))
    b = pd.Series([1.0], index=pd.date_range("2021-01-01", periods=1, name="date"))
    assert validate._agreement(a, b) == {"n": 0}


# --- the comparison's scope ---------------------------------------------------


def test_comparison_rejects_a_city_without_station_data():
    with pytest.raises(KeyError, match="no station record"):
        validate.compare_with_ghcn("madrid")


def test_valencia_offers_no_precipitation_comparison():
    """17% missing station PRCP cannot found a precipitation validation."""
    assert "prcp" not in validate._USABLE["valencia"]
    assert "prcp" in validate._USABLE["huesca"]


def test_huesca_tavg_carries_the_definitional_caveat():
    """Midrange vs true 24 h mean — a bias that is not reanalysis error."""
    assert "midrange" in validate._USABLE["huesca"]["tavg"]
    assert validate._USABLE["huesca"]["tmax"] == "", "tmax is like-for-like"


def test_wet_day_agreement_refuses_valencia():
    with pytest.raises(KeyError, match="only defensible for 'huesca'"):
        validate.wet_day_agreement("valencia")


def test_wet_day_contingency_table_is_consistent(staged):
    frame = _long_frame(years=70, slug="huesca")
    staged("huesca", frame)
    scores = validate.wet_day_agreement("huesca")
    assert scores["hits"] + scores["misses"] == scores["obs_wet_days"]
    assert scores["hits"] + scores["false_alarms"] == scores["era_wet_days"]
    total = scores["hits"] + scores["misses"] + scores["false_alarms"] + scores["correct_dry"]
    assert total == scores["n"]
    assert 0.0 <= scores["hit_rate"] <= 1.0


def test_a_corrupt_processed_file_fails_the_city_without_crashing_the_matrix(
    staged, isolated_paths
):
    """A gate that raises on bad data is not a gate."""
    staged("madrid", _long_frame())
    load.processed_path("porto").write_text("date,tavg\nnot-a-date,x\n")
    matrix = validate.quality_matrix()
    assert matrix.loc["porto", "verdict"] == "fail"
    assert "unreadable" in matrix.loc["porto", "reasons"]
    assert matrix.loc["madrid", "verdict"] == "pass"


def test_a_missing_raw_payload_is_not_fatal(staged, isolated_paths):
    """Provenance becomes unknown, but completeness can still be judged."""
    from wepr import paths

    staged("madrid", _long_frame())
    (paths.DATA_RAW / "madrid.json").unlink()
    row = validate.quality_matrix().loc["madrid"]
    assert row["verdict"] == "pass"
    assert pd.isna(row["grid_elevation_m"])


def test_reasons_is_always_a_string(staged):
    staged("madrid", _long_frame())
    assert validate.quality_matrix()["reasons"].map(type).eq(str).all()


# --- eligibility: which cities a modelling script may use ---------------------
#
# This lives in `validate` rather than in a script because `train_all` and
# `evaluate_all` must agree. If they disagreed, a model could be trained and never
# evaluated, or evaluated and never deployed — which is the class of mismatch the
# whole project is built to exclude.


def test_eligible_slugs_defaults_to_the_gates_verdict(staged):
    staged("madrid", _long_frame())
    staged("porto", _long_frame().tail(500))  # far too short to pass

    slugs, basis = validate.eligible_slugs()
    assert slugs == ["madrid"]
    assert "gate" in basis


def test_ignore_gate_accepts_everything_on_disk(staged):
    """The escape hatch for a partial archive, labelled as one."""
    staged("madrid", _long_frame())
    staged("porto", _long_frame().tail(500))

    slugs, basis = validate.eligible_slugs(ignore_gate=True)
    assert slugs == ["madrid", "porto"]
    assert "gate ignored" in basis


def test_requesting_a_failing_city_explains_the_override(staged):
    staged("porto", _long_frame().tail(500))
    with pytest.raises(KeyError, match="--ignore-gate"):
        validate.eligible_slugs(["porto"])


def test_requesting_a_failing_city_is_allowed_with_the_override(staged):
    staged("porto", _long_frame().tail(500))
    slugs, _ = validate.eligible_slugs(["porto"], ignore_gate=True)
    assert slugs == ["porto"]


def test_requesting_a_city_with_no_data_points_at_the_fetch_script(staged):
    staged("madrid", _long_frame())
    with pytest.raises(KeyError, match="fetch_data.py"):
        validate.eligible_slugs(["zaragoza"])


def test_no_eligible_cities_is_an_empty_list_not_an_error(isolated_paths):
    """Exactly today's situation: a partial archive and nothing passing the gate."""
    assert validate.eligible_slugs() == ([], "0 city/cities passing the data-quality gate")

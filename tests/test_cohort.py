"""Tests for GENIE BPC recurrent assessment interval construction and shift operations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from calendar_shift_audit.cohort import (
    build_frame,
    exclusion_summary,
    outer_split,
    shift_frame,
    simple_numeric_categorical_design,
    summarize_cohort,
)


def _sample_release_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    patients = pd.DataFrame(
        [
            {"record_id": "P1", "institution": "DFCI", "dob_lastalive_int": 150, "hybrid_death_int": np.nan},
            {"record_id": "P2", "institution": "MSK", "dob_lastalive_int": 90, "hybrid_death_int": 80},
            {"record_id": "P3", "institution": "UHN", "dob_lastalive_int": 200, "hybrid_death_int": np.nan},
            {"record_id": "P4", "institution": "VICC", "dob_lastalive_int": 200, "hybrid_death_int": np.nan},
        ]
    )
    marks = pd.DataFrame(
        [
            {"record_id": "P1", "day": 10, "is_imaging": 0},
            {"record_id": "P1", "day": 10, "is_imaging": 1},
            {"record_id": "P1", "day": 45, "is_imaging": 1},
            {"record_id": "P1", "day": 95, "is_imaging": 0},
            {"record_id": "P2", "day": 20, "is_imaging": 1},
            {"record_id": "P2", "day": 80, "is_imaging": 1},
            {"record_id": "P3", "day": 30, "is_imaging": 1},
            {"record_id": "P4", "day": np.nan, "is_imaging": 1},
        ]
    )
    return patients, marks


def test_build_frame_collapses_same_day_and_tracks_exclusions() -> None:
    patients, marks = _sample_release_tables()
    frame, audit, source_counts = build_frame(patients, marks)

    assert source_counts["same_patient_day_rows_collapsed"] == 1
    excl_all = exclusion_summary(audit)[0]
    assert excl_all["institution"] == "ALL"
    assert excl_all["release_patients"] == 4
    assert excl_all["retained_patients"] == 2
    by_reason = {item["reason"]: item["excluded"] for item in excl_all["sequential_flow"]}
    assert by_reason["no_valid_assessment_day"] == 1
    assert by_reason["only_one_distinct_assessment_day"] == 1

    p1 = frame[frame["record_id"] == "P1"].sort_values("order_in_patient")
    assert len(p1) == 3
    assert list(p1["event"]) == [1, 1, 0]
    assert int(p1.iloc[0]["start_is_imaging"]) == 1
    assert list(p1["exposure_days"]) == [35.0, 50.0, 55.0]

    p2 = frame[frame["record_id"] == "P2"]
    assert len(p2) == 1
    assert int(p2.iloc[0]["event"]) == 1

    counts = summarize_cohort(frame)
    assert counts["patients"] == 2
    assert counts["completed_intervals"] == 3
    assert counts["terminal_intervals"] == 1


def test_outer_split_and_shift_frame_preserve_exposures() -> None:
    patients, marks = _sample_release_tables()
    frame, _, _ = build_frame(patients, marks)
    dev, ev = outer_split(frame, "MSK")
    assert set(dev["record_id"]) == {"P1"}
    assert set(ev["record_id"]) == {"P2"}

    shifted_global = shift_frame(frame, 4)
    assert np.allclose(shifted_global["interval_start"] - frame["interval_start"], 4.0)
    assert np.allclose(shifted_global["interval_stop"] - frame["interval_stop"], 4.0)
    assert np.allclose(shifted_global["exposure_days"], frame["exposure_days"])

    shifted_map = shift_frame(frame, {"P1": 2, "P2": 6})
    p1_s = shifted_map[shifted_map["record_id"] == "P1"]
    p2_s = shifted_map[shifted_map["record_id"] == "P2"]
    assert np.allclose(p1_s["interval_start"] - frame[frame["record_id"] == "P1"]["interval_start"], 2.0)
    assert np.allclose(p2_s["interval_start"] - frame[frame["record_id"] == "P2"]["interval_start"], 6.0)


def test_design_uses_development_statistics_only() -> None:
    dev = pd.DataFrame({"x": [10.0, 20.0, 30.0], "cat": ["A", "B", "B"]})
    ev_a = pd.DataFrame({"x": [20.0, 40.0], "cat": ["A", "B"]})
    ev_b = pd.DataFrame({"x": [20.0, 1000.0], "cat": ["A", "B"]})

    _, X_a, names = simple_numeric_categorical_design(
        dev, ev_a, numeric=("x",), categorical=("cat",), minimum_count=1
    )
    _, X_b, _ = simple_numeric_categorical_design(
        dev, ev_b, numeric=("x",), categorical=("cat",), minimum_count=1
    )
    assert names == ("num:x", "cat:cat=B")
    assert np.allclose(X_a[0], X_b[0])

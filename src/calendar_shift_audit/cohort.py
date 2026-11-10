"""Recurrent assessment cohort construction and shift helpers for GENIE BPC."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


FRAME_COLUMNS: List[str] = [
    "record_id",
    "interval_start",
    "interval_stop",
    "exposure_days",
    "event",
    "order_in_patient",
    "start_is_imaging",
    "institution",
]

KNOWN_SITES = {"DFCI", "MSK", "UHN", "VICC"}
DISEASE_SITES: Dict[str, Tuple[str, ...]] = {
    "nsclc": ("DFCI", "MSK", "UHN", "VICC"),
    "crc": ("DFCI", "MSK", "VICC"),
}


def summarize_cohort(frame: pd.DataFrame) -> Dict[str, Any]:
    def _counts(block: pd.DataFrame) -> Dict[str, int]:
        return {
            "patients": int(block["record_id"].nunique()),
            "intervals": int(len(block)),
            "completed_intervals": int(block["event"].eq(1).sum()),
            "terminal_intervals": int(block["event"].eq(0).sum()),
        }

    result: Dict[str, Any] = _counts(frame)
    result["by_site"] = {
        str(site): _counts(block)
        for site, block in frame.groupby("institution", sort=True)
    }
    return result


def build_frame(
    patients: pd.DataFrame, marks: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Construct recurrent assessment intervals and terminal censoring rows.

    Each patient requires a known institution and at least two distinct finite
    numeric imaging or medical-oncology assessment days. Same-day imaging and
    oncology records collapse to a single assessment day with start_is_imaging
    equal to the maximum indicator on that day. A right-censored terminal
    interval is appended only when min(dob_lastalive_int, hybrid_death_int)
    exceeds the final observed assessment day.
    """
    if patients["record_id"].isna().any() or patients["record_id"].duplicated().any():
        raise ValueError("Patient table contains missing or duplicate record_id values")

    patients = patients.copy()
    patients["record_id"] = patients["record_id"].astype(str)

    marks = marks.copy()
    marks["day"] = pd.to_numeric(marks["day"], errors="coerce")
    usable = marks["record_id"].notna() & np.isfinite(marks["day"])
    source_counts: Dict[str, int] = {
        "source_rows": int(len(marks)),
        "rows_without_valid_patient_key_or_numeric_finite_day": int((~usable).sum()),
    }

    marks = marks.loc[usable].copy()
    marks["record_id"] = marks["record_id"].astype(str)
    source_counts["rows_with_noninteger_day_rounded"] = int(
        (marks["day"] != np.rint(marks["day"])).sum()
    )
    marks["day"] = np.rint(marks["day"]).astype(np.int64)

    orphan = ~marks["record_id"].isin(patients["record_id"])
    source_counts["valid_assessment_rows_outside_release_patient_table"] = int(orphan.sum())
    source_counts["assessment_identifiers_outside_release_patient_table"] = int(
        marks.loc[orphan, "record_id"].nunique()
    )
    marks = marks.loc[~orphan].copy()

    union = marks.groupby(["record_id", "day"], as_index=False)["is_imaging"].max()
    source_counts["same_patient_day_rows_collapsed"] = int(len(marks) - len(union))
    source_counts["unique_release_patient_assessment_days"] = int(len(union))

    groups = {
        pid: block.sort_values("day", kind="mergesort")
        for pid, block in union.groupby("record_id", sort=False)
    }

    rows: List[Tuple[Any, ...]] = []
    audit: List[Dict[str, Any]] = []
    for p in patients.itertuples(index=False):
        pid = str(p.record_id)
        site = str(p.institution)
        block = groups.get(pid)
        n_days = 0 if block is None else len(block)

        if site not in KNOWN_SITES:
            exclusion = "unknown_institution"
        elif n_days == 0:
            exclusion = "no_valid_assessment_day"
        elif n_days == 1:
            exclusion = "only_one_distinct_assessment_day"
        else:
            exclusion = ""

        audit_row: Dict[str, Any] = {
            "record_id": pid,
            "institution": site,
            "assessment_days": n_days,
            "included": not bool(exclusion),
            "exclusion_reason": exclusion,
            "completed_intervals": 0,
            "terminal_intervals": 0,
            "terminal_reason": "not_included",
        }
        if exclusion:
            audit.append(audit_row)
            continue

        day = block["day"].to_numpy(dtype=np.int64)
        imaging = block["is_imaging"].to_numpy(dtype=np.int64)
        candidates = [
            float(value)
            for value in (p.dob_lastalive_int, p.hybrid_death_int)
            if pd.notna(value) and np.isfinite(float(value))
        ]
        boundary = min(candidates) if candidates else np.nan

        for position in range(n_days - 1):
            rows.append(
                (
                    pid,
                    float(day[position]),
                    float(day[position + 1]),
                    float(day[position + 1] - day[position]),
                    1,
                    position,
                    int(imaging[position]),
                    site,
                )
            )

        if not candidates:
            reason = "missing_boundary"
        elif boundary < day[0]:
            reason = "boundary_before_first_assessment"
        elif boundary <= day[-1]:
            reason = "boundary_at_or_before_last_assessment"
        else:
            reason = "positive_terminal_interval"
            rows.append(
                (
                    pid,
                    float(day[-1]),
                    boundary,
                    float(boundary - day[-1]),
                    0,
                    n_days - 1,
                    int(imaging[-1]),
                    site,
                )
            )

        audit_row.update(
            {
                "completed_intervals": n_days - 1,
                "terminal_intervals": int(reason == "positive_terminal_interval"),
                "terminal_reason": reason,
                "first_assessment_day": float(day[0]),
                "last_assessment_day": float(day[-1]),
                "selected_terminal_boundary": boundary,
            }
        )
        audit.append(audit_row)

    frame = (
        pd.DataFrame(rows, columns=FRAME_COLUMNS)
        .sort_values(["record_id", "interval_start", "interval_stop"], kind="mergesort")
        .reset_index(drop=True)
    )
    patient_audit = (
        pd.DataFrame(audit)
        .sort_values("record_id", kind="mergesort")
        .reset_index(drop=True)
    )

    if not frame.empty:
        if not ((frame["exposure_days"] > 0) & (frame["interval_stop"] > frame["interval_start"])).all():
            raise AssertionError("Non-positive interval duration in constructed cohort frame")
        if frame.duplicated(["record_id", "interval_start", "event"]).any():
            raise AssertionError("Duplicate recurrent interval in constructed cohort frame")
        if (frame.loc[frame["event"].eq(0)].groupby("record_id").size() > 1).any():
            raise AssertionError("More than one terminal interval for a patient")
        if set(frame["record_id"]) != set(patient_audit.loc[patient_audit["included"], "record_id"]):
            raise AssertionError("Patient audit and interval frame disagree")

    return frame, patient_audit, source_counts


def exclusion_summary(patient_audit: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    groups = [("ALL", patient_audit)] + list(patient_audit.groupby("institution", sort=True))
    for site, block in groups:
        remaining = len(block)
        flow = []
        for reason in (
            "unknown_institution",
            "no_valid_assessment_day",
            "only_one_distinct_assessment_day",
        ):
            excluded = int(block["exclusion_reason"].eq(reason).sum())
            flow.append(
                {
                    "reason": reason,
                    "before": remaining,
                    "excluded": excluded,
                    "remaining": remaining - excluded,
                }
            )
            remaining -= excluded
        rows.append(
            {
                "institution": str(site),
                "release_patients": int(len(block)),
                "retained_patients": int(remaining),
                "excluded_patients": int(len(block) - remaining),
                "sequential_flow": flow,
            }
        )
    return rows


def load_genie_release_tables(release_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read patient, imaging, and medical-oncology CSVs from a GENIE BPC release folder."""
    release_dir = Path(release_dir)
    patient_path = release_dir / "patient_level_dataset.csv"
    patients = pd.read_csv(
        patient_path,
        usecols=["record_id", "institution", "dob_lastalive_int", "hybrid_death_int"],
        low_memory=False,
    )
    for column in ("dob_lastalive_int", "hybrid_death_int"):
        patients[column] = pd.to_numeric(patients[column], errors="coerce")

    blocks = []
    for filename, field, imaging in (
        ("imaging_level_dataset.csv", "image_scan_int", 1),
        ("med_onc_note_level_dataset.csv", "md_onc_visit_int", 0),
    ):
        table = pd.read_csv(
            release_dir / filename,
            usecols=["record_id", field],
            low_memory=False,
        ).rename(columns={field: "day"})
        table["is_imaging"] = imaging
        blocks.append(table)
    return patients, pd.concat(blocks, ignore_index=True)


def outer_split(
    frame: pd.DataFrame, held_out_institution: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-hospital-out whole-patient split."""
    mask = frame["institution"].astype(str) == str(held_out_institution)
    development = frame.loc[~mask].reset_index(drop=True)
    evaluation = frame.loc[mask].reset_index(drop=True)
    if development.empty or evaluation.empty:
        raise ValueError(f"Empty partition for held-out hospital {held_out_institution}")
    if set(development["record_id"]) & set(evaluation["record_id"]):
        raise AssertionError("Patient IDs overlap across hospital split")
    return development, evaluation


def shift_frame(frame: pd.DataFrame, shifts: Mapping[str, int] | int) -> pd.DataFrame:
    """Apply whole-day calendar origin shifts while holding durations and outcomes fixed."""
    out = frame.copy()
    if isinstance(shifts, int):
        delta = float(shifts)
        out["interval_start"] = out["interval_start"] + delta
        out["interval_stop"] = out["interval_stop"] + delta
    else:
        mapped = out["record_id"].astype(str).map(lambda pid: int(shifts.get(pid, 0)))
        delta_arr = mapped.to_numpy(dtype=float)
        out["interval_start"] = out["interval_start"] + delta_arr
        out["interval_stop"] = out["interval_stop"] + delta_arr
    return out


def simple_numeric_categorical_design(
    development: pd.DataFrame,
    evaluation: pd.DataFrame,
    numeric: Sequence[str] = (),
    categorical: Sequence[str] = (),
    minimum_count: int = 20,
) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...]]:
    """Standardize numeric features and encode categorical levels using development data only."""
    names: List[str] = []
    dev_cols: List[np.ndarray] = []
    ev_cols: List[np.ndarray] = []

    for column in numeric:
        if column not in development.columns:
            continue
        d = pd.to_numeric(development[column], errors="coerce").astype(float)
        e = pd.to_numeric(evaluation[column], errors="coerce").astype(float)
        median = float(d.median()) if np.isfinite(d.median()) else 0.0
        d = d.fillna(median)
        e = e.fillna(median)
        mean = float(d.mean())
        scale = float(d.std(ddof=0)) or 1.0
        dev_cols.append(((d - mean) / scale).to_numpy())
        ev_cols.append(((e - mean) / scale).to_numpy())
        names.append(f"num:{column}")

    for column in categorical:
        if column not in development.columns:
            continue
        text = development[column].astype(str).str.strip()
        counts = text.value_counts()
        keep = sorted(
            str(k)
            for k, v in counts.items()
            if int(v) >= minimum_count and str(k) not in ("", "nan", "None")
        )
        for level in keep[1:]:
            dev_cols.append((text == level).to_numpy(dtype=float))
            ev_cols.append(
                (evaluation[column].astype(str).str.strip() == level).to_numpy(dtype=float)
            )
            names.append(f"cat:{column}={level}")

    if not dev_cols:
        dev_cols = [np.ones(len(development), dtype=float)]
        ev_cols = [np.ones(len(evaluation), dtype=float)]
        names = ["num:intercept"]

    return np.column_stack(dev_cols), np.column_stack(ev_cols), tuple(names)

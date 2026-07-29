#!/usr/bin/env python3
"""Evaluate target construction and eligibility filtering at fixed alert capacity."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "patient_id",
    "state",
    "incident_event",
    "incident_score",
    "history_score",
}
VALID_STATES = {"never_exposed", "active", "resolved"}


def validate_input(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    invalid_states = set(frame["state"].dropna().unique()).difference(VALID_STATES)
    if invalid_states:
        raise ValueError(f"Invalid states: {sorted(invalid_states)}")
    if not frame["incident_event"].isin([0, 1]).all():
        raise ValueError("incident_event must contain only 0 and 1")
    if not np.isfinite(frame[["incident_score", "history_score"]]).all().all():
        raise ValueError("Scores must be finite")
    impossible = frame["incident_event"].eq(1) & frame["state"].ne("never_exposed")
    if impossible.any():
        raise ValueError("Incident events must occur within the never-exposed risk set")


def evaluate_cell(
    frame: pd.DataFrame,
    *,
    score_column: str,
    eligibility_filter: bool,
    alert_capacity: int,
) -> dict[str, float | int]:
    candidates = (
        frame.loc[frame["state"].eq("never_exposed")]
        if eligibility_filter
        else frame
    )
    if alert_capacity > len(candidates):
        raise ValueError("Alert capacity exceeds the number of eligible candidates")

    ranked = candidates.sort_values(
        [score_column, "patient_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    alerted = ranked.head(alert_capacity)
    total_events = int(frame["incident_event"].sum())
    incident_alerts = int(alerted["incident_event"].sum())
    ineligible_alerts = int(alerted["state"].ne("never_exposed").sum())

    return {
        "alerts": int(alert_capacity),
        "incident_alerts": incident_alerts,
        "ineligible_alerts": ineligible_alerts,
        "incident_sensitivity": incident_alerts / total_events if total_events else np.nan,
        "incident_ppv": incident_alerts / alert_capacity if alert_capacity else np.nan,
        "ineligible_alert_fraction": (
            ineligible_alerts / alert_capacity if alert_capacity else np.nan
        ),
        "alerts_per_incident_detected": (
            alert_capacity / incident_alerts if incident_alerts else np.inf
        ),
    }


def evaluate_factorial(frame: pd.DataFrame, budget_fraction: float) -> pd.DataFrame:
    validate_input(frame)
    if not 0 < budget_fraction <= 1:
        raise ValueError("budget_fraction must be in (0, 1]")
    capacity = math.ceil(len(frame) * budget_fraction)
    cells = [
        ("A", "incident", True, "incident_score"),
        ("B", "incident", False, "incident_score"),
        ("C", "history_inclusive", True, "history_score"),
        ("D", "history_inclusive", False, "history_score"),
    ]
    rows = []
    for cell, target, filtered, score_column in cells:
        result = evaluate_cell(
            frame,
            score_column=score_column,
            eligibility_filter=filtered,
            alert_capacity=capacity,
        )
        rows.append(
            {
                "cell": cell,
                "target_construction": target,
                "eligibility_filter": filtered,
                "budget_fraction": budget_fraction,
                "total_rows": len(frame),
                "total_incident_events": int(frame["incident_event"].sum()),
                **result,
            }
        )
    return pd.DataFrame(rows)


def self_test() -> None:
    frame = pd.DataFrame(
        {
            "patient_id": np.arange(20),
            "state": ["never_exposed"] * 12 + ["active"] * 4 + ["resolved"] * 4,
            "incident_event": [1, 1, 1] + [0] * 17,
            "incident_score": [
                0.95,
                0.90,
                0.85,
                0.80,
                0.70,
                0.60,
                0.50,
                0.40,
                0.30,
                0.20,
                0.10,
                0.05,
                0.75,
                0.65,
                0.55,
                0.45,
                0.35,
                0.25,
                0.15,
                0.01,
            ],
            "history_score": [
                0.60,
                0.55,
                0.50,
                0.45,
                0.40,
                0.35,
                0.30,
                0.25,
                0.20,
                0.15,
                0.10,
                0.05,
                0.99,
                0.98,
                0.97,
                0.96,
                0.95,
                0.94,
                0.93,
                0.92,
            ],
        }
    )
    result = evaluate_factorial(frame, 0.20).set_index("cell")
    assert result.loc["A", "incident_alerts"] == 3
    assert result.loc["A", "ineligible_alerts"] == 0
    assert result.loc["D", "ineligible_alerts"] == 4
    print("Self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--budget", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    if args.input is None or args.output is None:
        parser.error("--input and --output are required unless --self-test is used")

    frame = pd.read_csv(args.input)
    result = evaluate_factorial(frame, args.budget)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

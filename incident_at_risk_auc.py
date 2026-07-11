#!/usr/bin/env python3
"""Fixed-score AUROC decomposition for incident-at-risk evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import roc_auc_score


ALLOWED_GROUPS = ("negative", "incident", "active_support", "resolved_history")


def pairwise_auc(negative: np.ndarray, positive: np.ndarray) -> float:
    """Compute AUROC for one positive state against shared negatives."""
    if len(negative) == 0 or len(positive) == 0:
        raise ValueError("Both negative and positive score arrays must be non-empty")
    labels = np.r_[np.zeros(len(negative)), np.ones(len(positive))]
    scores = np.r_[negative, positive]
    return float(roc_auc_score(labels, scores))


def decompose_scores(groups: dict[str, np.ndarray]) -> dict[str, float | int]:
    """Reconstruct mixed-positive AUROC and state-specific contributions exactly."""
    required = set(ALLOWED_GROUPS)
    missing = required.difference(groups)
    if missing:
        raise ValueError(f"Missing event groups: {sorted(missing)}")
    if any(len(groups[name]) == 0 for name in ALLOWED_GROUPS):
        raise ValueError("Every event group must contain at least one score")

    negative = groups["negative"]
    positive_names = ("incident", "active_support", "resolved_history")
    positive_total = sum(len(groups[name]) for name in positive_names)
    aucs = {name: pairwise_auc(negative, groups[name]) for name in positive_names}
    weights = {name: len(groups[name]) / positive_total for name in positive_names}

    mixed_positive = np.concatenate([groups[name] for name in positive_names])
    observed = pairwise_auc(negative, mixed_positive)
    reconstructed = sum(weights[name] * aucs[name] for name in positive_names)
    incident_auc = aucs["incident"]

    result: dict[str, float | int] = {
        "n_negative": int(len(negative)),
        "n_positive_total": int(positive_total),
        "auc_observed_mixture": observed,
        "auc_reconstructed_mixture": float(reconstructed),
        "identity_error": float(observed - reconstructed),
        "fixed_mixture_increment": float(observed - incident_auc),
    }
    for name in positive_names:
        result[f"n_{name}"] = int(len(groups[name]))
        result[f"weight_{name}"] = float(weights[name])
        result[f"auc_{name}_vs_negative"] = float(aucs[name])
        if name != "incident":
            result[f"contribution_{name}"] = float(
                weights[name] * (aucs[name] - incident_auc)
            )
    return result


def load_scores(path: Path) -> dict[str, np.ndarray]:
    """Load authorized local scores without retaining identifiers or other columns."""
    values: dict[str, list[float]] = {name: [] for name in ALLOWED_GROUPS}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"event_group", "y_pred"}.issubset(reader.fieldnames):
            raise ValueError("Input must contain event_group and y_pred columns")
        for row in reader:
            group = row["event_group"]
            if group not in values:
                raise ValueError(f"Unsupported event_group: {group}")
            values[group].append(float(row["y_pred"]))
    return {name: np.asarray(scores, dtype=float) for name, scores in values.items()}


def self_test() -> dict[str, float | int]:
    """Verify the exact identity with synthetic scores only."""
    groups = {
        "negative": np.asarray([0.05, 0.15, 0.25, 0.35, 0.45]),
        "incident": np.asarray([0.20, 0.30, 0.50]),
        "active_support": np.asarray([0.75, 0.85, 0.95]),
        "resolved_history": np.asarray([0.55, 0.65]),
    }
    result = decompose_scores(groups)
    if abs(float(result["identity_error"])) > 1e-12:
        raise AssertionError("AUROC positive-mixture identity failed")
    contributions = float(result["contribution_active_support"]) + float(
        result["contribution_resolved_history"]
    )
    if not np.isclose(contributions, float(result["fixed_mixture_increment"])):
        raise AssertionError("State contributions did not sum to the mixture increment")
    return result


def write_result(result: dict[str, float | int], output: Path | None) -> None:
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(rendered, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Authorized local CSV with event_group,y_pred")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    parser.add_argument("--self-test", action="store_true", help="Run a synthetic-data identity check")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        write_result(self_test(), args.output)
        return 0
    if args.input is None:
        raise SystemExit("Provide --input or use --self-test")
    write_result(decompose_scores(load_scores(args.input)), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


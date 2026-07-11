#!/usr/bin/env python3
"""Regenerate the three main figures from aggregate source data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "figure_source_data.json"
COLORS = {"Logistic Regression": "#246B8E", "LightGBM": "#D97728"}
FIGURE_1_BOXES = {
    "full_cohort": (0.04, 0.42),
    "active_at_landmark": (0.38, 0.70),
    "resolved_before_landmark": (0.38, 0.42),
    "never_exposed_at_landmark": (0.38, 0.14),
    "strict_incident_12_36h": (0.72, 0.24),
    "no_first_initiation_12_36h": (0.72, 0.04),
}


def validate_figure_1_layout() -> None:
    """Prevent state partitions from appearing as one sequential pathway."""
    state_y = {
        FIGURE_1_BOXES[name][1]
        for name in (
            "active_at_landmark",
            "resolved_before_landmark",
            "never_exposed_at_landmark",
        )
    }
    if len(state_y) != 3:
        raise ValueError("Figure 1 state partitions must occupy three distinct branches")


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_1(data: dict, output_dir: Path) -> None:
    validate_figure_1_layout()
    rows = data["figure_1"]["groups"]
    by_name = {row["group"]: row for row in rows}
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axis("off")
    boxes = [
        ("full_cohort", "Full landmark cohort", "#E8EEF2"),
        ("active_at_landmark", "Active ventilation", "#F4D6D2"),
        ("resolved_before_landmark", "Resolved history", "#F7E7C6"),
        ("never_exposed_at_landmark", "Never exposed", "#DDEFE5"),
        ("strict_incident_12_36h", "Incident first initiation", "#CFE5F2"),
        ("no_first_initiation_12_36h", "No first initiation", "#E9ECEF"),
    ]
    width, height = 0.24, 0.18
    for key, label, color in boxes:
        x, y = FIGURE_1_BOXES[key]
        count = by_name[key]["n"]
        ax.add_patch(plt.Rectangle((x, y), 0.24, 0.18, facecolor=color, edgecolor="#4B5563"))
        ax.text(x + width / 2, y + 0.105, label, ha="center", va="center", fontsize=9)
        ax.text(x + width / 2, y + 0.045, f"n = {count:,}", ha="center", va="center", fontsize=10, weight="bold")

    def arrow(source: str, target: str) -> None:
        sx, sy = FIGURE_1_BOXES[source]
        tx, ty = FIGURE_1_BOXES[target]
        ax.annotate(
            "",
            xy=(tx, ty + height / 2),
            xytext=(sx + width, sy + height / 2),
            arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=0.0"},
        )

    for target in ("active_at_landmark", "resolved_before_landmark", "never_exposed_at_landmark"):
        arrow("full_cohort", target)
    for target in ("strict_incident_12_36h", "no_first_initiation_12_36h"):
        arrow("never_exposed_at_landmark", target)
    ax.text(0.51, -0.045, "Incident-at-risk evaluation excludes active and resolved prior states", ha="center", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.08, 1)
    save_figure(fig, output_dir, "Figure_1")


def figure_2(data: dict, output_dir: Path) -> None:
    decomposition = [row for row in data["figure_2"]["decomposition"] if row["component"] == "mv"]
    dose = data["figure_2"]["dose_response"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    models = ["Logistic Regression", "LightGBM"]
    x = np.arange(len(models))
    active = [next(row for row in decomposition if row["model"] == model)["contribution_active_support"] for model in models]
    resolved = [next(row for row in decomposition if row["model"] == model)["contribution_resolved_history"] for model in models]
    axes[0].bar(x, active, color="#C45D4A", label="Active state")
    axes[0].bar(x, resolved, bottom=active, color="#D6A44A", label="Resolved history")
    axes[0].set_xticks(x, ["LR", "LightGBM"])
    axes[0].set_ylabel("Contribution to AUROC increment")
    axes[0].set_title("A  State-resolved contribution", loc="left")
    axes[0].legend(frameon=False)

    incident = [next(row for row in decomposition if row["model"] == model)["auc_incident_vs_negative"] for model in models]
    mixture = [next(row for row in decomposition if row["model"] == model)["auc_observed_mixture"] for model in models]
    for index, model in enumerate(models):
        axes[1].plot([0, 1], [incident[index], mixture[index]], marker="o", color=COLORS[model], label=model)
    axes[1].set_xticks([0, 1], ["Incident only", "History inclusive"])
    axes[1].set_ylim(0.5, 1.0)
    axes[1].set_ylabel("Fixed-score AUROC")
    axes[1].set_title("B  Apparent discrimination", loc="left")
    axes[1].legend(frameon=False)

    for model in models:
        rows = sorted((row for row in dose if row["model"] == model), key=lambda row: row["prior_history_fraction_among_positive_labels"])
        fraction = np.asarray([row["prior_history_fraction_among_positive_labels"] for row in rows])
        estimate = np.asarray([row["expected_auroc"] for row in rows])
        lower = np.asarray([row["ci_lower"] for row in rows])
        upper = np.asarray([row["ci_upper"] for row in rows])
        axes[2].plot(fraction, estimate, color=COLORS[model], label=model)
        axes[2].fill_between(fraction, lower, upper, color=COLORS[model], alpha=0.15)
    axes[2].set_xlabel("Prior-state fraction among positive labels")
    axes[2].set_ylabel("Expected AUROC")
    axes[2].set_title("C  Positive-mixture dose response", loc="left")
    axes[2].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir, "Figure_2")


def figure_3(data: dict, output_dir: Path) -> None:
    defenses = data["figure_3"]["key_defenses"]
    contrast = data["figure_3"]["multilandmark_and_contrast"]
    folds = data["figure_3"]["eicu_folds"]
    global_rows = data["figure_3"]["eicu_global"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))

    labels = []
    positions = []
    cursor = 0
    for scenario in dict.fromkeys(row["display_label"] for row in defenses):
        rows = [row for row in defenses if row["display_label"] == scenario]
        for row in rows:
            positions.append(cursor)
            labels.append(f"{scenario} | {'LR' if row['model'] == 'Logistic Regression' else 'LightGBM'}")
            axes[0].errorbar(
                row["estimate"], cursor,
                xerr=[[row["estimate"] - row["ci_lower"]], [row["ci_upper"] - row["estimate"]]],
                fmt="o", color=COLORS[row["model"]], capsize=2,
            )
            cursor += 1
    axes[0].axvline(0, color="#6B7280", linewidth=1)
    axes[0].set_yticks(positions, labels, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Fixed-score AUROC increment")
    axes[0].set_title("A  Statistical defenses", loc="left")

    canonical = [row for row in contrast if row["scenario_id"] == "G1_L12"]
    for index, model in enumerate(["Logistic Regression", "LightGBM"]):
        mv = next(row for row in canonical if row["component"] == "mv" and row["model"] == model)
        crrt = next(row for row in canonical if row["component"] == "crrt" and row["model"] == model)
        offset = (-0.12, 0.12)[index]
        axes[1].errorbar(
            [mv["fixed_contamination_delta"], crrt["fixed_contamination_delta"]],
            np.asarray([0, 1]) + offset,
            xerr=[
                [mv["fixed_contamination_delta"] - mv["fixed_contamination_delta_ci_lower"], crrt["fixed_contamination_delta"] - crrt["fixed_contamination_delta_ci_lower"]],
                [mv["fixed_contamination_delta_ci_upper"] - mv["fixed_contamination_delta"], crrt["fixed_contamination_delta_ci_upper"] - crrt["fixed_contamination_delta"]],
            ],
            fmt="o", color=COLORS[model], label=model, capsize=3,
        )
    axes[1].axvline(0, color="#6B7280", linewidth=1)
    axes[1].set_yticks([0, 1], ["Mechanical ventilation", "CRRT"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Fixed-score AUROC increment")
    axes[1].set_title("B  Endpoint specificity", loc="left")
    axes[1].legend(frameon=False)

    for model in ["Logistic Regression", "LightGBM"]:
        model_rows = sorted((row for row in folds if row["model"] == model), key=lambda row: row["hospital_fold"])
        axes[2].plot(
            [row["hospital_fold"] + 1 for row in model_rows],
            [row["fixed_contamination_delta"] for row in model_rows],
            marker="o", color=COLORS[model], label=model,
        )
        pooled = next(row for row in global_rows if row["model"] == model)
        axes[2].axhline(pooled["fixed_contamination_delta"], color=COLORS[model], linestyle="--", alpha=0.7)
    axes[2].axhline(0, color="#6B7280", linewidth=1)
    axes[2].set_xticks(range(1, 6))
    axes[2].set_xlabel("Hospital-disjoint fold")
    axes[2].set_ylabel("Fixed-score AUROC increment")
    axes[2].set_title("C  eICU proxy endpoint", loc="left")
    axes[2].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir, "Figure_3")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reproduced_figures")
    args = parser.parse_args()
    data = json.loads(args.data.read_text(encoding="utf-8"))
    figure_1(data, args.output_dir)
    figure_2(data, args.output_dir)
    figure_3(data, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

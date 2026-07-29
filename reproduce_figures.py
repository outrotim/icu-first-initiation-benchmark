#!/usr/bin/env python3
"""Regenerate the three main figures from aggregate source data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "figure_source_data.json"
BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#E69F00"
RED = "#D55E00"
GRAY = "#8C8C8C"
MODEL_STYLE = {
    "Logistic Regression": (BLUE, "o"),
    "LightGBM": (GREEN, "s"),
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "legend.frameon": False,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, facecolor="white")
    fig.savefig(output_dir / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def figure1(data: dict, output_dir: Path) -> None:
    counts = pd.DataFrame(data["figure_1"]["landmark_counts"])
    metrics = pd.DataFrame(data["figure_1"]["model_metrics"])
    dca = pd.DataFrame(data["figure_1"]["decision_curves"])
    intervals = data["figure_1"]["auroc_intervals"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), constrained_layout=True)

    ax = axes[0, 0]
    ax.set_axis_off()
    ax.plot([1, 29], [0.72, 0.72], color="#252525", lw=1.2)
    for landmark in [6, 12, 18, 24]:
        ax.scatter(landmark, 0.72, color=BLUE)
        ax.text(landmark, 0.82, f"{landmark} h", ha="center")
    ax.annotate("", xy=(12, 0.52), xytext=(6, 0.52), arrowprops={"arrowstyle": "<->"})
    ax.text(8.6, 0.40, "6-h lookback", ha="center")
    ax.annotate("", xy=(18, 0.52), xytext=(12, 0.52), arrowprops={"arrowstyle": "<->"})
    ax.text(15.5, 0.40, "6-h outcome", ha="center")
    ax.text(
        7.0,
        0.12,
        "Eligible risk set\nnever ventilated",
        ha="center",
        va="center",
        color=GREEN,
        fontsize=8,
    )
    ax.text(
        22.5,
        0.12,
        "Excluded from first-initiation risk set\nactive or prior ventilation",
        ha="center",
        va="center",
        color=RED,
        fontsize=7.5,
    )
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 1)
    ax.set_title("A", loc="left", fontweight="bold")

    ax = axes[0, 1]
    x = np.arange(len(counts))
    bottom = np.zeros(len(counts))
    for column, color, label in [
        ("never_exposed", GREEN, "At risk"),
        ("active", RED, "Active"),
        ("resolved", ORANGE, "Prior/resolved"),
    ]:
        values = counts[column].to_numpy()
        ax.bar(x, values, bottom=bottom, color=color, label=label)
        bottom += values
    ax.set_xticks(x, [f"{int(value)} h" for value in counts["landmark_hours"]])
    ax.set_ylabel("Patient-landmarks")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        fontsize=6.5,
        columnspacing=0.7,
        handlelength=1.4,
    )
    ax.set_title("B", loc="left", fontweight="bold")

    ax = axes[1, 0]
    models = ["Logistic Regression", "LightGBM"]
    for offset, metric, color, marker in [
        (-0.08, "auroc", BLUE, "o"),
        (0.08, "auprc", ORANGE, "s"),
    ]:
        values = [
            float(metrics.loc[metrics["model"].eq(model), metric].iloc[0])
            for model in models
        ]
        ax.scatter(np.arange(2) + offset, values, color=color, marker=marker, label=metric.upper())
    for index, model in enumerate(models):
        value = float(metrics.loc[metrics["model"].eq(model), "auroc"].iloc[0])
        lower, upper = intervals[model]
        ax.errorbar(index - 0.08, value, yerr=[[value - lower], [upper - value]], fmt="none", color=BLUE)
    ax.set_xticks([0, 1], ["Logistic\nregression", "LightGBM"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Held-out performance")
    ax.legend()
    ax.set_title("C", loc="left", fontweight="bold")

    ax = axes[1, 1]
    for model, (color, marker) in MODEL_STYLE.items():
        block = dca.loc[dca["model"].eq(model)].sort_values("threshold")
        ax.plot(block["threshold"], block["net_benefit_model"], color=color, marker=marker, label=model)
    reference = dca.loc[dca["model"].eq("Logistic Regression")].sort_values("threshold")
    ax.plot(reference["threshold"], reference["net_benefit_all"], color=GRAY, linestyle="--", label="Alert all")
    ax.axhline(0, color="#252525", linestyle=":", label="Alert none")
    ax.set_xlabel("Risk threshold")
    ax.set_ylabel("Net benefit")
    ax.legend(fontsize=7)
    ax.set_title("D", loc="left", fontweight="bold")
    save(fig, output_dir, "Figure_1")


def figure2(data: dict, output_dir: Path) -> None:
    cells = pd.DataFrame(data["figure_2"]["cells"])
    contrasts = pd.DataFrame(data["figure_2"]["contrasts"])
    cell_order = ["A", "B", "C", "D"]
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.2), constrained_layout=True)
    offsets = {"Logistic Regression": -0.09, "LightGBM": 0.09}
    x = np.arange(4)
    labels = ["A\nIncident\nFilter on", "B\nIncident\nFilter off", "C\nHistory\nFilter on", "D\nHistory\nFilter off"]
    for ax, metric, lower, upper, ylabel in [
        (axes[0], "incident_sensitivity", "incident_sensitivity_ci_lower", "incident_sensitivity_ci_upper", "Incident events captured (%)"),
        (axes[1], "ineligible_alert_fraction", "ineligible_alert_fraction_ci_lower", "ineligible_alert_fraction_ci_upper", "Alerts outside risk set (%)"),
    ]:
        for model, (color, marker) in MODEL_STYLE.items():
            block = cells.loc[cells["model"].eq(model)].set_index("cell").loc[cell_order]
            values = 100 * block[metric].to_numpy()
            lowers = 100 * block[lower].to_numpy()
            uppers = 100 * block[upper].to_numpy()
            ax.errorbar(x + offsets[model], values, yerr=[values - lowers, uppers - values], fmt=marker, color=color, capsize=3, label=model)
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        ax.set_ylim(-3, 100)
    axes[0].legend(fontsize=7)
    contrast_order = [
        "filter_effect_incident_target",
        "filter_effect_history_target",
        "target_effect_with_filter",
        "target_effect_without_filter",
        "target_by_filter_interaction",
    ]
    y = np.arange(len(contrast_order))
    for model, (color, marker) in MODEL_STYLE.items():
        block = contrasts.loc[contrasts["model"].eq(model)].set_index("contrast").loc[contrast_order]
        estimate = 100 * block["estimate"].to_numpy()
        lower = 100 * block["ci_lower"].to_numpy()
        upper = 100 * block["ci_upper"].to_numpy()
        axes[2].errorbar(estimate, y + offsets[model], xerr=[estimate - lower, upper - estimate], fmt=marker, color=color, capsize=3)
    axes[2].axvline(0, color="#252525", linestyle=":")
    axes[2].set_yticks(y, ["Filter effect\nIncident", "Filter effect\nHistory", "Target effect\nFilter on", "Target effect\nFilter off", "Interaction"])
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Difference in event capture (percentage points)")
    for label, ax in zip(["A", "B", "C"], axes, strict=True):
        ax.set_title(label, loc="left", fontweight="bold")
    save(fig, output_dir, "Figure_2")


def figure3(data: dict, output_dir: Path) -> None:
    horizons = pd.DataFrame(data["figure_3"]["horizons"])
    ablation = pd.DataFrame(data["figure_3"]["ablation"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    for model, (color, marker) in MODEL_STYLE.items():
        block = horizons.loc[horizons["model"].eq(model)].sort_values("horizon_hours")
        axes[0].plot(block["horizon_hours"], block["auroc"], color=color, marker=marker, label=model)
    axes[0].set_xticks([6, 12, 24])
    axes[0].set_xlabel("Prediction horizon (h)")
    axes[0].set_ylabel("AUROC")
    axes[0].legend(fontsize=7)
    x = np.arange(2)
    width = 0.34
    axes[1].bar(x - width / 2, ablation["full_feature_auroc"], width, color=BLUE, label="Full feature set")
    axes[1].bar(x + width / 2, ablation["without_missingness_recency_auroc"], width, color=ORANGE, label="Without missingness/recency")
    axes[1].set_xticks(x, ["Logistic\nregression", "LightGBM"])
    axes[1].set_ylabel("AUROC")
    axes[1].legend(fontsize=7)
    axes[0].set_title("A", loc="left", fontweight="bold")
    axes[1].set_title("B", loc="left", fontweight="bold")
    save(fig, output_dir, "Figure_3")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reproduced_figures")
    args = parser.parse_args()
    data = json.loads(args.data.read_text(encoding="utf-8"))
    style()
    figure1(data, args.output_dir)
    figure2(data, args.output_dir)
    figure3(data, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

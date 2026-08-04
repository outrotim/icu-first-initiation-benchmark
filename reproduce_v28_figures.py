#!/usr/bin/env python3
"""Reproduce the v28 aggregate clinical-information figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"Logistic Regression": "#0072B2", "LightGBM": "#D55E00"}
MARKERS = {"Logistic Regression": "o", "LightGBM": "s"}


def frame(payload: dict[str, object], name: str) -> pd.DataFrame:
    return pd.DataFrame(payload["tables"][name])


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    axis.set_axisbelow(True)


def panel(axis: plt.Axes, label: str) -> None:
    axis.text(-0.13, 1.08, label, transform=axis.transAxes, fontsize=12, fontweight="bold")


def capacity_figure(payload: dict[str, object], output_dir: Path) -> None:
    point = frame(payload, "capacity_point_estimates")
    intervals = frame(payload, "capacity_intervals")
    opportunity = frame(payload, "patient_warning_opportunity")
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 7.2))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    for model in COLORS:
        group = intervals.loc[
            intervals["model"].eq(model) & intervals["metric"].eq("event_capture")
        ].sort_values("budget_fraction")
        x = group["budget_fraction"].to_numpy() * 100
        estimate = group["estimate"].to_numpy() * 100
        ax_a.plot(x, estimate, color=COLORS[model], marker=MARKERS[model], label=model)
        ax_a.fill_between(
            x,
            group["ci_lower"].to_numpy() * 100,
            group["ci_upper"].to_numpy() * 100,
            color=COLORS[model],
            alpha=0.14,
        )
        group_point = point.loc[point["model"].eq(model)].sort_values("budget_fraction")
        ax_b.plot(
            group_point["budget_fraction"] * 100,
            group_point["alerts_per_detected_event"],
            color=COLORS[model],
            marker=MARKERS[model],
        )
        ax_d.plot(
            group_point["budget_fraction"] * 100,
            group_point["repeat_alert_fraction"] * 100,
            color=COLORS[model],
            marker=MARKERS[model],
        )

    primary = opportunity.loc[
        np.isclose(opportunity["budget_fraction"], 0.05)
        & opportunity["warning_threshold_hours"].isin([0.5, 1.0, 2.0, 4.0])
    ]
    thresholds = sorted(primary["warning_threshold_hours"].unique())
    positions = np.arange(len(thresholds))
    for index, model in enumerate(COLORS):
        group = primary.loc[primary["model"].eq(model)].sort_values("warning_threshold_hours")
        ax_c.bar(
            positions + (index - 0.5) * 0.35,
            group["opportunity_yield_all_event_patients"] * 100,
            width=0.35,
            color=COLORS[model],
            label=model,
        )

    ax_a.set(title="Events identified within the alert budget", xlabel="Alert budget (% of test landmarks)", ylabel="First ventilation events identified (%)")
    ax_b.set(title="Clinical review workload", xlabel="Alert budget (% of test landmarks)", ylabel="Alerts per identified event")
    ax_c.set(title="Patient-level warning opportunities at 5%", xlabel="Minimum warning time (hours)", ylabel="All event patients identified in time (%)")
    ax_c.set_xticks(positions, [f"{value:g}" for value in thresholds])
    ax_d.set(title="Repeated-alert burden", xlabel="Alert budget (% of test landmarks)", ylabel="Alerts repeating a prior patient alert (%)")
    for label, axis in zip("ABCD", axes.ravel()):
        panel(axis, label)
        style_axis(axis)
    handles, labels = ax_a.get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, loc="upper center", ncol=2)
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    for extension in ["png", "pdf"]:
        figure.savefig(output_dir / f"figure_v28_capacity_warning_value.{extension}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def clinical_figure(payload: dict[str, object], output_dir: Path) -> None:
    detection = frame(payload, "event_detection_groups")
    experience = frame(payload, "patient_warning_experience")
    signal = frame(payload, "event_signal_features")
    context = frame(payload, "clinical_heterogeneity")
    figure, axes = plt.subplots(2, 2, figsize=(10.6, 7.4))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    labels = ["Both models", "One model", "Neither model"]
    colors = ["#009E73", "#E69F00", "#999999"]
    ax_a.bar(labels, detection["events"], color=colors)

    for model in COLORS:
        group = experience.loc[experience["model"].eq(model)].sort_values("budget_fraction")
        ax_b.plot(
            group["budget_fraction"] * 100,
            group["patient_level_pre_event_alert_capture"] * 100,
            color=COLORS[model], marker=MARKERS[model], label=model,
        )

    physiologic = signal.loc[signal["feature_type"].eq("physiological_summary")].nlargest(
        8, "absolute_standardized_mean_difference"
    ).sort_values("standardized_mean_difference")
    ax_c.barh(
        physiologic["feature"],
        physiologic["standardized_mean_difference"],
        color=np.where(physiologic["standardized_mean_difference"] >= 0, "#E67E22", "#2E86C1"),
    )
    ax_c.axvline(0, color="black", linewidth=0.8)

    units = context.loc[context["stratum"].eq("ICU type")]
    for model in COLORS:
        group = units.loc[units["model"].eq(model)]
        ax_d.scatter(group["event_capture"] * 100, group["level"], color=COLORS[model], marker=MARKERS[model], label=model)

    ax_a.set(title="Agreement in event identification at the 5% budget", ylabel="First ventilation events")
    ax_b.set(title="Patient-level surveillance reach", xlabel="Alert budget (% of test landmarks)", ylabel="Event patients reached before ventilation (%)")
    ax_c.set(title="Physiologic profile of consistently identified events", xlabel="Standardized mean difference")
    ax_d.set(title="Clinical setting heterogeneity", xlabel="Events identified at the 5% budget (%)")
    ax_b.legend(frameon=False)
    ax_d.legend(frameon=False)
    for label, axis in zip("ABCD", axes.ravel()):
        panel(axis, label)
        style_axis(axis)
    figure.tight_layout()
    for extension in ["png", "pdf"]:
        figure.savefig(output_dir / f"figure_v28_clinical_signal_patient_experience.{extension}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(__file__).with_name("v28_figure_source_data.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reproduced_figures"))
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(arguments.source.read_text(encoding="utf-8"))
    capacity_figure(payload, arguments.output_dir)
    clinical_figure(payload, arguments.output_dir)


if __name__ == "__main__":
    main()

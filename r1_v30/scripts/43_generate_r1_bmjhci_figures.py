#!/usr/bin/env python3
"""
Generate all publication-grade Main Figures (1-3) and Supplementary Figures (S1-S8)
for BMJHCI-2026-102465 Major Revision (R1, v30).
All panels strictly match their captions; outputs saved as 300-dpi PNG and vector PDF.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


BASE_DIR = Path(__file__).resolve().parent.parent
PROC_R1 = BASE_DIR / "data" / "processed" / "v30_bmjhci_r1"
RES_DIR = BASE_DIR / "results" / "high_tier_upgrade" / "v30_bmjhci_r1"
SUB_DIR = BASE_DIR / "submission_BMJHCI_R1_v30_202611"
MAIN_FIG_DIR = SUB_DIR / "04_figures" / "main"
SUPP_FIG_DIR = SUB_DIR / "04_figures" / "supplementary"

# Clean BMJ / Okabe-Ito journal-safe color palette
PALETTE = {
    "lgb": "#0072B2",      # deep blue
    "lr": "#D55E00",       # vermilion
    "news2": "#009E73",    # bluish green
    "rr": "#CC79A7",       # reddish purple
    "spo2": "#7F7F7F",     # neutral gray
    "accent": "#E69F00",   # amber
    "dark": "#1F2937",
    "light_bg": "#F8FAFC",
    "grid": "#E2E8F0",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.edgecolor": "#334155",
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "legend.frameon": True,
    "legend.facecolor": "white",
    "legend.edgecolor": "#CBD5E1",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"  Saved {out_dir.name}/{stem}.png & .pdf")


def style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.45, color="#CBD5E1")
    ax.set_axisbelow(True)


def make_figure1(audit: dict, lm_flow: pd.DataFrame) -> None:
    """Figure 1: (A) Patient & landmark CONSORT/STROBE flow; (B) Dynamic risk set & event rate."""
    fig = plt.figure(figsize=(13.2, 6.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.32, 1.0], wspace=0.22)

    # Panel A: Patient & Landmark Flow Diagram
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 10.2)
    ax.axis("off")
    ax.set_title("A. Cohort Selection and Dynamic Eligibility Flow", loc="left", pad=8)

    fl = audit["flow"]

    def box(x, y, w, h, text, fc="#F1F5F9", ec="#334155", bold=False, fs=7.4):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.10,rounding_size=0.14",
            facecolor=fc, edgecolor=ec, linewidth=1.0,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", color="#0F172A", linespacing=1.28,
        )

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.1))

    # Main vertical chain (left)
    box(0.15, 8.75, 5.75, 1.05, f"MIMIC-IV v3.1 Adult ICU Stays\n{fl['total_icustays_raw']:,} stays ({fl['total_unique_patients_raw']:,} unique adults)", fc="#EFF6FF", ec="#1D4ED8", bold=True, fs=7.8)
    arrow(3.0, 8.75, 3.0, 8.0)
    box(6.35, 8.65, 4.1, 1.2, f"Excluded:\n• Subsequent stays: {fl['excluded_subsequent_icustays']:,}\n  (incl. {fl['patients_with_contiguous_icu_transfer_le24h']:,} ≤24 h transfers)\n• Missing outtime / LOS ≤6 h: {fl['excluded_missing_outtime'] + fl['excluded_los_le_6h']:,}", fc="#FFF7ED", ec="#C2410C", fs=7.0)
    arrow(5.9, 9.25, 6.35, 9.25)

    box(0.15, 6.9, 5.75, 1.05, f"First Adult ICU Stays with LOS >6 h\n{fl['eligible_los_gt_6h_patients']:,} patients (238,123 scheduled assessments)", fc="#F8FAFC", ec="#334155", fs=7.4)
    arrow(3.0, 6.9, 3.0, 6.15)
    box(6.35, 6.8, 4.1, 1.2, f"Excluded from primary cohort:\n• CVICU stays: {fl['cvicu_patients']:,}\n• Elective / same-day surgical:\n  {fl['planned_surgical_patients']:,} (overlap removed)", fc="#FFF7ED", ec="#C2410C", fs=7.0)
    arrow(5.9, 7.42, 6.35, 7.42)

    box(0.15, 4.95, 5.75, 1.15, f"Unplanned Non-CVICU Cohort\n{fl['unplanned_non_cvicu_patients']:,} patients (175,003 scheduled assessments\nacross 6, 12, 18, and 24 h landmarks)", fc="#EFF6FF", ec="#1D4ED8", bold=True, fs=7.3)
    arrow(3.0, 4.95, 3.0, 4.2)
    box(6.35, 4.8, 4.1, 1.4, "Landmark-level exclusions:\n• No bedside vital before landmark:\n  407 assessments (0.2%)\n• Prior IMV (procedure or chart):\n  55,978 assessments (incl. 9,534\n  caught via chartevents only)", fc="#FEF2F2", ec="#B91C1C", fs=6.9)
    arrow(5.9, 5.52, 6.35, 5.52)

    box(0.15, 2.75, 5.75, 1.38, "Primary Dynamically Eligible Risk Set\n34,281 unique patients | 118,618 eligible assessments\n1,841 first ICU-recorded IMV events (1.55%)\n(6 h: 729 | 12 h: 530 | 18 h: 355 | 24 h: 227)", fc="#ECFDF5", ec="#047857", bold=True, fs=7.2)
    arrow(3.0, 2.75, 3.0, 2.12)

    # Temporal split boxes at bottom
    box(0.15, 0.25, 3.15, 1.7, "Train (2008-2013)\n17,554 patients\n60,286 eligible\n939 events (1.56%)", fc="#F8FAFC", ec="#475569", fs=7.1)
    box(3.55, 0.25, 3.15, 1.7, "Validation (2014-2016)\n6,421 patients\n22,217 eligible\n384 events (1.73%)", fc="#F8FAFC", ec="#475569", fs=7.1)
    box(6.95, 0.25, 3.5, 1.7, "Temporal Test (2017-2022)\n10,306 patients | 36,115 eligible\n518 events (1.43%)\n(2017-19: 266 | 2020-22: 252)", fc="#EFF6FF", ec="#0072B2", bold=True, fs=6.8)
    arrow(3.0, 2.12, 1.72, 1.98)
    arrow(3.0, 2.12, 5.12, 1.98)
    arrow(3.0, 2.12, 8.70, 1.98)

    # Panel B: Dynamic Landmark Schematic + Eligible Risk Set & Incidence
    gs_r = gs[0, 1].subgridspec(2, 1, height_ratios=[1.05, 1.35], hspace=0.40)
    ax_b1 = fig.add_subplot(gs_r[0, 0])
    ax_b1.set_xlim(-0.5, 31.5)
    ax_b1.set_ylim(-0.3, 4.7)
    ax_b1.set_title("B. Leak-Free Repeated Landmark Design (First 24 h)", loc="left", pad=6)
    ax_b1.set_xlabel("Hours after ICU admission")
    ax_b1.set_yticks([0.5, 1.5, 2.5, 3.5])
    ax_b1.set_yticklabels(["24 h", "18 h", "12 h", "6 h"])
    ax_b1.spines["top"].set_visible(False)
    ax_b1.spines["right"].set_visible(False)
    ax_b1.spines["left"].set_visible(False)
    ax_b1.grid(axis="x", linestyle=":", alpha=0.5)

    for idx, lm in enumerate([6, 12, 18, 24]):
        y = 3.5 - idx
        ax_b1.barh(y, 6, left=lm - 6, height=0.44, color="#BFDBFE", edgecolor="#1D4ED8", lw=0.8, label="6 h lookback (labs storetime < L)" if idx == 0 else None)
        ax_b1.barh(y, 6, left=lm, height=0.44, color="#FDE68A", edgecolor="#B45309", lw=0.8, label="6 h first-IMV horizon (L, L+6]" if idx == 0 else None)
        ax_b1.plot(lm, y, "o", color="#DC2626", markersize=5.5, zorder=5, label="Eligibility check & prediction landmark" if idx == 0 else None)
    ax_b1.legend(loc="upper right", fontsize=7.2, ncol=1, framealpha=0.95)

    ax_b2 = fig.add_subplot(gs_r[1, 0])
    style_ax(ax_b2)
    x = np.arange(len(lm_flow))
    w = 0.50
    bars = ax_b2.bar(x, lm_flow["r1_eligible"] / 1000.0, width=w, color="#BFDBFE", edgecolor="#1D4ED8", lw=0.9, label="Dynamically eligible patients (thousands)")
    ax_b2.set_xticks(x)
    ax_b2.set_xticklabels([f"{int(h)} h" for h in lm_flow["landmark_hours"]])
    ax_b2.set_xlabel("Prediction landmark after ICU admission")
    ax_b2.set_ylabel("Eligible patients (thousands)", color="#1D4ED8")
    ax_b2.set_ylim(0, 44)
    for b, cnt in zip(bars, lm_flow["r1_eligible"]):
        ax_b2.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.1, f"{int(cnt):,}", ha="center", va="bottom", fontsize=7.8, color="#1E3A8A", fontweight="bold")

    ax_b2r = ax_b2.twinx()
    ax_b2r.spines["top"].set_visible(False)
    ax_b2r.plot(x, lm_flow["event_rate_pct"], "o-", color="#D55E00", lw=2.0, markersize=6, label="6 h first-IMV incidence (%)")
    ax_b2r.set_ylabel("6 h first-IMV incidence (%)", color="#D55E00")
    ax_b2r.set_ylim(0, 3.1)
    for xi, rate, ev in zip(x, lm_flow["event_rate_pct"], lm_flow["events_6h"]):
        ax_b2r.text(xi, rate - 0.16, f"{rate:.2f}%\n(n={int(ev)})", ha="center", va="top", fontsize=7.4, color="#9A3412", fontweight="bold")
    ax_b2.set_title("C. Eligible Risk Set and First-IMV Rate Across Landmarks", loc="left", pad=6)

    save_fig(fig, MAIN_FIG_DIR, "figure1_cohort_flow_and_framework")


def make_figure2(curves: pd.DataFrame, disc_ci: pd.DataFrame, cd_df: pd.DataFrame) -> None:
    """Figure 2: Capacity-response vs baselines (A-B), Lead-time reach (C), Cooldown trade-off (D)."""
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.6))
    plt.subplots_adjust(wspace=0.26, hspace=0.34)

    elig = curves[curves["budget_basis"] == "eligible"].copy()
    models_plot = [
        ("LightGBM", PALETTE["lgb"], "-", "o"),
        ("Logistic Regression", PALETTE["lr"], "--", "s"),
        ("Modified NEWS2 (no O2)", PALETTE["news2"], "-.", "^"),
        ("RR max alone", PALETTE["rr"], ":", "D"),
        ("SpO2 min alone", PALETTE["spo2"], ":", "v"),
    ]

    # Panel A: Event-Landmark Capture vs Eligible Quota
    ax = axes[0, 0]
    style_ax(ax)
    for m, col, ls, mk in models_plot:
        d = elig[elig["model"] == m].sort_values("budget_pct")
        ax.plot(d["budget_pct"], d["event_capture_pct"], ls=ls, marker=mk, color=col, lw=1.9, markersize=5, label=m)
        ax.fill_between(d["budget_pct"], d["event_capture_lo"], d["event_capture_hi"], color=col, alpha=0.12)
    ax.axvline(5.0, color="#64748B", ls="--", lw=0.9, alpha=0.8)
    ax.text(5.35, 58.0, "Primary 5%\neligible quota\n(3.3% scheduled)", fontsize=7.5, color="#475569", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CBD5E1", alpha=0.9))
    ax.set_xlabel("Retrospective alert quota (% of eligible assessments)")
    ax.set_ylabel("Event-landmark capture (% of 518 events)")
    ax.set_xlim(0.5, 20.8)
    ax.set_ylim(0, 76)
    ax.set_title("A. Event-Landmark Capture Across Alert Quotas", loc="left")
    ax.legend(loc="lower right", fontsize=7.6)

    # Panel B: PPV vs Eligible Quota
    ax = axes[0, 1]
    style_ax(ax)
    for m, col, ls, mk in models_plot:
        d = elig[elig["model"] == m].sort_values("budget_pct")
        ax.plot(d["budget_pct"], d["ppv_pct"], ls=ls, marker=mk, color=col, lw=1.9, markersize=5, label=m)
        ax.fill_between(d["budget_pct"], d["ppv_lo"], d["ppv_hi"], color=col, alpha=0.12)
    ax.axhline(1.434, color="#334155", ls=":", lw=1.0, label="Baseline event rate (1.43%)")
    ax.axvline(5.0, color="#64748B", ls="--", lw=0.9, alpha=0.8)
    ax.set_xlabel("Retrospective alert quota (% of eligible assessments)")
    ax.set_ylabel("Positive predictive value (%)")
    ax.set_xlim(0.5, 20.8)
    ax.set_ylim(0, 34)
    ax.set_title("B. Positive Predictive Value Across Alert Quotas", loc="left")
    ax.legend(loc="upper right", fontsize=7.6)

    # Panel C: Separated Warning-Time Reach at Primary 5% Eligible Quota
    ax = axes[1, 0]
    style_ax(ax)
    lgb_5 = elig[(elig["model"] == "LightGBM") & (elig["budget_pct"] == 5.0)].iloc[0]
    lr_5 = elig[(elig["model"] == "Logistic Regression") & (elig["budget_pct"] == 5.0)].iloc[0]
    nw_5 = elig[(elig["model"] == "Modified NEWS2 (no O2)") & (elig["budget_pct"] == 5.0)].iloc[0]

    thrs = ["≥0.5 h", "≥1.0 h", "≥2.0 h", "≥4.0 h"]
    x = np.arange(len(thrs))
    w = 0.25
    lgb_any = [lgb_5[f"lead_ge_{t}h_any_prior_pct"] for t in [0.5, 1.0, 2.0, 4.0]]
    lgb_lm = [lgb_5[f"lead_ge_{t}h_event_landmark_pct"] for t in [0.5, 1.0, 2.0, 4.0]]
    lr_any = [lr_5[f"lead_ge_{t}h_any_prior_pct"] for t in [0.5, 1.0, 2.0, 4.0]]
    nw_any = [nw_5[f"lead_ge_{t}h_any_prior_pct"] for t in [0.5, 1.0, 2.0, 4.0]]

    ax.bar(x - w, lgb_any, width=w, color=PALETTE["lgb"], label="LightGBM (any prior alert)")
    ax.bar(x, lgb_lm, width=w, color="#93C5FD", edgecolor=PALETTE["lgb"], hatch="//", label="LightGBM (event-landmark only)")
    ax.bar(x + w, nw_any, width=w, color=PALETTE["news2"], alpha=0.85, label="Modified NEWS2 (any prior alert)")
    for i in range(len(thrs)):
        ax.text(x[i] - w, lgb_any[i] + 0.8, f"{lgb_any[i]:.1f}%", ha="center", va="bottom", fontsize=7.2, color=PALETTE["lgb"], fontweight="bold")
        ax.text(x[i], lgb_lm[i] + 0.8, f"{lgb_lm[i]:.1f}%", ha="center", va="bottom", fontsize=7.2, color="#1E3A8A")
        ax.text(x[i] + w, nw_any[i] + 0.8, f"{nw_any[i]:.1f}%", ha="center", va="bottom", fontsize=7.2, color="#065F46")
    ax.set_xticks(x)
    ax.set_xticklabels(thrs)
    ax.set_xlabel("Minimum warning lead time before first IMV")
    ax.set_ylabel("Proportion of all 518 event patients (%)")
    ax.set_ylim(0, 48)
    ax.set_title("C. Lead-Time-Specific Reach at 5% Eligible Quota", loc="left")
    ax.legend(loc="upper right", fontsize=7.5)

    # Panel D: Cooldown Trade-off (Foregrounding Event-Window & Timely <=6h Closest-Alert Loss)
    ax = axes[1, 1]
    style_ax(ax)
    cd5 = cd_df[cd_df["budget_basis"] == "eligible_5pct"].copy()
    cds = [0, 6, 12]
    x = np.arange(len(cds))
    w = 0.32
    for idx, (m, col) in enumerate([("LightGBM", PALETTE["lgb"]), ("Logistic Regression", PALETTE["lr"])]):
        dm = cd5[cd5["model"] == m].sort_values("cooldown_hours")
        offset = -w / 2 if idx == 0 else w / 2
        bars = ax.bar(x + offset, dm["timely_closest_alert_le_6h_pct"], width=w, color=col, alpha=0.88, label=f"{m}: timely ≤6h closest alert")
        for b, red in zip(bars, dm["alert_reduction_pct"]):
            lbl = f"{b.get_height():.1f}%" if red == 0 else f"{b.get_height():.1f}%\n(-{red:.0f}%)"
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 0.42, lbl, ha="center", va="center", fontsize=7.3, color="white", fontweight="bold")
    # Overlay closest-alert median lead time on twin axis for LightGBM
    ax_r = ax.twinx()
    ax_r.spines["top"].set_visible(False)
    dm_lgb = cd5[cd5["model"] == "LightGBM"].sort_values("cooldown_hours")
    ax_r.plot(x, dm_lgb["closest_alert_lead_median_h"], "D--", color="#0F172A", lw=1.7, markersize=5.5, label="Median closest-alert gap (h, LGBM)")
    for xi, gap in zip(x, dm_lgb["closest_alert_lead_median_h"]):
        ax_r.text(xi, gap + 0.25, f"{gap:.2f} h", ha="center", va="bottom", fontsize=7.5, color="#0F172A", fontweight="bold")
    ax_r.set_ylabel("Median closest-alert lead time (h)", color="#0F172A")
    ax_r.set_ylim(0, 7.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["No cooldown\n(0 h)", "6 h\ncooldown", "12 h\ncooldown"])
    ax.set_xlabel("Post-alert suppression policy (5% eligible quota)")
    ax.set_ylabel("Event-window / timely ≤6h reach (%)")
    ax.set_ylim(0, 48)
    ax.set_title("D. Cooldown Trade-off: Alert Volume vs Timely Warning", loc="left")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7.0)

    save_fig(fig, MAIN_FIG_DIR, "figure2_capacity_response_and_baselines")


def make_figure3(era_lm: pd.DataFrame, disc_ci: pd.DataFrame, sub_df: pd.DataFrame, fact_df: pd.DataFrame) -> None:
    """Figure 3: (A) Per-landmark AUROC & calibration; (B) Feature/charting ablations; (C) Subgroups; (D) 2x2 Factorial."""
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.2))
    plt.subplots_adjust(wspace=0.58, hspace=0.36)

    # Panel A: Per-Landmark AUROC & E/O Calibration
    ax = axes[0, 0]
    style_ax(ax)
    lm_ord = ["6h", "12h", "18h", "24h"]
    x = np.arange(len(lm_ord))
    for m, col, mk in [("LightGBM", PALETTE["lgb"], "o"), ("Logistic Regression", PALETTE["lr"], "s")]:
        d = era_lm[(era_lm["model"] == m) & (era_lm["stratum_type"] == "landmark") & (era_lm["calibration"] == "landmark_cal")].set_index("stratum").loc[lm_ord]
        ax.plot(x, d["auroc"], f"{mk}-", color=col, lw=2.0, markersize=6, label=f"{m} AUROC")
    ax.set_xticks(x)
    ax.set_xticklabels(lm_ord)
    ax.set_xlabel("Prediction landmark after ICU admission")
    ax.set_ylabel("AUROC (2017-2022 temporal test)")
    ax.set_ylim(0.68, 0.88)

    ax_r = ax.twinx()
    ax_r.spines["top"].set_visible(False)
    d_p = era_lm[(era_lm["model"] == "LightGBM") & (era_lm["stratum_type"] == "landmark") & (era_lm["calibration"] == "pooled_cal")].set_index("stratum").loc[lm_ord]
    d_l = era_lm[(era_lm["model"] == "LightGBM") & (era_lm["stratum_type"] == "landmark") & (era_lm["calibration"] == "landmark_cal")].set_index("stratum").loc[lm_ord]
    ax_r.plot(x, d_p["eo_ratio"], "^:", color="#94A3B8", lw=1.4, markersize=5, label="LGBM E/O (pooled cal.)")
    ax_r.plot(x, d_l["eo_ratio"], "v--", color="#0F172A", lw=1.5, markersize=5, label="LGBM E/O (landmark cal.)")
    ax_r.axhline(1.0, color="#64748B", ls=":", lw=0.8)
    ax_r.set_ylabel("Expected-to-observed (E/O) ratio", color="#0F172A")
    ax_r.set_ylim(0.8, 1.85)
    ax.set_title("A. Per-Landmark Discrimination and Calibration", loc="left")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7.2, ncol=2)

    # Panel B: Feature & Charting-Pattern Ablations (AUROC with 95% CI)
    ax = axes[0, 1]
    style_ax(ax)
    order_models = [
        ("LightGBM (+O2 support)", "LightGBM + non-invasive O2"),
        ("LightGBM", "LightGBM (primary 16-var)"),
        ("LightGBM (physiology-only)", "LightGBM (physiology-only)"),
        ("Logistic Regression", "Logistic reg. (primary 16-var)"),
        ("Logistic Regression (physiology-only)", "Logistic reg. (physiology-only)"),
        ("Logistic Regression (charting-only)", "Logistic reg. (charting-only)"),
        ("LightGBM (charting-only)", "LightGBM (charting-only)"),
        ("Modified NEWS2 (with O2)", "Modified NEWS2 (with O2)"),
        ("Modified NEWS2 (no O2)", "Modified NEWS2 (no O2)"),
    ]
    d_idx = disc_ci.set_index("model")
    y_pos = np.arange(len(order_models))[::-1]
    for yi, (mkey, mlbl) in zip(y_pos, order_models):
        r = d_idx.loc[mkey]
        col = PALETTE["lgb"] if "LightGBM" in mkey else (PALETTE["lr"] if "Logistic" in mkey else PALETTE["news2"])
        ax.errorbar(
            r["auroc"], yi,
            xerr=[[r["auroc"] - r["auroc_lo"]], [r["auroc_hi"] - r["auroc"]]],
            fmt="o", color=col, ecolor=col, elinewidth=1.6, capsize=3, markersize=5.5,
        )
        ax.text(r["auroc_hi"] + 0.005, yi, f"{r['auroc']:.3f} (cap {r['elig_5pct_cap']:.1f}%)", va="center", fontsize=7.3, color="#1E293B")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([lbl for _, lbl in order_models], fontsize=7.8)
    ax.set_xlabel("AUROC with 95% patient-cluster bootstrap CI")
    ax.set_xlim(0.62, 0.90)
    ax.set_title("B. Feature-Set and Charting-Pattern Ablations", loc="left")

    # Panel C: Subgroup Capture (with 95% CI) for Primary LightGBM at 5% Eligible Quota
    ax = axes[1, 0]
    style_ax(ax)
    sg = sub_df[(sub_df["model"] == "LightGBM") & (sub_df["subgroup_type"].isin(["Sex", "Age group", "ICU type", "Temporal era"]))].copy()
    sg_order = [
        ("Sex", "Female"),
        ("Sex", "Male"),
        ("Age group", "<65 years"),
        ("Age group", "65-79 years"),
        ("Age group", ">=80 years"),
        ("ICU type", "SICU"),
        ("ICU type", "TSICU"),
        ("ICU type", "MICU"),
        ("ICU type", "MICU/SICU"),
        ("ICU type", "CCU"),
        ("ICU type", "Neuro / Other ICU"),
        ("Temporal era", "2017 - 2019"),
        ("Temporal era", "2020 - 2022"),
    ]
    sg_i = sg.set_index(["subgroup_type", "subgroup"])
    y_pos = np.arange(len(sg_order))[::-1]
    ax.axvline(35.135, color="#64748B", ls="--", lw=1.0, label="Overall 5% capture (35.1%)")
    type_colors = {"Sex": "#0072B2", "Age group": "#009E73", "ICU type": "#D55E00", "Temporal era": "#7C3AED"}
    for yi, (st, sv) in zip(y_pos, sg_order):
        r = sg_i.loc[(st, sv)]
        c = type_colors[st]
        ax.errorbar(
            r["capture_pct"], yi,
            xerr=[[r["capture_pct"] - r["capture_lo"]], [r["capture_hi"] - r["capture_pct"]]],
            fmt="o", color=c, ecolor=c, elinewidth=1.5, capsize=2.8, markersize=5,
        )
        ax.text(r["capture_hi"] + 1.0, yi, f"{r['capture_pct']:.1f}% (n={int(r['events'])})", va="center", fontsize=7.0, color="#1E293B")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{sv}" for _, sv in sg_order], fontsize=7.8)
    ax.set_xlabel("Event-landmark capture at global 5% eligible quota (%, 95% CI)")
    ax.set_xlim(8, 72)
    ax.set_title("C. Subgroup Event Capture (LightGBM, Dashed = Overall 35.1%)", loc="left")

    # Panel D: 2x2 Factorial Target x Eligibility on Clean Cohort (5% eligible = 1,806 alerts)
    ax = axes[1, 1]
    style_ax(ax)
    f5 = fact_df[fact_df["budget_basis"] == "5pct_eligible"].copy()
    cells = [
        ("A_incident_filter_on", "Incident target\n+ Filter ON"),
        ("B_incident_filter_off", "Incident target\n+ Filter OFF"),
        ("C_history_filter_on", "History target\n+ Filter ON"),
        ("D_history_filter_off", "History target\n+ Filter OFF"),
    ]
    x = np.arange(len(cells))
    w = 0.36
    for idx, (m, col) in enumerate([("LightGBM", PALETTE["lgb"]), ("Logistic Regression", PALETTE["lr"])]):
        dm = f5[f5["model"] == m].set_index("cell").loc[[c[0] for c in cells]]
        off = -w / 2 if idx == 0 else w / 2
        bars = ax.bar(x + off, dm["incident_capture_pct"], width=w, color=col, alpha=0.88, label=m)
        for b, inelig in zip(bars, dm["ineligible_alert_pct"]):
            h_val = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h_val + 0.7, f"{h_val:.1f}%", ha="center", va="bottom", fontsize=7.4, color=col, fontweight="bold")
            if inelig > 1.0:
                if h_val >= 10.0:
                    ax.text(b.get_x() + b.get_width() / 2, h_val * 0.45, f"{inelig:.0f}%\ninelig.", ha="center", va="center", fontsize=6.8, color="white", fontweight="bold")
                else:
                    ax.text(b.get_x() + b.get_width() / 2, h_val + (3.0 if idx == 0 else 5.8), f"({inelig:.0f}% inel.)", ha="center", va="bottom", fontsize=6.5, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in cells], fontsize=7.8)
    ax.set_ylabel("Incident first-IMV capture (% of 518 events)")
    ax.set_ylim(0, 44)
    ax.set_title("D. Target Construction × Dynamic Eligibility (1,806 Alerts)", loc="left")
    ax.legend(loc="upper right", fontsize=7.8)

    save_fig(fig, MAIN_FIG_DIR, "figure3_landmarks_ablations_and_subgroups")


def make_supplementary_figures(
    states: pd.DataFrame,
    preds: pd.DataFrame,
    dca_df: pd.DataFrame,
    imp_df: pd.DataFrame,
    sd_df: pd.DataFrame,
    out_sens: pd.DataFrame,
    comp_df: pd.DataFrame,
    mod_comp: pd.DataFrame,
    lab_delay: pd.DataFrame,
) -> None:
    """Generate Supplementary Figures S1-S8 cleanly."""
    # Figure S1: OR-Arrival Contamination Audit (v29 artifact diagnosis)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    plt.subplots_adjust(wspace=0.28)
    st6_v29 = states[(states["landmark_hours"] == 6.0) & (states["v29_eligible"] == 1)].copy()
    st6_v29["no_vital"] = st6_v29["has_vital_before_lm"] == 0
    cats = ["Non-CVICU\nNon-events", "Non-CVICU\n6h Events", "CVICU\nNon-events", "CVICU\n6h Events"]
    vals = [
        100.0 * st6_v29[(st6_v29["is_cvicu"] == 0) & (st6_v29["v29_outcome_6h"] == 0)]["no_vital"].mean(),
        100.0 * st6_v29[(st6_v29["is_cvicu"] == 0) & (st6_v29["v29_outcome_6h"] == 1)]["no_vital"].mean(),
        100.0 * st6_v29[(st6_v29["is_cvicu"] == 1) & (st6_v29["v29_outcome_6h"] == 0)]["no_vital"].mean(),
        100.0 * st6_v29[(st6_v29["is_cvicu"] == 1) & (st6_v29["v29_outcome_6h"] == 1)]["no_vital"].mean(),
    ]
    ax = axes[0]
    style_ax(ax)
    bars = ax.bar(cats, vals, color=["#94A3B8", "#F59E0B", "#94A3B8", "#DC2626"], width=0.55)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_ylabel("No bedside vital sign before 6 h landmark (%)")
    ax.set_ylim(0, 96)
    ax.set_title("A. Absence of Bedside Vitals Before 6 h (Original v29 Cohort)", loc="left")

    ax = axes[1]
    style_ax(ax)
    ev6_nv = st6_v29[(st6_v29["v29_outcome_6h"] == 1) & (st6_v29["no_vital"])].copy()
    dt_full = (ev6_nv["proc_imv_start_hours"] - ev6_nv["first_vital_exact_hours"]).dropna()
    dt = dt_full[(dt_full >= -2) & (dt_full <= 4)]
    ax.hist(dt, bins=32, color="#DC2626", alpha=0.8, edgecolor="white")
    ax.axvline(dt_full.median(), color="#0F172A", ls="--", lw=1.4, label=f"Median = +{dt_full.median():.2f} h ({100.0*(dt_full.abs()<=1.0).mean():.1f}% within ±1 h)")
    ax.set_xlabel("IMV start minus first recorded ICU vital sign (hours)")
    ax.set_ylabel("Number of 6 h v29 events without prior vitals")
    ax.set_title("B. Arrival-Coincident Ventilation Among Unmonitored 6 h Events", loc="left")
    ax.legend(loc="upper right")
    save_fig(fig, SUPP_FIG_DIR, "figure_s1_or_arrival_audit")

    # Figure S2: ROC & Precision-Recall Curves (Temporal Test)
    te = preds[preds["split"] == "test"].copy()
    va = preds[preds["split"] == "validation"].copy()
    from sklearn.linear_model import LogisticRegression
    va_lgb = va[va["model"] == "LightGBM"]
    te_lgb = te[te["model"] == "LightGBM"].copy()
    te_lr = te[te["model"] == "Logistic Regression"].copy()
    cal_nw = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(va_lgb[["baseline_mnews2"]], va_lgb["outcome_r1_6h"])
    p_nw = cal_nw.predict_proba(te_lgb[["baseline_mnews2"]])[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    plt.subplots_adjust(wspace=0.25)
    y = te_lgb["outcome_r1_6h"].astype(int).to_numpy()
    for lbl, p, col, ls in [
        ("LightGBM (AUROC 0.803, AUPRC 0.141)", te_lgb["y_pred_lm_cal"].to_numpy(), PALETTE["lgb"], "-"),
        ("Logistic regression (AUROC 0.788, AUPRC 0.118)", te_lr["y_pred_lm_cal"].to_numpy(), PALETTE["lr"], "--"),
        ("Modified NEWS2 (AUROC 0.673, AUPRC 0.035)", p_nw, PALETTE["news2"], "-."),
    ]:
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, ls=ls, color=col, lw=1.9, label=lbl)
        prec, rec, _ = precision_recall_curve(y, p)
        axes[1].plot(rec, prec, ls=ls, color=col, lw=1.9, label=lbl)
    axes[0].plot([0, 1], [0, 1], ":", color="#94A3B8", lw=1.0)
    style_ax(axes[0]); style_ax(axes[1])
    axes[0].set_xlabel("1 - Specificity (False Positive Rate)"); axes[0].set_ylabel("Sensitivity")
    axes[0].set_title("A. Receiver Operating Characteristic (Temporal Test)", loc="left")
    axes[0].legend(loc="lower right", fontsize=7.6)
    axes[1].axhline(y.mean(), color="#94A3B8", ls=":", lw=1.0, label=f"Prevalence ({100*y.mean():.2f}%)")
    axes[1].set_xlabel("Recall (Event-Landmark Capture)"); axes[1].set_ylabel("Precision (PPV)")
    axes[1].set_title("B. Precision-Recall Curves (Temporal Test)", loc="left")
    axes[1].set_ylim(0, 0.65)
    axes[1].legend(loc="upper right", fontsize=7.6)
    save_fig(fig, SUPP_FIG_DIR, "figure_s2_roc_pr")

    # Figure S3: Calibration Plots (Pooled vs Era 2017-2019 & 2020-2022)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    plt.subplots_adjust(wspace=0.25)
    for ax_i, (title, sub_lgb, sub_lr) in enumerate([
        ("A. Temporal Test Calibration (2017-2022 Pooled)", te_lgb, te_lr),
        ("B. Era Calibration Drift (LightGBM: 2017-2019 vs 2020-2022)", te_lgb[te_lgb["anchor_year_group"] == "2017 - 2019"], te_lgb[te_lgb["anchor_year_group"] == "2020 - 2022"]),
    ]):
        ax = axes[ax_i]
        style_ax(ax)
        ax.plot([0, 0.16], [0, 0.16], ":", color="#64748B", lw=1.0, label="Ideal calibration")
        series = [
            ("LightGBM (landmark cal.)", sub_lgb["outcome_r1_6h"], sub_lgb["y_pred_lm_cal"], PALETTE["lgb"], "o-"),
            ("Logistic reg. (landmark cal.)", sub_lr["outcome_r1_6h"], sub_lr["y_pred_lm_cal"], PALETTE["lr"], "s--"),
        ] if ax_i == 0 else [
            ("2017-2019 (pre-COVID, E/O 1.40)", sub_lgb["outcome_r1_6h"], sub_lgb["y_pred_lm_cal"], "#0072B2", "o-"),
            ("2020-2022 (COVID era, E/O 1.03)", sub_lr["outcome_r1_6h"], sub_lr["y_pred_lm_cal"], "#D55E00", "s--"),
        ]
        for lbl, yy, pp, col, fmt in series:
            bins = pd.qcut(pd.Series(pp.to_numpy()), q=10, duplicates="drop")
            g = pd.DataFrame({"y": yy.to_numpy(), "p": pp.to_numpy(), "b": bins}).groupby("b", observed=True).mean()
            ax.plot(g["p"], g["y"], fmt, color=col, lw=1.8, markersize=5, label=lbl)
        ax.set_xlabel("Mean predicted 6h risk"); ax.set_ylabel("Observed 6h first-IMV rate")
        ax.set_xlim(0, 0.14); ax.set_ylim(0, 0.14)
        ax.set_title(title, loc="left")
        ax.legend(loc="upper left", fontsize=7.8)
    save_fig(fig, SUPP_FIG_DIR, "figure_s3_calibration")

    # Figure S4: Decision Curve Analysis
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    style_ax(ax)
    for m, col, ls in [
        ("LightGBM", PALETTE["lgb"], "-"),
        ("Logistic Regression", PALETTE["lr"], "--"),
        ("Modified NEWS2 (no O2)", PALETTE["news2"], "-."),
    ]:
        d = dca_df[dca_df["model"] == m].sort_values("threshold")
        ax.plot(100 * d["threshold"], 1000 * d["net_benefit_model"], ls=ls, color=col, lw=2.0, marker="o", markersize=4.5, label=m)
    d0 = dca_df[dca_df["model"] == "LightGBM"].sort_values("threshold")
    ax.plot(100 * d0["threshold"], 1000 * d0["net_benefit_all"], ":", color="#64748B", lw=1.3, label="Alert all eligible")
    ax.axhline(0, color="#0F172A", lw=1.0, label="Alert none")
    ax.set_xlabel("Risk threshold probability (%)")
    ax.set_ylabel("Net benefit (per 1,000 eligible assessments)")
    ax.set_xlim(0.3, 15.3); ax.set_ylim(-2.5, 11.5)
    ax.set_title("Decision Curve Analysis in the 2017-2022 Temporal Test Set", loc="left")
    ax.legend(loc="upper right")
    save_fig(fig, SUPP_FIG_DIR, "figure_s4_decision_curve")

    # Figure S5: Grouped Permutation Importance
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    style_ax(ax)
    imp_s = imp_df.sort_values("lgbm_auroc_drop_mean", ascending=True)
    y_pos = np.arange(len(imp_s))
    h = 0.36
    ax.barh(y_pos + h / 2, imp_s["lgbm_auroc_drop_mean"], height=h, xerr=imp_s["lgbm_auroc_drop_sd"], color=PALETTE["lgb"], alpha=0.9, label="LightGBM", capsize=2)
    ax.barh(y_pos - h / 2, imp_s["lr_auroc_drop_mean"], height=h, xerr=imp_s["lr_auroc_drop_sd"], color=PALETTE["lr"], alpha=0.8, label="Logistic regression", capsize=2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(imp_s["feature_group"].str.replace("_", " ").str.title())
    ax.set_xlabel("AUROC drop after grouped permutation (mean ± SD, 10 repeats)")
    ax.set_title("Grouped Predictor Importance in the 2017-2022 Temporal Test Set", loc="left")
    ax.legend(loc="lower right")
    save_fig(fig, SUPP_FIG_DIR, "figure_s5_grouped_importance")

    # Figure S6: Static-6h vs Dynamic-Updating Ablation
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    plt.subplots_adjust(wspace=0.25)
    wins = ["landmark_12h", "landmark_18h", "landmark_24h", "pooled_12_18_24h"]
    lbls = ["12 h", "18 h", "24 h", "Pooled\n(12-24 h)"]
    x = np.arange(len(wins))
    w = 0.34
    sd_lgb = sd_df[sd_df["model"] == "LightGBM"].set_index("window").loc[wins]
    for ax_i, (dyn_c, sta_c, ylab, title) in enumerate([
        ("dynamic_auroc", "static6h_auroc", "AUROC", "A. Dynamic Updating vs Static 6 h Score (AUROC)"),
        ("dynamic_5pct_elig_cap_pct", "static6h_5pct_elig_cap_pct", "5% Eligible Capture (%)", "B. Dynamic Updating vs Static 6 h Score (5% Capture)"),
    ]):
        ax = axes[ax_i]
        style_ax(ax)
        b1 = ax.bar(x - w / 2, sd_lgb[dyn_c], width=w, color=PALETTE["lgb"], label="Dynamic rolling features")
        b2 = ax.bar(x + w / 2, sd_lgb[sta_c], width=w, color="#CBD5E1", edgecolor="#475569", label="Static 6 h baseline score")
        for bar in list(b1) + list(b2):
            fmt = f"{bar.get_height():.3f}" if ax_i == 0 else f"{bar.get_height():.1f}%"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (0.008 if ax_i == 0 else 0.9), fmt, ha="center", va="bottom", fontsize=7.6)
        ax.set_xticks(x); ax.set_xticklabels(lbls)
        ax.set_ylabel(ylab)
        ax.set_ylim(0.55, 0.88) if ax_i == 0 else ax.set_ylim(0, 60)
        ax.set_title(title, loc="left")
        ax.legend(loc="upper left" if ax_i == 1 else "lower left")
    save_fig(fig, SUPP_FIG_DIR, "figure_s6_static_vs_dynamic")

    # Figure S7: Signed Timing Concordance & Lab Storetime Turnaround Delay
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    plt.subplots_adjust(wspace=0.28)
    both = states.drop_duplicates("case_id")
    both = both[both["proc_imv_start_hours"].notna() & both["chart_imv_start_hours"].notna()]
    sdiff = (both["proc_imv_start_hours"] - both["chart_imv_start_hours"]).clip(-3, 3)
    ax = axes[0]
    style_ax(ax)
    ax.hist(sdiff, bins=48, color=PALETTE["lgb"], alpha=0.82, edgecolor="white")
    ax.axvline(0, color="#DC2626", ls="--", lw=1.2, label="Zero difference")
    ax.axvline(sdiff.median(), color="#0F172A", ls="-", lw=1.3, label=f"Median = +{sdiff.median():.2f} h")
    ax.set_xlabel("Procedure IMV start minus chartevents IMV start (hours, clipped ±3 h)")
    ax.set_ylabel("Matched ventilated stays (n = 22,941)")
    ax.set_title("A. Signed Timing Concordance Between Ventilation Sources", loc="left")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    style_ax(ax)
    ld = lab_delay.sort_values("median_turnaround_h", ascending=True)
    y_pos = np.arange(len(ld))
    ax.barh(y_pos, ld["median_turnaround_h"], color="#60A5FA", edgecolor="#1D4ED8", height=0.55, label="Median turnaround")
    ax.hlines(y_pos, ld["q25_turnaround_h"], ld["p90_turnaround_h"], color="#0F172A", lw=2.0, label="Q25 to P90 range")
    ax.plot(ld["p90_turnaround_h"], y_pos, "o", color="#DC2626", markersize=5, label="90th percentile")
    ax.set_yticks(y_pos); ax.set_yticklabels(ld["feature"].str.upper())
    ax.set_xlabel("Lab result turnaround delay: storetime − charttime (hours)")
    ax.set_title("B. Laboratory Result Availability Delay (storetime Audit)", loc="left")
    ax.legend(loc="lower right", fontsize=7.6)
    save_fig(fig, SUPP_FIG_DIR, "figure_s7_timing_and_storetime")

    # Figure S8: Comprehensive Robustness Forest Summary (LightGBM)
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    style_ax(ax)
    rows = []
    os_l = out_sens[out_sens["model"] == "LightGBM"].set_index("outcome_sensitivity")
    cp_l = comp_df[comp_df["model"] == "LightGBM"].set_index("sensitivity")
    mc_l = mod_comp[(mod_comp["model"] == "LightGBM") & (mod_comp["calibration"] == "landmark_specific_validation")].set_index("suite")

    specs = [
        ("Primary: Unplanned non-CVICU, 6h harmonized (Temporal)", os_l.loc["primary_harmonized_6h", "auroc"], os_l.loc["primary_harmonized_6h", "elig_5pct_capture_pct"]),
        ("1-hour prediction gap: (L+1h, L+6h]", os_l.loc["prediction_gap_1h_(L+1,L+6]", "auroc"], os_l.loc["prediction_gap_1h_(L+1,L+6]", "elig_5pct_capture_pct"]),
        ("2-hour prediction gap: (L+2h, L+6h]", os_l.loc["prediction_gap_2h_(L+2,L+6]", "auroc"], os_l.loc["prediction_gap_2h_(L+2,L+6]", "elig_5pct_capture_pct"]),
        ("Procedure-confirmed IMV (225792) only", os_l.loc["procedure_confirmed_225792_6h", "auroc"], os_l.loc["procedure_confirmed_225792_6h", "elig_5pct_capture_pct"]),
        ("Intubation-confirmed (224385) only", os_l.loc["intubation_confirmed_224385_6h", "auroc"], os_l.loc["intubation_confirmed_224385_6h", "elig_5pct_capture_pct"]),
        ("12-hour prediction horizon", os_l.loc["horizon_12h", "auroc"], os_l.loc["horizon_12h", "elig_5pct_capture_pct"]),
        ("24-hour prediction horizon", os_l.loc["horizon_24h", "auroc"], os_l.loc["horizon_24h", "elig_5pct_capture_pct"]),
        ("Exclude 6h competing ICU death & discharge", cp_l.loc["Exclude both 6h competing death & discharge", "auroc"], cp_l.loc["Exclude both 6h competing death & discharge", "elig_5pct_capture_pct"]),
        ("Composite outcome: IMV or 6h ICU death", cp_l.loc["Composite outcome: IMV or 6h ICU death", "auroc"], cp_l.loc["Composite outcome: IMV or 6h ICU death", "elig_5pct_capture_pct"]),
        ("Exclude documented DNI / comfort-measures", cp_l.loc["Exclude documented DNI / comfort-measures-only", "auroc"], cp_l.loc["Exclude documented DNI / comfort-measures-only", "elig_5pct_capture_pct"]),
        ("Random patient-grouped split (seed 42)", mc_l.loc["sensitivity_primary_random_split_16var", "auroc"], mc_l.loc["sensitivity_primary_random_split_16var", "elig_5pct_capture_pct"]),
        ("Retrained: All monitored ICU (incl. CVICU/planned)", mc_l.loc["sensitivity_all_monitored_icu_temporal_16var", "auroc"], mc_l.loc["sensitivity_all_monitored_icu_temporal_16var", "elig_5pct_capture_pct"]),
        ("Retrained: Strict non-surgical unplanned ICU", mc_l.loc["sensitivity_strict_non_surgical_temporal_16var", "auroc"], mc_l.loc["sensitivity_strict_non_surgical_temporal_16var", "elig_5pct_capture_pct"]),
    ]
    y_pos = np.arange(len(specs))[::-1]
    aucs = [s[1] for s in specs]
    caps = [s[2] for s in specs]
    ax.axvline(aucs[0], color="#64748B", ls="--", lw=1.0)
    ax.barh(y_pos, aucs, height=0.58, color="#93C5FD", edgecolor=PALETTE["lgb"], lw=0.9)
    for yi, a, c in zip(y_pos, aucs, caps):
        ax.text(a + 0.004, yi, f"AUROC {a:.3f} | 5% cap {c:.1f}%", va="center", fontsize=7.6, color="#0F172A")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([s[0] for s in specs], fontsize=8.0)
    ax.set_xlim(0.65, 0.94)
    ax.set_xlabel("Held-out AUROC (annotated with 5% eligible event capture)")
    ax.set_title("Sensitivity and Robustness Summary Across Outcomes, Gaps, Competing Risks, and Cohorts (LightGBM)", loc="left")
    save_fig(fig, SUPP_FIG_DIR, "figure_s8_robustness")


def main() -> None:
    print("Loading R1 result tables to render Figures 1-3 and S1-S8...")
    audit = json.loads((RES_DIR / "audit" / "r1_extract_and_audit_summary.json").read_text())
    lm_flow = pd.read_csv(RES_DIR / "audit" / "r1_primary_landmark_flow.csv")
    lab_delay = pd.read_csv(RES_DIR / "audit" / "r1_lab_turnaround_by_feature.csv")
    mod_comp = pd.read_csv(RES_DIR / "models" / "r1_model_comparison_and_ablations.csv")
    era_lm = pd.read_csv(RES_DIR / "models" / "r1_era_and_landmark_performance.csv")
    sd_df = pd.read_csv(RES_DIR / "models" / "r1_static_vs_dynamic_ablation.csv")
    out_sens = pd.read_csv(RES_DIR / "models" / "r1_outcome_and_gap_sensitivities.csv")
    preds = pd.read_parquet(RES_DIR / "models" / "r1_primary_temporal_predictions.parquet")
    disc_ci = pd.read_csv(RES_DIR / "operational" / "r1_model_performance_with_ci.csv")
    curves = pd.read_csv(RES_DIR / "operational" / "r1_capacity_response_curves_dual_denom.csv")
    cd_df = pd.read_csv(RES_DIR / "operational" / "r1_cooldown_closest_alert.csv")
    fact_df = pd.read_csv(RES_DIR / "operational" / "r1_factorial_2x2_clean.csv")
    sub_df = pd.read_csv(RES_DIR / "operational" / "r1_subgroups_with_ci.csv")
    comp_df = pd.read_csv(RES_DIR / "operational" / "r1_competing_and_code_status.csv")
    dca_df = pd.read_csv(RES_DIR / "operational" / "r1_decision_curve.csv")
    imp_df = pd.read_csv(RES_DIR / "descriptives" / "r1_grouped_permutation_importance.csv")
    states = pd.read_parquet(PROC_R1 / "r1_landmark_states.parquet")

    make_figure1(audit, lm_flow)
    make_figure2(curves, disc_ci, cd_df)
    make_figure3(era_lm, disc_ci, sub_df, fact_df)
    make_supplementary_figures(states, preds, dca_df, imp_df, sd_df, out_sens, comp_df, mod_comp, lab_delay)
    print("ALL 11 R1 FIGURES GENERATED SUCCESSFULLY.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Study 5 BMJHCI R1 (v30): Complete Operational Evaluation, Patient-Cluster Bootstrap CIs,
Paired Model Comparisons, Closest-Alert Cooldown Re-Analysis, 2x2 Factorial Eligibility,
Competing Risks / Code-Status Sensitivities, Full Subgroup Audit, and Decision Curve Analysis.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


BASE_DIR = Path(__file__).resolve().parent.parent
PROC_R1 = BASE_DIR / "data" / "processed" / "v30_bmjhci_r1"
PROC_V25 = BASE_DIR / "data" / "processed" / "v25_clinical_upgrade"
MOD_DIR = BASE_DIR / "results" / "high_tier_upgrade" / "v30_bmjhci_r1" / "models"
OUT_DIR = BASE_DIR / "results" / "high_tier_upgrade" / "v30_bmjhci_r1" / "operational"

SEED = 42
N_BOOT = 1000
LANDMARKS = [6.0, 12.0, 18.0, 24.0]
BUDGETS = [0.01, 0.02, 0.05, 0.10, 0.20]
LEAD_THRESHOLDS = [0.5, 1.0, 2.0, 4.0]


def ci_bounds(arr: np.ndarray) -> tuple[float, float]:
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return np.nan, np.nan
    return float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))


def build_bootstrap_indices(case_ids: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Precompute patient clusters and bootstrap patient draws for fast vectorized resampling."""
    uniq_ids, inv = np.unique(case_ids, return_inverse=True)
    n_pts = len(uniq_ids)
    # Map patient index -> row indices
    order = np.argsort(inv, kind="mergesort")
    counts = np.bincount(inv, minlength=n_pts)
    starts = np.cumsum(counts) - counts
    pt_rows = [order[starts[i] : starts[i] + counts[i]] for i in range(n_pts)]
    rng = np.random.RandomState(seed)
    draws = rng.randint(0, n_pts, size=(n_boot, n_pts))
    return uniq_ids, pt_rows, draws


def eval_quota_point(
    df: pd.DataFrame,
    score_col: str,
    y_col: str,
    k_alerts: int,
    denom_n: int,
    event_time_col: str = "event_time_r1_hours",
) -> dict[str, float | int]:
    """Evaluate single fixed-quota operating point cleanly separating capture, reach, and lead time."""
    ordered = df.sort_values([score_col, "case_id", "landmark_hours"], ascending=[False, True, True], kind="mergesort")
    alerts = ordered.head(k_alerts)
    total_events = int(df[y_col].sum())
    det_events = int(alerts[y_col].sum())
    uniq_alerted = int(alerts["case_id"].nunique())
    rep_alerts = int(len(alerts) - uniq_alerted)

    # Event-landmark lead time (among detected event-positive landmarks)
    det_rows = alerts[alerts[y_col] == 1]
    lm_leads = (det_rows[event_time_col] - det_rows["landmark_hours"]).dropna()

    # Patient-level any-prior-alert reach & earliest/closest alert before event
    ev_pts = df.loc[df[y_col] == 1, ["case_id", event_time_col]].drop_duplicates("case_id")
    n_ev_pts = len(ev_pts)
    m = alerts[["case_id", "landmark_hours"]].merge(ev_pts, on="case_id", how="inner")
    m = m[m["landmark_hours"] < m[event_time_col]].copy()
    if len(m) > 0:
        per_pt = m.groupby("case_id", sort=False).agg(
            ev_t=(event_time_col, "first"),
            earliest_lm=("landmark_hours", "min"),
            closest_lm=("landmark_hours", "max"),
        )
        earliest_lead = per_pt["ev_t"] - per_pt["earliest_lm"]
        closest_lead = per_pt["ev_t"] - per_pt["closest_lm"]
        reached = int(len(per_pt))
    else:
        earliest_lead = pd.Series(dtype=float)
        closest_lead = pd.Series(dtype=float)
        reached = 0

    scale = 1000.0 / max(denom_n, 1)
    out: dict[str, float | int] = {
        "alerts": int(len(alerts)),
        "alerts_per_1000": float(len(alerts) * scale),
        "unique_patients_alerted": uniq_alerted,
        "unique_patients_per_1000": float(uniq_alerted * scale),
        "repeated_alerts": rep_alerts,
        "repeated_alert_frac_pct": float(100.0 * rep_alerts / max(len(alerts), 1)),
        "events_total": total_events,
        "events_detected": det_events,
        "event_capture_pct": float(100.0 * det_events / max(total_events, 1)),
        "ppv_pct": float(100.0 * det_events / max(len(alerts), 1)),
        "alerts_per_detected_event": float(len(alerts) / det_events) if det_events else np.nan,
        "event_patients_reached": reached,
        "patient_reach_pct": float(100.0 * reached / max(n_ev_pts, 1)),
        "landmark_warning_median_h": float(lm_leads.median()) if len(lm_leads) else np.nan,
        "landmark_warning_q25_h": float(lm_leads.quantile(0.25)) if len(lm_leads) else np.nan,
        "landmark_warning_q75_h": float(lm_leads.quantile(0.75)) if len(lm_leads) else np.nan,
        "earliest_warning_median_h": float(earliest_lead.median()) if len(earliest_lead) else np.nan,
        "earliest_warning_q25_h": float(earliest_lead.quantile(0.25)) if len(earliest_lead) else np.nan,
        "earliest_warning_q75_h": float(earliest_lead.quantile(0.75)) if len(earliest_lead) else np.nan,
        "closest_warning_median_h": float(closest_lead.median()) if len(closest_lead) else np.nan,
    }
    for thr in LEAD_THRESHOLDS:
        out[f"lead_ge_{thr}h_event_landmark_pct"] = float(100.0 * (lm_leads >= thr).sum() / max(n_ev_pts, 1))
        out[f"lead_ge_{thr}h_any_prior_pct"] = float(100.0 * (earliest_lead >= thr).sum() / max(n_ev_pts, 1))
    return out


def bootstrap_quota_and_discrimination(
    df_wide: pd.DataFrame,
    model_cols: dict[str, str],
    y_col: str,
    total_scheduled_n: int,
    n_boot: int = N_BOOT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fast patient-cluster bootstrap for:
    1) Discrimination & calibration (with paired contrasts)
    2) Dual-denominator capacity-response curves (eligible basis & all-scheduled basis)
    3) Lead-time-specific reach CIs at 5% eligible quota
    """
    n_elig = len(df_wide)
    y_all = df_wide[y_col].astype(int).to_numpy()
    lm_all = df_wide["landmark_hours"].to_numpy()
    ev_t_all = df_wide["event_time_r1_hours"].to_numpy()
    cid_all = df_wide["case_id"].astype(str).to_numpy()
    uniq_ids, pt_rows, draws = build_bootstrap_indices(cid_all, n_boot=n_boot, seed=SEED)
    # Build integer patient id per row
    pt_id_all = np.searchsorted(uniq_ids, cid_all)
    n_pts = len(uniq_ids)
    # Pre-extract arrays per model
    scores = {m: df_wide[c].astype(float).to_numpy() for m, c in model_cols.items()}

    # Point estimates for discrimination
    disc_boot: dict[str, dict[str, np.ndarray]] = {
        m: {k: np.full(n_boot, np.nan) for k in ["auroc", "auprc", "brier", "eo", "cap5", "ppv5", "reach5", "cap1", "ppv1", "lead_ge_2h_lm", "lead_ge_2h_any", "lead_ge_1h_any", "lead_ge_4h_any", "earliest_med"]}
        for m in model_cols
    }
    # Also store capacity curve bootstrap for key models
    curve_models = [m for m in ["LightGBM", "Logistic Regression", "Modified NEWS2 (no O2)", "Modified NEWS2 (with O2)", "SpO2 min alone", "RR max alone"] if m in model_cols]
    curve_boot: dict[tuple[str, str, float], dict[str, np.ndarray]] = {}
    for m in curve_models:
        for basis in ["eligible", "all_scheduled"]:
            for b in BUDGETS:
                curve_boot[(m, basis, b)] = {
                    k: np.full(n_boot, np.nan) for k in ["cap", "ppv", "reach", "rep_frac"]
                }

    # Pre-build patient-level arrays (each patient has 1..4 rows; flatten via numpy repeat!)
    # To make 1000 bootstraps super fast (<15s), use patient-level weight vector (multinomial counts)!
    # Wait: exact row replication with index concatenation is ~12s for 1000 boots on 36k rows if done in numpy!
    # Let's check: concatenate indices using flat array of row positions per patient.
    pt_counts = np.array([len(r) for r in pt_rows], dtype=np.int32)
    pt_offsets = np.cumsum(pt_counts) - pt_counts
    flat_order = np.concatenate(pt_rows)

    def sample_rows(pt_draw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return sampled row indices and unique bootstrap patient IDs (distinguishing duplicate draws)."""
        lens = pt_counts[pt_draw]
        tot = int(lens.sum())
        # Build row indices vectorized
        starts = pt_offsets[pt_draw]
        # Cumulative offsets in output
        out_starts = np.cumsum(lens) - lens
        # Vectorized range expansion
        row_pos = np.repeat(starts, lens) + (np.arange(tot) - np.repeat(out_starts, lens))
        row_idx = flat_order[row_pos]
        boot_pid = np.repeat(np.arange(len(pt_draw), dtype=np.int32), lens)
        return row_idx, boot_pid

    # Ratio of total_scheduled_n to n_elig (for all_scheduled basis scaling in bootstrap)
    sched_ratio = float(total_scheduled_n) / float(n_elig)

    t0 = time.time()
    for b_idx in range(n_boot):
        r_idx, b_pid = sample_rows(draws[b_idx])
        yb = y_all[r_idx]
        lmb = lm_all[r_idx]
        evtb = ev_t_all[r_idx]
        n_b = len(yb)
        ev_b = int(yb.sum())
        if ev_b == 0:
            continue
        # Event patient lookup (unique boot_pid -> event_time)
        ev_mask_b = yb == 1
        ev_pids = b_pid[ev_mask_b]
        ev_times = evtb[ev_mask_b]
        ev_map = np.full(n_pts, np.nan, dtype=float)
        ev_map[ev_pids] = ev_times

        for m, sc_arr in scores.items():
            pb = sc_arr[r_idx]
            disc_boot[m]["auroc"][b_idx] = roc_auc_score(yb, pb)
            disc_boot[m]["auprc"][b_idx] = average_precision_score(yb, pb)
            if (pb >= 0.0).all() and (pb <= 1.0).all():
                disc_boot[m]["brier"][b_idx] = brier_score_loss(yb, pb)
                disc_boot[m]["eo"][b_idx] = pb.sum() / ev_b

            # Sort descending once per model per bootstrap!
            ord_idx = np.argsort(-pb, kind="mergesort")
            y_ord = yb[ord_idx]
            pid_ord = b_pid[ord_idx]
            lm_ord = lmb[ord_idx]
            evt_ord = evtb[ord_idx]

            # 5% eligible & 1% eligible quick metrics for all models
            k5 = int(np.ceil(n_b * 0.05))
            k1 = int(np.ceil(n_b * 0.01))
            y5 = y_ord[:k5]
            disc_boot[m]["cap5"][b_idx] = 100.0 * y5.sum() / ev_b
            disc_boot[m]["ppv5"][b_idx] = 100.0 * y5.mean()
            disc_boot[m]["cap1"][b_idx] = 100.0 * y_ord[:k1].sum() / ev_b
            disc_boot[m]["ppv1"][b_idx] = 100.0 * y_ord[:k1].mean()

            # Patient reach & lead-time at 5% eligible
            p5 = pid_ord[:k5]
            l5 = lm_ord[:k5]
            et5 = ev_map[p5]
            prior_m = np.isfinite(et5) & (l5 < et5)
            if prior_m.any():
                # Earliest alert per reached patient
                rp = p5[prior_m]
                rl = l5[prior_m]
                ret = et5[prior_m]
                # Min landmark per boot_pid via sorting by (pid, lm)
                s_o = np.lexsort((rl, rp))
                rp_s = rp[s_o]
                rl_s = rl[s_o]
                ret_s = ret[s_o]
                first_mask = np.concatenate(([True], rp_s[1:] != rp_s[:-1]))
                earliest_leads = ret_s[first_mask] - rl_s[first_mask]
                disc_boot[m]["reach5"][b_idx] = 100.0 * len(earliest_leads) / ev_b
                disc_boot[m]["lead_ge_1h_any"][b_idx] = 100.0 * (earliest_leads >= 1.0).sum() / ev_b
                disc_boot[m]["lead_ge_2h_any"][b_idx] = 100.0 * (earliest_leads >= 2.0).sum() / ev_b
                disc_boot[m]["lead_ge_4h_any"][b_idx] = 100.0 * (earliest_leads >= 4.0).sum() / ev_b
                disc_boot[m]["earliest_med"][b_idx] = float(np.median(earliest_leads))
            else:
                disc_boot[m]["reach5"][b_idx] = 0.0
                disc_boot[m]["lead_ge_1h_any"][b_idx] = 0.0
                disc_boot[m]["lead_ge_2h_any"][b_idx] = 0.0
                disc_boot[m]["lead_ge_4h_any"][b_idx] = 0.0

            det5_m = y5 == 1
            if det5_m.any():
                lm_leads5 = evt_ord[:k5][det5_m] - l5[det5_m]
                disc_boot[m]["lead_ge_2h_lm"][b_idx] = 100.0 * (lm_leads5 >= 2.0).sum() / ev_b
            else:
                disc_boot[m]["lead_ge_2h_lm"][b_idx] = 0.0

            # Full curve bootstrap for curve_models
            if m in curve_models:
                for basis in ["eligible", "all_scheduled"]:
                    denom_b = n_b if basis == "eligible" else int(round(n_b * sched_ratio))
                    for bud in BUDGETS:
                        kb = min(int(np.ceil(denom_b * bud)), n_b)
                        yk = y_ord[:kb]
                        pk = pid_ord[:kb]
                        lk = lm_ord[:kb]
                        curve_boot[(m, basis, bud)]["cap"][b_idx] = 100.0 * yk.sum() / ev_b
                        curve_boot[(m, basis, bud)]["ppv"][b_idx] = 100.0 * yk.mean()
                        uniq_k = len(np.unique(pk))
                        curve_boot[(m, basis, bud)]["rep_frac"][b_idx] = 100.0 * (kb - uniq_k) / max(kb, 1)
                        etk = ev_map[pk]
                        pm = np.isfinite(etk) & (lk < etk)
                        r_k = len(np.unique(pk[pm])) if pm.any() else 0
                        curve_boot[(m, basis, bud)]["reach"][b_idx] = 100.0 * r_k / ev_b

    print(f"  Completed {n_boot} patient-cluster bootstraps across {len(model_cols)} models in {time.time()-t0:.1f}s")

    # Assemble summary table with CIs
    disc_rows = []
    for m, col in model_cols.items():
        pt_m = eval_quota_point(df_wide, col, y_col, int(np.ceil(n_elig * 0.05)), n_elig)
        pt_1 = eval_quota_point(df_wide, col, y_col, int(np.ceil(n_elig * 0.01)), n_elig)
        p_arr = df_wide[col].astype(float).to_numpy()
        auc_pt = float(roc_auc_score(y_all, p_arr))
        ap_pt = float(average_precision_score(y_all, p_arr))
        row: dict[str, Any] = {
            "model": m,
            "auroc": auc_pt, "auroc_lo": ci_bounds(disc_boot[m]["auroc"])[0], "auroc_hi": ci_bounds(disc_boot[m]["auroc"])[1],
            "auprc": ap_pt, "auprc_lo": ci_bounds(disc_boot[m]["auprc"])[0], "auprc_hi": ci_bounds(disc_boot[m]["auprc"])[1],
            "brier": float(brier_score_loss(y_all, p_arr)) if (p_arr.min() >= 0 and p_arr.max() <= 1) else np.nan,
            "brier_lo": ci_bounds(disc_boot[m]["brier"])[0], "brier_hi": ci_bounds(disc_boot[m]["brier"])[1],
            "eo_ratio": float(p_arr.sum() / y_all.sum()) if (p_arr.min() >= 0 and p_arr.max() <= 1) else np.nan,
            "eo_lo": ci_bounds(disc_boot[m]["eo"])[0], "eo_hi": ci_bounds(disc_boot[m]["eo"])[1],
            "elig_1pct_cap": pt_1["event_capture_pct"], "elig_1pct_cap_lo": ci_bounds(disc_boot[m]["cap1"])[0], "elig_1pct_cap_hi": ci_bounds(disc_boot[m]["cap1"])[1],
            "elig_1pct_ppv": pt_1["ppv_pct"], "elig_1pct_ppv_lo": ci_bounds(disc_boot[m]["ppv1"])[0], "elig_1pct_ppv_hi": ci_bounds(disc_boot[m]["ppv1"])[1],
            "elig_5pct_cap": pt_m["event_capture_pct"], "elig_5pct_cap_lo": ci_bounds(disc_boot[m]["cap5"])[0], "elig_5pct_cap_hi": ci_bounds(disc_boot[m]["cap5"])[1],
            "elig_5pct_ppv": pt_m["ppv_pct"], "elig_5pct_ppv_lo": ci_bounds(disc_boot[m]["ppv5"])[0], "elig_5pct_ppv_hi": ci_bounds(disc_boot[m]["ppv5"])[1],
            "elig_5pct_reach": pt_m["patient_reach_pct"], "elig_5pct_reach_lo": ci_bounds(disc_boot[m]["reach5"])[0], "elig_5pct_reach_hi": ci_bounds(disc_boot[m]["reach5"])[1],
            "elig_5pct_lead_ge_1h_any": pt_m["lead_ge_1.0h_any_prior_pct"], "elig_5pct_lead_ge_1h_any_lo": ci_bounds(disc_boot[m]["lead_ge_1h_any"])[0], "elig_5pct_lead_ge_1h_any_hi": ci_bounds(disc_boot[m]["lead_ge_1h_any"])[1],
            "elig_5pct_lead_ge_2h_any": pt_m["lead_ge_2.0h_any_prior_pct"], "elig_5pct_lead_ge_2h_any_lo": ci_bounds(disc_boot[m]["lead_ge_2h_any"])[0], "elig_5pct_lead_ge_2h_any_hi": ci_bounds(disc_boot[m]["lead_ge_2h_any"])[1],
            "elig_5pct_lead_ge_4h_any": pt_m["lead_ge_4.0h_any_prior_pct"], "elig_5pct_lead_ge_4h_any_lo": ci_bounds(disc_boot[m]["lead_ge_4h_any"])[0], "elig_5pct_lead_ge_4h_any_hi": ci_bounds(disc_boot[m]["lead_ge_4h_any"])[1],
            "elig_5pct_lead_ge_2h_lm": pt_m["lead_ge_2.0h_event_landmark_pct"], "elig_5pct_lead_ge_2h_lm_lo": ci_bounds(disc_boot[m]["lead_ge_2h_lm"])[0], "elig_5pct_lead_ge_2h_lm_hi": ci_bounds(disc_boot[m]["lead_ge_2h_lm"])[1],
            "elig_5pct_earliest_med_h": pt_m["earliest_warning_median_h"], "elig_5pct_earliest_med_lo": ci_bounds(disc_boot[m]["earliest_med"])[0], "elig_5pct_earliest_med_hi": ci_bounds(disc_boot[m]["earliest_med"])[1],
        }
        disc_rows.append(row)

    # Paired contrasts
    pairs = [
        ("LightGBM", "Logistic Regression"),
        ("LightGBM", "Modified NEWS2 (no O2)"),
        ("Logistic Regression", "Modified NEWS2 (no O2)"),
        ("LightGBM", "Modified NEWS2 (with O2)"),
        ("LightGBM", "SpO2 min alone"),
        ("LightGBM", "RR max alone"),
        ("LightGBM (+O2 support)", "LightGBM"),
        ("LightGBM (physiology-only)", "LightGBM"),
        ("LightGBM (charting-only)", "LightGBM"),
        ("Logistic Regression (physiology-only)", "Logistic Regression"),
        ("Logistic Regression (charting-only)", "Logistic Regression"),
    ]
    disc_lookup = {r["model"]: r for r in disc_rows}
    contrast_rows = []
    for m_a, m_b in pairs:
        if m_a not in disc_boot or m_b not in disc_boot:
            continue
        for met_pt, met_b in [
            ("auroc", "auroc"),
            ("auprc", "auprc"),
            ("elig_5pct_cap", "cap5"),
            ("elig_5pct_ppv", "ppv5"),
            ("elig_5pct_reach", "reach5"),
        ]:
            diff_pt = float(disc_lookup[m_a][met_pt] - disc_lookup[m_b][met_pt])
            diff_b = disc_boot[m_a][met_b] - disc_boot[m_b][met_b]
            lo, hi = ci_bounds(diff_b)
            # Two-sided bootstrap p-value approximation
            valid = diff_b[np.isfinite(diff_b)]
            p_two = float(2.0 * min((valid <= 0).mean(), (valid >= 0).mean())) if len(valid) else np.nan
            contrast_rows.append({
                "model_a": m_a, "model_b": m_b, "metric": met_pt,
                "value_a": disc_lookup[m_a][met_pt], "value_b": disc_lookup[m_b][met_pt],
                "diff_a_minus_b": diff_pt, "ci_lo": lo, "ci_hi": hi, "boot_p": min(p_two, 1.0),
            })

    # Full capacity curve table (both denominators)
    curve_rows = []
    for m in curve_models:
        col = model_cols[m]
        for basis, denom_n in [("eligible", n_elig), ("all_scheduled", total_scheduled_n)]:
            for bud in BUDGETS:
                k = min(int(np.ceil(denom_n * bud)), n_elig)
                pt = eval_quota_point(df_wide, col, y_col, k, denom_n)
                cb = curve_boot[(m, basis, bud)]
                curve_rows.append({
                    "model": m,
                    "budget_basis": basis,
                    "budget_pct": 100.0 * bud,
                    "equiv_eligible_pct": float(100.0 * k / n_elig),
                    "equiv_scheduled_pct": float(100.0 * k / total_scheduled_n),
                    **pt,
                    "event_capture_lo": ci_bounds(cb["cap"])[0], "event_capture_hi": ci_bounds(cb["cap"])[1],
                    "ppv_lo": ci_bounds(cb["ppv"])[0], "ppv_hi": ci_bounds(cb["ppv"])[1],
                    "patient_reach_lo": ci_bounds(cb["reach"])[0], "patient_reach_hi": ci_bounds(cb["reach"])[1],
                    "rep_frac_lo": ci_bounds(cb["rep_frac"])[0], "rep_frac_hi": ci_bounds(cb["rep_frac"])[1],
                })

    return pd.DataFrame(disc_rows), pd.DataFrame(contrast_rows), pd.DataFrame(curve_rows)


def apply_cooldown(alerts: pd.DataFrame, cooldown_hours: float) -> pd.DataFrame:
    ordered = alerts.sort_values(["case_id", "landmark_hours"], kind="mergesort").copy()
    if cooldown_hours <= 0:
        return ordered
    keep = np.zeros(len(ordered), dtype=bool)
    for _, pos in ordered.groupby("case_id", sort=False).indices.items():
        last_kept = -np.inf
        for p in pos:
            lm = float(ordered.iloc[p]["landmark_hours"])
            if lm - last_kept > cooldown_hours:
                keep[p] = True
                last_kept = lm
    return ordered.loc[keep].reset_index(drop=True)


def run_cooldown_reanalysis(df_wide: pd.DataFrame, total_scheduled_n: int) -> pd.DataFrame:
    """
    R1 #5 & Results #7: Foreground event-window capture loss AND closest clinically relevant
    alert (timely alert within 6h / 12h before ventilation + closest-alert lead time).
    """
    n_elig = len(df_wide)
    ev_pts = df_wide.loc[df_wide["outcome_r1_6h"] == 1, ["case_id", "event_time_r1_hours"]].drop_duplicates("case_id")
    n_ev = len(ev_pts)
    rows = []
    for basis, denom_n, bud in [("eligible_5pct", n_elig, 0.05), ("all_scheduled_5pct", total_scheduled_n, 0.05)]:
        k = int(np.ceil(denom_n * bud))
        for mname, col in [("LightGBM", "p_lgb"), ("Logistic Regression", "p_lr")]:
            base_alerts = df_wide.sort_values([col, "case_id", "landmark_hours"], ascending=[False, True, True], kind="mergesort").head(k)
            base_cap = float(100.0 * base_alerts["outcome_r1_6h"].sum() / n_ev)
            for cd in [0.0, 6.0, 12.0]:
                kept = apply_cooldown(base_alerts, cd)
                det = int(kept["outcome_r1_6h"].sum())
                cap = float(100.0 * det / n_ev)
                ppv = float(100.0 * det / max(len(kept), 1))
                m = kept[["case_id", "landmark_hours"]].merge(ev_pts, on="case_id", how="inner")
                m = m[m["landmark_hours"] < m["event_time_r1_hours"]].copy()
                if len(m) > 0:
                    pp = m.groupby("case_id", sort=False).agg(
                        ev_t=("event_time_r1_hours", "first"),
                        earliest_lm=("landmark_hours", "min"),
                        closest_lm=("landmark_hours", "max"),
                    )
                    earliest_gap = pp["ev_t"] - pp["earliest_lm"]
                    closest_gap = pp["ev_t"] - pp["closest_lm"]
                    any_reach = int(len(pp))
                    timely_6h = int((closest_gap <= 6.0).sum())
                    timely_12h = int((closest_gap <= 12.0).sum())
                    stale_gt_6h = int((closest_gap > 6.0).sum())
                else:
                    earliest_gap = closest_gap = pd.Series(dtype=float)
                    any_reach = timely_6h = timely_12h = stale_gt_6h = 0

                rows.append({
                    "budget_basis": basis,
                    "model": mname,
                    "cooldown_hours": int(cd),
                    "alerts_initial": int(len(base_alerts)),
                    "alerts_retained": int(len(kept)),
                    "alerts_suppressed": int(len(base_alerts) - len(kept)),
                    "alert_reduction_pct": float(100.0 * (len(base_alerts) - len(kept)) / max(len(base_alerts), 1)),
                    "event_window_detected": det,
                    "event_window_capture_pct": cap,
                    "capture_loss_pp": float(base_cap - cap),
                    "ppv_pct": ppv,
                    "any_prior_reach_n": any_reach,
                    "any_prior_reach_pct": float(100.0 * any_reach / n_ev),
                    "timely_closest_alert_le_6h_n": timely_6h,
                    "timely_closest_alert_le_6h_pct": float(100.0 * timely_6h / n_ev),
                    "timely_closest_alert_le_12h_n": timely_12h,
                    "timely_closest_alert_le_12h_pct": float(100.0 * timely_12h / n_ev),
                    "stale_closest_alert_gt_6h_n": stale_gt_6h,
                    "closest_alert_lead_median_h": float(closest_gap.median()) if len(closest_gap) else np.nan,
                    "closest_alert_lead_q25_h": float(closest_gap.quantile(0.25)) if len(closest_gap) else np.nan,
                    "closest_alert_lead_q75_h": float(closest_gap.quantile(0.75)) if len(closest_gap) else np.nan,
                    "earliest_alert_lead_median_h": float(earliest_gap.median()) if len(earliest_gap) else np.nan,
                })
    return pd.DataFrame(rows)


def run_factorial_2x2(fact_df: pd.DataFrame) -> pd.DataFrame:
    """2x2 target x eligibility under equal capacity (both 5% all scheduled and 5% eligible)."""
    n_all = len(fact_df)
    n_elig = int(fact_df["r1_eligible"].sum())
    ev_total = int(fact_df["outcome_r1_6h"].sum())
    rows = []
    for basis, k in [("5pct_all_scheduled", int(np.ceil(n_all * 0.05))), ("5pct_eligible", int(np.ceil(n_elig * 0.05)))]:
        for mname, inc_col, hist_col in [
            ("LightGBM", "lgb_inc_score", "lgb_hist_score"),
            ("Logistic Regression", "lr_inc_score", "lr_hist_score"),
        ]:
            for cell, target, filt, col in [
                ("A_incident_filter_on", "incident", "on", inc_col),
                ("B_incident_filter_off", "incident", "off", inc_col),
                ("C_history_filter_on", "history_inclusive", "on", hist_col),
                ("D_history_filter_off", "history_inclusive", "off", hist_col),
            ]:
                pool = fact_df[fact_df["r1_eligible"] == 1] if filt == "on" else fact_df
                top = pool.sort_values([col, "case_id", "landmark_hours"], ascending=[False, True, True], kind="mergesort").head(k)
                det = int(top["outcome_r1_6h"].sum())
                inelig = int((top["r1_eligible"] == 0).sum())
                rows.append({
                    "budget_basis": basis, "model": mname, "cell": cell,
                    "target": target, "eligibility_filter": filt,
                    "alerts": k, "ineligible_alerts": inelig,
                    "ineligible_alert_pct": float(100.0 * inelig / max(k, 1)),
                    "incident_events_detected": det,
                    "incident_capture_pct": float(100.0 * det / max(ev_total, 1)),
                    "incident_ppv_pct": float(100.0 * det / max(k, 1)),
                })
    return pd.DataFrame(rows)


def run_subgroups_and_competing(df_wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    1) Full subgroup performance (Sex, Age, ICU type, Admission type, Era, Landmark) with global 5% eligible quota + 1,000 bootstrap CIs
    2) Competing risks & DNI/CMO code-status sensitivities
    3) Alert allocation breakdown by landmark & ICU type
    """
    n_elig = len(df_wide)
    k5 = int(np.ceil(n_elig * 0.05))
    # Precompute global 5% alert flags
    for mcol, fcol in [("p_lgb", "alert5_lgb"), ("p_lr", "alert5_lr")]:
        top_idx = df_wide.sort_values([mcol, "case_id", "landmark_hours"], ascending=[False, True, True], kind="mergesort").head(k5).index
        df_wide[fcol] = 0
        df_wide.loc[top_idx, fcol] = 1

    # Define readable subgroup columns
    df_wide["sex_label"] = np.where(df_wide["sex"] == 1, "Male", "Female")
    df_wide["age_group"] = pd.cut(df_wide["age"], bins=[0, 64.999, 79.999, 200], labels=["<65 years", "65-79 years", ">=80 years"]).astype(str)
    icu_map = {
        "Medical Intensive Care Unit (MICU)": "MICU",
        "Medical/Surgical Intensive Care Unit (MICU/SICU)": "MICU/SICU",
        "Surgical Intensive Care Unit (SICU)": "SICU",
        "Trauma SICU (TSICU)": "TSICU",
        "Coronary Care Unit (CCU)": "CCU",
    }
    df_wide["icu_group"] = df_wide["first_careunit"].map(icu_map).fillna("Neuro / Other ICU")
    df_wide["adm_group"] = df_wide["admission_type"].where(
        df_wide["admission_type"].isin(["EW EMER.", "URGENT", "DIRECT EMER."]), "Observation / Other"
    )
    df_wide["landmark_label"] = df_wide["landmark_hours"].astype(int).astype(str) + "h"

    # Bootstrap patient draws once for subgroup CIs (recomputing global 5% threshold inside each bootstrap!)
    cid_all = df_wide["case_id"].astype(str).to_numpy()
    uniq_ids, pt_rows, draws = build_bootstrap_indices(cid_all, n_boot=N_BOOT, seed=SEED + 7)
    pt_counts = np.array([len(r) for r in pt_rows], dtype=np.int32)
    pt_offsets = np.cumsum(pt_counts) - pt_counts
    flat_order = np.concatenate(pt_rows)

    def sample_row_idx(pt_draw: np.ndarray) -> np.ndarray:
        lens = pt_counts[pt_draw]
        tot = int(lens.sum())
        starts = pt_offsets[pt_draw]
        out_starts = np.cumsum(lens) - lens
        return flat_order[np.repeat(starts, lens) + (np.arange(tot) - np.repeat(out_starts, lens))]

    # Collect subgroup masks
    subgroup_specs: list[tuple[str, str, np.ndarray]] = []
    for gtype, gcol in [
        ("Sex", "sex_label"),
        ("Age group", "age_group"),
        ("ICU type", "icu_group"),
        ("Admission type", "adm_group"),
        ("Temporal era", "anchor_year_group"),
        ("Landmark", "landmark_label"),
    ]:
        for val in sorted(df_wide[gcol].unique()):
            subgroup_specs.append((gtype, str(val), df_wide[gcol].eq(val).to_numpy()))

    y_all = df_wide["outcome_r1_6h"].astype(int).to_numpy()
    scores = {"LightGBM": df_wide["p_lgb"].to_numpy(), "Logistic Regression": df_wide["p_lr"].to_numpy()}

    sub_boot: dict[tuple[str, str, str], dict[str, np.ndarray]] = {
        (m, gt, gv): {k: np.full(N_BOOT, np.nan) for k in ["auroc", "auprc", "cap", "ppv", "eo"]}
        for m in scores for gt, gv, _ in subgroup_specs
    }

    for b_idx in range(N_BOOT):
        r_idx = sample_row_idx(draws[b_idx])
        yb = y_all[r_idx]
        kb = int(np.ceil(len(yb) * 0.05))
        for m, sc in scores.items():
            pb = sc[r_idx]
            # Global 5% alert mask
            top_pos = np.argpartition(-pb, kb - 1)[:kb]
            al_b = np.zeros(len(yb), dtype=bool)
            al_b[top_pos] = True
            for gt, gv, full_mask in subgroup_specs:
                sm = full_mask[r_idx]
                ys = yb[sm]
                ps = pb[sm]
                als = al_b[sm]
                evs = int(ys.sum())
                if evs > 0 and evs < len(ys):
                    sub_boot[(m, gt, gv)]["auroc"][b_idx] = roc_auc_score(ys, ps)
                    sub_boot[(m, gt, gv)]["auprc"][b_idx] = average_precision_score(ys, ps)
                    sub_boot[(m, gt, gv)]["eo"][b_idx] = ps.sum() / evs
                    sub_boot[(m, gt, gv)]["cap"][b_idx] = 100.0 * ys[als].sum() / evs
                if als.sum() > 0:
                    sub_boot[(m, gt, gv)]["ppv"][b_idx] = 100.0 * ys[als].mean()

    sub_rows = []
    alloc_rows = []
    for m, sc_col, al_col in [("LightGBM", "p_lgb", "alert5_lgb"), ("Logistic Regression", "p_lr", "alert5_lr")]:
        tot_al = int(df_wide[al_col].sum())
        for gt, gv, mask in subgroup_specs:
            d = df_wide[mask]
            ys = d["outcome_r1_6h"].astype(int).to_numpy()
            ps = d[sc_col].to_numpy()
            als = d[al_col].to_numpy().astype(bool)
            evs = int(ys.sum())
            al_n = int(als.sum())
            det_n = int(ys[als].sum())
            sb = sub_boot[(m, gt, gv)]
            row = {
                "model": m, "subgroup_type": gt, "subgroup": gv,
                "n_assessments": int(len(d)), "events": evs,
                "prevalence_pct": float(100.0 * evs / max(len(d), 1)),
                "alerts_at_global_5pct": al_n,
                "share_of_all_alerts_pct": float(100.0 * al_n / max(tot_al, 1)),
                "alert_rate_within_subgroup_pct": float(100.0 * al_n / max(len(d), 1)),
                "events_detected": det_n,
                "auroc": float(roc_auc_score(ys, ps)) if (evs > 0 and evs < len(ys)) else np.nan,
                "auroc_lo": ci_bounds(sb["auroc"])[0], "auroc_hi": ci_bounds(sb["auroc"])[1],
                "auprc": float(average_precision_score(ys, ps)) if evs > 0 else np.nan,
                "auprc_lo": ci_bounds(sb["auprc"])[0], "auprc_hi": ci_bounds(sb["auprc"])[1],
                "eo_ratio": float(ps.sum() / evs) if evs > 0 else np.nan,
                "eo_lo": ci_bounds(sb["eo"])[0], "eo_hi": ci_bounds(sb["eo"])[1],
                "capture_pct": float(100.0 * det_n / evs) if evs > 0 else np.nan,
                "capture_lo": ci_bounds(sb["cap"])[0], "capture_hi": ci_bounds(sb["cap"])[1],
                "ppv_pct": float(100.0 * det_n / al_n) if al_n > 0 else np.nan,
                "ppv_lo": ci_bounds(sb["ppv"])[0], "ppv_hi": ci_bounds(sb["ppv"])[1],
            }
            sub_rows.append(row)
            if gt in {"Landmark", "ICU type"}:
                alloc_rows.append(row)

    # Competing events & DNI/CMO code-status sensitivities
    code_lm = pd.read_parquet(PROC_V25 / "v25_code_status_landmarks.parquet")
    cs_col = [c for c in code_lm.columns if "limitation" in c.lower() or "dni" in c.lower() or "comfort" in c.lower()]
    df_cs = df_wide.merge(code_lm, on=["case_id", "landmark_hours"], how="left")
    if "intubation_limitation_active" in df_cs.columns:
        dni_mask = df_cs["intubation_limitation_active"].fillna(0).astype(int) == 1
    elif cs_col:
        dni_mask = df_cs[cs_col[0]].fillna(0).astype(int) == 1
    else:
        dni_mask = pd.Series(False, index=df_cs.index)

    comp_rows = []
    sens_defs = [
        ("Primary (all eligible)", pd.Series(True, index=df_cs.index), "outcome_r1_6h"),
        ("Exclude 6h competing ICU death without IMV", df_cs["competing_death_6h"] == 0, "outcome_r1_6h"),
        ("Exclude 6h competing ICU discharge without IMV", df_cs["competing_discharge_6h"] == 0, "outcome_r1_6h"),
        ("Exclude both 6h competing death & discharge", (df_cs["competing_death_6h"] == 0) & (df_cs["competing_discharge_6h"] == 0), "outcome_r1_6h"),
        ("Composite outcome: IMV or 6h ICU death", pd.Series(True, index=df_cs.index), "_comp_death_or_imv"),
        ("Exclude documented DNI / comfort-measures-only", ~dni_mask, "outcome_r1_6h"),
    ]
    df_cs["_comp_death_or_imv"] = ((df_cs["outcome_r1_6h"] == 1) | (df_cs["competing_death_6h"] == 1)).astype(int)

    for sname, smask, ycol in sens_defs:
        d = df_cs[smask].copy()
        ks = int(np.ceil(len(d) * 0.05))
        for mname, col in [("LightGBM", "p_lgb"), ("Logistic Regression", "p_lr")]:
            ys = d[ycol].astype(int).to_numpy()
            ps = d[col].to_numpy()
            top = d.sort_values([col, "case_id", "landmark_hours"], ascending=[False, True, True], kind="mergesort").head(ks)
            comp_rows.append({
                "sensitivity": sname,
                "model": mname,
                "n_assessments": int(len(d)),
                "excluded_n": int(len(df_cs) - len(d)),
                "events": int(ys.sum()),
                "prevalence_pct": float(100.0 * ys.mean()),
                "auroc": float(roc_auc_score(ys, ps)),
                "auprc": float(average_precision_score(ys, ps)),
                "elig_5pct_capture_pct": float(100.0 * top[ycol].sum() / max(ys.sum(), 1)),
                "elig_5pct_ppv_pct": float(100.0 * top[ycol].mean()),
            })

    return pd.DataFrame(sub_rows), pd.DataFrame(alloc_rows), pd.DataFrame(comp_rows)


def run_dca(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Decision curve analysis across thresholds 0.5% to 15%."""
    y = df_wide["outcome_r1_6h"].astype(int).to_numpy()
    n = len(y)
    prev = float(y.mean())
    thresholds = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15]
    rows = []
    for thr in thresholds:
        odds = thr / (1.0 - thr)
        nb_all = prev - (1.0 - prev) * odds
        for mname, col in [
            ("LightGBM", "p_lgb"),
            ("Logistic Regression", "p_lr"),
            ("Modified NEWS2 (no O2)", "p_news2"),
        ]:
            p = df_wide[col].to_numpy()
            al = p >= thr
            tp = int(np.sum(al & (y == 1)))
            fp = int(np.sum(al & (y == 0)))
            nb = tp / n - (fp / n) * odds
            rows.append({
                "threshold": thr, "model": mname,
                "net_benefit_model": float(nb),
                "net_benefit_all": float(nb_all),
                "net_benefit_none": 0.0,
                "net_benefit_gain_over_best_default": float(nb - max(0.0, nb_all)),
                "alert_rate_pct": float(100.0 * al.mean()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading R1 temporal test predictions & ablations...")

    states = pd.read_parquet(PROC_R1 / "r1_landmark_states.parquet")
    total_scheduled_test_n = int(
        len(states[(states["is_unplanned_non_cvicu"] == 1) & (states["anchor_year_group"].isin({"2017 - 2019", "2020 - 2022"}))])
    )

    prim_preds = pd.read_parquet(MOD_DIR / "r1_primary_temporal_predictions.parquet")
    te = prim_preds[prim_preds["split"] == "test"].copy()
    va = prim_preds[prim_preds["split"] == "validation"].copy()

    # Build wide test table keyed on (case_id, landmark_hours)
    lgb_te = te[te["model"] == "LightGBM"].copy().reset_index(drop=True)
    lr_te = te[te["model"] == "Logistic Regression"][["case_id", "landmark_hours", "y_pred_pooled_cal", "y_pred_lm_cal"]].rename(
        columns={"y_pred_pooled_cal": "p_lr_pooled", "y_pred_lm_cal": "p_lr"}
    )
    wide = lgb_te.rename(columns={"y_pred_pooled_cal": "p_lgb_pooled", "y_pred_lm_cal": "p_lgb"}).merge(
        lr_te, on=["case_id", "landmark_hours"], how="inner", validate="one_to_one"
    )

    # Fit validation logistic mapping for NEWS2 so DCA can use calibrated probabilities
    from sklearn.linear_model import LogisticRegression
    va_lgb = va[va["model"] == "LightGBM"]
    for bcol, pcol in [
        ("baseline_mnews2", "p_news2"),
        ("baseline_mnews2_o2", "p_news2_o2"),
        ("baseline_neg_spo2_min", "p_spo2"),
        ("baseline_rr_max", "p_rr"),
    ]:
        cal_b = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        cal_b.fit(va_lgb[[bcol]].to_numpy(), va_lgb["outcome_r1_6h"].to_numpy())
        wide[pcol] = cal_b.predict_proba(wide[[bcol]].to_numpy())[:, 1]

    # Merge ablation model predictions
    ab_map = {
        "r1_sensitivity_plus_o2_support_temporal_predictions.parquet": ("p_lgb_o2", "p_lr_o2"),
        "r1_ablation_physiology_only_no_charting_temporal_predictions.parquet": ("p_lgb_phys", "p_lr_phys"),
        "r1_ablation_charting_pattern_only_temporal_predictions.parquet": ("p_lgb_chart", "p_lr_chart"),
    }
    for fname, (c_lgb, c_lr) in ab_map.items():
        fpath = MOD_DIR / fname
        if fpath.exists():
            ab = pd.read_parquet(fpath)
            ab_te = ab[ab["split"] == "test"]
            wide = wide.merge(
                ab_te[ab_te["model"] == "LightGBM"][["case_id", "landmark_hours", "y_pred_lm_cal"]].rename(columns={"y_pred_lm_cal": c_lgb}),
                on=["case_id", "landmark_hours"], how="left", validate="one_to_one"
            ).merge(
                ab_te[ab_te["model"] == "Logistic Regression"][["case_id", "landmark_hours", "y_pred_lm_cal"]].rename(columns={"y_pred_lm_cal": c_lr}),
                on=["case_id", "landmark_hours"], how="left", validate="one_to_one"
            )

    model_cols = {
        "LightGBM": "p_lgb",
        "Logistic Regression": "p_lr",
        "Modified NEWS2 (no O2)": "p_news2",
        "Modified NEWS2 (with O2)": "p_news2_o2",
        "SpO2 min alone": "p_spo2",
        "RR max alone": "p_rr",
    }
    if "p_lgb_o2" in wide.columns:
        model_cols["LightGBM (+O2 support)"] = "p_lgb_o2"
        model_cols["Logistic Regression (+O2 support)"] = "p_lr_o2"
    if "p_lgb_phys" in wide.columns:
        model_cols["LightGBM (physiology-only)"] = "p_lgb_phys"
        model_cols["Logistic Regression (physiology-only)"] = "p_lr_phys"
    if "p_lgb_chart" in wide.columns:
        model_cols["LightGBM (charting-only)"] = "p_lgb_chart"
        model_cols["Logistic Regression (charting-only)"] = "p_lr_chart"

    print(f"Temporal test eligible assessments: {len(wide):,}; all scheduled unplanned non-CVICU test assessments: {total_scheduled_test_n:,}; events: {int(wide['outcome_r1_6h'].sum())}")

    print("[1/5] Bootstrap discrimination, paired comparisons, and dual-denominator quota curves...")
    disc_df, contrast_df, curve_df = bootstrap_quota_and_discrimination(
        wide, model_cols, "outcome_r1_6h", total_scheduled_test_n, n_boot=N_BOOT
    )

    print("[2/5] Closest-alert cooldown re-analysis...")
    cd_df = run_cooldown_reanalysis(wide, total_scheduled_test_n)

    print("[3/5] 2x2 factorial target x eligibility...")
    fact_df = pd.read_parquet(MOD_DIR / "r1_factorial_all_scheduled_test.parquet")
    fact_res = run_factorial_2x2(fact_df)

    print("[4/5] Subgroups (with 1,000 bootstrap CIs), alert allocation, and competing-risk/DNI sensitivities...")
    sub_df, alloc_df, comp_df = run_subgroups_and_competing(wide)

    print("[5/5] Decision curve analysis...")
    dca_df = run_dca(wide)

    # Save all aggregate tables
    disc_df.to_csv(OUT_DIR / "r1_model_performance_with_ci.csv", index=False)
    contrast_df.to_csv(OUT_DIR / "r1_paired_model_contrasts.csv", index=False)
    curve_df.to_csv(OUT_DIR / "r1_capacity_response_curves_dual_denom.csv", index=False)
    cd_df.to_csv(OUT_DIR / "r1_cooldown_closest_alert.csv", index=False)
    fact_res.to_csv(OUT_DIR / "r1_factorial_2x2_clean.csv", index=False)
    sub_df.to_csv(OUT_DIR / "r1_subgroups_with_ci.csv", index=False)
    alloc_df.to_csv(OUT_DIR / "r1_alert_allocation_landmark_icu.csv", index=False)
    comp_df.to_csv(OUT_DIR / "r1_competing_and_code_status.csv", index=False)
    dca_df.to_csv(OUT_DIR / "r1_decision_curve.csv", index=False)

    print("\n=== MODEL PERFORMANCE WITH 95% BOOTSTRAP CI ===")
    print(disc_df[["model", "auroc", "auroc_lo", "auroc_hi", "auprc", "auprc_lo", "auprc_hi", "elig_5pct_cap", "elig_5pct_ppv", "elig_5pct_reach", "elig_5pct_lead_ge_2h_any"]].to_string(index=False))

    print("\n=== KEY PAIRED CONTRASTS ===")
    print(contrast_df[contrast_df["metric"].isin(["auroc", "auprc", "elig_5pct_cap"])].to_string(index=False))

    print("\n=== COOLDOWN RE-ANALYSIS (ELIGIBLE 5% QUOTA) ===")
    print(cd_df[cd_df["budget_basis"] == "eligible_5pct"][["model", "cooldown_hours", "alert_reduction_pct", "event_window_capture_pct", "capture_loss_pp", "any_prior_reach_pct", "timely_closest_alert_le_6h_pct", "closest_alert_lead_median_h"]].to_string(index=False))

    print("\n=== SUBGROUP SEX & ICU TYPE (LIGHTGBM) ===")
    print(sub_df[(sub_df["model"] == "LightGBM") & (sub_df["subgroup_type"].isin(["Sex", "ICU type", "Landmark"]))][["subgroup_type", "subgroup", "n_assessments", "events", "share_of_all_alerts_pct", "auroc", "capture_pct", "capture_lo", "capture_hi", "ppv_pct"]].to_string(index=False))

    print("\n=== COMPETING EVENTS & DNI SENSITIVITY (LIGHTGBM) ===")
    print(comp_df[comp_df["model"] == "LightGBM"].to_string(index=False))

    print(f"\nALL OPERATIONAL ANALYSES DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

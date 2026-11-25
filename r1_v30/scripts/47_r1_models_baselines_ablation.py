#!/usr/bin/env python3
"""
Study 5 BMJHCI R1 (v30): Model Development, Temporal & Random Validation,
Clinical Baselines (mNEWS2, SpO2, RR), Feature/Charting-Pattern Ablations,
Static-vs-Dynamic Ablation, Landmark-Specific Calibration, and Outcome/Gap Sensitivities.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed" / "v30_bmjhci_r1"
OUT_DIR = BASE_DIR / "results" / "high_tier_upgrade" / "v30_bmjhci_r1" / "models"

SEED = 42
LANDMARKS = [6.0, 12.0, 18.0, 24.0]
TRAIN_ERAS = {"2008 - 2010", "2011 - 2013"}
VAL_ERAS = {"2014 - 2016"}
TEST_ERAS = {"2017 - 2019", "2020 - 2022"}

ORIG_12_VARS = [
    "heart_rate", "map", "sbp", "spo2", "temperature",
    "lactate", "creatinine", "bilirubin", "platelets", "wbc", "pao2", "urine_output",
]
NEW_4_VARS = ["resp_rate", "gcs_total", "paco2", "ph"]
PRIMARY_16_VARS = ORIG_12_VARS + NEW_4_VARS
ALL_SUMMARIES = ["last", "mean", "min", "max", "slope", "missing_rate", "time_since_last", "delta6h"]
PHYS_SUMMARIES = ["last", "mean", "min", "max", "slope", "delta6h"]
CHART_SUMMARIES = ["missing_rate", "time_since_last"]
O2_COLS = ["o2_support_max", "o2_support_last", "o2_any_flag", "hfnc_flag", "niv_flag"]
STATIC_COLS = ["age", "sex", "landmark_hours"]


def feature_list(vars_list: list[str], summaries: list[str], extra: list[str] | None = None) -> list[str]:
    cols = [f"{v}_{s}" for v in vars_list for s in summaries] + list(STATIC_COLS)
    if extra:
        cols.extend(extra)
    return cols


def clip_p(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)


def fit_recalibrator(y: np.ndarray, p: np.ndarray) -> LogisticRegression:
    pc = clip_p(p)
    logit = np.log(pc / (1.0 - pc)).reshape(-1, 1)
    cal = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    cal.fit(logit, np.asarray(y, dtype=int))
    return cal


def apply_recalibrator(cal: LogisticRegression, p: np.ndarray) -> np.ndarray:
    pc = clip_p(p)
    logit = np.log(pc / (1.0 - pc)).reshape(-1, 1)
    return cal.predict_proba(logit)[:, 1]


def compute_cal_stats(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    pc = clip_p(p)
    if y.sum() == 0 or y.sum() == len(y):
        return {"brier": np.nan, "eo_ratio": np.nan, "cal_intercept": np.nan, "cal_slope": np.nan, "ece_10": np.nan}
    cal = fit_recalibrator(y, pc)
    q = pd.qcut(pd.Series(pc), q=min(10, len(np.unique(pc))), duplicates="drop")
    g = pd.DataFrame({"y": y, "p": pc, "q": q}).groupby("q", observed=True).agg(o=("y", "mean"), e=("p", "mean"))
    return {
        "brier": float(brier_score_loss(y, pc)),
        "eo_ratio": float(pc.sum() / max(float(y.sum()), 1.0)),
        "cal_intercept": float(cal.intercept_[0]),
        "cal_slope": float(cal.coef_[0, 0]),
        "ece_10": float(np.mean(np.abs(g["o"] - g["e"]))),
    }


def compute_modified_news2(df: pd.DataFrame, include_o2: bool = False) -> np.ndarray:
    """Royal College of Physicians NEWS2 built strictly from non-ventilator bedside variables."""
    score = np.zeros(len(df), dtype=float)
    # Respiratory rate
    rr = pd.to_numeric(df["resp_rate_last"], errors="coerce").to_numpy()
    score += np.where(np.isnan(rr), 0, np.where((rr <= 8) | (rr >= 25), 3, np.where((rr >= 21) & (rr <= 24), 2, np.where((rr >= 9) & (rr <= 11), 1, 0))))
    # SpO2 (Scale 1)
    spo2 = pd.to_numeric(df["spo2_last"], errors="coerce").to_numpy()
    score += np.where(np.isnan(spo2), 0, np.where(spo2 <= 91, 3, np.where(spo2 <= 93, 2, np.where(spo2 <= 95, 1, 0))))
    # Systolic BP
    sbp = pd.to_numeric(df["sbp_last"], errors="coerce").to_numpy()
    score += np.where(np.isnan(sbp), 0, np.where((sbp <= 90) | (sbp >= 220), 3, np.where(sbp <= 100, 2, np.where(sbp <= 110, 1, 0))))
    # Heart rate
    hr = pd.to_numeric(df["heart_rate_last"], errors="coerce").to_numpy()
    score += np.where(np.isnan(hr), 0, np.where((hr <= 40) | (hr >= 131), 3, np.where((hr >= 111) & (hr <= 130), 2, np.where(((hr >= 41) & (hr <= 50)) | ((hr >= 91) & (hr <= 110)), 1, 0))))
    # Consciousness (GCS < 15 => CVPU = 3 pts)
    gcs = pd.to_numeric(df["gcs_total_last"], errors="coerce").to_numpy()
    score += np.where(np.isnan(gcs), 0, np.where(gcs < 15, 3, 0))
    # Temperature
    temp = pd.to_numeric(df["temperature_last"], errors="coerce").to_numpy()
    score += np.where(np.isnan(temp), 0, np.where(temp <= 35.0, 3, np.where(temp >= 39.1, 2, np.where(((temp >= 35.1) & (temp <= 36.0)) | ((temp >= 38.1) & (temp <= 39.0)), 1, 0))))
    if include_o2:
        o2 = pd.to_numeric(df["o2_any_flag"], errors="coerce").fillna(0).to_numpy()
        score += np.where(o2 >= 1, 2, 0)
    return score


def tune_and_fit_lr(X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame, y_va: np.ndarray) -> tuple[Pipeline, dict[str, Any]]:
    best_c = 1.0
    best_ll = np.inf
    best_pipe = None
    grid_rows = []
    for c_val in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(penalty="l2", C=c_val, max_iter=2000, random_state=SEED)),
        ])
        pipe.fit(X_tr, y_tr)
        p_va = clip_p(pipe.predict_proba(X_va)[:, 1])
        ll = float(log_loss(y_va, p_va))
        auc = float(roc_auc_score(y_va, p_va))
        ap = float(average_precision_score(y_va, p_va))
        grid_rows.append({"C": c_val, "val_logloss": ll, "val_auroc": auc, "val_auprc": ap})
        if ll < best_ll:
            best_ll = ll
            best_c = c_val
            best_pipe = pipe
    assert best_pipe is not None
    return best_pipe, {"selected_C": best_c, "best_val_logloss": best_ll, "grid": grid_rows}


def tune_and_fit_lgbm(X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame, y_va: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    imp = SimpleImputer(strategy="median")
    X_tr_i = imp.fit_transform(X_tr)
    X_va_i = imp.transform(X_va)

    best_cfg = None
    best_ll = np.inf
    best_model = None
    grid_rows = []
    for max_depth in [3, 5, 6]:
        for min_child in [30, 60, 100]:
            for num_leaves in [15, 31]:
                if num_leaves > 2 ** max_depth:
                    continue
                clf = lgb.LGBMClassifier(
                    n_estimators=1000,
                    learning_rate=0.03,
                    max_depth=max_depth,
                    num_leaves=num_leaves,
                    min_child_samples=min_child,
                    subsample=0.8,
                    subsample_freq=1,
                    colsample_bytree=0.8,
                    reg_alpha=0.5,
                    reg_lambda=1.0,
                    objective="binary",
                    random_state=SEED,
                    n_jobs=-1,
                    verbosity=-1,
                )
                clf.fit(
                    X_tr_i, y_tr,
                    eval_set=[(X_va_i, y_va)],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
                p_va = clip_p(clf.predict_proba(X_va_i)[:, 1])
                ll = float(log_loss(y_va, p_va))
                auc = float(roc_auc_score(y_va, p_va))
                ap = float(average_precision_score(y_va, p_va))
                row = {
                    "max_depth": max_depth,
                    "num_leaves": num_leaves,
                    "min_child_samples": min_child,
                    "best_iteration": int(clf.best_iteration_),
                    "val_logloss": ll,
                    "val_auroc": auc,
                    "val_auprc": ap,
                }
                grid_rows.append(row)
                if ll < best_ll:
                    best_ll = ll
                    best_cfg = row
                    best_model = clf
    assert best_model is not None and best_cfg is not None
    return {"imputer": imp, "model": best_model}, {"selected": best_cfg, "grid": grid_rows}


def predict_model(obj: Any, X: pd.DataFrame) -> np.ndarray:
    if isinstance(obj, Pipeline):
        return obj.predict_proba(X)[:, 1]
    X_i = obj["imputer"].transform(X)
    return obj["model"].predict_proba(X_i)[:, 1]


def evaluate_predictions_with_quota(
    df: pd.DataFrame,
    y_col: str,
    p_col: str,
    event_time_col: str = "event_time_r1_hours",
    quota_elig_frac: float = 0.05,
) -> dict[str, float | int]:
    y = df[y_col].astype(int).to_numpy()
    p = df[p_col].astype(float).to_numpy()
    n = len(df)
    ev = int(y.sum())
    if ev == 0:
        return {"n": n, "events": 0, "auroc": np.nan, "auprc": np.nan}
    auc = float(roc_auc_score(y, p))
    ap = float(average_precision_score(y, p))
    cal = compute_cal_stats(y, p)

    # 5% eligible quota capture, PPV, and patient reach
    k5 = int(np.ceil(n * quota_elig_frac))
    ord_df = df.assign(_p=p, _y=y).sort_values(["_p", "case_id", "landmark_hours"], ascending=[False, True, True], kind="mergesort")
    top5 = ord_df.head(k5)
    cap5 = float(top5["_y"].sum() / ev)
    ppv5 = float(top5["_y"].mean())

    # Patient-level any-prior-alert reach at 5% eligible quota
    ev_pts = df.loc[df[y_col] == 1, ["case_id", event_time_col]].drop_duplicates("case_id")
    m = top5[["case_id", "landmark_hours"]].merge(ev_pts, on="case_id", how="inner")
    m = m[m["landmark_hours"] < m[event_time_col]]
    reach5 = float(m["case_id"].nunique() / max(len(ev_pts), 1))

    # 1% eligible quota
    k1 = int(np.ceil(n * 0.01))
    top1 = ord_df.head(k1)
    cap1 = float(top1["_y"].sum() / ev)
    ppv1 = float(top1["_y"].mean())

    return {
        "n": n,
        "events": ev,
        "prevalence_pct": float(100.0 * ev / n),
        "auroc": auc,
        "auprc": ap,
        **cal,
        "elig_1pct_alerts": k1,
        "elig_1pct_capture_pct": 100.0 * cap1,
        "elig_1pct_ppv_pct": 100.0 * ppv1,
        "elig_5pct_alerts": k5,
        "elig_5pct_capture_pct": 100.0 * cap5,
        "elig_5pct_ppv_pct": 100.0 * ppv5,
        "elig_5pct_patient_reach_pct": 100.0 * reach5,
    }


def apply_landmark_recalibration(
    va_df: pd.DataFrame,
    te_df: pd.DataFrame,
    raw_col: str,
    y_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit separate logistic recalibrators at each landmark on validation, apply to test."""
    p_va_out = np.zeros(len(va_df), dtype=float)
    p_te_out = np.zeros(len(te_df), dtype=float)
    for lm in LANDMARKS:
        m_va = va_df["landmark_hours"].eq(lm).to_numpy()
        m_te = te_df["landmark_hours"].eq(lm).to_numpy()
        cal = fit_recalibrator(va_df.loc[m_va, y_col].to_numpy(), va_df.loc[m_va, raw_col].to_numpy())
        p_va_out[m_va] = apply_recalibrator(cal, va_df.loc[m_va, raw_col].to_numpy())
        p_te_out[m_te] = apply_recalibrator(cal, te_df.loc[m_te, raw_col].to_numpy())
    return p_va_out, p_te_out


def build_random_split(case_ids: pd.Series, seed: int = SEED) -> pd.Series:
    ids = np.sort(case_ids.astype(str).unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_te = int(n * 0.15)
    n_va = int(n * 0.15)
    mapping = {}
    for i, cid in enumerate(ids):
        if i < n_te:
            mapping[cid] = "test"
        elif i < n_te + n_va:
            mapping[cid] = "validation"
        else:
            mapping[cid] = "train"
    return case_ids.astype(str).map(mapping)


def run_suite_on_dataframe(
    df_all: pd.DataFrame,
    split_col: str,
    feat_cols: list[str],
    y_col: str,
    suite_name: str,
    save_preds: bool = False,
) -> tuple[list[dict[str, Any]], pd.DataFrame | None, dict[str, Any]]:
    tr = df_all[df_all[split_col] == "train"].copy()
    va = df_all[df_all[split_col] == "validation"].copy()
    te = df_all[df_all[split_col] == "test"].copy()

    lr_obj, lr_tune = tune_and_fit_lr(tr[feat_cols], tr[y_col].to_numpy(), va[feat_cols], va[y_col].to_numpy())
    lgb_obj, lgb_tune = tune_and_fit_lgbm(tr[feat_cols], tr[y_col].to_numpy(), va[feat_cols], va[y_col].to_numpy())

    rows = []
    pred_frames = []
    for mname, obj in [("Logistic Regression", lr_obj), ("LightGBM", lgb_obj)]:
        p_tr_raw = predict_model(obj, tr[feat_cols])
        p_va_raw = predict_model(obj, va[feat_cols])
        p_te_raw = predict_model(obj, te[feat_cols])

        # Global recalibration on validation
        cal_glob = fit_recalibrator(va[y_col].to_numpy(), p_va_raw)
        p_va_glob = apply_recalibrator(cal_glob, p_va_raw)
        p_te_glob = apply_recalibrator(cal_glob, p_te_raw)

        # Landmark-specific recalibration on validation
        va_tmp = va.assign(_raw=p_va_raw)
        te_tmp = te.assign(_raw=p_te_raw)
        p_va_lm, p_te_lm = apply_landmark_recalibration(va_tmp, te_tmp, "_raw", y_col)

        te_eval = te.assign(p_raw=p_te_raw, p_glob=p_te_glob, p_lm=p_te_lm)
        m_glob = evaluate_predictions_with_quota(te_eval, y_col, "p_glob")
        m_lm = evaluate_predictions_with_quota(te_eval, y_col, "p_lm")
        rows.append({
            "suite": suite_name,
            "model": mname,
            "calibration": "pooled_validation",
            "n_features": len(feat_cols),
            **m_glob,
        })
        rows.append({
            "suite": suite_name,
            "model": mname,
            "calibration": "landmark_specific_validation",
            "n_features": len(feat_cols),
            **m_lm,
        })

        if save_preds:
            for sp_name, part, pr, pg, plm in [
                ("train", tr, p_tr_raw, apply_recalibrator(cal_glob, p_tr_raw), p_tr_raw),
                ("validation", va, p_va_raw, p_va_glob, p_va_lm),
                ("test", te, p_te_raw, p_te_glob, p_te_lm),
            ]:
                pf = part[[
                    "case_id", "subject_id", "landmark_hours", "anchor_year_group",
                    "first_careunit", "admission_type", "curr_service", "age", "sex",
                    "r1_eligible", "is_unplanned_non_cvicu", "is_unplanned_non_surgical",
                    "outcome_r1_6h", "outcome_r1_12h", "outcome_r1_24h",
                    "outcome_r1_gap1h", "outcome_r1_gap2h",
                    "outcome_proc_confirmed_6h", "outcome_intub_confirmed_6h",
                    "history_inclusive_outcome_r1",
                    "event_time_r1_hours", "lead_time_r1_hours",
                    "competing_death_6h", "competing_discharge_6h",
                ]].copy()
                pf["split"] = sp_name
                pf["model"] = mname
                pf["y_pred_raw"] = pr
                pf["y_pred_pooled_cal"] = pg
                pf["y_pred_lm_cal"] = plm
                pred_frames.append(pf)

    preds_out = pd.concat(pred_frames, ignore_index=True) if pred_frames else None
    tune_info = {"lr": lr_tune, "lgbm": lgb_tune}
    return rows, preds_out, tune_info


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading R1 states & rolling features...")
    states = pd.read_parquet(PROC_DIR / "r1_landmark_states.parquet")
    feats = pd.read_parquet(PROC_DIR / "r1_rolling_features.parquet")

    # Merge features onto states (drop duplicate static columns from feats)
    drop_dup = [c for c in ["subject_id", "age", "sex", "first_careunit", "admission_type", "anchor_year_group"] if c in feats.columns]
    df = states.merge(feats.drop(columns=drop_dup), on=["case_id", "landmark_hours"], how="inner", validate="one_to_one")
    assert len(df) == len(states), "State-feature merge mismatch!"

    # Assign temporal split and random patient-level split
    df["temporal_split"] = np.where(
        df["anchor_year_group"].isin(TRAIN_ERAS), "train",
        np.where(df["anchor_year_group"].isin(VAL_ERAS), "validation", "test")
    )
    df["random_split"] = build_random_split(df["case_id"], seed=SEED)

    # Compute clinical baseline scores on full dataframe
    df["baseline_mnews2"] = compute_modified_news2(df, include_o2=False)
    df["baseline_mnews2_o2"] = compute_modified_news2(df, include_o2=True)
    df["baseline_neg_spo2_min"] = -pd.to_numeric(df["spo2_min"], errors="coerce").fillna(98.0)
    df["baseline_rr_max"] = pd.to_numeric(df["resp_rate_max"], errors="coerce").fillna(18.0)

    # Feature sets
    f_primary_16 = feature_list(PRIMARY_16_VARS, ALL_SUMMARIES)
    f_orig_12 = feature_list(ORIG_12_VARS, ALL_SUMMARIES)
    f_plus_o2 = feature_list(PRIMARY_16_VARS, ALL_SUMMARIES, extra=O2_COLS)
    f_phys_only = feature_list(PRIMARY_16_VARS, PHYS_SUMMARIES)
    f_chart_only = feature_list(PRIMARY_16_VARS, CHART_SUMMARIES)

    # Primary cohort: r1_eligible == 1 & is_unplanned_non_cvicu == 1
    prim = df[(df["r1_eligible"] == 1) & (df["is_unplanned_non_cvicu"] == 1)].copy()
    print(f"Primary cohort (unplanned non-CVICU, r1_eligible): {len(prim):,} rows, {int(prim['outcome_r1_6h'].sum())} events")
    print(prim.groupby("temporal_split")["outcome_r1_6h"].agg(["count", "sum", "mean"]))

    all_rows: list[dict[str, Any]] = []
    tuning_logs: dict[str, Any] = {}

    # 1. PRIMARY ANALYSIS: Unplanned Non-CVICU, Temporal Validation, Primary 16-var set
    print("\n[1/8] Primary cohort (Temporal split, 16-var physiological+lab)...")
    r_prim, preds_prim, tune_prim = run_suite_on_dataframe(
        prim, "temporal_split", f_primary_16, "outcome_r1_6h", "primary_unplanned_non_cvicu_temporal_16var", save_preds=True
    )
    all_rows.extend(r_prim)
    tuning_logs["primary_temporal_16var"] = tune_prim
    assert preds_prim is not None
    # Attach baseline scores onto test/all predictions for downstream quota curves
    preds_prim = preds_prim.merge(
        prim[["case_id", "landmark_hours", "baseline_mnews2", "baseline_mnews2_o2", "baseline_neg_spo2_min", "baseline_rr_max"]],
        on=["case_id", "landmark_hours"], how="left", validate="many_to_one",
    )
    preds_prim.to_parquet(OUT_DIR / "r1_primary_temporal_predictions.parquet", index=False)

    # Evaluate clinical baselines on primary temporal test set
    prim_te = prim[prim["temporal_split"] == "test"].copy()
    prim_va = prim[prim["temporal_split"] == "validation"].copy()
    baseline_rows = []
    for bname, bcol in [
        ("Modified NEWS2 (no O2)", "baseline_mnews2"),
        ("Modified NEWS2 (with O2)", "baseline_mnews2_o2"),
        ("SpO2 min alone", "baseline_neg_spo2_min"),
        ("Respiratory rate max alone", "baseline_rr_max"),
    ]:
        # Fit monotonic logistic mapping on validation for calibration comparability (preserves exact rank)
        cal_b = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        cal_b.fit(prim_va[[bcol]].to_numpy(), prim_va["outcome_r1_6h"].to_numpy())
        p_te_b = cal_b.predict_proba(prim_te[[bcol]].to_numpy())[:, 1]
        te_b = prim_te.assign(p_b=p_te_b)
        mb = evaluate_predictions_with_quota(te_b, "outcome_r1_6h", "p_b")
        baseline_rows.append({"suite": "clinical_baseline_temporal", "model": bname, "calibration": "validation_logistic", "n_features": 1, **mb})
    all_rows.extend(baseline_rows)

    # 2. FEATURE ABLATIONS (on primary temporal split): 12-var, +O2, Phys-only, Charting-only
    print("[2/8] Feature & charting-pattern ablations (Temporal split)...")
    for sname, fcols in [
        ("ablation_orig_12var_temporal", f_orig_12),
        ("sensitivity_plus_o2_support_temporal", f_plus_o2),
        ("ablation_physiology_only_no_charting_temporal", f_phys_only),
        ("ablation_charting_pattern_only_temporal", f_chart_only),
    ]:
        r_ab, preds_ab, _ = run_suite_on_dataframe(prim, "temporal_split", fcols, "outcome_r1_6h", sname, save_preds=(sname in {"ablation_physiology_only_no_charting_temporal", "ablation_charting_pattern_only_temporal", "sensitivity_plus_o2_support_temporal"}))
        all_rows.extend(r_ab)
        if preds_ab is not None:
            preds_ab.to_parquet(OUT_DIR / f"r1_{sname}_predictions.parquet", index=False)

    # 3. RANDOM SPLIT SENSITIVITY (Primary cohort, 16-var)
    print("[3/8] Random patient-grouped split sensitivity (Primary cohort, 16-var)...")
    r_rand, preds_rand, tune_rand = run_suite_on_dataframe(
        prim, "random_split", f_primary_16, "outcome_r1_6h", "sensitivity_primary_random_split_16var", save_preds=True
    )
    all_rows.extend(r_rand)
    tuning_logs["primary_random_16var"] = tune_rand
    assert preds_rand is not None
    preds_rand.to_parquet(OUT_DIR / "r1_primary_random_predictions.parquet", index=False)

    # 4. RETRAINED SENSITIVITY COHORTS (All monitored ICU, Strict non-surgical unplanned)
    print("[4/8] Retrained sensitivity cohorts (All monitored ICU & Strict non-surgical)...")
    all_mon = df[df["r1_eligible"] == 1].copy()
    strict_med = df[(df["r1_eligible"] == 1) & (df["is_unplanned_non_surgical"] == 1)].copy()
    r_allmon, preds_allmon, _ = run_suite_on_dataframe(
        all_mon, "temporal_split", f_primary_16, "outcome_r1_6h", "sensitivity_all_monitored_icu_temporal_16var", save_preds=True
    )
    r_med, _, _ = run_suite_on_dataframe(
        strict_med, "temporal_split", f_primary_16, "outcome_r1_6h", "sensitivity_strict_non_surgical_temporal_16var", save_preds=False
    )
    all_rows.extend(r_allmon)
    all_rows.extend(r_med)
    assert preds_allmon is not None
    preds_allmon.to_parquet(OUT_DIR / "r1_all_monitored_temporal_predictions.parquet", index=False)

    # 5. STATIC-6H vs DYNAMIC-UPDATING ABLATION at landmarks 12, 18, 24h (Reviewer 1 Discussion!)
    print("[5/8] Static-6h vs Dynamic-updating ablation at landmarks 12, 18, 24h...")
    # Train on 6h landmark features only (static baseline risk at 6h), apply frozen 6h score to landmarks 12, 18, 24h
    prim_6h = prim[prim["landmark_hours"] == 6.0].copy()
    f_no_lm = [c for c in f_primary_16 if c != "landmark_hours"]
    tr6 = prim_6h[prim_6h["temporal_split"] == "train"]
    va6 = prim_6h[prim_6h["temporal_split"] == "validation"]
    lr6, _ = tune_and_fit_lr(tr6[f_no_lm], tr6["outcome_r1_6h"].to_numpy(), va6[f_no_lm], va6["outcome_r1_6h"].to_numpy())
    lgb6, _ = tune_and_fit_lgbm(tr6[f_no_lm], tr6["outcome_r1_6h"].to_numpy(), va6[f_no_lm], va6["outcome_r1_6h"].to_numpy())

    # Score every patient's 6h feature vector once and broadcast to landmarks 12, 18, 24h
    static_scores = prim_6h[["case_id"]].copy()
    static_scores["static_lr"] = predict_model(lr6, prim_6h[f_no_lm])
    static_scores["static_lgb"] = predict_model(lgb6, prim_6h[f_no_lm])

    later_te = preds_prim[(preds_prim["split"] == "test") & (preds_prim["landmark_hours"].isin([12.0, 18.0, 24.0]))].merge(
        static_scores, on="case_id", how="inner"
    )
    static_dyn_rows = []
    for mname, scol in [("Logistic Regression", "static_lr"), ("LightGBM", "static_lgb")]:
        sub_m = later_te[later_te["model"] == mname]
        for lm_label, sub in [("pooled_12_18_24h", sub_m), *[(f"landmark_{int(l)}h", sub_m[sub_m["landmark_hours"] == l]) for l in [12.0, 18.0, 24.0]]]:
            dyn_auc = float(roc_auc_score(sub["outcome_r1_6h"], sub["y_pred_lm_cal"]))
            dyn_ap = float(average_precision_score(sub["outcome_r1_6h"], sub["y_pred_lm_cal"]))
            sta_auc = float(roc_auc_score(sub["outcome_r1_6h"], sub[scol]))
            sta_ap = float(average_precision_score(sub["outcome_r1_6h"], sub[scol]))
            # 5% eligible capture
            k5 = int(np.ceil(len(sub) * 0.05))
            ev_n = int(sub["outcome_r1_6h"].sum())
            dyn_cap = float(sub.sort_values("y_pred_lm_cal", ascending=False).head(k5)["outcome_r1_6h"].sum() / ev_n)
            sta_cap = float(sub.sort_values(scol, ascending=False).head(k5)["outcome_r1_6h"].sum() / ev_n)
            static_dyn_rows.append({
                "model": mname,
                "window": lm_label,
                "n": len(sub),
                "events": ev_n,
                "dynamic_auroc": dyn_auc,
                "static6h_auroc": sta_auc,
                "delta_auroc": dyn_auc - sta_auc,
                "dynamic_auprc": dyn_ap,
                "static6h_auprc": sta_ap,
                "delta_auprc": dyn_ap - sta_ap,
                "dynamic_5pct_elig_cap_pct": 100.0 * dyn_cap,
                "static6h_5pct_elig_cap_pct": 100.0 * sta_cap,
                "delta_cap_pp": 100.0 * (dyn_cap - sta_cap),
            })
    pd.DataFrame(static_dyn_rows).to_csv(OUT_DIR / "r1_static_vs_dynamic_ablation.csv", index=False)

    # 6. ALTERNATIVE OUTCOMES, PREDICTION GAPS (1h, 2h), AND HORIZONS (12h, 24h)
    print("[6/8] Evaluating prediction gaps (1h, 2h), alternative outcomes, and 12h/24h horizons...")
    te_p = preds_prim[preds_prim["split"] == "test"].copy()
    outcome_sens_rows = []
    for mname in ["Logistic Regression", "LightGBM"]:
        sub = te_p[te_p["model"] == mname]
        for oname, ocol, filt_note in [
            ("primary_harmonized_6h", "outcome_r1_6h", "all eligible"),
            ("prediction_gap_1h_(L+1,L+6]", "outcome_r1_gap1h", "exclude events in (L, L+1h]"),
            ("prediction_gap_2h_(L+2,L+6]", "outcome_r1_gap2h", "exclude events in (L, L+2h]"),
            ("procedure_confirmed_225792_6h", "outcome_proc_confirmed_6h", "all eligible"),
            ("intubation_confirmed_224385_6h", "outcome_intub_confirmed_6h", "all eligible"),
            ("horizon_12h", "outcome_r1_12h", "all eligible"),
            ("horizon_24h", "outcome_r1_24h", "all eligible"),
        ]:
            # For prediction-gap sensitivity, exclude immediate events occurring inside the gap!
            if oname.startswith("prediction_gap_1h"):
                eval_df = sub[~((sub["outcome_r1_6h"] == 1) & (sub["outcome_r1_gap1h"] == 0))].copy()
            elif oname.startswith("prediction_gap_2h"):
                eval_df = sub[~((sub["outcome_r1_6h"] == 1) & (sub["outcome_r1_gap2h"] == 0))].copy()
            else:
                eval_df = sub.copy()
            m_res = evaluate_predictions_with_quota(eval_df, ocol, "y_pred_lm_cal")
            outcome_sens_rows.append({"model": mname, "outcome_sensitivity": oname, "note": filt_note, **m_res})
    pd.DataFrame(outcome_sens_rows).to_csv(OUT_DIR / "r1_outcome_and_gap_sensitivities.csv", index=False)

    # 7. FACTORIAL 2x2 TARGET x ELIGIBILITY ON CLEAN UNPLANNED NON-CVICU COHORT (All scheduled landmarks!)
    print("[7/8] Running 2x2 factorial target x eligibility analysis on clean unplanned non-CVICU cohort...")
    # Train history-inclusive model on ALL unplanned non-CVICU scheduled landmarks (including already ventilated!)
    all_sched_prim = df[df["is_unplanned_non_cvicu"] == 1].copy()
    tr_s = all_sched_prim[all_sched_prim["temporal_split"] == "train"]
    va_s = all_sched_prim[all_sched_prim["temporal_split"] == "validation"]
    te_s = all_sched_prim[all_sched_prim["temporal_split"] == "test"].copy()

    # Incident models (re-fit on prim train/val with selected tuned params to score ALL scheduled test landmarks!)
    tr_e = prim[prim["temporal_split"] == "train"]
    va_e = prim[prim["temporal_split"] == "validation"]
    lr_inc, _ = tune_and_fit_lr(tr_e[f_primary_16], tr_e["outcome_r1_6h"].to_numpy(), va_e[f_primary_16], va_e["outcome_r1_6h"].to_numpy())
    lgb_inc, _ = tune_and_fit_lgbm(tr_e[f_primary_16], tr_e["outcome_r1_6h"].to_numpy(), va_e[f_primary_16], va_e["outcome_r1_6h"].to_numpy())

    # History-inclusive models (trained on all scheduled landmarks targeting prior_any_imv | outcome_r1_6h)
    lr_hist, _ = tune_and_fit_lr(tr_s[f_primary_16], tr_s["history_inclusive_outcome_r1"].to_numpy(), va_s[f_primary_16], va_s["history_inclusive_outcome_r1"].to_numpy())
    lgb_hist, _ = tune_and_fit_lgbm(tr_s[f_primary_16], tr_s["history_inclusive_outcome_r1"].to_numpy(), va_s[f_primary_16], va_s["history_inclusive_outcome_r1"].to_numpy())

    te_s["lr_inc_score"] = predict_model(lr_inc, te_s[f_primary_16])
    te_s["lgb_inc_score"] = predict_model(lgb_inc, te_s[f_primary_16])
    te_s["lr_hist_score"] = predict_model(lr_hist, te_s[f_primary_16])
    te_s["lgb_hist_score"] = predict_model(lgb_hist, te_s[f_primary_16])
    te_s[[
        "case_id", "subject_id", "landmark_hours", "anchor_year_group", "r1_eligible",
        "outcome_r1_6h", "history_inclusive_outcome_r1", "event_time_r1_hours", "lead_time_r1_hours",
        "lr_inc_score", "lgb_inc_score", "lr_hist_score", "lgb_hist_score",
    ]].to_parquet(OUT_DIR / "r1_factorial_all_scheduled_test.parquet", index=False)

    # 8. ERA-SPECIFIC & LANDMARK-SPECIFIC BREAKDOWN ON PRIMARY TEMPORAL TEST SET
    print("[8/8] Computing era-specific (2017-2019 vs 2020-2022) and landmark-specific (6/12/18/24h) metrics...")
    era_lm_rows = []
    for mname in ["Logistic Regression", "LightGBM"]:
        sub = te_p[te_p["model"] == mname]
        for cal_label, pcol in [("pooled_cal", "y_pred_pooled_cal"), ("landmark_cal", "y_pred_lm_cal")]:
            for era in ["2017 - 2019", "2020 - 2022", "2017 - 2022 (pooled)"]:
                d = sub if era.endswith("(pooled)") else sub[sub["anchor_year_group"] == era]
                res = evaluate_predictions_with_quota(d, "outcome_r1_6h", pcol)
                era_lm_rows.append({"model": mname, "calibration": cal_label, "stratum_type": "era", "stratum": era, **res})
            for lm in LANDMARKS:
                d = sub[sub["landmark_hours"] == lm]
                res = evaluate_predictions_with_quota(d, "outcome_r1_6h", pcol)
                era_lm_rows.append({"model": mname, "calibration": cal_label, "stratum_type": "landmark", "stratum": f"{int(lm)}h", **res})
    era_lm_df = pd.DataFrame(era_lm_rows)
    era_lm_df.to_csv(OUT_DIR / "r1_era_and_landmark_performance.csv", index=False)

    # Save main model summary table and tuning logs
    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv(OUT_DIR / "r1_model_comparison_and_ablations.csv", index=False)
    (OUT_DIR / "r1_tuning_and_spec.json").write_text(
        json.dumps({"elapsed_seconds": round(time.time() - t0, 2), "tuning": tuning_logs}, indent=2), encoding="utf-8"
    )

    print("\n=== MAIN MODEL, BASELINE, AND ABLATION SUMMARY (LANDMARK-CALIBRATED OR BASELINE) ===")
    view = summary_df[summary_df["calibration"].isin(["landmark_specific_validation", "validation_logistic"])][
        ["suite", "model", "n", "events", "auroc", "auprc", "eo_ratio", "cal_slope", "elig_1pct_capture_pct", "elig_1pct_ppv_pct", "elig_5pct_capture_pct", "elig_5pct_ppv_pct", "elig_5pct_patient_reach_pct"]
    ]
    print(view.to_string(index=False))

    print("\n=== LANDMARK-SPECIFIC PERFORMANCE (PRIMARY TEMPORAL TEST, POOLED vs LANDMARK CAL) ===")
    lm_v = era_lm_df[era_lm_df["stratum_type"] == "landmark"][
        ["model", "calibration", "stratum", "n", "events", "auroc", "auprc", "eo_ratio", "cal_slope"]
    ]
    print(lm_v.to_string(index=False))

    print("\n=== ERA-SPECIFIC PERFORMANCE (2017-2019 vs 2020-2022) ===")
    era_v = era_lm_df[(era_lm_df["stratum_type"] == "era") & (era_lm_df["calibration"] == "landmark_cal")][
        ["model", "stratum", "n", "events", "auroc", "auprc", "eo_ratio", "cal_slope", "elig_5pct_capture_pct", "elig_5pct_ppv_pct"]
    ]
    print(era_v.to_string(index=False))

    print("\n=== STATIC-6H vs DYNAMIC ABLATION ===")
    print(pd.DataFrame(static_dyn_rows).to_string(index=False))

    print("\n=== OUTCOME & PREDICTION-GAP SENSITIVITIES ===")
    print(pd.DataFrame(outcome_sens_rows)[["model", "outcome_sensitivity", "n", "events", "auroc", "auprc", "elig_5pct_capture_pct", "elig_5pct_ppv_pct"]].to_string(index=False))

    print(f"\nALL DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

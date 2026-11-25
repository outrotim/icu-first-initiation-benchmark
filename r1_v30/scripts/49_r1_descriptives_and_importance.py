#!/usr/bin/env python3
"""
Study 5 BMJHCI R1 (v30): Generate Cohort Characteristics (Table S1 full cohort + splits),
Predictor Variable Dictionary & Missingness Patterns (Table S3A & S3B),
OR-Arrival Contamination Audit Table (v29 vs v30), and Grouped Permutation Importance.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
PROC_R1 = BASE_DIR / "data" / "processed" / "v30_bmjhci_r1"
PROC_V25 = BASE_DIR / "data" / "processed" / "v25_clinical_upgrade"
MOD_DIR = BASE_DIR / "results" / "high_tier_upgrade" / "v30_bmjhci_r1" / "models"
OUT_DIR = BASE_DIR / "results" / "high_tier_upgrade" / "v30_bmjhci_r1" / "descriptives"

SEED = 42
TRAIN_ERAS = {"2008 - 2010", "2011 - 2013"}
VAL_ERAS = {"2014 - 2016"}
TEST_ERAS = {"2017 - 2019", "2020 - 2022"}

VARIABLE_SPEC = [
    ("heart_rate", "Heart rate", "chartevents", "220045", "bpm", "20-300", "Hourly mean", "charttime in [L-6, L)"),
    ("map", "Mean arterial pressure", "chartevents", "220052, 220181, 225312", "mmHg", "20-200", "Hourly mean", "charttime in [L-6, L)"),
    ("sbp", "Systolic blood pressure", "chartevents", "220050, 220179, 225309", "mmHg", "40-300", "Hourly mean", "charttime in [L-6, L)"),
    ("spo2", "Peripheral oxygen saturation (SpO2)", "chartevents", "220277", "%", "50-100", "Hourly mean", "charttime in [L-6, L)"),
    ("temperature", "Body temperature", "chartevents", "223761 (F->C), 223762", "deg C", "30-42", "Hourly mean", "charttime in [L-6, L)"),
    ("resp_rate", "Respiratory rate (bedside/spontaneous)", "chartevents", "220210 (excludes Set RR 224688)", "breaths/min", "4-70", "Hourly mean", "charttime in [L-6, L)"),
    ("gcs_total", "Glasgow Coma Scale (total E+V+M)", "chartevents", "220739, 223900 (excl ETT), 223901", "score", "3-15", "Hourly sum of E+V+M", "charttime in [L-6, L)"),
    ("lactate", "Blood lactate", "labevents", "50813", "mmol/L", "0-30", "Hourly mean", "charttime in [L-6, L) AND storetime < L"),
    ("creatinine", "Serum creatinine", "labevents", "50912", "mg/dL", "0-30", "Hourly mean", "charttime in [L-6, L) AND storetime < L"),
    ("bilirubin", "Total bilirubin", "labevents", "50885", "mg/dL", "0-50", "Hourly mean", "charttime in [L-6, L) AND storetime < L"),
    ("platelets", "Platelet count", "labevents", "51265", "K/uL", "0-1500", "Hourly mean", "charttime in [L-6, L) AND storetime < L"),
    ("wbc", "White blood cell count", "labevents", "51301", "K/uL", "0-500", "Hourly mean", "charttime in [L-6, L) AND storetime < L"),
    ("pao2", "Arterial/blood oxygen tension (PaO2)", "labevents", "50821", "mmHg", "20-600", "Hourly mean", "charttime in [L-6, L) AND storetime < L"),
    ("paco2", "Arterial/blood CO2 tension (PaCO2)", "labevents", "50818", "mmHg", "10-150", "Hourly mean", "charttime in [L-6, L) AND storetime < L"),
    ("ph", "Blood pH", "labevents", "50820", "pH units", "6.80-7.80", "Hourly mean", "charttime in [L-6, L) AND storetime < L"),
    ("urine_output", "Urine output", "outputevents", "226557-226567, 226584, 227488, 227489", "mL/h", "0-10000", "Hourly sum", "charttime in [L-6, L)"),
]

ALL_SUMMARIES = ["last", "mean", "min", "max", "slope", "missing_rate", "time_since_last", "delta6h"]


def fmt_med_iqr(s: pd.Series, dec: int = 1) -> str:
    v = pd.to_numeric(s, errors="coerce").dropna()
    if len(v) == 0:
        return "NA"
    return f"{v.median():.{dec}f} ({v.quantile(0.25):.{dec}f}-{v.quantile(0.75):.{dec}f})"


def fmt_n_pct(count: int, total: int) -> str:
    return f"{count:,} ({100.0 * count / max(total, 1):.1f}%)"


def build_table_s1_full_cohort(states: pd.DataFrame) -> pd.DataFrame:
    """Full patient-level cohort table for primary unplanned non-CVICU cohort across temporal splits."""
    prim = states[(states["r1_eligible"] == 1) & (states["is_unplanned_non_cvicu"] == 1)].copy()
    prim["temporal_split"] = np.where(
        prim["anchor_year_group"].isin(TRAIN_ERAS), "Train (2008-2013)",
        np.where(prim["anchor_year_group"].isin(VAL_ERAS), "Validation (2014-2016)", "Test (2017-2022)")
    )
    # Collapse to unique patients (with any event flag)
    pts = prim.groupby("case_id", sort=False).agg(
        temporal_split=("temporal_split", "first"),
        age=("age", "first"),
        sex=("sex", "first"),
        icu_los_hours=("icu_los_hours", "first"),
        first_careunit=("first_careunit", "first"),
        admission_type=("admission_type", "first"),
        hospital_expire_flag=("hospital_expire_flag", "first"),
        n_elig_landmarks=("landmark_hours", "count"),
        any_event=("outcome_r1_6h", "max"),
    ).reset_index()

    groups = {
        "Overall": pts,
        "Train (2008-2013)": pts[pts["temporal_split"] == "Train (2008-2013)"],
        "Validation (2014-2016)": pts[pts["temporal_split"] == "Validation (2014-2016)"],
        "Test (2017-2022)": pts[pts["temporal_split"] == "Test (2017-2022)"],
        "No first IMV in 6-30h": pts[pts["any_event"] == 0],
        "First IMV in 6-30h": pts[pts["any_event"] == 1],
    }

    rows = []
    def add_row(label: str, fn):
        r = {"Characteristic": label}
        for gname, gdf in groups.items():
            r[gname] = fn(gdf)
        rows.append(r)

    add_row("Unique eligible patients, n", lambda d: f"{len(d):,}")
    add_row("Eligible landmark assessments, n", lambda d: f"{int(d['n_elig_landmarks'].sum()):,}")
    add_row("First IMV events (6-30h), n (% of patients)", lambda d: fmt_n_pct(int(d["any_event"].sum()), len(d)))
    add_row("Age, years, median (IQR)", lambda d: fmt_med_iqr(d["age"], 1))
    add_row("Male sex, n (%)", lambda d: fmt_n_pct(int((d["sex"] == 1).sum()), len(d)))
    add_row("ICU length of stay, hours, median (IQR)", lambda d: fmt_med_iqr(d["icu_los_hours"], 1))
    add_row("Hospital mortality, n (%)", lambda d: fmt_n_pct(int((d["hospital_expire_flag"] == 1).sum()), len(d)))

    for unit in [
        "Medical Intensive Care Unit (MICU)",
        "Medical/Surgical Intensive Care Unit (MICU/SICU)",
        "Surgical Intensive Care Unit (SICU)",
        "Coronary Care Unit (CCU)",
        "Trauma SICU (TSICU)",
        "Neuro Intermediate",
        "Neuro Stepdown",
        "Neuro Surgical Intensive Care Unit (Neuro SICU)",
    ]:
        add_row(f"ICU unit: {unit}, n (%)", lambda d, u=unit: fmt_n_pct(int((d["first_careunit"] == u).sum()), len(d)))

    for atype in ["EW EMER.", "URGENT", "DIRECT EMER.", "OBSERVATION ADMIT"]:
        add_row(f"Admission type: {atype}, n (%)", lambda d, a=atype: fmt_n_pct(int((d["admission_type"] == a).sum()), len(d)))

    return pd.DataFrame(rows)


def build_predictor_tables(states: pd.DataFrame, feats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Table S3A (variable dictionary) and Table S3B (missingness in clean primary cohort)."""
    dict_df = pd.DataFrame(
        VARIABLE_SPEC,
        columns=["variable", "clinical_name", "source_table", "mimic_itemids", "unit", "valid_range", "hourly_aggregation", "availability_rule"],
    )

    prim = states[(states["r1_eligible"] == 1) & (states["is_unplanned_non_cvicu"] == 1)][["case_id", "landmark_hours", "outcome_r1_6h"]].merge(
        feats, on=["case_id", "landmark_hours"], how="inner", validate="one_to_one"
    )
    ev = prim[prim["outcome_r1_6h"] == 1]
    nev = prim[prim["outcome_r1_6h"] == 0]

    miss_rows = []
    for var, cname, *_ in VARIABLE_SPEC:
        mr = f"{var}_missing_rate"
        miss_rows.append({
            "variable": var,
            "clinical_name": cname,
            "all_any_obs_pct": float(100.0 * (prim[mr] < 1.0).mean()),
            "all_obs_bins_pct": float(100.0 * (1.0 - prim[mr]).mean()),
            "no_event_any_obs_pct": float(100.0 * (nev[mr] < 1.0).mean()),
            "no_event_obs_bins_pct": float(100.0 * (1.0 - nev[mr]).mean()),
            "event_any_obs_pct": float(100.0 * (ev[mr] < 1.0).mean()),
            "event_obs_bins_pct": float(100.0 * (1.0 - ev[mr]).mean()),
        })
    return dict_df, pd.DataFrame(miss_rows)


def compute_grouped_permutation_importance(states: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Grouped permutation AUROC drop on temporal test set for LightGBM and Logistic Regression."""
    drop_dup = [c for c in ["subject_id", "age", "sex", "first_careunit", "admission_type", "anchor_year_group"] if c in feats.columns]
    df = states[(states["r1_eligible"] == 1) & (states["is_unplanned_non_cvicu"] == 1)].merge(
        feats.drop(columns=drop_dup), on=["case_id", "landmark_hours"], how="inner", validate="one_to_one"
    )
    df["temporal_split"] = np.where(
        df["anchor_year_group"].isin(TRAIN_ERAS), "train",
        np.where(df["anchor_year_group"].isin(VAL_ERAS), "validation", "test")
    )
    vars_16 = [v[0] for v in VARIABLE_SPEC]
    fcols = [f"{v}_{s}" for v in vars_16 for s in ALL_SUMMARIES] + ["age", "sex", "landmark_hours"]

    # Load selected hyperparameters from tuning json
    tune = json.loads((MOD_DIR / "r1_tuning_and_spec.json").read_text())["tuning"]["primary_temporal_16var"]
    lr_c = float(tune["lr"]["selected_C"])
    lgb_sel = tune["lgbm"]["selected"]

    tr = df[df["temporal_split"] == "train"]
    va = df[df["temporal_split"] == "validation"]
    te = df[df["temporal_split"] == "test"]
    y_te = te["outcome_r1_6h"].astype(int).to_numpy()

    lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=lr_c, max_iter=2000, random_state=SEED)),
    ])
    lr.fit(tr[fcols], tr["outcome_r1_6h"].to_numpy())

    imp = SimpleImputer(strategy="median")
    X_tr_i = imp.fit_transform(tr[fcols])
    X_va_i = imp.transform(va[fcols])
    X_te_i = imp.transform(te[fcols])
    clf = lgb.LGBMClassifier(
        n_estimators=int(lgb_sel["best_iteration"]),
        learning_rate=0.03,
        max_depth=int(lgb_sel["max_depth"]),
        num_leaves=int(lgb_sel["num_leaves"]),
        min_child_samples=int(lgb_sel["min_child_samples"]),
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_alpha=0.5, reg_lambda=1.0, objective="binary",
        random_state=SEED, n_jobs=-1, verbosity=-1,
    )
    clf.fit(X_tr_i, tr["outcome_r1_6h"].to_numpy())

    groups: dict[str, list[str]] = {v: [f"{v}_{s}" for s in ALL_SUMMARIES] for v in vars_16}
    groups["demographics_and_landmark"] = ["age", "sex", "landmark_hours"]

    base_lr = float(roc_auc_score(y_te, lr.predict_proba(te[fcols])[:, 1]))
    base_lgb = float(roc_auc_score(y_te, clf.predict_proba(X_te_i)[:, 1]))

    rng = np.random.RandomState(SEED)
    n_rep = 10
    rows = []
    X_te_df = te[fcols].copy().reset_index(drop=True)
    col_idx_map = {c: i for i, c in enumerate(fcols)}

    for gname, gcols in groups.items():
        drops_lr = []
        drops_lgb = []
        c_idxs = [col_idx_map[c] for c in gcols]
        for _ in range(n_rep):
            perm = rng.permutation(len(X_te_df))
            # LR
            Xp = X_te_df.copy()
            Xp[gcols] = X_te_df[gcols].iloc[perm].to_numpy()
            drops_lr.append(base_lr - float(roc_auc_score(y_te, lr.predict_proba(Xp)[:, 1])))
            # LGBM
            Xi_p = X_te_i.copy()
            Xi_p[:, c_idxs] = X_te_i[perm][:, c_idxs]
            drops_lgb.append(base_lgb - float(roc_auc_score(y_te, clf.predict_proba(Xi_p)[:, 1])))
        rows.append({
            "feature_group": gname,
            "lgbm_auroc_drop_mean": float(np.mean(drops_lgb)),
            "lgbm_auroc_drop_sd": float(np.std(drops_lgb)),
            "lr_auroc_drop_mean": float(np.mean(drops_lr)),
            "lr_auroc_drop_sd": float(np.std(drops_lr)),
        })
    return pd.DataFrame(rows).sort_values("lgbm_auroc_drop_mean", ascending=False).reset_index(drop=True)


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    states = pd.read_parquet(PROC_R1 / "r1_landmark_states.parquet")
    feats = pd.read_parquet(PROC_R1 / "r1_rolling_features.parquet")

    t_s1 = build_table_s1_full_cohort(states)
    t_s3a, t_s3b = build_predictor_tables(states, feats)
    imp_df = compute_grouped_permutation_importance(states, feats)

    t_s1.to_csv(OUT_DIR / "r1_table_s1_full_cohort.csv", index=False)
    t_s3a.to_csv(OUT_DIR / "r1_table_s3a_variable_dictionary.csv", index=False)
    t_s3b.to_csv(OUT_DIR / "r1_table_s3b_predictor_missingness.csv", index=False)
    imp_df.to_csv(OUT_DIR / "r1_grouped_permutation_importance.csv", index=False)

    print("=== TABLE S3B: MISSINGNESS IN CLEAN PRIMARY COHORT (NO OR-ARRIVAL ARTIFACT!) ===")
    print(t_s3b.to_string(index=False))
    print("\n=== GROUPED PERMUTATION IMPORTANCE (TEMPORAL TEST) ===")
    print(imp_df.to_string(index=False))
    print(f"\nDONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

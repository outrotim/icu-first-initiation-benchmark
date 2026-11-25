#!/usr/bin/env python3
"""
Study 5 BMJHCI R1 (v30): Unified Cohort Audit, Leak-Free Outcome Reconstruction,
and Expanded Respiratory/Neurological Predictor Extraction.

Addresses:
- R1 #1: Validate ventilation outcome, signed timing comparisons, prediction-gap analyses
- R1 #2: Lab storetime < landmark availability audit and enforcement
- R1 #6 & R2 Major #1/#7: Post-op/OR-arrival audit via transfers, services, admissions,
  first bedside vital sign timestamp, and intubation procedure (224385)
- R1 Method: Complete patient-level and landmark-level flow diagram counts
- R2 Major #3: Add Respiratory Rate (220210), GCS (E/V/M), PaCO2 (50818), pH (50820),
  and non-invasive O2/respiratory support (226732 / 225794)
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "mimic_iv"
SSD_HOSP_DIR = Path(os.environ.get("MIMIC_IV_HOSP_DIR", str(RAW_DIR / "hosp")))
OUT_PROC_DIR = BASE_DIR / "data" / "processed" / "v30_bmjhci_r1"
OUT_RES_DIR = BASE_DIR / "results" / "high_tier_upgrade" / "v30_bmjhci_r1" / "audit"

CVICU_NAME = "Cardiac Vascular Intensive Care Unit (CVICU)"
PLANNED_SURGICAL_ADMISSIONS = {"ELECTIVE", "SURGICAL SAME DAY ADMISSION"}
SURGICAL_SERVICES = {
    "CSURG", "SURG", "NSURG", "TSURG", "VSURG",
    "ORTHO", "TRAUM", "GU", "GYN", "ENT", "PSURG", "OBS",
}

LANDMARKS = [6.0, 12.0, 18.0, 24.0]
HORIZONS = [6.0, 12.0, 24.0]
LOOKBACK = 6.0

# Valid physiological ranges (12 original + 4 new core respiratory/neuro)
VALID_RANGES: dict[str, tuple[float, float]] = {
    "heart_rate": (20.0, 300.0),
    "map": (20.0, 200.0),
    "sbp": (40.0, 300.0),
    "spo2": (50.0, 100.0),
    "temperature": (30.0, 42.0),
    "resp_rate": (4.0, 70.0),
    "gcs_total": (3.0, 15.0),
    "lactate": (0.0, 30.0),
    "creatinine": (0.0, 30.0),
    "bilirubin": (0.0, 50.0),
    "platelets": (0.0, 1500.0),
    "wbc": (0.0, 500.0),
    "pao2": (20.0, 600.0),
    "paco2": (10.0, 150.0),
    "ph": (6.80, 7.80),
    "urine_output": (0.0, 10000.0),
}

CHART_VITAL_MAP: dict[int, str] = {
    220045: "heart_rate",
    220052: "map",
    220181: "map",
    225312: "map",
    220050: "sbp",
    220179: "sbp",
    225309: "sbp",
    220277: "spo2",
    223761: "temperature_f",
    223762: "temperature",
    220210: "resp_rate",
    220739: "gcs_eye",
    223900: "gcs_verbal",
    223901: "gcs_motor",
}

CHART_MARKER_IDS = {226732, 223848, 223849, 229314}
ALL_CHART_IDS = set(CHART_VITAL_MAP.keys()) | CHART_MARKER_IDS

LAB_ITEM_MAP: dict[int, str] = {
    50813: "lactate",
    50912: "creatinine",
    50885: "bilirubin",
    51265: "platelets",
    51301: "wbc",
    50821: "pao2",
    50818: "paco2",
    50820: "ph",
}

URINE_ITEMIDS = {
    226559, 226560, 226561, 226584, 226563, 226564,
    226565, 226567, 226557, 227488, 227489,
}

# Invasive ventilator modes (standard MIMIC-IV ventilation.sql excluding Standby/NIV)
INVASIVE_VENT_MODES_EXACT = {
    "CMV/ASSIST/AutoFlow",
    "CMV/ASSIST",
    "CMV/AutoFlow",
    "CMV",
    "PRVC/AC",
    "PCV+Assist",
    "PCV+",
    "PCV+/PSV",
    "VOL/AC",
    "PRES/AC",
    "APV (cmv)",
    "APV (simv)",
    "SIMV/PSV/AutoFlow",
    "SIMV/PSV",
    "SIMV/AutoFlow",
    "SIMV",
    "SIMV/PRES",
    "SIMV/VOL",
    "MMV/PSV/AutoFlow",
    "MMV/PSV",
    "MMV/AutoFlow",
    "MMV",
    "APRV",
    "APRV/Biphasic+ApnPress",
    "APRV/Biphasic+ApnVol",
    "ASV",
    "SYNCHRON MASTER",
    "SYNCHRON SLAVE",
    "DuoPaP",
    "INTELLiVENT",
}

O2_INVASIVE_VALUES = {"Endotracheal tube", "Tracheostomy tube"}
O2_NIV_VALUES = {"Bipap mask", "CPAP mask"}
O2_HFNC_VALUES = {"High flow nasal cannula", "High flow neb"}
O2_LOWFLOW_VALUES = {
    "Nasal cannula", "Non-rebreather", "Face tent", "Aerosol-cool",
    "Oxymizer", "Medium conc mask", "Venti mask", "Ultrasonic neb",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(blk)
    return h.hexdigest()


def build_cohort_and_flow() -> tuple[pd.DataFrame, dict[str, Any]]:
    t0 = time.time()
    print("[1/5] Loading cohort, transfers, services, and computing flow diagram counts...")
    patients = pd.read_csv(
        RAW_DIR / "hosp" / "patients.csv.gz",
        usecols=["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group"],
    )
    admissions = pd.read_csv(
        RAW_DIR / "hosp" / "admissions.csv.gz",
        usecols=[
            "hadm_id", "admittime", "dischtime", "deathtime",
            "admission_type", "admission_location", "discharge_location",
            "hospital_expire_flag",
        ],
    )
    icustays = pd.read_csv(RAW_DIR / "icu" / "icustays.csv.gz")
    icustays["intime"] = pd.to_datetime(icustays["intime"], format="mixed")
    icustays["outtime"] = pd.to_datetime(icustays["outtime"], format="mixed")
    icustays["icu_los_hours"] = (icustays["outtime"] - icustays["intime"]).dt.total_seconds() / 3600.0

    flow: dict[str, Any] = {
        "total_icustays_raw": int(len(icustays)),
        "total_unique_patients_raw": int(icustays["subject_id"].nunique()),
    }

    stays = icustays.merge(admissions, on="hadm_id", how="left", validate="many_to_one")
    stays = stays.merge(patients, on="subject_id", how="left", validate="many_to_one")

    # Adult filter
    n_non_adult_stays = int((stays["anchor_age"] < 18).sum())
    stays_adult = stays[stays["anchor_age"] >= 18].copy()
    flow["excluded_non_adult_stays"] = n_non_adult_stays
    flow["adult_icustays"] = int(len(stays_adult))
    flow["adult_unique_patients"] = int(stays_adult["subject_id"].nunique())

    # Audit contiguous ICU stays (next stay starts <= 24h after first stay outtime)
    stays_sorted = stays_adult.sort_values(["subject_id", "intime", "stay_id"]).copy()
    stays_sorted["stay_rank"] = stays_sorted.groupby("subject_id").cumcount() + 1
    first_stays = stays_sorted[stays_sorted["stay_rank"] == 1].copy()
    second_stays = stays_sorted[stays_sorted["stay_rank"] == 2][["subject_id", "hadm_id", "intime"]].rename(
        columns={"hadm_id": "next_hadm_id", "intime": "next_intime"}
    )
    contig_check = first_stays[["subject_id", "hadm_id", "outtime"]].merge(second_stays, on="subject_id", how="left")
    contig_gap_h = (contig_check["next_intime"] - contig_check["outtime"]).dt.total_seconds() / 3600.0
    flow["excluded_subsequent_icustays"] = int(len(stays_adult) - len(first_stays))
    flow["patients_with_same_hadm_subsequent_icu_stay"] = int(
        (contig_check["hadm_id"] == contig_check["next_hadm_id"]).sum()
    )
    flow["patients_with_contiguous_icu_transfer_le24h"] = int(
        ((contig_check["hadm_id"] == contig_check["next_hadm_id"]) & (contig_gap_h <= 24.0)).sum()
    )
    flow["first_adult_icustays"] = int(len(first_stays))

    # Minimum LOS > 6h filter (also exclude missing outtime / NaN LOS)
    missing_outtime_mask = first_stays["icu_los_hours"].isna()
    short_stay_mask = first_stays["icu_los_hours"].le(6.0).fillna(False)
    flow["excluded_missing_outtime"] = int(missing_outtime_mask.sum())
    flow["excluded_los_le_6h"] = int(short_stay_mask.sum())
    cohort = first_stays[first_stays["icu_los_hours"].gt(6.0)].copy().reset_index(drop=True)
    cohort["case_id"] = "mimic_" + cohort["stay_id"].astype(str)
    cohort["age"] = cohort["anchor_age"].astype(float)
    cohort["sex"] = cohort["gender"].eq("M").astype(int)
    cohort["deathtime"] = pd.to_datetime(cohort["deathtime"], format="mixed", errors="coerce")
    cohort["death_hours"] = (cohort["deathtime"] - cohort["intime"]).dt.total_seconds() / 3600.0
    cohort["admittime"] = pd.to_datetime(cohort["admittime"], format="mixed", errors="coerce")
    cohort["hours_hosp_to_icu"] = (cohort["intime"] - cohort["admittime"]).dt.total_seconds() / 3600.0
    flow["eligible_los_gt_6h_patients"] = int(len(cohort))

    # Unit transfer within first stay check
    flow["patients_with_unit_change_during_stay"] = int(
        (cohort["first_careunit"] != cohort["last_careunit"]).sum()
    )

    # Merge previous careunit from transfers.csv.gz (canonical SSD master)
    transfers = pd.read_csv(
        SSD_HOSP_DIR / "transfers.csv.gz",
        usecols=["hadm_id", "eventtype", "careunit", "intime", "outtime"],
    )
    transfers = transfers[transfers["hadm_id"].isin(set(cohort["hadm_id"]))].copy()
    transfers["intime"] = pd.to_datetime(transfers["intime"], format="mixed", errors="coerce")
    transfers["outtime"] = pd.to_datetime(transfers["outtime"], format="mixed", errors="coerce")
    tr_m = cohort[["case_id", "hadm_id", "intime"]].merge(
        transfers, on="hadm_id", how="left", suffixes=("_icu", "_tr")
    )
    prev_tr = (
        tr_m[tr_m["intime_tr"] < tr_m["intime_icu"]]
        .sort_values(["case_id", "intime_tr"])
        .groupby("case_id", sort=False)
        .last()
        .reset_index()[["case_id", "careunit", "outtime"]]
        .rename(columns={"careunit": "prev_careunit", "outtime": "prev_outtime"})
    )
    cohort = cohort.merge(prev_tr, on="case_id", how="left", validate="one_to_one")
    cohort["prev_careunit"] = cohort["prev_careunit"].fillna("Direct/None")

    # Merge clinical service at ICU entry from services.csv.gz
    services = pd.read_csv(
        SSD_HOSP_DIR / "services.csv.gz",
        usecols=["hadm_id", "transfertime", "curr_service"],
    )
    services = services[services["hadm_id"].isin(set(cohort["hadm_id"]))].copy()
    services["transfertime"] = pd.to_datetime(services["transfertime"], format="mixed", errors="coerce")
    sv_m = cohort[["case_id", "hadm_id", "intime"]].merge(services, on="hadm_id", how="left")
    first_sv = (
        sv_m[sv_m["transfertime"] <= sv_m["intime"] + pd.Timedelta(hours=6)]
        .sort_values(["case_id", "transfertime"])
        .groupby("case_id", sort=False)
        .last()
        .reset_index()[["case_id", "curr_service"]]
    )
    cohort = cohort.merge(first_sv, on="case_id", how="left", validate="one_to_one")
    cohort["curr_service"] = cohort["curr_service"].fillna("UNKNOWN")

    # Care-setting flags
    cohort["is_cvicu"] = cohort["first_careunit"].eq(CVICU_NAME).astype(int)
    cohort["is_planned_surgical"] = cohort["admission_type"].isin(PLANNED_SURGICAL_ADMISSIONS).astype(int)
    cohort["is_surgical_service"] = cohort["curr_service"].isin(SURGICAL_SERVICES).astype(int)
    cohort["is_or_pacu_transfer"] = cohort["prev_careunit"].isin({"PACU", "Discharge Lounge"}).astype(int)
    cohort["is_unplanned_non_cvicu"] = (
        (cohort["is_cvicu"] == 0) & (cohort["is_planned_surgical"] == 0)
    ).astype(int)
    cohort["is_unplanned_non_surgical"] = (
        (cohort["is_cvicu"] == 0)
        & (cohort["is_planned_surgical"] == 0)
        & (cohort["is_surgical_service"] == 0)
    ).astype(int)

    flow["cvicu_patients"] = int(cohort["is_cvicu"].sum())
    flow["planned_surgical_patients"] = int(cohort["is_planned_surgical"].sum())
    flow["unplanned_non_cvicu_patients"] = int(cohort["is_unplanned_non_cvicu"].sum())
    flow["unplanned_non_surgical_patients"] = int(cohort["is_unplanned_non_surgical"].sum())
    print(f"  Cohort built: {len(cohort):,} patients (unplanned non-CVICU: {flow['unplanned_non_cvicu_patients']:,}) in {time.time()-t0:.1f}s")
    return cohort, flow


def extract_procedure_ventilation(cohort: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    t0 = time.time()
    print("[2/5] Extracting procedureevents (IMV 225792, Intubation 224385, NIV 225794)...")
    pe = pd.read_csv(
        RAW_DIR / "icu" / "procedureevents.csv.gz",
        usecols=["stay_id", "starttime", "endtime", "storetime", "itemid", "statusdescription"],
    )
    pe = pe[pe["itemid"].isin([225792, 224385, 225794])].copy()
    pe = pe.merge(cohort[["stay_id", "case_id", "intime"]], on="stay_id", how="inner")
    pe["starttime"] = pd.to_datetime(pe["starttime"], format="mixed")
    pe["endtime"] = pd.to_datetime(pe["endtime"], format="mixed")
    pe["storetime"] = pd.to_datetime(pe["storetime"], format="mixed")
    pe["start_hours"] = (pe["starttime"] - pe["intime"]).dt.total_seconds() / 3600.0
    pe["end_hours"] = (pe["endtime"] - pe["intime"]).dt.total_seconds() / 3600.0
    pe["store_hours"] = (pe["storetime"] - pe["intime"]).dt.total_seconds() / 3600.0

    # IMV 225792 episodes (same accepted statuses as v25/v29)
    imv_raw = pe[
        (pe["itemid"] == 225792)
        & pe["statusdescription"].isin({"FinishedRunning", "Stopped", "Paused"})
        & np.isfinite(pe["start_hours"])
        & np.isfinite(pe["end_hours"])
        & (pe["end_hours"] > pe["start_hours"])
    ].sort_values(["stay_id", "start_hours", "end_hours"]).copy()

    # Merge intervals within 1h gap
    merged = []
    for (stay_id, case_id), grp in imv_raw.groupby(["stay_id", "case_id"], sort=False):
        cur_s = cur_e = None
        cur_store = None
        for row in grp.itertuples(index=False):
            if cur_s is None:
                cur_s, cur_e, cur_store = row.start_hours, row.end_hours, row.store_hours
            elif row.start_hours <= cur_e + 1.0:
                cur_e = max(cur_e, row.end_hours)
            else:
                merged.append((stay_id, case_id, cur_s, cur_e, cur_store))
                cur_s, cur_e, cur_store = row.start_hours, row.end_hours, row.store_hours
        if cur_s is not None:
            merged.append((stay_id, case_id, cur_s, cur_e, cur_store))
    imv_episodes = pd.DataFrame(
        merged, columns=["stay_id", "case_id", "start_hours", "end_hours", "first_store_hours"]
    )

    # Intubation procedure (224385)
    intub = (
        pe[pe["itemid"] == 224385]
        .groupby("case_id", sort=False)["start_hours"]
        .min()
        .reset_index()
        .rename(columns={"start_hours": "intub_proc_hours"})
    )

    # NIV procedure (225794)
    niv_proc = pe[
        (pe["itemid"] == 225794)
        & pe["statusdescription"].isin({"FinishedRunning", "Stopped", "Paused"})
        & np.isfinite(pe["start_hours"])
        & np.isfinite(pe["end_hours"])
    ][["case_id", "start_hours", "end_hours", "store_hours"]].copy()

    print(f"  IMV episodes: {len(imv_episodes):,}, intubation stays: {len(intub):,}, NIV rows: {len(niv_proc):,} ({time.time()-t0:.1f}s)")
    return imv_episodes, intub, niv_proc


def extract_chartevents_single_pass(cohort: pd.DataFrame) -> dict[str, pd.DataFrame]:
    t0 = time.time()
    print("[3/5] Single-pass Polars scan of chartevents.csv.gz (vitals, RR, GCS, O2 device, IMV markers)...")
    coh_pl = pl.from_pandas(cohort[["stay_id", "case_id", "intime"]]).with_columns(
        pl.col("stay_id").cast(pl.Int64),
        pl.col("intime").cast(pl.Datetime("us")),
    )

    # Stream filter chartevents with Polars
    q = (
        pl.scan_csv(
            str(RAW_DIR / "icu" / "chartevents.csv.gz"),
            schema_overrides={
                "stay_id": pl.Int64,
                "itemid": pl.Int64,
                "charttime": pl.Utf8,
                "storetime": pl.Utf8,
                "value": pl.Utf8,
                "valuenum": pl.Float64,
            },
            ignore_errors=True,
        )
        .select(["stay_id", "itemid", "charttime", "storetime", "value", "valuenum"])
        .filter(pl.col("itemid").is_in(list(ALL_CHART_IDS)))
        .join(coh_pl.lazy(), on="stay_id", how="inner")
        .with_columns(
            pl.col("charttime").str.to_datetime(strict=False).alias("chart_dt"),
            pl.col("storetime").str.to_datetime(strict=False).alias("store_dt"),
        )
        .with_columns(
            ((pl.col("chart_dt") - pl.col("intime")).dt.total_seconds() / 3600.0).alias("time_hours"),
            ((pl.col("store_dt") - pl.col("intime")).dt.total_seconds() / 3600.0).alias("store_hours"),
        )
        .filter((pl.col("time_hours") >= -2.0) & (pl.col("time_hours") <= 48.0))
        .select(["case_id", "stay_id", "itemid", "time_hours", "store_hours", "value", "valuenum"])
    )
    ce = q.collect(engine="streaming")
    print(f"  Collected {ce.height:,} relevant chartevents rows in {time.time()-t0:.1f}s")

    # 3A. Bedside first vital sign timestamp (strictly >= 0 and valid numeric)
    vital_ids_core = [220045, 220052, 220181, 225312, 220050, 220179, 225309, 220277, 220210]
    first_vital = (
        ce.filter(
            pl.col("itemid").is_in(vital_ids_core)
            & pl.col("valuenum").is_not_null()
            & (pl.col("time_hours") >= 0.0)
        )
        .group_by("case_id")
        .agg(
            pl.col("time_hours").min().alias("first_vital_exact_hours"),
            pl.col("store_hours").min().alias("first_vital_store_hours"),
        )
        .to_pandas()
    )

    # 3B. Independent chartevents IMV markers
    # (1) O2 Delivery Device in Endotracheal tube / Tracheostomy tube
    # (2) GCS Verbal == No Response-ETT
    # (3) Ventilator Mode in INVASIVE_VENT_MODES_EXACT
    ce_val_clean = ce.with_columns(pl.col("value").str.strip_chars().alias("val_clean"))
    imv_o2 = ce_val_clean.filter(
        (pl.col("itemid") == 226732) & pl.col("val_clean").is_in(list(O2_INVASIVE_VALUES))
    ).with_columns(pl.lit("o2_ett_trach").alias("marker_source"))

    imv_gcs_ett = ce_val_clean.filter(
        (pl.col("itemid") == 223900) & pl.col("val_clean").str.contains("ETT")
    ).with_columns(pl.lit("gcs_verbal_ett").alias("marker_source"))

    imv_mode = ce_val_clean.filter(
        pl.col("itemid").is_in([223849, 229314]) & pl.col("val_clean").is_in(list(INVASIVE_VENT_MODES_EXACT))
    ).with_columns(pl.lit("invasive_vent_mode").alias("marker_source"))

    imv_markers = pl.concat([imv_o2, imv_gcs_ett, imv_mode])
    first_chart_imv = (
        imv_markers.filter(pl.col("time_hours") >= -2.0)
        .group_by("case_id")
        .agg(
            pl.col("time_hours").min().alias("chart_imv_start_hours"),
            pl.col("time_hours").filter(pl.col("marker_source") == "o2_ett_trach").min().alias("o2_ett_start_hours"),
            pl.col("time_hours").filter(pl.col("marker_source") == "gcs_verbal_ett").min().alias("gcs_ett_start_hours"),
            pl.col("time_hours").filter(pl.col("marker_source") == "invasive_vent_mode").min().alias("vent_mode_start_hours"),
        )
        .to_pandas()
    )

    # 3C. Non-invasive oxygen support hourly flags in [0, 24h) (using charttime in [0,24) and storetime audit)
    o2_supp = (
        ce_val_clean.filter(
            (pl.col("itemid") == 226732)
            & (pl.col("time_hours") >= 0.0)
            & (pl.col("time_hours") < 24.0)
        )
        .with_columns(
            pl.col("time_hours").floor().cast(pl.Int32).alias("hour_bin"),
            pl.when(pl.col("val_clean").is_in(list(O2_NIV_VALUES))).then(3)
            .when(pl.col("val_clean").is_in(list(O2_HFNC_VALUES))).then(2)
            .when(pl.col("val_clean").is_in(list(O2_LOWFLOW_VALUES))).then(1)
            .when(pl.col("val_clean") == "None").then(0)
            .otherwise(None)
            .alias("o2_support_level"),
        )
        .filter(pl.col("o2_support_level").is_not_null())
        .group_by(["case_id", "hour_bin"])
        .agg(
            pl.col("o2_support_level").max().alias("o2_support_level"),
            (pl.col("o2_support_level") == 3).any().cast(pl.Int8).alias("has_niv_chart"),
            (pl.col("o2_support_level") == 2).any().cast(pl.Int8).alias("has_hfnc_chart"),
            (pl.col("o2_support_level") >= 1).any().cast(pl.Int8).alias("has_any_o2_chart"),
        )
        .to_pandas()
    )

    # 3D. Hourly vitals + RR + GCS in [0, 24h)
    # Audit charttime vs storetime delay for vitals
    vital_ce = ce_val_clean.filter(
        pl.col("itemid").is_in(list(CHART_VITAL_MAP.keys()))
        & (pl.col("time_hours") >= 0.0)
        & (pl.col("time_hours") < 24.0)
        & pl.col("valuenum").is_not_null()
    )
    # Exclude GCS Verbal "No Response-ETT" from unintubated GCS feature!
    vital_ce = vital_ce.filter(
        ~((pl.col("itemid") == 223900) & pl.col("val_clean").str.contains("ETT"))
    )
    vital_delay_stats = (
        vital_ce.with_columns((pl.col("store_hours") - pl.col("time_hours")).alias("delay_h"))
        .filter(pl.col("delay_h").is_not_null())
        .select([
            pl.len().alias("n"),
            pl.col("delay_h").median().alias("median_h"),
            pl.col("delay_h").quantile(0.25).alias("q25_h"),
            pl.col("delay_h").quantile(0.75).alias("q75_h"),
            pl.col("delay_h").quantile(0.90).alias("p90_h"),
            (pl.col("delay_h") <= 0.5).mean().alias("frac_le_30min"),
            (pl.col("delay_h") <= 1.0).mean().alias("frac_le_1h"),
        ])
        .to_pandas()
    )

    # Map itemid -> feature name and convert Temp F -> C
    vital_ce = vital_ce.with_columns(
        pl.col("itemid").replace_strict(CHART_VITAL_MAP, default=None).alias("feature"),
        pl.when(pl.col("itemid") == 223761)
        .then((pl.col("valuenum") - 32.0) * 5.0 / 9.0)
        .otherwise(pl.col("valuenum"))
        .alias("val_num"),
        pl.col("time_hours").floor().cast(pl.Int32).alias("hour_bin"),
    ).with_columns(
        pl.when(pl.col("feature") == "temperature_f")
        .then(pl.lit("temperature"))
        .otherwise(pl.col("feature"))
        .alias("feature")
    )

    # Apply physiological range filtering
    range_expr = pl.lit(True)
    for feat, (lo, hi) in VALID_RANGES.items():
        if feat in {"heart_rate", "map", "sbp", "spo2", "temperature", "resp_rate"}:
            range_expr = range_expr & (
                (pl.col("feature") != feat) | ((pl.col("val_num") >= lo) & (pl.col("val_num") <= hi))
            )
    range_expr = range_expr & (
        (pl.col("feature") != "gcs_eye") | ((pl.col("val_num") >= 1) & (pl.col("val_num") <= 4))
    ) & (
        (pl.col("feature") != "gcs_verbal") | ((pl.col("val_num") >= 1) & (pl.col("val_num") <= 5))
    ) & (
        (pl.col("feature") != "gcs_motor") | ((pl.col("val_num") >= 1) & (pl.col("val_num") <= 6))
    )

    vital_clean = vital_ce.filter(range_expr)

    # Aggregate hourly mean by (case_id, hour_bin, feature)
    hourly_vitals_long = (
        vital_clean.group_by(["case_id", "hour_bin", "feature"])
        .agg(pl.col("val_num").mean().alias("val"))
    )
    hourly_vitals = (
        hourly_vitals_long.pivot(
            on="feature", index=["case_id", "hour_bin"], values="val"
        )
        .to_pandas()
    )
    # Compute gcs_total when all 3 E/V/M are observed in that hour
    for col in ["gcs_eye", "gcs_verbal", "gcs_motor"]:
        if col not in hourly_vitals.columns:
            hourly_vitals[col] = np.nan
    hourly_vitals["gcs_total"] = (
        hourly_vitals["gcs_eye"] + hourly_vitals["gcs_verbal"] + hourly_vitals["gcs_motor"]
    )

    print(f"  Chartevents hourly vitals pivoted: {hourly_vitals.shape} ({time.time()-t0:.1f}s)")
    return {
        "first_vital": first_vital,
        "first_chart_imv": first_chart_imv,
        "o2_supp": o2_supp,
        "hourly_vitals": hourly_vitals,
        "vital_delay_stats": vital_delay_stats,
    }


def extract_labs_and_urine(cohort: pd.DataFrame) -> dict[str, pd.DataFrame]:
    t0 = time.time()
    print("[4/5] Single-pass Polars scan of labevents.csv.gz (with storetime audit) & outputevents.csv.gz...")
    coh_hadm = pl.from_pandas(cohort[["hadm_id", "case_id", "intime"]]).with_columns(
        pl.col("hadm_id").cast(pl.Int64),
        pl.col("intime").cast(pl.Datetime("us")),
    )

    q_lab = (
        pl.scan_csv(
            str(RAW_DIR / "hosp" / "labevents.csv.gz"),
            schema_overrides={
                "hadm_id": pl.Int64,
                "itemid": pl.Int64,
                "charttime": pl.Utf8,
                "storetime": pl.Utf8,
                "valuenum": pl.Float64,
            },
            ignore_errors=True,
        )
        .select(["hadm_id", "itemid", "charttime", "storetime", "valuenum"])
        .filter(pl.col("itemid").is_in(list(LAB_ITEM_MAP.keys())) & pl.col("valuenum").is_not_null())
        .join(coh_hadm.lazy(), on="hadm_id", how="inner")
        .with_columns(
            pl.col("charttime").str.to_datetime(strict=False).alias("chart_dt"),
            pl.col("storetime").str.to_datetime(strict=False).alias("store_dt"),
        )
        .with_columns(
            ((pl.col("chart_dt") - pl.col("intime")).dt.total_seconds() / 3600.0).alias("chart_hours"),
            ((pl.col("store_dt") - pl.col("intime")).dt.total_seconds() / 3600.0).alias("store_hours"),
        )
        .filter((pl.col("chart_hours") >= -6.0) & (pl.col("chart_hours") < 24.0))
        .with_columns(
            pl.col("itemid").replace_strict(LAB_ITEM_MAP, default=None).alias("feature"),
            (pl.col("store_hours") - pl.col("chart_hours")).alias("turnaround_hours"),
        )
        .select(["case_id", "feature", "chart_hours", "store_hours", "turnaround_hours", "valuenum"])
    )
    labs_pl = q_lab.collect(engine="streaming")

    # Apply physiological range filter
    range_expr = pl.lit(True)
    for feat in set(LAB_ITEM_MAP.values()):
        lo, hi = VALID_RANGES[feat]
        range_expr = range_expr & (
            (pl.col("feature") != feat) | ((pl.col("valuenum") >= lo) & (pl.col("valuenum") <= hi))
        )
    labs_pl = labs_pl.filter(range_expr)

    lab_delay_audit = (
        labs_pl.filter((pl.col("chart_hours") >= 0.0) & (pl.col("chart_hours") < 24.0))
        .group_by("feature")
        .agg(
            pl.len().alias("n_obs"),
            pl.col("store_hours").is_null().mean().alias("storetime_missing_frac"),
            pl.col("turnaround_hours").median().alias("median_turnaround_h"),
            pl.col("turnaround_hours").quantile(0.25).alias("q25_turnaround_h"),
            pl.col("turnaround_hours").quantile(0.75).alias("q75_turnaround_h"),
            pl.col("turnaround_hours").quantile(0.90).alias("p90_turnaround_h"),
            (pl.col("turnaround_hours") > 1.0).mean().alias("frac_gt_1h"),
            (pl.col("turnaround_hours") > 2.0).mean().alias("frac_gt_2h"),
        )
        .sort("feature")
        .to_pandas()
    )

    # Urine output from outputevents
    coh_stay = pl.from_pandas(cohort[["stay_id", "case_id", "intime"]]).with_columns(
        pl.col("stay_id").cast(pl.Int64),
        pl.col("intime").cast(pl.Datetime("us")),
    )
    q_uo = (
        pl.scan_csv(
            str(RAW_DIR / "icu" / "outputevents.csv.gz"),
            schema_overrides={
                "stay_id": pl.Int64,
                "itemid": pl.Int64,
                "charttime": pl.Utf8,
                "storetime": pl.Utf8,
                "value": pl.Float64,
            },
            ignore_errors=True,
        )
        .select(["stay_id", "itemid", "charttime", "storetime", "value"])
        .filter(pl.col("itemid").is_in(list(URINE_ITEMIDS)) & pl.col("value").is_not_null())
        .join(coh_stay.lazy(), on="stay_id", how="inner")
        .with_columns(
            pl.col("charttime").str.to_datetime(strict=False).alias("chart_dt"),
            pl.col("storetime").str.to_datetime(strict=False).alias("store_dt"),
        )
        .with_columns(
            ((pl.col("chart_dt") - pl.col("intime")).dt.total_seconds() / 3600.0).alias("chart_hours"),
            ((pl.col("store_dt") - pl.col("intime")).dt.total_seconds() / 3600.0).alias("store_hours"),
        )
        .filter((pl.col("chart_hours") >= 0.0) & (pl.col("chart_hours") < 24.0) & (pl.col("value") >= 0.0))
        .with_columns(pl.col("chart_hours").floor().cast(pl.Int32).alias("hour_bin"))
        .group_by(["case_id", "hour_bin"])
        .agg(pl.col("value").sum().alias("urine_output"))
        .filter(pl.col("urine_output") <= 10000.0)
    )
    hourly_uo = q_uo.collect(engine="streaming").to_pandas()
    print(f"  Labs collected: {labs_pl.height:,} rows; urine hourly rows: {len(hourly_uo):,} ({time.time()-t0:.1f}s)")
    return {
        "labs_df": labs_pl.to_pandas(),
        "lab_delay_audit": lab_delay_audit,
        "hourly_uo": hourly_uo,
    }


def _summarize_window_feature(
    df_feat: pd.DataFrame,
    case_ids: pd.Index,
    feature: str,
    landmark: float,
    lookback: float,
) -> pd.DataFrame:
    """Compute last, mean, min, max, slope, missing_rate, time_since_last, delta6h."""
    idx = pd.Index(case_ids, name="case_id")
    out = pd.DataFrame(index=idx)
    if df_feat.empty:
        for suf in ("last", "mean", "min", "max", "slope", "delta6h"):
            out[f"{feature}_{suf}"] = np.nan
        out[f"{feature}_missing_rate"] = 1.0
        out[f"{feature}_time_since_last"] = float(lookback)
        return out

    df_s = df_feat.sort_values(["case_id", "hour_bin"])
    grp = df_s.groupby("case_id", sort=False)
    desc = grp["val"].agg(["last", "mean", "min", "max", "first", "count"])
    last_t = grp["hour_bin"].max()

    # Vectorized OLS slope
    t = df_s["hour_bin"].astype(float).to_numpy()
    v = df_s["val"].astype(float).to_numpy()
    tmp = pd.DataFrame({
        "case_id": df_s["case_id"].to_numpy(),
        "t": t, "v": v, "tt": t * t, "tv": t * v,
    })
    sums = tmp.groupby("case_id", sort=False).agg(
        n=("v", "count"),
        st=("t", "sum"),
        sv=("v", "sum"),
        stt=("tt", "sum"),
        stv=("tv", "sum"),
    )
    den = sums["stt"] - sums["st"].pow(2) / sums["n"]
    num = sums["stv"] - sums["st"] * sums["sv"] / sums["n"]
    slope = (num / den.replace(0.0, np.nan)).where(sums["n"] >= 2, 0.0)
    delta = (desc["last"] - desc["first"]).where(desc["count"] >= 2, 0.0)

    out[f"{feature}_last"] = desc["last"].reindex(idx)
    out[f"{feature}_mean"] = desc["mean"].reindex(idx)
    out[f"{feature}_min"] = desc["min"].reindex(idx)
    out[f"{feature}_max"] = desc["max"].reindex(idx)
    out[f"{feature}_slope"] = slope.reindex(idx)
    out[f"{feature}_missing_rate"] = (1.0 - desc["count"].reindex(idx).fillna(0) / float(lookback)).clip(0.0, 1.0)
    out[f"{feature}_time_since_last"] = (landmark - last_t.reindex(idx)).fillna(float(lookback))
    out[f"{feature}_delta6h"] = delta.reindex(idx)
    return out


def build_states_and_features(
    cohort: pd.DataFrame,
    imv_episodes: pd.DataFrame,
    intub_df: pd.DataFrame,
    niv_proc: pd.DataFrame,
    chart_res: dict[str, pd.DataFrame],
    lab_res: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    t0 = time.time()
    print("[5/5] Building leak-free landmark states, signed timing audits, and 16-variable + O2 features...")

    # Merge onset sources onto cohort
    proc_first = imv_episodes.groupby("case_id")["start_hours"].min().rename("proc_imv_start_hours")
    coh = cohort.merge(proc_first, on="case_id", how="left")
    coh = coh.merge(chart_res["first_vital"], on="case_id", how="left")
    coh = coh.merge(chart_res["first_chart_imv"], on="case_id", how="left")
    coh = coh.merge(intub_df, on="case_id", how="left")

    # Earliest invasive ventilation evidence across procedureevents (225792),
    # validated chartevents IMV markers (ETT/Trach/GCS-ETT/invasive mode), and intubation (224385)
    coh["any_chart_or_intub_imv_hours"] = coh[["chart_imv_start_hours", "intub_proc_hours"]].min(axis=1)
    coh["harmonized_imv_start_hours"] = coh[["proc_imv_start_hours", "any_chart_or_intub_imv_hours"]].min(axis=1)

    # Signed timing difference audit (procedure_start - chartevents_imv_start)
    both = coh[coh["proc_imv_start_hours"].notna() & coh["chart_imv_start_hours"].notna()].copy()
    both["signed_diff_h"] = both["proc_imv_start_hours"] - both["chart_imv_start_hours"]
    timing_audit = {
        "n_proc_imv_stays": int(coh["proc_imv_start_hours"].notna().sum()),
        "n_chart_imv_stays": int(coh["chart_imv_start_hours"].notna().sum()),
        "n_both_matched": int(len(both)),
        "signed_diff_proc_minus_chart_median_h": float(both["signed_diff_h"].median()),
        "signed_diff_q25_h": float(both["signed_diff_h"].quantile(0.25)),
        "signed_diff_q75_h": float(both["signed_diff_h"].quantile(0.75)),
        "signed_diff_p05_h": float(both["signed_diff_h"].quantile(0.05)),
        "signed_diff_p95_h": float(both["signed_diff_h"].quantile(0.95)),
        "abs_diff_median_h": float(both["signed_diff_h"].abs().median()),
        "within_1h_frac": float((both["signed_diff_h"].abs() <= 1.0).mean()),
        "within_2h_frac": float((both["signed_diff_h"].abs() <= 2.0).mean()),
        "chart_precedes_proc_frac": float((both["signed_diff_h"] > 0).mean()),
    }

    # Build landmark state rows across landmarks [6, 12, 18, 24]
    # Include:
    # - v29_eligible (procedure-only never_exposed, no bedside vital check)
    # - r1_eligible (physically monitored before L AND never exposed to procedure OR chartevents IMV at <= L)
    # - outcomes at 6h, 12h, 24h, plus 1h-gap and 2h-gap, plus procedure-confirmed and intubation-confirmed outcomes
    state_frames = []
    for lm in LANDMARKS:
        present = coh[coh["icu_los_hours"] > lm].copy()
        present["landmark_hours"] = lm

        # Procedure-only state (v29)
        active_proc = set(
            imv_episodes.loc[
                (imv_episodes["start_hours"] <= lm) & (imv_episodes["end_hours"] > lm), "case_id"
            ]
        )
        prior_proc = set(imv_episodes.loc[imv_episodes["start_hours"] <= lm, "case_id"])
        is_act_p = present["case_id"].isin(active_proc)
        is_pri_p = present["case_id"].isin(prior_proc)
        present["v29_state"] = np.select([is_act_p, is_pri_p & ~is_act_p], ["active", "resolved"], default="never_exposed")
        present["v29_eligible"] = (present["v29_state"] == "never_exposed").astype(int)
        present["v29_outcome_6h"] = (
            (present["v29_eligible"] == 1)
            & (present["proc_imv_start_hours"] > lm)
            & (present["proc_imv_start_hours"] <= lm + 6.0)
        ).astype(int)

        # R1 leak-free eligibility:
        # 1. Physically present at ICU bedside (at least 1 core vital strictly before L)
        present["has_vital_before_lm"] = (
            present["first_vital_exact_hours"].notna() & (present["first_vital_exact_hours"] < lm)
        ).astype(int)
        # 2. No prior invasive ventilation from EITHER procedureevents OR chartevents/intubation at <= L
        prior_chart_imv = present["any_chart_or_intub_imv_hours"].notna() & (present["any_chart_or_intub_imv_hours"] <= lm)
        present["prior_any_imv"] = (is_pri_p | prior_chart_imv).astype(int)
        present["prior_chart_only_imv"] = (~is_pri_p & prior_chart_imv).astype(int)
        present["r1_eligible"] = (
            (present["has_vital_before_lm"] == 1) & (present["prior_any_imv"] == 0)
        ).astype(int)

        # Harmonized first-IMV onset (must be > lm for r1_eligible)
        onset = present["harmonized_imv_start_hours"]
        proc_onset = present["proc_imv_start_hours"]

        # Primary R1 outcome: procedure-confirmed OR validated invasive marker onset in (L, L+6h],
        # AND let's store BOTH harmonized and procedure-confirmed!
        # Wait: let's check how often chart_imv occurs WITHOUT procedureevents 225792!
        # Primary: harmonized first IMV onset in (L, L+6h] among r1_eligible
        # Sensitivity A: procedure-confirmed IMV (proc_imv_start_hours exists & harmonized onset in (L, L+6h])
        # Sensitivity B: intubation-procedure-confirmed (intub_proc_hours in (L, L+6h])
        present["outcome_r1_6h"] = ((present["r1_eligible"] == 1) & (onset > lm) & (onset <= lm + 6.0)).astype(int)
        present["outcome_r1_12h"] = ((present["r1_eligible"] == 1) & (onset > lm) & (onset <= lm + 12.0)).astype(int)
        present["outcome_r1_24h"] = ((present["r1_eligible"] == 1) & (onset > lm) & (onset <= lm + 24.0)).astype(int)
        present["outcome_r1_gap1h"] = ((present["r1_eligible"] == 1) & (onset > lm + 1.0) & (onset <= lm + 6.0)).astype(int)
        present["outcome_r1_gap2h"] = ((present["r1_eligible"] == 1) & (onset > lm + 2.0) & (onset <= lm + 6.0)).astype(int)
        present["outcome_proc_confirmed_6h"] = (
            (present["r1_eligible"] == 1)
            & proc_onset.notna()
            & (onset > lm)
            & (onset <= lm + 6.0)
        ).astype(int)
        present["outcome_intub_confirmed_6h"] = (
            (present["r1_eligible"] == 1)
            & (present["intub_proc_hours"] > lm)
            & (present["intub_proc_hours"] <= lm + 6.0)
        ).astype(int)
        present["history_inclusive_outcome_r1"] = (
            (present["prior_any_imv"] == 1) | (present["outcome_r1_6h"] == 1)
        ).astype(int)

        present["event_time_r1_hours"] = onset.where(present["outcome_r1_6h"] == 1)
        present["lead_time_r1_hours"] = (present["event_time_r1_hours"] - lm).where(present["outcome_r1_6h"] == 1)

        # Competing events within 6h horizon
        present["competing_death_6h"] = (
            (present["r1_eligible"] == 1)
            & (present["outcome_r1_6h"] == 0)
            & (present["death_hours"] > lm)
            & (present["death_hours"] <= lm + 6.0)
        ).astype(int)
        present["competing_discharge_6h"] = (
            (present["r1_eligible"] == 1)
            & (present["outcome_r1_6h"] == 0)
            & (present["competing_death_6h"] == 0)
            & (present["icu_los_hours"] <= lm + 6.0)
        ).astype(int)

        state_frames.append(present)

    states = pd.concat(state_frames, ignore_index=True)

    # Assert uniqueness: each patient has <= 1 positive primary 6h outcome across all 4 landmarks!
    pos_counts = states.groupby("case_id")["outcome_r1_6h"].sum()
    assert pos_counts.le(1).all(), "ERROR: Multiple positive 6h outcomes for a patient!"

    # Merge hourly vitals + hourly urine output
    hv = chart_res["hourly_vitals"]
    huo = lab_res["hourly_uo"]
    o2_supp = chart_res["o2_supp"]
    labs_df = lab_res["labs_df"]

    # Build rolling features at each landmark (strict storetime < landmark for labs!)
    # Also compute charttime-only lab features for availability sensitivity audit!
    vital_feats = ["heart_rate", "map", "sbp", "spo2", "temperature", "resp_rate", "gcs_total"]
    lab_feats = ["lactate", "creatinine", "bilirubin", "platelets", "wbc", "pao2", "paco2", "ph"]

    feat_frames = []
    lab_leak_audit_rows = []
    for lm in LANDMARKS:
        ids = pd.Index(states.loc[states["landmark_hours"] == lm, "case_id"].unique(), name="case_id")
        win_lo = lm - LOOKBACK
        pieces = []

        # Vitals in [lm-6, lm)
        hv_win = hv[(hv["hour_bin"] >= win_lo) & (hv["hour_bin"] < lm) & hv["case_id"].isin(ids)]
        for f in vital_feats:
            sub = hv_win[["case_id", "hour_bin", f]].dropna(subset=[f]).rename(columns={f: "val"})
            pieces.append(_summarize_window_feature(sub, ids, f, lm, LOOKBACK))

        # Urine output in [lm-6, lm)
        uo_win = huo[(huo["hour_bin"] >= win_lo) & (huo["hour_bin"] < lm) & huo["case_id"].isin(ids)]
        pieces.append(_summarize_window_feature(
            uo_win.rename(columns={"urine_output": "val"}), ids, "urine_output", lm, LOOKBACK
        ))

        # Labs with STRICT storetime < lm (and chart_hours in [lm-6, lm), or if storetime missing exclude!)
        lab_chart_win = labs_df[
            (labs_df["chart_hours"] >= win_lo)
            & (labs_df["chart_hours"] < lm)
            & labs_df["case_id"].isin(ids)
        ].copy()
        lab_chart_win["available_at_lm"] = (
            lab_chart_win["store_hours"].notna() & (lab_chart_win["store_hours"] < lm)
        )
        lab_leak_audit_rows.append({
            "landmark_hours": lm,
            "total_labs_chart_in_window": int(len(lab_chart_win)),
            "available_by_storetime_lt_lm": int(lab_chart_win["available_at_lm"].sum()),
            "excluded_pending_result_at_lm": int((~lab_chart_win["available_at_lm"]).sum()),
            "pending_fraction": float((~lab_chart_win["available_at_lm"]).mean()) if len(lab_chart_win) else 0.0,
        })

        lab_avail = lab_chart_win[lab_chart_win["available_at_lm"]].copy()
        lab_avail["hour_bin"] = np.floor(lab_avail["chart_hours"]).astype(int)
        lab_hourly = (
            lab_avail.groupby(["case_id", "hour_bin", "feature"], sort=False)["valuenum"]
            .mean()
            .reset_index()
            .rename(columns={"valuenum": "val"})
        )
        for f in lab_feats:
            sub = lab_hourly[lab_hourly["feature"] == f][["case_id", "hour_bin", "val"]]
            pieces.append(_summarize_window_feature(sub, ids, f, lm, LOOKBACK))

        # Non-invasive O2/respiratory support summary in [lm-6, lm) (for sensitivity analysis!)
        o2_win = o2_supp[(o2_supp["hour_bin"] >= win_lo) & (o2_supp["hour_bin"] < lm) & o2_supp["case_id"].isin(ids)]
        niv_win_ids = set(
            niv_proc.loc[
                (niv_proc["start_hours"] < lm) & (niv_proc["end_hours"] > win_lo), "case_id"
            ]
        )
        o2_agg = o2_win.sort_values(["case_id", "hour_bin"]).groupby("case_id", sort=False).agg(
            o2_support_max=("o2_support_level", "max"),
            o2_support_last=("o2_support_level", "last"),
            o2_any_flag=("has_any_o2_chart", "max"),
            hfnc_flag=("has_hfnc_chart", "max"),
            niv_chart_flag=("has_niv_chart", "max"),
        ).reindex(ids)
        o2_out = pd.DataFrame(index=ids)
        o2_out["o2_support_max"] = o2_agg["o2_support_max"].fillna(0.0)
        o2_out["o2_support_last"] = o2_agg["o2_support_last"].fillna(0.0)
        o2_out["o2_any_flag"] = o2_agg["o2_any_flag"].fillna(0).astype(int)
        o2_out["hfnc_flag"] = o2_agg["hfnc_flag"].fillna(0).astype(int)
        o2_out["niv_flag"] = np.maximum(
            o2_agg["niv_chart_flag"].fillna(0).astype(int),
            o2_out.index.isin(niv_win_ids).astype(int),
        )
        o2_out["o2_support_max"] = np.where(o2_out["niv_flag"] == 1, 3.0, o2_out["o2_support_max"])
        pieces.append(o2_out)

        lm_frame = pd.concat(pieces, axis=1).reset_index()
        lm_frame["landmark_hours"] = lm
        feat_frames.append(lm_frame)

    features = pd.concat(feat_frames, ignore_index=True)
    # Merge static features (age, sex)
    features = features.merge(
        coh[["case_id", "subject_id", "age", "sex", "first_careunit", "admission_type", "anchor_year_group"]],
        on="case_id",
        how="left",
        validate="many_to_one",
    )

    extra_audit = {
        "timing_audit": timing_audit,
        "lab_storetime_landmark_audit": lab_leak_audit_rows,
        "vital_storetime_delay": chart_res["vital_delay_stats"].to_dict(orient="records"),
        "lab_turnaround_by_feature": lab_res["lab_delay_audit"].to_dict(orient="records"),
    }
    print(f"  Built states {states.shape} & features {features.shape} in {time.time()-t0:.1f}s")
    return states, features, extra_audit


def summarize_cohorts_and_audit(states: pd.DataFrame) -> pd.DataFrame:
    """Summarize landmark counts across v29 vs R1 cleaned cohorts."""
    rows = []
    subsets = {
        "v29_original_uncleaned": states["v29_eligible"] == 1,
        "r1_all_monitored_icu": states["r1_eligible"] == 1,
        "r1_primary_unplanned_non_cvicu": (states["r1_eligible"] == 1) & (states["is_unplanned_non_cvicu"] == 1),
        "r1_strict_unplanned_non_surgical": (states["r1_eligible"] == 1) & (states["is_unplanned_non_surgical"] == 1),
        "r1_cvicu_or_planned_surgical_monitored": (states["r1_eligible"] == 1) & (states["is_unplanned_non_cvicu"] == 0),
    }
    for sname, mask in subsets.items():
        outcome_col = "v29_outcome_6h" if sname == "v29_original_uncleaned" else "outcome_r1_6h"
        sub = states[mask]
        for lm in LANDMARKS + [0.0]:  # 0.0 = pooled
            d = sub if lm == 0.0 else sub[sub["landmark_hours"] == lm]
            rows.append({
                "cohort": sname,
                "landmark_hours": "pooled" if lm == 0.0 else int(lm),
                "assessments": int(len(d)),
                "unique_patients": int(d["case_id"].nunique()),
                "events_6h": int(d[outcome_col].sum()),
                "event_rate_pct": float(100.0 * d[outcome_col].mean()) if len(d) else 0.0,
                "events_proc_confirmed_6h": int(d["outcome_proc_confirmed_6h"].sum()),
                "events_intub_confirmed_6h": int(d["outcome_intub_confirmed_6h"].sum()),
                "events_gap1h": int(d["outcome_r1_gap1h"].sum()),
                "events_gap2h": int(d["outcome_r1_gap2h"].sum()),
                "competing_death_6h": int(d["competing_death_6h"].sum()),
                "competing_discharge_6h": int(d["competing_discharge_6h"].sum()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    t_start = time.time()
    OUT_PROC_DIR.mkdir(parents=True, exist_ok=True)
    OUT_RES_DIR.mkdir(parents=True, exist_ok=True)

    cohort, flow = build_cohort_and_flow()
    imv_episodes, intub_df, niv_proc = extract_procedure_ventilation(cohort)
    chart_res = extract_chartevents_single_pass(cohort)
    lab_res = extract_labs_and_urine(cohort)
    states, features, extra_audit = build_states_and_features(
        cohort, imv_episodes, intub_df, niv_proc, chart_res, lab_res
    )

    cohort_audit_df = summarize_cohorts_and_audit(states)

    # Save processed parquets
    cohort.to_parquet(OUT_PROC_DIR / "r1_cohort.parquet", index=False)
    states.to_parquet(OUT_PROC_DIR / "r1_landmark_states.parquet", index=False)
    features.to_parquet(OUT_PROC_DIR / "r1_rolling_features.parquet", index=False)
    imv_episodes.to_parquet(OUT_PROC_DIR / "r1_imv_episodes.parquet", index=False)

    # Save aggregate-only audit artifacts
    cohort_audit_df.to_csv(OUT_RES_DIR / "r1_cohort_landmark_audit.csv", index=False)
    pd.DataFrame(extra_audit["lab_storetime_landmark_audit"]).to_csv(
        OUT_RES_DIR / "r1_lab_storetime_landmark_audit.csv", index=False
    )
    lab_res["lab_delay_audit"].to_csv(OUT_RES_DIR / "r1_lab_turnaround_by_feature.csv", index=False)

    # Era distribution for primary cohort
    prim = states[(states["r1_eligible"] == 1) & (states["is_unplanned_non_cvicu"] == 1)]
    era_table = (
        prim.groupby("anchor_year_group", sort=True)
        .agg(
            unique_patients=("case_id", "nunique"),
            eligible_assessments=("outcome_r1_6h", "size"),
            events_6h=("outcome_r1_6h", "sum"),
            events_proc_confirmed=("outcome_proc_confirmed_6h", "sum"),
            events_gap1h=("outcome_r1_gap1h", "sum"),
        )
        .reset_index()
    )
    era_table["event_rate_pct"] = 100.0 * era_table["events_6h"] / era_table["eligible_assessments"]
    era_table.to_csv(OUT_RES_DIR / "r1_primary_era_distribution.csv", index=False)

    # Landmark flow counts for Primary Unplanned Non-CVICU Cohort
    prim_all = states[states["is_unplanned_non_cvicu"] == 1]
    lm_flow_rows = []
    for lm in LANDMARKS:
        d = prim_all[prim_all["landmark_hours"] == lm]
        lm_flow_rows.append({
            "landmark_hours": int(lm),
            "present_unplanned_non_cvicu": int(len(d)),
            "excluded_no_bedside_vital_before_lm": int((d["has_vital_before_lm"] == 0).sum()),
            "excluded_prior_imv_proc_or_chart": int(((d["has_vital_before_lm"] == 1) & (d["prior_any_imv"] == 1)).sum()),
            "excluded_prior_imv_chart_only_caught": int(((d["has_vital_before_lm"] == 1) & (d["prior_chart_only_imv"] == 1)).sum()),
            "r1_eligible": int((d["r1_eligible"] == 1).sum()),
            "events_6h": int(d["outcome_r1_6h"].sum()),
            "event_rate_pct": float(100.0 * d.loc[d["r1_eligible"] == 1, "outcome_r1_6h"].mean()),
            "competing_death_6h": int(d["competing_death_6h"].sum()),
            "competing_discharge_6h": int(d["competing_discharge_6h"].sum()),
        })
    lm_flow_df = pd.DataFrame(lm_flow_rows)
    lm_flow_df.to_csv(OUT_RES_DIR / "r1_primary_landmark_flow.csv", index=False)

    summary = {
        "status": "PASS",
        "elapsed_seconds": round(time.time() - t_start, 2),
        "flow": flow,
        **extra_audit,
        "primary_landmark_flow": lm_flow_rows,
        "primary_era_distribution": era_table.to_dict(orient="records"),
    }
    (OUT_RES_DIR / "r1_extract_and_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("\n=== COHORT & LANDMARK AUDIT SUMMARY ===")
    print(cohort_audit_df[cohort_audit_df["landmark_hours"] == "pooled"].to_string(index=False))
    print("\n=== PRIMARY UNPLANNED NON-CVICU LANDMARK FLOW ===")
    print(lm_flow_df.to_string(index=False))
    print("\n=== PRIMARY ERA DISTRIBUTION ===")
    print(era_table.to_string(index=False))
    print("\n=== LAB STORETIME AUDIT ===")
    print(pd.DataFrame(extra_audit["lab_storetime_landmark_audit"]).to_string(index=False))
    print(f"\nDONE in {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()

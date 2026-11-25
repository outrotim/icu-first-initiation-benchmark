# Eligibility-Aware Dynamic Prediction of First ICU-Recorded Invasive Mechanical Ventilation (BMJHCI R1, Release v30-bmjhci-r1)

Reproducible analysis code and aggregate-only source tables for:

> *Eligibility-Aware Dynamic Prediction of First ICU-Recorded Invasive Mechanical Ventilation in the First 24 Hours Under Fixed Alert Quotas: A Retrospective MIMIC-IV Development and Temporal Validation Study*

## Contents
- `scripts/46_r1_extract_and_audit.py`: Cohort selection, OR-arrival audit, harmonised IMV onset, signed timing audit, and strict `storetime < landmark` 16-variable + non-invasive O2 extraction.
- `scripts/47_r1_models_baselines_ablation.py`: Chronological temporal validation (2008-2013 train / 2014-2016 val / 2017-2022 test), random split sensitivity, clinical baselines (Modified NEWS2, RR, SpO2), charting ablations, static-vs-dynamic ablation, and 1h/2h prediction gaps.
- `scripts/48_r1_operational_cooldown_subgroups.py`: 1,000 patient-cluster bootstrap CIs, paired contrasts, dual-denominator quota curves, closest-alert cooldown re-analysis, 2x2 factorial eligibility, competing-risk/DNI sensitivities, subgroups, and DCA.
- `scripts/49_r1_descriptives_and_importance.py`: Full cohort characteristics, variable dictionary, missingness audit, and grouped permutation importance.
- `scripts/43_generate_r1_bmjhci_figures.py`: Publication-grade Figures 1-3 and Figures S1-S8.
- `aggregate_results/`: Aggregate-only CSV/JSON outputs (zero patient-level data or identifiers).

## Data availability
MIMIC-IV v3.1 is accessible exclusively to credentialed researchers via PhysioNet under its data-use agreement. No patient-level records are distributed here.

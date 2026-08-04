# Eligibility-Aware Dynamic Prediction of First Invasive Mechanical Ventilation Under Fixed ICU Review Capacity

Minimal reproducibility assets for the manuscript:

> *Eligibility-Aware Dynamic Prediction of First Invasive Mechanical Ventilation Under Fixed ICU Review Capacity*

The study evaluates how routine ICU data can concentrate near-term first-ventilation events under limited review capacity. It reports capacity-response curves, patient-level pre-event reach, warning-time opportunities, repeated-alert burden, and care-context heterogeneity. Dynamic eligibility restricts risk ranking to patients who can still experience a first initiation.

## Repository contents

- `v28_figure_source_data.json`: aggregate, non-patient-level source data for the capacity-response and patient-level clinical figures.
- `reproduce_v28_figures.py`: regenerates the two clinical-information figures from the aggregate JSON file.
- `eligibility_aware_evaluation.py`: reusable fixed-budget target-by-eligibility evaluation and synthetic self-test.
- `study_specification.csv`: estimand, predictor, preprocessing, model, and uncertainty settings.
- `figure_source_data.json` and `reproduce_figures.py`: aggregate sources and redraw script for the supporting eligibility analysis.
- `requirements.txt`: minimal Python dependencies.
- `LICENSE`: MIT for code and CC BY 4.0 for aggregate data and parameter tables.

## Data availability

The released JSON contains only aggregate counts, estimates, uncertainty intervals, and descriptive summaries. It does not contain patient-level records, predictions, timestamps, split manifests, hospital identifiers, or restricted intermediate data.

The study used MIMIC-IV version 3.1. MIMIC-IV is available only to credentialed users through PhysioNet under its data-use agreement and cannot be redistributed here.

## Reproduce the clinical-information figures

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce_v28_figures.py --output-dir reproduced_figures
```

The command writes PNG and PDF versions of the capacity-response figure and the patient-level clinical-context figure.

## Verify the supporting AUROC identity

```bash
python eligibility_aware_evaluation.py --self-test
```

To apply the framework to authorized local data, follow the column specification printed by `eligibility_aware_evaluation.py`. Local patient-level inputs must never be committed to this repository.

## Scope

These assets support retrospective model evaluation. They do not provide a fitted bedside product, establish that displaying an alert changes care or outcomes, or authorize redistribution of MIMIC-IV data.

## License

Source code is released under the MIT License. `v28_figure_source_data.json`, `figure_source_data.json`, and `study_specification.csv` are released under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

## Citation

Citation details will be added after publication. Until then, cite the manuscript title above and this repository:

`https://github.com/outrotim/icu-first-initiation-benchmark`

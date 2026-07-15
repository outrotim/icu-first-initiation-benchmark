# ICU First-Initiation Eligibility Benchmark

Minimal reproducibility assets for:

> *Prevalent Ventilation State and History Inflate Apparent Discrimination for First-Initiation Prognosis: A Fixed-Score Empirical Benchmark*

This repository does not propose a deployable clinical prediction model. It provides a reusable eligibility and evaluation framework for separating future first invasive-ventilation initiation from recognition of active or prior ventilation.

## Repository contents

- `incident_at_risk_auc.py`: generic fixed-score AUROC positive-mixture decomposition and self-test.
- `model_specification.csv`: locked logistic-regression, LightGBM, and preprocessing settings used in the benchmark.
- `figure_source_data.json`: aggregate, non-patient-level source data for the three main figures.
- `reproduce_figures.py`: regenerates the three main figures from the aggregate JSON file.
- `requirements.txt`: minimal Python dependencies.
- `LICENSE`: MIT for code; CC BY 4.0 for aggregate data and parameter tables.

## Data availability

The aggregate numerical results needed to redraw the main figures are openly available in this repository. No patient-level records, predictions, timestamps, split manifests, hospital identifiers, or restricted intermediate datasets are included.

The study used MIMIC-IV v3.1 and eICU Collaborative Research Database v2.0. These credentialed-access datasets cannot be redistributed here. Eligible researchers can request access directly through PhysioNet and must comply with the applicable credentialing, training, and data-use requirements.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Verify the AUROC identity

Run the built-in synthetic-data self-test:

```bash
python incident_at_risk_auc.py --self-test
```

To apply the decomposition to authorized local data, prepare a CSV containing only these columns:

- `event_group`: `negative`, `incident`, `active_support`, or `resolved_history`.
- `y_pred`: a fixed model score, evaluated without refitting between positive-state comparisons.

Then run:

```bash
python incident_at_risk_auc.py --input authorized_predictions.csv --output decomposition.json
```

The input file remains local and is never included in this repository.

## Reproduce the main figures

```bash
python reproduce_figures.py --output-dir reproduced_figures
```

The command writes PNG and PDF versions of Figures 1-3 using only `figure_source_data.json`.

## Interpretation caveats

- The estimand is future first invasive-ventilation initiation among patients still eligible to experience that event at the landmark.
- Inflation refers to the deliberately constructed fixed-score stress test; it is not an estimate of how common target mixing is in published studies.
- The eICU analysis provides directional consistency under a non-equivalent proxy endpoint. It is not external validation of a transported MIMIC model.
- The CRRT comparison is an endpoint-specific contrast and does not establish a universal correction.
- The released estimator settings are for methodological transparency. No fitted clinical model or deployment claim is provided.

## License

Source code is released under the MIT License. `figure_source_data.json` and `model_specification.csv` are released under the Creative Commons Attribution 4.0 International License (CC BY 4.0). See `LICENSE` for scope.

## Citation

Citation details will be added after publication. Until then, cite the manuscript title above and this repository URL:

`https://github.com/outrotim/icu-first-initiation-benchmark`

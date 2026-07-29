# Eligibility Before Prediction

Minimal reproducibility assets for:

> *Eligibility Before Prediction: Dynamic Warning for First Invasive Mechanical Ventilation in Critical Care*

This repository accompanies a retrospective predictive modeling study of
dynamic warning for first initiation of invasive mechanical ventilation. The
study treats target construction and eligibility filtering as linked parts of
alert design: a first-event model should predict the future event and rank only
patients who remain able to experience it.

## Repository contents

- `study_specification.csv`: estimand, split, preprocessing, model, uncertainty,
  and predictor dictionary.
- `eligibility_aware_evaluation.py`: reusable fixed-budget 2 x 2 evaluation for
  incident versus history-inclusive targets and eligibility filtering on versus
  off.
- `figure_source_data.json`: aggregate, non-patient-level source data for the
  three main figures.
- `reproduce_figures.py`: redraws the three main figures from the aggregate JSON.
- `requirements.txt`: minimal Python environment.
- `LICENSE`: MIT for code and CC BY 4.0 for aggregate data and specifications.

## Data availability

The aggregate numerical results needed to redraw the main figures are openly
available here. No patient-level records, predictions, timestamps, split
manifests, hospital identifiers, fitted models, or restricted intermediate
datasets are included.

The study used MIMIC-IV version 3.1. This credentialed-access dataset cannot be
redistributed. Eligible researchers may request access through PhysioNet and
must comply with its credentialing, training, and data-use requirements.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Verify the alert-allocation framework

Run the synthetic-data self-test:

```bash
python eligibility_aware_evaluation.py --self-test
```

To evaluate authorized local predictions, prepare a CSV with:

- `patient_id`: patient-level clustering identifier.
- `state`: `never_exposed`, `active`, or `resolved`.
- `incident_event`: 1 for a future first initiation in the horizon, otherwise 0.
- `incident_score`: score from the incident-target model.
- `history_score`: score from the history-inclusive model.

Then run:

```bash
python eligibility_aware_evaluation.py \
  --input authorized_predictions.csv \
  --budget 0.05 \
  --output factorial_results.csv
```

The input remains local and is never written to this repository.

## Reproduce the main figures

```bash
python reproduce_figures.py --output-dir reproduced_figures
```

The command writes PNG and PDF versions of Figures 1-3 using only aggregate
source data.

## Interpretation

The estimand is ICU-recorded first invasive mechanical ventilation initiation
after a landmark among patients never previously exposed at that landmark. The
released framework evaluates alert allocation; it is not a deployable model.
Clinical transportability and prospective utility require external validation.

## License and citation

Code is released under the MIT License. `figure_source_data.json` and
`study_specification.csv` are released under CC BY 4.0. See `LICENSE`.

Citation details will be added after publication. Until then, cite the manuscript
title above and the archived release linked from the repository release page.

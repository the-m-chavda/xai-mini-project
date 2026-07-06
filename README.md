# XAI Mini Project: AIFB R-GCN with Global DL Explanations

This repository contains the final AIFB-only version of the XAI mini project.
It implements Strategy 3:

1. train an R-GCN on the AIFB RDF knowledge graph,
2. convert the model predictions into positive/negative examples,
3. learn global description-logic explanations with Ontolearn/CELOE.

The final report uses two AIFB experiments only:

| Experiment | Config | Purpose |
| --- | --- | --- |
| AIFB best model | `configs/aifb.yaml` | RDF-neighborhood features, residual 2-layer R-GCN, early stopping |
| AIFB baseline | `configs/aifb_baseline.yaml` | node-id embeddings, original simpler R-GCN baseline |

## Repository Layout

```text
configs/
  aifb.yaml               # final model used in the report
  aifb_baseline.yaml      # node-id baseline for comparison
data/aifb/                # AIFB RDF graph and label splits
src/xai_miniproject/      # package source code
tests/                    # lightweight regression tests
requirements.txt          # core dependencies
requirements-ontolearn.txt # optional CELOE dependencies
```

Generated outputs are ignored by git and are written to:

```text
artifacts/aifb/
artifacts/aifb_baseline/
logs/
```

Important generated files:

```text
dataset_stats.json
metrics.json
predictions.csv
model.pt
learning_problems.json
explanation_results.json
explanation_results.csv
```

## Setup

Use Python 3.10 or newer. Python 3.12 has been tested on macOS.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Conda alternative:

```bash
conda create -n xai python=3.12
conda activate xai
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Optional: Install Ontolearn / CELOE

The classifier can be trained without Ontolearn. For the report-style symbolic explanations, install
Ontolearn as follows:

```bash
python -m pip install -r requirements-ontolearn.txt
python -m pip install --no-deps ontolearn==0.10.0
```

The two-step install avoids Ontolearn's old pinned `python-sat` dependency, which can fail to compile on
some macOS/Python combinations. If Ontolearn is unavailable, the code falls back to a simple feature-based
explainer so the pipeline still runs end-to-end.

## Quick Reproduction

Run the final AIFB model used in the report:

```bash
xai-mini --config configs/aifb.yaml run-all
```

Run the node-id baseline:

```bash
xai-mini --config configs/aifb_baseline.yaml train
```

Run individual stages for the final model:

```bash
xai-mini --config configs/aifb.yaml analyze
xai-mini --config configs/aifb.yaml train
xai-mini --config configs/aifb.yaml explain
```

If the package is not installed with `pip install -e .`, use the module form:

```bash
PYTHONPATH=src python -m xai_miniproject.cli --config configs/aifb.yaml run-all
PYTHONPATH=src python -m xai_miniproject.cli --config configs/aifb_baseline.yaml train
```

Add `--no-log` for quick local debugging:

```bash
xai-mini --config configs/aifb.yaml --no-log train
```

## Expected Results

Results can vary slightly by platform and PyTorch version. On the tested CPU environment:

| Experiment | Initial features | Test accuracy | Test macro-F1 |
| --- | --- | ---: | ---: |
| AIFB baseline | node-id embedding | about 86.1% | about 0.842 |
| AIFB best model | RDF-neighborhood features | about 94.4% | about 0.912 |

The final model normally reaches 34/36 correct test predictions. It writes explanations for the four
predicted research groups under `artifacts/aifb/explanation_results.json`.

## Development Checks

```bash
python -m compileall -q src
python -m pytest
```

If the package has not been installed:

```bash
PYTHONPATH=src python -m pytest
```

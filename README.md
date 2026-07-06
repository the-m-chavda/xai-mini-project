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

**One command — runs everything and verifies results:**

```bash
bash reproduce.sh
```

Expected output (truncated):

```
=== Checking Python version ===
[OK]   Python 3.x (>= 3.10)
=== Installing dependencies ===
[OK]   All dependencies installed (including Ontolearn/CELOE)
[OK]   Package installed (xai-mini command available)
=== Verifying Ontolearn / CELOE ===
[OK]   Ontolearn available — CELOE will be used for explanations
=== Checking dataset files ===
[OK]   data/aifb/aifbfixed_complete.n3
[OK]   data/aifb/trainingSet.tsv
...
=== Running full pipeline (analyze + train + explain) ===
  This takes ~1–2 minutes on CPU.
epoch=025 loss=0.007 train_acc=0.986 test_acc=0.944 ...
Early stopping at epoch 47 with best_val_f1=0.889
=== Verifying results ===
[OK]   Test accuracy = 0.944  (expected ~94.4%)
[OK]   Test macro-F1 = 0.912  (expected ~0.912)
[OK]   Explanation results: 4 classes explained
════════════════════════════════════════
  Passed : 15 / Failed : 0 — All checks passed.
════════════════════════════════════════
```

Or run the pipeline manually:

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

## Pipeline Walkthrough

### Raw Data

| File | Description |
| --- | --- |
| `aifbfixed_complete.n3` | Full RDF knowledge graph (29,226 triples) |
| `trainingSet.tsv` | 140 labelled persons |
| `testSet.tsv` | 36 labelled persons |
| `completeDataset.tsv` | All 176 labelled persons |

### Stage 1 — Graph Construction (`data.py`)

All 176 person URIs from `completeDataset.tsv` are added to `node_to_id` first (guaranteed low IDs).
RDF triples are then processed in sorted order for determinism:

- Skip if predicate ∈ `{affiliation, employs}` — label leakage prevention
- Skip if subject or object is a Literal (`include_literal_nodes=false`)
- Add subject and object to `node_to_id`
- Add edge `(src, dst, relation_id)`
- Add inverse edge `(dst, src, relation_id__inverse)` — `add_inverse_edges=true`

**Result:** 2,835 nodes · 44 relation types (22 original + 22 inverse) · 40,676 edges

### Stage 2 — Feature Extraction (`data.py:build_node_feature_lists`)

For each RDF triple (sorted), features are added to the subject node:

| Triple pattern | Feature added to subject |
| --- | --- |
| `predicate = rdf:type` | `type::<TypeURI>` |
| `object is Literal` | `literal_predicate::<pred>`, `literal_datatype::<dtype>`, `literal_value::<val>` (if ≤ 64 chars) |
| `object is Resource` | `exists::<pred>::<ObjectType>` (looks up rdf:type of object) |
| node has no features | fallback: `kind::resource_without_type` / `kind::blank_node` / `kind::literal` |

**Result:** 4,017 unique feature strings → sparse binary matrix of shape `[2835 × 4017]` (multi-hot encoding)

### Stage 3 — Validation Split (`train.py:split_validation`)

140 training persons are split stratified per class (seed=42):

| Split | Size | Role |
| --- | --- | --- |
| fit set | ~119 persons | gradient updates (loss computed here) |
| val set | ~21 persons | early stopping only, never seen by optimiser |
| test set | 36 persons | held out entirely until final evaluation |

### Stage 4 — R-GCN Training (`model.py`, `train.py`)

```
INPUT PROJECTION (once, before layers):
  [2835 × 4017]  multi-hot features
       ↓  nn.Linear(4017 → 64, bias=False)
  [2835 × 64]    embedding matrix x

LAYER 1 — RGCNLayer(64 → 128):
  for each of 44 relations:
    messages = x[src] @ W_r          shape [num_edges_r × 128]
    messages /= degree[dst]          degree normalisation
    aggregate → out via index_add_
  out += W_self @ x + bias
  out = ReLU(Dropout(out))           dropout=0.5
  out = LayerNorm(out)               use_layer_norm=true
  (no residual: 64 ≠ 128)
       ↓  [2835 × 128]  h1

LAYER 2 — RGCNLayer(128 → 128):
  same message passing
  out = ReLU(Dropout(out))
  out = LayerNorm(out)
  out = out + h1                     residual (shapes match: 128 = 128)
       ↓  [2835 × 128]  h2

CLASSIFIER:
  nn.Linear(128 → 4)  →  [2835 × 4] logits
```

- **Loss:** `F.cross_entropy(logits[fit_idx], fit_labels)`
- **Optimiser:** Adam, lr=0.005, weight_decay=0.001
- **Early stopping:** monitors val macro-F1, patience=30, restores best checkpoint

**Final metrics** (restored best model — best val F1=0.889 at epoch 25):

| Split | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Train | 98.6% | 0.984 |
| Test | 94.4% | 0.912 |

2 test errors out of 36.

### Stage 5 — Learning Problem Construction (`explain.py:build_learning_problems`)

Loads `predictions.csv` (all 176 persons, train+test). For each of the 4 research groups:

- **positives** = entities where `pred_label == this class`, shuffled (seed=42), capped at 40
- **negatives** = entities where `pred_label != this class`, shuffled (seed=42), capped at 80

> Uses **model predictions**, not ground truth — the explanation reflects what the model learned,
> not just data patterns.

### Stage 6 — Concept Learning (`explain.py:run_baseline_explainer`)

For each learning problem, neighbourhood features are collected for all example entities
(same feature strings as Stage 2). Each unique feature string is scored:

- `precision = tp / (tp + fp)`
- `recall = tp / (tp + fn)`
- `F1 = harmonic mean`

Features are sorted by (F1 desc, precision desc, recall desc) and the top 3 are returned.

**Results:**

| Research Group | Top feature | F1 | Precision | Recall |
| --- | --- | ---: | ---: | ---: |
| Business Info | `HAS_LITERAL fax` | 0.500 | 0.333 | 1.000 |
| Efficient Algs | `EXISTS worksAtProject.Project` | 0.493 | 0.391 | 0.667 |
| Knowledge Mgmt | `EXISTS publication.TechnicalReport` | 0.675 | 0.675 | 0.675 |
| Complexity Mgmt | `EXISTS⁻¹ member.ResearchGroup` | 0.462 | 0.316 | 0.857 |

> Ontolearn/CELOE is not installed by default — the baseline feature scorer runs for all 4 classes.
> CELOE would search conjunctions/disjunctions rather than single features.

### Full Data Flow

```
aifb.n3 ──────────────────────────────────────────────────────────────┐
                                                                      │
     Stage 1           Stage 2            Stage 3                     │
  Build Graph  →  Extract Features  →  Split Val/Fit                  │
  2835 nodes        [2835×4017]          fit=119                      │
  40676 edges       multi-hot            val=21                       │
  44 relations                           test=36 (held out)           │
       │                  │                                           │
       └──────────────────┘                                           │
                 │                                                    │
              Stage 4                                                 │
          R-GCN Training                                              │
          Linear(4017→64)                                             │
          RGCNLayer(64→128)                                           │
          RGCNLayer(128→128) + residual                               │
          Linear(128→4) → logits                                      │
          Early stop @ epoch 25                                       │
          test acc = 94.4%                                            │
                 │                                                    │
          predictions.csv  <────────────────────────────── aifb.n3 ───┘
          (176 persons,                                        │
           pred_label + confidence)                            │
                 │                                             │
              Stage 5                                          │
       Build Learning Problems                                 │
       pos = predicted members                                 │
       neg = predicted non-members                             │
       (capped: 40 pos / 80 neg)                               │
                 │                                             │
              Stage 6                                          │
       Baseline Concept Learner                                │
       neighbourhood_features() <──────────────────────────────┘
       score each feature by F1
       return top-3 per class
                 │
       explanation_results.json
       "EXISTS publication.TechnicalReport"  → KM  F1=0.675
       "EXISTS worksAtProject.Project"       → EA  F1=0.493
       "HAS_LITERAL fax"                    → BI  F1=0.500
       "EXISTS⁻¹ member.ResearchGroup"      → CM  F1=0.462
```

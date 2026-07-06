# XAI Mini Project — Onboarding Guide for New Members

> **⚠️ Historical document — does not reflect the current codebase.** This was written during an
> earlier phase of the project that also explored the MUTAG dataset and a feature scheme called
> `rdf_types`. The final repository is **AIFB-only** (see `README.md`) and the shipped feature scheme
> is `rdf_neighborhood` — a superset of `rdf_types` that additionally includes typed-neighbourhood
> (`exists::<pred>::<ObjectType>`) features (see `README.md`'s "Pipeline Walkthrough" and the report for
> the current, accurate description). `configs/mutag.yaml` and `configs/mutag_node_id.yaml` are dead
> leftovers from that phase: `data/mutag/` no longer exists in this repo, and `initial_features:
> rdf_types` is no longer a value the code accepts (`train.py` only accepts `node_id` or
> `rdf_neighborhood`). Kept here for historical context on why `rdf_neighborhood` was chosen over
> per-node ID embeddings — treat `README.md` and the report as the source of truth for anything that
> conflicts with what's below.

> : This document follows the chronological order of problem discovery → solution design. It covers project background, datasets, system architecture, the core innovation, and the exact code changes made. 

---

## Table of Contents

1. [Project Background: What Are We Doing?](#1-project-background-what-are-we-doing)
2. [Datasets](#2-datasets)
3. [System Architecture](#3-system-architecture)
4. [Phase 1: AIFB + node_id — It Runs, but Problems Emerge](#4-phase-1-aifb--node_id--it-runs-but-problems-emerge)
5. [Phase 2: Switching to MUTAG — The Fundamental Flaw Is Exposed](#5-phase-2-switching-to-mutag--the-fundamental-flaw-is-exposed)
6. [Phase 3: Identifying the Root Cause — The Essential Difference Between Two Initialization Approaches](#6-phase-3-identifying-the-root-cause--the-essential-difference-between-two-initialization-approaches)
7. [Phase 4: Implementing the rdf_types Approach](#7-phase-4-implementing-the-rdf_types-approach)
8. [Controlled Experiment: node_id vs rdf_types](#8-controlled-experiment-node_id-vs-rdf_types)
9. [Exact Code Changes](#9-exact-code-changes)
10. [Key Focus Points](#10-key-focus-points)
11. [Quick Start](#11-quick-start)

---

## 1. Project Background: What Are We Doing?

### 1.1 Three Strategies of Explainable AI

Deep neural networks like GNNs achieve excellent performance but are **black boxes**: they give you a prediction but cannot explain **why**. GNN explainability is categorized into three strategies:

| Strategy | Granularity | Method | Our Choice |
|---|---|---|---|
| Strategy 1 | Instance-level | Identify key input subgraphs for a single prediction (e.g., GNNExplainer) | ❌ |
| Strategy 2 | Instance-level | Use knowledge distillation or surrogate models per prediction | ❌ |
| Strategy 3 | **Global-level** | Extract human-readable symbolic rules from the model's overall behavior | ✅ |

### 1.2 Our Task

```
Input:  RDF knowledge graph + class labels for some nodes
        ↓
Step 1: Train R-GCN for node classification
        ↓
Step 2: GNN predictions → CELOE concept learner → Description Logic formulas
        ↓
Output: Human-readable rules explaining "what features characterize each class"
```

For example, CELOE outputs `∀ member⁻.(¬Project)` → "people in this research group are not members of any Project."

---

## 2. Datasets

### 2.1 AIFB

An academic knowledge graph describing the AIFB research institute at the University of Karlsruhe, Germany.

```
Statistics:
├── RDF triples:      29,226
├── Model nodes:       2,835 (Person 1,045 / Publication 1,222 / InProceedings 687 / ...)
├── Model edges:      40,676 (includes inverse relations)
├── Relation types:       44
├── Target entities:     176 researchers
└── Labels:                4 research groups
    ├── Business Information and Communication Systems (73 people)
    ├── Knowledge Management (60 people)
    ├── Efficient Algorithms (28 people)
    └── Complexity Management (15 people)
```

**Key characteristic**: Nodes only have URI strings — no initial feature vectors (unlike Cora with bag-of-words features). AIFB is heterogeneous: it contains Person, Publication, Project, and other node types.

### 2.2 MUTAG

A chemical compound mutagenicity dataset.

```
Statistics:
├── RDF triples:      ~74,000
├── Model nodes:       ~10,000 (188 Compounds / Atoms / Bonds ...)
├── Relation types:     multiple (bond types, atom properties, etc.)
├── Target entities:     188 compounds
└── Labels:                2 classes (Mutagenic / Non-mutagenic)
```

**Key differences from AIFB**:

| Dimension | AIFB | MUTAG |
|---|---|---|
| Graph structure | One large connected graph | 188 independent molecular subgraphs |
| Label-leaking predicates | `affiliation`, `employs` | `isMutagenic` |
| Literal properties | Titles, years, etc. (8,705) | Atomic charges, bond energies, etc. |
| Node types | Person, Publication... | Compound, Atom, Bond... |
| Node feature requirement | Moderate | **High** (molecular subgraphs are isolated and must be distinguished by features) |

### 2.3 What RDF Triples Look Like

```turtle
# A researcher published a conference paper
:person123    :publication    :paper456 .
:paper456     rdf:type        :InProceedings .
:paper456     :author         :person123 .
:paper456     :title          "Graph Neural Networks for Knowledge Graphs" .

# Label data (TSV file)
person                                              label_affiliation
http://.../id1909instance                           http://.../id1instance   # Belongs to Group 1
```

**The label leakage problem**: The original RDF contains triples like `:person :affiliation :group1`. If not excluded, the model simply reads this edge and knows the answer — it learns nothing. Hence `exclude_predicates` removes them.

---

## 3. System Architecture

```
configs/*.yaml
      │
      ▼
┌─────────┐    ┌──────────┐    ┌──────────┐    ┌───────────┐
│config.py│    │ data.py  │    │model.py  │    │ train.py  │
│YAML→    │───→│RDF parse │───→│R-GCN     │───→│Train+eval │
│dataclass│    │graph build│   │classifier│    │predictions│
└─────────┘    │features   │    └──────────┘    └─────┬─────┘
               └──────────┘                          │
                                                     ▼
                                             ┌──────────────┐
                                             │ explain.py   │
                                             │ CELOE learner│
                                             │ → DL rules   │
                                             └──────────────┘
```

**Core file responsibilities**:

| File | Responsibility |
|---|---|
| `config.py` | YAML config → immutable dataclass, controls all behavior |
| `data.py` | RDF parsing, heterogeneous graph construction, **node feature extraction** (where the innovation lives) |
| `model.py` | R-GCN model: message passing + classifier, supports two initialization modes |
| `train.py` | Training loop, validation split, early stopping, evaluation |
| `explain.py` | Converts GNN predictions to positive/negative examples → CELOE learns DL rules |
| `cli.py` | CLI entry point (`xai-mini analyze / train / explain / run-all`) |

---

## 4. Phase 1: AIFB + node_id — It Runs, but Problems Emerge

### 4.1 Running

```bash
xai-mini --config configs/aifb.yaml run-all
```

The AIFB config has **no** `initial_features` field → defaults to `node_id` mode.

### 4.2 Results

```
epoch=001 loss=1.4317 train_acc=0.236 test_acc=0.167 train_f1=0.312 test_f1=0.071
epoch=025 loss=0.0019 train_acc=1.000 test_acc=0.889 train_f1=1.000 test_f1=0.875
...
epoch=250 loss=0.0016 train_acc=1.000 test_acc=0.861 train_f1=1.000 test_f1=0.842

Final: test_accuracy=0.861, test_macro_f1=0.842
```

### 4.3 Two Observed Problems

**Problem 1: Severe overfitting**

Training accuracy hits 100% by epoch 25, but test accuracy peaks at 88.9% and eventually settles at 86.1%. There is a ~14 percentage point gap between training and testing. This indicates the model is **memorizing training nodes** rather than learning **transferable patterns**.

**Problem 2: Poor explanation quality for small classes**

| Research Group | Samples | CELOE Best F1 |
|---|---|---|
| Business Information | 73 | 0.667 |
| Knowledge Management | 60 | 0.506 |
| Efficient Algorithms | 28 | 0.429 |
| Complexity Management | 15 | **0.321** |

Complexity Management has only 15 people. The best rule CELOE finds (`≤ 5 publication.(¬InProceedings)`) has an F1 of 0.32 — essentially random.

### 4.4 What Is the node_id Mode?

```python
# model.py — RGCNClassifier initialization
self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)

# This is a lookup table:
# Node 0 → random vector [0.03, -0.12, 0.07, ...]
# Node 1 → random vector [0.11,  0.04, -0.02, ...]
# Node 2 → random vector [-0.05, 0.09, 0.01, ...]
# ...
```

Every node gets a **completely independent, from-scratch-trained vector**. The initial positions of Node 0 and Node 1 in embedding space are purely random, regardless of their roles in the graph. The model must learn each node's representation independently through backpropagation.

**This barely works on AIFB** because it is a single large connected graph where 176 target nodes share the same graph structure, and message passing can propagate information. But even then, the model leans toward memorization rather than generalization.

---

## 5. Phase 2: Switching to MUTAG — The Fundamental Flaw Is Exposed

### 5.1 Attempting MUTAG with node_id

To test the node_id approach's generalization ability on heterogeneous datasets, we created `configs/mutag_node_id.yaml`:

```yaml
model:
  name: rgcn
  initial_features: node_id    # ← Explicitly use node_id mode
  embedding_dim: 64
  ...
```

### 5.2 Why node_id Fails on MUTAG

MUTAG has a fundamentally different graph structure from AIFB:

```
AIFB: One large connected graph         MUTAG: 188 isolated molecular subgraphs
┌──────────────────────┐              ┌────┐ ┌────┐ ┌────┐
│   Person ── Paper     │              │Mol1│ │Mol2│ │Mol3│ ...
│     │        │        │              └────┘ └────┘ └────┘
│   Person ── Paper     │              No edges between them
│     │        │        │
│   ...  ...  ...       │
└──────────────────────┘
```

In AIFB, even without semantic features in node_id, message passing propagates information through the graph's connectivity. But in MUTAG:
- The 188 compounds are **isolated from each other** — messages cannot propagate between molecules
- Each compound's node_id embedding is an independent random vector, unable to learn shared patterns across molecules
- The model must learn on 188 isolated subgraphs separately → **severely insufficient information**

**Result**: node_id performs poorly on MUTAG because node ID embeddings cannot share any knowledge across different molecules. Two structurally similar molecules (e.g., both containing nitrogen heterocycles) have completely unrelated embeddings in the initial space.

### 5.3 The Core Contradiction, Clarified

```
The fundamental flaw of node_id mode:
┌──────────────────────────────────────────────────────┐
│ nn.Embedding gives each node an independent vector    │
│ → Same-type nodes (two PhDStudents) start unrelated   │
│ → Same-structure subgraphs (two benzene rings) start  │
│   unrelated                                          │
│ → Model must learn all node semantics from scratch    │
│ → On small datasets or isolated subgraphs, there is   │
│   not enough signal                                  │
└──────────────────────────────────────────────────────┘
```

---

## 6. Phase 3: Identifying the Root Cause — The Essential Difference Between Two Initialization Approaches

### 6.1 The Root Cause

Go back to the code and look at what feeds into the GNN:

**node_id approach** (original):
```python
# Each node gets an independent trainable vector
x = self.node_embeddings(node_ids)  # shape: (num_nodes, embedding_dim)
# Node 0 → vec_0, Node 1 → vec_1, Node 2 → vec_2, ...
# vec_0, vec_1, vec_2 share no parameters
```

**The problem**: If the graph has 1,000 nodes of type `PhDStudent`, each learns a completely independent 64-dimensional vector. The only connection between them comes from R-GCN message passing (two hops). But if two PhDStudents are far apart in the graph (e.g., in different connected components), messages never reach.

### 6.2 The Core Insight

> **A node's initial representation should not be determined by its ID (which is arbitrary), but by its RDF properties (which carry semantics).**

In an RDF graph, every node has rich property information:
- `rdf:type` → what type the node is (PhDStudent, Article, Project...)
- Literal properties (title, year, abstract, charge...)
- Literal datatypes (xsd:integer, xsd:string...)
- Literal values ("Graph Neural Networks", "2020"...)

Two nodes that are both PhDStudents should start from **similar initial embeddings** because they share the `type::PhDStudent` feature.

### 6.3 Visual Comparison of the Two Approaches

```
node_id (independent per-node vectors):    rdf_types (shared feature representation):

Node_0 (PhDStudent)   → [0.03, -0.12]      Node_0: {type::PhDStudent}
Node_1 (PhDStudent)   → [0.11,  0.04]                ↓
Node_2 (Lecturer)     → [-0.05, 0.09]     Feature vocab: [type::PhDStudent,
                         ↑                 type::Lecturer, type::Article,
                         No shared          ...]
                         knowledge                  ↓
                                           Node_0 → [1, 0, 0, ...] → Linear → [0.5, -0.1]
                                           Node_1 → [1, 0, 0, ...] → Linear → [0.5, -0.1]
                                           Node_2 → [0, 1, 0, ...] → Linear → [0.2,  0.3]
                                                     ↑
                                    Same-type nodes → same initial vector → shared knowledge
```

**The critical difference**:
- `node_id`: Node_0 and Node_1 learn two independent embeddings with no shared parameters
- `rdf_types`: Node_0 and Node_1 (both PhDStudents) are projected through the **same** `Linear` layer, producing **identical** initial embeddings; the model learns not "what Node_0 is" but "what behavioral patterns PhDStudent-type nodes typically exhibit"

---

## 7. Phase 4: Implementing the rdf_types Approach

### 7.1 Design

The core change is a single modification: **replace independent ID embeddings with shared feature projection**.

```
Before (node_id):
  Node ID → nn.Embedding(num_nodes, dim) → initial vector → R-GCN → classification

After (rdf_types):
  Node URI → extract RDF features → multi-hot vector → nn.Linear(feat_dim, dim) → initial vector → R-GCN → classification
```

### 7.2 data.py — New Function `build_node_feature_lists()`

This function iterates over all triples and collects features for each node. There are four feature categories:

```python
# 1. Type features — from rdf:type
#    "type::<TypeURI>"
#    Example: "type::http://...#PhDStudent"

# 2. Literal predicate features — names of literal properties on the node
#    "literal_predicate::<PredicateURI>"
#    Example: "literal_predicate::http://...#charge"

# 3. Literal datatype features — datatypes of literal values
#    "literal_datatype::<DatatypeURI>"
#    Example: "literal_datatype::http://...#float"

# 4. Literal value features — short literal values (≤ 64 chars)
#    "literal_value::<lexical_value>"
#    Example: "literal_value::0.35"

# 5. Fallback features — for nodes without any features
#    "kind::resource_without_type"  (resource nodes without a type)
#    "kind::blank_node"             (blank nodes)
#    "kind::literal"                (literal nodes)
```

Key implementation (`data.py:183-221`):

```python
def build_node_feature_lists(data, config):
    feature_to_id: dict[str, int] = {}
    feature_sets: list[set[int]] = [set() for _ in range(data.num_nodes)]

    for subject, predicate, obj in sorted_graph_triples(data.graph):
        if predicate == RDF.type:
            add_feature(subject, f"type::{obj}")              # Type feature
        elif isinstance(obj, Literal):
            add_feature(subject, f"literal_predicate::{predicate}")  # Property feature
            if obj.datatype:
                add_feature(subject, f"literal_datatype::{obj.datatype}")  # Datatype
            if len(str(obj)) <= 64:
                add_feature(subject, f"literal_value::{obj}")  # Specific value

    # Fallback for featureless nodes
    for node_id, node_key in enumerate(data.id_to_node):
        if not feature_sets[node_id]:
            if node_key.startswith("http"):
                feature_sets[node_id].add(feature_id("kind::resource_without_type"))
            ...

    return feature_sets, feature_to_id
```

**Important detail**: Features are only extracted from triples where the node is the **subject**. This means features describe "what properties does this node have / what type is it," not "who points to this node." This avoids indirectly introducing label leakage.

### 7.3 model.py — Dual-Path Architecture

`RGCNClassifier.__init__` selects between two paths based on whether `input_feature_dim` is provided:

```python
class RGCNClassifier(nn.Module):
    def __init__(self, ..., input_feature_dim: int | None = None):
        if input_feature_dim is None:
            # Path A: node_id mode (original)
            self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)
            self.input_projection = None
        else:
            # Path B: rdf_types mode (innovation)
            self.node_embeddings = None
            self.input_projection = nn.Linear(input_feature_dim, embedding_dim, bias=False)

    def forward(self, graph, input_features=None):
        if self.input_projection is not None:
            x = self.input_projection(input_features)  # Shared features → projection
        else:
            x = self.node_embeddings(node_ids)          # Independent ID embeddings
        x = self.layer1(x, graph)
        x = self.layer2(x, graph)
        return self.classifier(x)
```

### 7.4 train.py — Config-Driven Feature Construction

```python
def run_training(config):
    ...
    if config.model.initial_features == "rdf_types":
        feature_lists, feature_mapping = build_node_feature_lists(data, config.dataset)
        input_feature_dim = len(feature_mapping)
        input_features = torch.zeros((data.num_nodes, input_feature_dim))
        for node_id, feature_ids in enumerate(feature_lists):
            input_features[node_id, feature_ids] = 1.0
    elif config.model.initial_features != "node_id":
        raise ValueError(...)
    # node_id mode: input_features is None, input_feature_dim is None
```

### 7.5 Config File Control

Three config files form a controlled comparison matrix:

```yaml
# configs/aifb.yaml — AIFB + node_id (original approach)
model:
  # initial_features omitted → defaults to "node_id"

# configs/mutag_node_id.yaml — MUTAG + node_id (comparison baseline)
model:
  initial_features: node_id

# configs/mutag.yaml — MUTAG + rdf_types (innovation)
model:
  initial_features: rdf_types
  include_literal_nodes: true   # Enable literal properties
  validation_fraction: 0.2      # 20% of training set used for validation
  early_stopping_patience: 50   # Stop if no improvement for 50 epochs
```

**Why does MUTAG need validation and early stopping?** Because MUTAG's node_id mode easily overfits to the 188 training compounds. Validation + early stopping prevents overfitting in the node_id baseline, and ensures both approaches are compared under fair conditions.

---

## 8. Controlled Experiment: node_id vs rdf_types

### 8.1 Experimental Design

```
Experiment A: configs/mutag_node_id.yaml  → initial_features: node_id
Experiment B: configs/mutag.yaml          → initial_features: rdf_types

Comparison metrics:
├── test_accuracy / test_macro_f1
├── Training convergence speed
├── Overfitting gap (train_acc − test_acc)
├── Input feature dimensionality (rdf_types)
└── CELOE explanation F1 (if Ontolearn is installed)
```

### 8.2 Expected Results

| Metric | node_id | rdf_types | Reason |
|---|---|---|---|
| test_accuracy | Lower | **Higher** | Shared features → same-type nodes share parameters → better generalization |
| train − test gap | Large | **Smaller** | Cannot memorize IDs, must learn semantic patterns |
| Convergence speed | Slow | **Faster** | Features provide a good initialization direction |
| Feature dim | None (ID only) | ~hundreds | Multi-hot encoding from RDF types |

### 8.3 Why rdf_types Is Better — The Core Intuition

```
Two molecules both containing carbon (C) and nitrogen (N) atoms:

node_id mode:
  Molecule_A carbon atom: embedding_vec_47  (random init)
  Molecule_B carbon atom: embedding_vec_129 (random init)
  → The model doesn't know these are the same type of atom!

rdf_types mode:
  Molecule_A carbon atom: multi_hot{type::Carbon, literal_predicate::charge, ...}
  Molecule_B carbon atom: multi_hot{type::Carbon, literal_predicate::charge, ...}
  → Identical multi-hot → Linear → identical initial embedding
  → Model learns "typical behavior of carbon atoms," not "behavior of node #47"
```

---

## 9. Exact Code Changes

The entire innovation involves changes across 4 files with minimal code delta:

### 9.1 `config.py` — New Config Fields

```python
# ModelConfig gained three new fields
initial_features: str              # "node_id" or "rdf_types", defaults to "node_id"
validation_fraction: float          # Validation set proportion, defaults to 0.0 (no validation)
early_stopping_patience: int        # Early stopping patience, defaults to 0 (no early stopping)
```

### 9.2 `data.py` — New Function `build_node_feature_lists()`

- Location: `data.py:183-221`
- Input: `RdfGraphData` + `DatasetConfig`
- Output: `(List[List[int]], Dict[str, int])` — per-node feature ID lists + global feature vocabulary
- Key: iterate over all triples in the graph, extract four categories of features from the subject

### 9.3 `model.py` — RGCNClassifier Dual-Path

- Location: `model.py:66-104`
- `__init__` gained `input_feature_dim: int | None = None` parameter
  - `None` → creates `nn.Embedding` (node_id path)
  - `int` → creates `nn.Linear(feat_dim, embed_dim, bias=False)` (rdf_types path)
- `forward` gained `input_features: torch.Tensor | None = None` parameter
  - Has `input_projection` → `input_projection(input_features)`
  - None → `node_embeddings(node_ids)`

### 9.4 `train.py` — Config-Driven Feature Construction

- Location: `train.py:57-67`
- Checks `config.model.initial_features`:
  - `"rdf_types"` → calls `build_node_feature_lists()` → builds multi-hot matrix → passes to model
  - `"node_id"` → does not build features, `input_features=None`
- Added validation split (`split_validation()`) and early stopping logic

---

## 10. Key Focus Points

### 10.1 Whether Features Introduce Label Leakage

For every feature type added, you must verify it doesn't directly expose the label. Examples:
- ✅ `type::PhDStudent` — safe, role ≠ research group
- ❌ `exists::affiliation::ResearchGroup1` — **leaks the label!** (already excluded by `exclude_predicates`)

The current `build_node_feature_lists()` only extracts `rdf:type` and `literal` features from triples where the node is the subject. It does not extract `exists`-type features onto target entities, avoiding indirect leakage.

### 10.2 Controlling Feature Dimensionality

Different RDF graphs produce different feature vocabulary sizes:
- AIFB: ~200-300 dimensions
- MUTAG: potentially larger (numeric properties produce many `literal_value` features)

The current code limits `literal_value` features: only values ≤ 64 characters are included, preventing dimensionality explosion. If dimension is still too high, consider feature hashing or PCA.

### 10.3 Reproducibility

`sorted_graph_triples()` in `data.py` guarantees a fixed triple iteration order. All randomness is controlled by `seed_everything(config.project.seed)`. This ensures identical features on every run.

### 10.4 End-to-End Evaluation

Innovation cannot be judged by accuracy alone. Strategy 3 requires three-level evaluation:
- **GNN performance**: test_accuracy, test_macro_f1
- **Explanation quality**: CELOE hypothesis F1
- **Transferability**: same model/feature approach performance comparison across AIFB and MUTAG

### 10.5 Backward Compatibility

Both approaches coexist through a config toggle. The conditional branches total under 20 lines. The original `xai-mini --config configs/aifb.yaml run-all` runs without any changes.

---

## 11. Quick Start

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # includes Ontolearn/CELOE
pip install -e .

# 2. Download MUTAG data into data/mutag/ directory

# 3. Run experiments
# AIFB + node_id (original approach)
xai-mini --config configs/aifb.yaml run-all

# MUTAG + node_id (comparison baseline)
xai-mini --config configs/mutag_node_id.yaml run-all

# MUTAG + rdf_types (innovation)
xai-mini --config configs/mutag.yaml run-all

# 5. View results
cat artifacts/mutag/metrics.json
cat artifacts/mutag_node_id/metrics.json
# Compare test_accuracy, test_macro_f1, num_input_features, etc.

# 6. Test
python -m pytest tests/
```

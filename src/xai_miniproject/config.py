from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    seed: int
    artifacts_dir: Path


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    rdf_path: Path
    rdf_format: str
    train_path: Path
    test_path: Path
    all_labels_path: Path
    entity_column: str
    label_column: str
    target_node_type: str | None = None
    label_names: dict[str, str] = field(default_factory=dict)
    exclude_predicates: set[str] = field(default_factory=set)
    include_literal_nodes: bool = False
    add_inverse_edges: bool = True


@dataclass(frozen=True)
class ModelConfig:
    name: str
    initial_features: str
    embedding_dim: int
    hidden_dim: int
    dropout: float
    epochs: int
    learning_rate: float
    weight_decay: float
    validation_fraction: float
    early_stopping_patience: int
    log_every: int
    device: str
    num_layers: int = 2
    use_residual: bool = False
    use_layer_norm: bool = False


@dataclass(frozen=True)
class ExplanationConfig:
    learner: str
    backend: str
    fallback_baseline: bool
    target_classes: str | list[str]
    example_source: str
    example_split: str
    max_positive_examples: int
    max_negative_examples: int
    random_seed: int
    max_runtime_seconds: int
    top_k: int
    ontology_output: Path
    examples_output: Path


@dataclass(frozen=True)
class Config:
    project: ProjectConfig
    dataset: DatasetConfig
    model: ModelConfig
    explanation: ExplanationConfig
    path: Path


def _resolve_path(config_path: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent.parent / path).resolve()


def load_config(path: str | Path) -> Config:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw: dict[str, Any] = yaml.safe_load(stream)

    project_raw = raw["project"]
    dataset_raw = raw["dataset"]
    model_raw = raw["model"]
    explanation_raw = raw["explanation"]

    project = ProjectConfig(
        name=project_raw["name"],
        seed=int(project_raw.get("seed", 42)),
        artifacts_dir=_resolve_path(config_path, project_raw["artifacts_dir"]),
    )
    dataset = DatasetConfig(
        name=dataset_raw["name"],
        rdf_path=_resolve_path(config_path, dataset_raw["rdf_path"]),
        rdf_format=dataset_raw.get("rdf_format", "n3"),
        train_path=_resolve_path(config_path, dataset_raw["train_path"]),
        test_path=_resolve_path(config_path, dataset_raw["test_path"]),
        all_labels_path=_resolve_path(config_path, dataset_raw["all_labels_path"]),
        entity_column=dataset_raw["entity_column"],
        label_column=dataset_raw["label_column"],
        target_node_type=dataset_raw.get("target_node_type"),
        label_names=dict(dataset_raw.get("label_names", {})),
        exclude_predicates=set(dataset_raw.get("exclude_predicates", [])),
        include_literal_nodes=bool(dataset_raw.get("include_literal_nodes", False)),
        add_inverse_edges=bool(dataset_raw.get("add_inverse_edges", True)),
    )
    model = ModelConfig(
        name=model_raw.get("name", "rgcn"),
        initial_features=model_raw.get("initial_features", "node_id"),
        embedding_dim=int(model_raw.get("embedding_dim", 64)),
        hidden_dim=int(model_raw.get("hidden_dim", 64)),
        dropout=float(model_raw.get("dropout", 0.25)),
        epochs=int(model_raw.get("epochs", 250)),
        learning_rate=float(model_raw.get("learning_rate", 0.01)),
        weight_decay=float(model_raw.get("weight_decay", 5e-4)),
        validation_fraction=float(model_raw.get("validation_fraction", 0.0)),
        early_stopping_patience=int(model_raw.get("early_stopping_patience", 0)),
        log_every=int(model_raw.get("log_every", 25)),
        device=model_raw.get("device", "auto"),
        num_layers=int(model_raw.get("num_layers", 2)),
        use_residual=bool(model_raw.get("use_residual", False)),
        use_layer_norm=bool(model_raw.get("use_layer_norm", False)),
    )
    explanation = ExplanationConfig(
        learner=explanation_raw.get("learner", "CELOE"),
        backend=explanation_raw.get("backend", "ontolearn"),
        fallback_baseline=bool(explanation_raw.get("fallback_baseline", True)),
        target_classes=explanation_raw.get("target_classes", "all"),
        example_source=explanation_raw.get("example_source", "predictions"),
        example_split=explanation_raw.get("example_split", "all"),
        max_positive_examples=int(explanation_raw.get("max_positive_examples", 40)),
        max_negative_examples=int(explanation_raw.get("max_negative_examples", 80)),
        random_seed=int(explanation_raw.get("random_seed", project.seed)),
        max_runtime_seconds=int(explanation_raw.get("max_runtime_seconds", 60)),
        top_k=int(explanation_raw.get("top_k", 3)),
        ontology_output=_resolve_path(config_path, explanation_raw["ontology_output"]),
        examples_output=_resolve_path(config_path, explanation_raw["examples_output"]),
    )
    return Config(
        project=project,
        dataset=dataset,
        model=model,
        explanation=explanation,
        path=config_path,
    )

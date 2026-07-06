from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from xai_miniproject.config import Config
from xai_miniproject.data import (
    build_graph_data,
    build_node_feature_lists,
    build_split_tensors,
    relation_edge_groups,
)
from xai_miniproject.metrics import accuracy, macro_f1
from xai_miniproject.model import RGCNClassifier, build_tensor_graph
from xai_miniproject.utils import ensure_dir, seed_everything, short_uri, write_json


def choose_device(device_name: str) -> torch.device:
    normalized = device_name.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device is set to 'cuda', but CUDA is not available.")
    if normalized == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "device is set to 'mps', but PyTorch reports MPS is not available. "
            "Use device: cpu, or fix the PyTorch/macOS MPS environment."
        )
    if normalized != "cpu" and not normalized.startswith("cuda"):
        raise ValueError("model.device must be one of: auto, cpu, cuda, cuda:N, mps.")
    return torch.device(normalized)


def run_training(config: Config) -> dict[str, object]:
    seed_everything(config.project.seed)
    output_dir = ensure_dir(config.project.artifacts_dir)
    data = build_graph_data(config.dataset)
    splits = build_split_tensors(data, config.dataset)
    device = choose_device(config.model.device)
    tensor_graph = build_tensor_graph(
        relation_edge_groups(data),
        num_nodes=data.num_nodes,
        device=device,
    )
    input_features = None
    input_feature_dim = None
    feature_mapping = None
    if config.model.initial_features == "rdf_neighborhood":
        feature_lists, feature_mapping = build_node_feature_lists(
            data, config.dataset, initial_features=config.model.initial_features
        )
        input_feature_dim = len(feature_mapping)
        input_features = torch.zeros((data.num_nodes, input_feature_dim), dtype=torch.float32, device=device)
        for node_id, feature_ids in enumerate(feature_lists):
            input_features[node_id, feature_ids] = 1.0
    elif config.model.initial_features != "node_id":
        raise ValueError(
            "model.initial_features must be either 'node_id' or 'rdf_neighborhood', "
            f"got {config.model.initial_features!r}."
        )

    train_nodes = splits["train_idx"]
    train_label_values = splits["train_labels"]
    fit_nodes, fit_labels, val_nodes, val_labels = split_validation(
        train_nodes,
        train_label_values,
        validation_fraction=config.model.validation_fraction,
        seed=config.project.seed,
    )
    train_idx = torch.tensor(train_nodes, dtype=torch.long, device=device)
    train_labels = torch.tensor(train_label_values, dtype=torch.long, device=device)
    fit_idx = torch.tensor(fit_nodes, dtype=torch.long, device=device)
    fit_label_tensor = torch.tensor(fit_labels, dtype=torch.long, device=device)
    val_idx = torch.tensor(val_nodes, dtype=torch.long, device=device) if val_nodes else None
    val_label_tensor = (
        torch.tensor(val_labels, dtype=torch.long, device=device) if val_labels else None
    )
    test_idx = torch.tensor(splits["test_idx"], dtype=torch.long, device=device)
    test_labels = torch.tensor(splits["test_labels"], dtype=torch.long, device=device)

    model = RGCNClassifier(
        num_nodes=data.num_nodes,
        num_relations=data.num_relations,
        num_classes=data.num_classes,
        embedding_dim=config.model.embedding_dim,
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
        input_feature_dim=input_feature_dim,
        num_layers=config.model.num_layers,
        use_residual=config.model.use_residual,
        use_layer_norm=config.model.use_layer_norm,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.model.learning_rate,
        weight_decay=config.model.weight_decay,
    )

    history: list[dict[str, float]] = []
    best_state = None
    best_val_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, config.model.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(tensor_graph, input_features=input_features)
        loss = F.cross_entropy(logits[fit_idx], fit_label_tensor)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(tensor_graph, input_features=input_features)
            train_pred = logits[train_idx].argmax(dim=1).cpu().tolist()
            test_pred = logits[test_idx].argmax(dim=1).cpu().tolist()
            train_true = train_labels.cpu().tolist()
            test_true = test_labels.cpu().tolist()
            train_acc = accuracy(train_true, train_pred)
            test_acc = accuracy(test_true, test_pred)
            train_f1 = macro_f1(train_true, train_pred, list(range(data.num_classes)))
            test_f1 = macro_f1(test_true, test_pred, list(range(data.num_classes)))
            val_acc = None
            val_f1 = None
            if val_idx is not None and val_label_tensor is not None:
                val_pred = logits[val_idx].argmax(dim=1).cpu().tolist()
                val_true = val_label_tensor.cpu().tolist()
                val_acc = accuracy(val_true, val_pred)
                val_f1 = macro_f1(val_true, val_pred, list(range(data.num_classes)))

        if val_f1 is not None:
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                epochs_without_improvement = 0
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
            else:
                epochs_without_improvement += 1

        if epoch == 1 or epoch % config.model.log_every == 0 or epoch == config.model.epochs:
            row = {
                "epoch": float(epoch),
                "loss": float(loss.item()),
                "train_accuracy": train_acc,
                "train_macro_f1": train_f1,
                "test_accuracy": test_acc,
                "test_macro_f1": test_f1,
            }
            if val_acc is not None and val_f1 is not None:
                row["val_accuracy"] = val_acc
                row["val_macro_f1"] = val_f1
            history.append(row)
            val_part = f" val_acc={val_acc:.3f} val_f1={val_f1:.3f}" if val_f1 is not None else ""
            print(
                f"epoch={epoch:03d} loss={loss.item():.4f} "
                f"train_acc={train_acc:.3f} test_acc={test_acc:.3f} "
                f"train_f1={train_f1:.3f} test_f1={test_f1:.3f}{val_part}"
            )
        if (
            val_f1 is not None
            and config.model.early_stopping_patience > 0
            and epochs_without_improvement >= config.model.early_stopping_patience
        ):
            print(f"Early stopping at epoch {epoch} with best_val_f1={best_val_f1:.3f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(tensor_graph, input_features=input_features)
        probabilities = torch.softmax(logits, dim=1)
        all_target_idx = torch.tensor(splits["all_idx"], dtype=torch.long, device=device)
        all_pred_ids = probabilities[all_target_idx].argmax(dim=1).cpu().tolist()
        all_conf = probabilities[all_target_idx].max(dim=1).values.cpu().tolist()
        train_pred = logits[train_idx].argmax(dim=1).cpu().tolist()
        test_pred = logits[test_idx].argmax(dim=1).cpu().tolist()

    train_true = train_labels.cpu().tolist()
    test_true = test_labels.cpu().tolist()
    metrics = {
        "dataset": config.dataset.name,
        "device": str(device),
        "num_nodes": data.num_nodes,
        "num_edges": len(data.edges),
        "num_relations": data.num_relations,
        "num_classes": data.num_classes,
        "initial_features": config.model.initial_features,
        "num_input_features": input_feature_dim,
        "validation_fraction": config.model.validation_fraction,
        "best_val_macro_f1": best_val_f1 if best_state is not None else None,
        "train_accuracy": accuracy(train_true, train_pred),
        "train_macro_f1": macro_f1(train_true, train_pred, list(range(data.num_classes))),
        "test_accuracy": accuracy(test_true, test_pred),
        "test_macro_f1": macro_f1(test_true, test_pred, list(range(data.num_classes))),
        "history": history,
    }

    predictions = _prediction_frame(config, data, all_pred_ids, all_conf)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    write_json(output_dir / "metrics.json", metrics)
    write_json(
        output_dir / "label_mapping.json",
        {
            "id_to_label": data.id_to_label,
            "label_to_id": data.label_to_id,
            "label_names": config.dataset.label_names,
        },
    )
    write_json(
        output_dir / "graph_metadata.json",
        {
            "id_to_node": data.id_to_node,
            "id_to_relation": data.id_to_relation,
            "feature_to_id": feature_mapping,
            "excluded_predicates": sorted(config.dataset.exclude_predicates),
        },
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config_path": str(config.path),
            "metrics": metrics,
        },
        output_dir / "model.pt",
    )
    print(f"Saved artifacts to {output_dir}")
    return metrics


def _prediction_frame(
    config: Config,
    data,
    pred_ids: list[int],
    confidences: list[float],
) -> pd.DataFrame:
    split_by_entity: dict[str, str] = {
        row[config.dataset.entity_column]: "train" for _, row in data.train_df.iterrows()
    }
    split_by_entity.update(
        {row[config.dataset.entity_column]: "test" for _, row in data.test_df.iterrows()}
    )

    rows = []
    for offset, (_, row) in enumerate(data.all_labels_df.iterrows()):
        entity = row[config.dataset.entity_column]
        true_label = row[config.dataset.label_column]
        pred_label = data.id_to_label[pred_ids[offset]]
        rows.append(
            {
                "entity_uri": entity,
                "entity_name": short_uri(entity),
                "true_label_uri": true_label,
                "true_label_name": config.dataset.label_names.get(true_label, short_uri(true_label)),
                "pred_label_uri": pred_label,
                "pred_label_name": config.dataset.label_names.get(pred_label, short_uri(pred_label)),
                "confidence": confidences[offset],
                "split": split_by_entity.get(entity, "unknown"),
                "correct": true_label == pred_label,
            }
        )
    return pd.DataFrame(rows)


def load_metrics(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def split_validation(
    nodes: list[int],
    labels: list[int],
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], list[int], list[int]]:
    if validation_fraction <= 0:
        return nodes, labels, [], []
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1).")

    grouped_positions: dict[int, list[int]] = {}
    for pos, label in enumerate(labels):
        grouped_positions.setdefault(label, []).append(pos)

    rng = random.Random(seed)
    val_positions: set[int] = set()
    for positions in grouped_positions.values():
        shuffled = positions[:]
        rng.shuffle(shuffled)
        if len(shuffled) <= 1:
            continue
        n_val = max(1, round(len(shuffled) * validation_fraction))
        n_val = min(n_val, len(shuffled) - 1)
        val_positions.update(shuffled[:n_val])

    fit_nodes, fit_labels, val_nodes, val_labels = [], [], [], []
    for pos, (node, label) in enumerate(zip(nodes, labels, strict=True)):
        if pos in val_positions:
            val_nodes.append(node)
            val_labels.append(label)
        else:
            fit_nodes.append(node)
            fit_labels.append(label)
    return fit_nodes, fit_labels, val_nodes, val_labels

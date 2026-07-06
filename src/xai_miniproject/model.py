from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class TensorGraph:
    relation_edges: list[torch.Tensor]
    num_nodes: int
    num_relations: int

    def to(self, device: torch.device) -> "TensorGraph":
        return TensorGraph(
            relation_edges=[edge_index.to(device) for edge_index in self.relation_edges],
            num_nodes=self.num_nodes,
            num_relations=self.num_relations,
        )


def build_tensor_graph(
    relation_groups: list[list[tuple[int, int]]],
    num_nodes: int,
    device: torch.device,
) -> TensorGraph:
    edge_tensors = []
    for group in relation_groups:
        if group:
            edge_tensors.append(torch.tensor(group, dtype=torch.long, device=device).t().contiguous())
        else:
            edge_tensors.append(torch.empty((2, 0), dtype=torch.long, device=device))
    return TensorGraph(
        relation_edges=edge_tensors,
        num_nodes=num_nodes,
        num_relations=len(relation_groups),
    )


class RGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_relations: int, dropout: float) -> None:
        super().__init__()
        self.relation_weights = nn.Parameter(torch.empty(num_relations, in_dim, out_dim))
        self.self_loop = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.relation_weights)
        nn.init.xavier_uniform_(self.self_loop.weight)

    def forward(self, x: torch.Tensor, graph: TensorGraph) -> torch.Tensor:
        out = self.self_loop(x)
        for relation_id, edge_index in enumerate(graph.relation_edges):
            if edge_index.numel() == 0:
                continue
            src, dst = edge_index
            messages = x[src] @ self.relation_weights[relation_id]
            degree = torch.bincount(dst, minlength=graph.num_nodes).clamp(min=1).to(messages.dtype)
            messages = messages / degree[dst].unsqueeze(1)
            out.index_add_(0, dst, messages)
        out = out + self.bias
        return self.dropout(F.relu(out))


class RGCNClassifier(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        num_classes: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        input_feature_dim: int | None = None,
        num_layers: int = 2,
        use_residual: bool = False,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.node_embeddings = (
            nn.Embedding(num_nodes, embedding_dim) if input_feature_dim is None else None
        )
        self.input_projection = (
            nn.Linear(input_feature_dim, embedding_dim, bias=False)
            if input_feature_dim is not None
            else None
        )
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for layer_idx in range(num_layers):
            in_dim = embedding_dim if layer_idx == 0 else hidden_dim
            self.layers.append(RGCNLayer(in_dim, hidden_dim, num_relations, dropout))
            self.layer_norms.append(nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity())
        self.classifier = nn.Linear(hidden_dim, num_classes)
        if self.node_embeddings is not None:
            nn.init.xavier_uniform_(self.node_embeddings.weight)

    def forward(self, graph: TensorGraph, input_features: torch.Tensor | None = None) -> torch.Tensor:
        if self.input_projection is not None:
            if input_features is None:
                raise ValueError("input_features must be provided when input_feature_dim is set.")
            x = self.input_projection(input_features)
        else:
            if self.node_embeddings is None:
                raise ValueError("node_embeddings are not initialized.")
            node_ids = torch.arange(graph.num_nodes, device=self.node_embeddings.weight.device)
            x = self.node_embeddings(node_ids)
        for layer, norm in zip(self.layers, self.layer_norms, strict=True):
            h = norm(layer(x, graph))
            if self.use_residual and h.shape == x.shape:
                h = h + x
            x = h
        return self.classifier(x)

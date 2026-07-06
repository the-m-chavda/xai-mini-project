from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from rdflib import BNode, Graph, Literal, RDF, URIRef

from xai_miniproject.config import DatasetConfig
from xai_miniproject.utils import short_uri

ResourceNode = URIRef | BNode


@dataclass(frozen=True)
class Edge:
    src: int
    dst: int
    relation: int


@dataclass
class RdfGraphData:
    graph: Graph
    node_to_id: dict[str, int]
    id_to_node: list[str]
    relation_to_id: dict[str, int]
    id_to_relation: list[str]
    edges: list[Edge]
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    all_labels_df: pd.DataFrame
    label_to_id: dict[str, int]
    id_to_label: list[str]

    @property
    def num_nodes(self) -> int:
        return len(self.id_to_node)

    @property
    def num_relations(self) -> int:
        return len(self.id_to_relation)

    @property
    def num_classes(self) -> int:
        return len(self.id_to_label)


def load_rdf_graph(config: DatasetConfig) -> Graph:
    graph = Graph()
    graph.parse(str(config.rdf_path), format=config.rdf_format)
    return graph


def load_label_table(path: Path, config: DatasetConfig) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    # DGL's trainingSet.tsv has columns ordered differently from completeDataset.tsv.
    required = {config.entity_column, config.label_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame[config.entity_column] = frame[config.entity_column].astype(str)
    frame[config.label_column] = frame[config.label_column].map(normalize_label_value)
    return frame


def normalize_label_value(value: object) -> str:
    """Keep URI labels unchanged while normalizing numeric labels such as 1.0 -> 1."""
    if pd.isna(value):
        raise ValueError("Label values must not be NA.")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value)
    try:
        numeric = float(text)
    except ValueError:
        return text
    if numeric.is_integer() and text.strip().replace(".", "", 1).isdigit():
        return str(int(numeric))
    return text


def _is_graph_resource(term: object, include_literal_nodes: bool) -> bool:
    if isinstance(term, (URIRef, BNode)):
        return True
    return include_literal_nodes and isinstance(term, Literal)


def _term_key(term: object) -> str:
    if isinstance(term, BNode):
        return f"_:{term}"
    return str(term)


def _add_mapping(mapping: dict[str, int], values: list[str], key: str) -> int:
    if key not in mapping:
        mapping[key] = len(values)
        values.append(key)
    return mapping[key]


def sorted_graph_triples(graph: Graph) -> list[tuple[object, object, object]]:
    return sorted(graph, key=lambda triple: tuple(_term_key(term) for term in triple))


def build_graph_data(config: DatasetConfig) -> RdfGraphData:
    graph = load_rdf_graph(config)
    train_df = load_label_table(config.train_path, config)
    test_df = load_label_table(config.test_path, config)
    all_labels_df = load_label_table(config.all_labels_path, config)

    label_values = sorted(all_labels_df[config.label_column].unique().tolist())
    label_to_id = {label: idx for idx, label in enumerate(label_values)}

    node_to_id: dict[str, int] = {}
    id_to_node: list[str] = []
    relation_to_id: dict[str, int] = {}
    id_to_relation: list[str] = []
    edges: list[Edge] = []

    for entity in all_labels_df[config.entity_column]:
        _add_mapping(node_to_id, id_to_node, entity)

    excluded = set(config.exclude_predicates)
    for subject, predicate, obj in sorted_graph_triples(graph):
        predicate_uri = str(predicate)
        if predicate_uri in excluded:
            continue
        if not _is_graph_resource(subject, config.include_literal_nodes):
            continue
        if not _is_graph_resource(obj, config.include_literal_nodes):
            continue

        src = _add_mapping(node_to_id, id_to_node, _term_key(subject))
        dst = _add_mapping(node_to_id, id_to_node, _term_key(obj))
        rel = _add_mapping(relation_to_id, id_to_relation, predicate_uri)
        edges.append(Edge(src=src, dst=dst, relation=rel))

        if config.add_inverse_edges:
            inverse_rel = _add_mapping(relation_to_id, id_to_relation, f"{predicate_uri}__inverse")
            edges.append(Edge(src=dst, dst=src, relation=inverse_rel))

    return RdfGraphData(
        graph=graph,
        node_to_id=node_to_id,
        id_to_node=id_to_node,
        relation_to_id=relation_to_id,
        id_to_relation=id_to_relation,
        edges=edges,
        train_df=train_df,
        test_df=test_df,
        all_labels_df=all_labels_df,
        label_to_id=label_to_id,
        id_to_label=label_values,
    )


def build_split_tensors(data: RdfGraphData, config: DatasetConfig) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for split, frame in (("train", data.train_df), ("test", data.test_df), ("all", data.all_labels_df)):
        indices = []
        labels = []
        for _, row in frame.iterrows():
            entity = row[config.entity_column]
            if entity not in data.node_to_id:
                raise KeyError(f"Entity {entity} is not present in the graph node mapping.")
            indices.append(data.node_to_id[entity])
            labels.append(data.label_to_id[row[config.label_column]])
        result[f"{split}_idx"] = indices
        result[f"{split}_labels"] = labels
    return result


def relation_edge_groups(data: RdfGraphData) -> list[list[tuple[int, int]]]:
    groups: list[list[tuple[int, int]]] = [[] for _ in range(data.num_relations)]
    for edge in data.edges:
        groups[edge.relation].append((edge.src, edge.dst))
    return groups


def build_node_feature_lists(
    data: RdfGraphData, config: DatasetConfig, initial_features: str = "rdf_neighborhood"
) -> tuple[list[list[int]], dict[str, int]]:
    feature_to_id: dict[str, int] = {}
    feature_sets: list[set[int]] = [set() for _ in range(data.num_nodes)]
    use_neighborhood = initial_features == "rdf_neighborhood"

    def feature_id(name: str) -> int:
        if name not in feature_to_id:
            feature_to_id[name] = len(feature_to_id)
        return feature_to_id[name]

    def add_feature(node_key: str, feature_name: str) -> None:
        node_id = data.node_to_id.get(node_key)
        if node_id is not None:
            feature_sets[node_id].add(feature_id(feature_name))

    object_types: dict[str, set[str]] = defaultdict(set)
    if use_neighborhood:
        for subject, predicate, obj in sorted_graph_triples(data.graph):
            if predicate == RDF.type:
                object_types[_term_key(subject)].add(str(obj))

    for subject, predicate, obj in sorted_graph_triples(data.graph):
        predicate_uri = str(predicate)
        if predicate_uri in config.exclude_predicates:
            continue
        subject_key = _term_key(subject)
        if predicate == RDF.type:
            add_feature(subject_key, f"type::{obj}")
        elif isinstance(obj, Literal):
            if config.include_literal_nodes:
                literal_key = _term_key(obj)
                add_feature(literal_key, f"literal_predicate::{predicate}")
                if obj.datatype is not None:
                    add_feature(literal_key, f"literal_datatype::{obj.datatype}")
                lexical = str(obj)
                if len(lexical) <= 64:
                    add_feature(literal_key, f"literal_value::{lexical}")
            if use_neighborhood:
                add_feature(subject_key, f"literal_predicate::{predicate}")
                if obj.datatype is not None:
                    add_feature(subject_key, f"literal_datatype::{obj.datatype}")
                lexical = str(obj)
                if len(lexical) <= 64:
                    add_feature(subject_key, f"literal_value::{lexical}")
        elif use_neighborhood and isinstance(obj, (URIRef, BNode)):
            obj_types = object_types.get(_term_key(obj), set())
            if obj_types:
                for obj_type in obj_types:
                    add_feature(subject_key, f"exists::{predicate_uri}::{obj_type}")
            else:
                add_feature(subject_key, f"exists::{predicate_uri}::Thing")

    for node_id, node_key in enumerate(data.id_to_node):
        if not feature_sets[node_id]:
            if node_key.startswith("http://") or node_key.startswith("https://"):
                feature_sets[node_id].add(feature_id("kind::resource_without_type"))
            elif node_key.startswith("_:"):
                feature_sets[node_id].add(feature_id("kind::blank_node"))
            else:
                feature_sets[node_id].add(feature_id("kind::literal"))

    return [sorted(features) for features in feature_sets], feature_to_id


def dataset_statistics(data: RdfGraphData, config: DatasetConfig) -> dict[str, object]:
    type_counter: Counter[str] = Counter()
    predicate_counter: Counter[str] = Counter()
    resource_edges = 0
    literal_edges = 0

    for subject, predicate, obj in sorted_graph_triples(data.graph):
        predicate_uri = str(predicate)
        predicate_counter[predicate_uri] += 1
        if predicate == RDF.type:
            type_counter[str(obj)] += 1
        if isinstance(subject, (URIRef, BNode)) and isinstance(obj, (URIRef, BNode)):
            resource_edges += 1
        else:
            literal_edges += 1

    labels = data.all_labels_df[config.label_column].value_counts().to_dict()
    readable_labels = {
        config.label_names.get(label, short_uri(label)): int(count) for label, count in labels.items()
    }

    return {
        "dataset": config.name,
        "rdf_path": str(config.rdf_path),
        "triples": len(data.graph),
        "resource_triples": resource_edges,
        "literal_triples": literal_edges,
        "model_nodes": data.num_nodes,
        "model_edges": len(data.edges),
        "model_relations": data.num_relations,
        "train_examples": len(data.train_df),
        "test_examples": len(data.test_df),
        "target_examples": len(data.all_labels_df),
        "label_distribution": readable_labels,
        "top_node_types": _top_counter(type_counter),
        "top_predicates": _top_counter(predicate_counter),
        "excluded_predicates": sorted(config.exclude_predicates),
    }


def _top_counter(counter: Counter[str], limit: int = 20) -> list[dict[str, object]]:
    return [
        {"uri": uri, "name": short_uri(uri), "count": int(count)}
        for uri, count in counter.most_common(limit)
    ]


def filtered_graph(graph: Graph, excluded_predicates: Iterable[str]) -> Graph:
    output = Graph()
    for prefix, namespace in graph.namespaces():
        output.bind(prefix, namespace)
    excluded = set(excluded_predicates)
    for triple in sorted_graph_triples(graph):
        if str(triple[1]) not in excluded:
            output.add(triple)
    return output


def neighborhood_features(
    graph: Graph,
    entity_uris: Iterable[str],
    excluded_predicates: Iterable[str],
) -> dict[str, set[str]]:
    excluded = set(excluded_predicates)
    feature_map: dict[str, set[str]] = defaultdict(set)
    entity_set = set(entity_uris)
    rdf_type = str(RDF.type)

    object_types: dict[str, set[str]] = defaultdict(set)
    for subject, predicate, obj in sorted(
        graph.triples((None, RDF.type, None)),
        key=lambda triple: tuple(_term_key(term) for term in triple),
    ):
        object_types[str(subject)].add(str(obj))

    for uri in sorted(entity_set):
        node = URIRef(uri)
        for _, predicate, obj in sorted(
            graph.triples((node, None, None)),
            key=lambda triple: tuple(_term_key(term) for term in triple),
        ):
            predicate_uri = str(predicate)
            if predicate_uri in excluded:
                continue
            if predicate_uri == rdf_type:
                feature_map[uri].add(f"type::{obj}")
            elif isinstance(obj, (URIRef, BNode)):
                types = object_types.get(str(obj), set())
                if types:
                    for object_type in types:
                        feature_map[uri].add(f"exists::{predicate_uri}::{object_type}")
                else:
                    feature_map[uri].add(f"exists::{predicate_uri}::Thing")
            elif isinstance(obj, Literal):
                feature_map[uri].add(f"literal::{predicate_uri}")
        for subject, predicate, _ in sorted(
            graph.triples((None, None, node)),
            key=lambda triple: tuple(_term_key(term) for term in triple),
        ):
            predicate_uri = str(predicate)
            if predicate_uri in excluded:
                continue
            subject_types = object_types.get(str(subject), set())
            if subject_types:
                for subject_type in subject_types:
                    feature_map[uri].add(f"exists_inverse::{predicate_uri}::{subject_type}")
            else:
                feature_map[uri].add(f"exists_inverse::{predicate_uri}::Thing")
    return feature_map

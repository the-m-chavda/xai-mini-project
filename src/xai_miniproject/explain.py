from __future__ import annotations

import inspect
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from rdflib import OWL, RDF, URIRef

from xai_miniproject.config import Config
from xai_miniproject.data import filtered_graph, load_rdf_graph, neighborhood_features
from xai_miniproject.utils import ensure_dir, seed_everything, short_uri, write_json


@dataclass(frozen=True)
class LearningProblem:
    target_class_uri: str
    target_class_name: str
    positive_examples: list[str]
    negative_examples: list[str]


def run_explanations(config: Config) -> dict[str, Any]:
    seed_everything(config.explanation.random_seed)
    output_dir = ensure_dir(config.project.artifacts_dir)
    predictions_path = output_dir / "predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"{predictions_path} does not exist. Run "
            f"`xai-mini --config {config.path} train` first."
        )

    graph = load_rdf_graph(config.dataset)
    ontology_path = serialize_filtered_ontology(config, graph)
    predictions = pd.read_csv(predictions_path)
    problems = build_learning_problems(config, predictions)
    write_json(config.explanation.examples_output, {"problems": [p.__dict__ for p in problems]})

    results: list[dict[str, Any]] = []
    for problem in problems:
        print(
            f"Explaining {problem.target_class_name}: "
            f"{len(problem.positive_examples)} positive / {len(problem.negative_examples)} negative examples"
        )
        try:
            if config.explanation.backend.lower() != "ontolearn":
                raise ValueError(f"Unsupported backend: {config.explanation.backend}")
            learner_results = run_ontolearn(config, ontology_path, problem)
            results.append(
                {
                    "target_class_uri": problem.target_class_uri,
                    "target_class_name": problem.target_class_name,
                    "backend": "ontolearn",
                    "hypotheses": learner_results,
                }
            )
        except Exception as exc:
            if not config.explanation.fallback_baseline:
                raise
            baseline = run_baseline_explainer(config, graph, problem)
            results.append(
                {
                    "target_class_uri": problem.target_class_uri,
                    "target_class_name": problem.target_class_name,
                    "backend": "baseline",
                    "ontolearn_error": str(exc),
                    "hypotheses": baseline,
                }
            )
            print(f"Ontolearn unavailable or failed; wrote baseline explanation for this class: {exc}")

    payload = {
        "dataset": config.dataset.name,
        "learner": config.explanation.learner,
        "ontology_path": str(ontology_path),
        "learning_problems_path": str(config.explanation.examples_output),
        "results": results,
    }
    write_json(output_dir / "explanation_results.json", payload)
    write_flat_results(output_dir / "explanation_results.csv", results)
    print(f"Saved explanation results to {output_dir / 'explanation_results.json'}")
    return payload


def serialize_filtered_ontology(config: Config, graph) -> Path:
    output = config.explanation.ontology_output
    output.parent.mkdir(parents=True, exist_ok=True)
    filtered = filtered_graph(graph, config.dataset.exclude_predicates)
    for entity_uri in pd.read_csv(config.dataset.all_labels_path, sep="\t")[
        config.dataset.entity_column
    ].astype(str):
        filtered.add((URIRef(entity_uri), RDF.type, OWL.NamedIndividual))
    filtered.serialize(destination=str(output), format="xml")
    return output


def build_learning_problems(config: Config, predictions: pd.DataFrame) -> list[LearningProblem]:
    split = config.explanation.example_split.lower()
    if split != "all":
        predictions = predictions[predictions["split"] == split].copy()
    if predictions.empty:
        raise ValueError(f"No prediction rows available for explanation split: {split}")

    if config.explanation.example_source != "predictions":
        raise ValueError("Only example_source='predictions' is implemented.")

    configured_targets = config.explanation.target_classes
    if configured_targets == "all":
        target_classes = sorted(predictions["pred_label_uri"].unique().tolist())
    elif isinstance(configured_targets, str):
        target_classes = [configured_targets]
    else:
        target_classes = list(configured_targets)

    rng = random.Random(config.explanation.random_seed)
    problems = []
    for target_class_uri in target_classes:
        positives = predictions[predictions["pred_label_uri"] == target_class_uri][
            "entity_uri"
        ].tolist()
        negatives = predictions[predictions["pred_label_uri"] != target_class_uri][
            "entity_uri"
        ].tolist()
        if not positives or not negatives:
            continue
        rng.shuffle(positives)
        rng.shuffle(negatives)
        positives = sorted(positives[: config.explanation.max_positive_examples])
        negatives = sorted(negatives[: config.explanation.max_negative_examples])
        label_name = config.dataset.label_names.get(target_class_uri, short_uri(target_class_uri))
        problems.append(
            LearningProblem(
                target_class_uri=target_class_uri,
                target_class_name=label_name,
                positive_examples=positives,
                negative_examples=negatives,
            )
        )
    if not problems:
        raise ValueError("No valid positive/negative learning problems could be built.")
    return problems


def run_ontolearn(config: Config, ontology_path: Path, problem: LearningProblem) -> list[dict[str, Any]]:
    KnowledgeBase, PosNegLPStandard, OWLNamedIndividual, learner_class = _load_ontolearn_symbols(
        config.explanation.learner
    )
    renderer = _load_renderer()
    kb = KnowledgeBase(path=str(ontology_path))

    learner_kwargs: dict[str, Any] = {"knowledge_base": kb}
    signature = inspect.signature(learner_class)
    if "max_runtime" in signature.parameters:
        learner_kwargs["max_runtime"] = config.explanation.max_runtime_seconds
    elif "max_runtime_seconds" in signature.parameters:
        learner_kwargs["max_runtime_seconds"] = config.explanation.max_runtime_seconds
    model = learner_class(**learner_kwargs)

    learning_problem = PosNegLPStandard(
        pos={OWLNamedIndividual(uri) for uri in problem.positive_examples},
        neg={OWLNamedIndividual(uri) for uri in problem.negative_examples},
    )
    fitted = model.fit(learning_problem=learning_problem)
    hypothesis_source = fitted if hasattr(fitted, "best_hypotheses") else model
    hypotheses = _best_hypotheses(hypothesis_source, config.explanation.top_k)

    results = []
    for hypothesis in hypotheses:
        concept = getattr(hypothesis, "concept", hypothesis)
        quality = getattr(hypothesis, "quality", None)
        results.append(
            {
                "concept": renderer(concept),
                "quality": float(quality) if quality is not None else None,
                "raw": str(concept),
            }
        )
    return results


def _load_ontolearn_symbols(learner_name: str):
    os.environ.setdefault("MPLCONFIGDIR", str(ensure_dir(Path(".matplotlib-cache").resolve())))
    try:
        from ontolearn.knowledge_base import KnowledgeBase
        from ontolearn.learning_problem import PosNegLPStandard
    except ImportError as exc:
        raise ImportError(
            "Ontolearn is not installed. Install it with "
            "`python -m pip install -r requirements-ontolearn.txt`."
        ) from exc

    try:
        from owlapy.owl_individual import OWLNamedIndividual
    except ImportError:
        from owlapy.model import OWLNamedIndividual

    learner_class = None
    try:
        import ontolearn.learners as learners

        learner_class = getattr(learners, learner_name, None)
    except ImportError:
        learner_class = None
    if learner_class is None:
        try:
            from ontolearn.concept_learner import CELOE

            if learner_name.upper() == "CELOE":
                learner_class = CELOE
        except ImportError:
            learner_class = None
    if learner_class is None:
        raise ImportError(f"Cannot find Ontolearn learner class {learner_name}.")
    return KnowledgeBase, PosNegLPStandard, OWLNamedIndividual, learner_class


def _load_renderer():
    try:
        from owlapy import owl_expression_to_dl

        return owl_expression_to_dl
    except ImportError:
        pass
    try:
        from owlapy.render import DLSyntaxObjectRenderer

        renderer = DLSyntaxObjectRenderer()
        return renderer.render
    except ImportError:
        return str


def _best_hypotheses(model, top_k: int):
    method = model.best_hypotheses
    try:
        result = method(top_k)
    except TypeError:
        result = method()
    if isinstance(result, list):
        return result[:top_k]
    if isinstance(result, tuple):
        return list(result[:top_k])
    return [result]


def run_baseline_explainer(config: Config, graph, problem: LearningProblem) -> list[dict[str, Any]]:
    examples = problem.positive_examples + problem.negative_examples
    features = neighborhood_features(graph, examples, config.dataset.exclude_predicates)
    positive_set = set(problem.positive_examples)
    negative_set = set(problem.negative_examples)
    all_features = sorted(set().union(*(features.get(uri, set()) for uri in examples)))

    scored = []
    for feature in all_features:
        covered_pos = {uri for uri in positive_set if feature in features.get(uri, set())}
        covered_neg = {uri for uri in negative_set if feature in features.get(uri, set())}
        tp = len(covered_pos)
        fp = len(covered_neg)
        fn = len(positive_set - covered_pos)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if tp:
            scored.append(
                {
                    "concept": render_feature(feature),
                    "quality": f1,
                    "precision": precision,
                    "recall": recall,
                    "covered_positive": tp,
                    "covered_negative": fp,
                    "raw": feature,
                }
            )
    scored.sort(
        key=lambda item: (
            item["quality"],
            item["precision"],
            item["recall"],
            -item["covered_negative"],
        ),
        reverse=True,
    )
    return scored[: config.explanation.top_k]


def render_feature(feature: str) -> str:
    parts = feature.split("::")
    if parts[0] == "type" and len(parts) == 2:
        return short_uri(parts[1])
    if parts[0] == "exists" and len(parts) == 3:
        return f"EXISTS {short_uri(parts[1])}.{short_uri(parts[2])}"
    if parts[0] == "exists_inverse" and len(parts) == 3:
        return f"EXISTS inverse({short_uri(parts[1])}).{short_uri(parts[2])}"
    if parts[0] == "literal" and len(parts) == 2:
        return f"HAS_LITERAL {short_uri(parts[1])}"
    return feature


def write_flat_results(path: Path, results: list[dict[str, Any]]) -> None:
    rows = []
    for result in results:
        for rank, hypothesis in enumerate(result["hypotheses"], start=1):
            rows.append(
                {
                    "target_class_uri": result["target_class_uri"],
                    "target_class_name": result["target_class_name"],
                    "backend": result["backend"],
                    "rank": rank,
                    **hypothesis,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)

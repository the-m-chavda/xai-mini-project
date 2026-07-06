from __future__ import annotations

from xai_miniproject.config import Config
from xai_miniproject.data import build_graph_data, dataset_statistics
from xai_miniproject.utils import ensure_dir, write_json


def run_analysis(config: Config) -> dict[str, object]:
    output_dir = ensure_dir(config.project.artifacts_dir)
    data = build_graph_data(config.dataset)
    stats = dataset_statistics(data, config.dataset)
    write_json(output_dir / "dataset_stats.json", stats)

    print(f"Dataset: {stats['dataset']}")
    print(f"Triples: {stats['triples']}")
    print(f"Model nodes: {stats['model_nodes']}")
    print(f"Model edges: {stats['model_edges']}")
    print(f"Relations: {stats['model_relations']}")
    print(f"Train/test examples: {stats['train_examples']}/{stats['test_examples']}")
    print(f"Saved statistics to {output_dir / 'dataset_stats.json'}")
    return stats

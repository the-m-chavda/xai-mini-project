from pathlib import Path

from xai_miniproject.config import load_config


def test_load_aifb_config_resolves_paths() -> None:
    config = load_config("configs/aifb.yaml")
    assert config.dataset.name == "aifb"
    assert config.dataset.rdf_path.is_absolute()
    assert config.dataset.rdf_path.name == "aifbfixed_complete.n3"
    assert config.project.artifacts_dir == Path("artifacts/aifb").resolve()
    assert "http://swrc.ontoware.org/ontology#affiliation" in config.dataset.exclude_predicates
    assert config.model.initial_features == "rdf_neighborhood"
    assert config.model.use_residual is True
    assert config.model.use_layer_norm is True


def test_load_aifb_baseline_config() -> None:
    config = load_config("configs/aifb_baseline.yaml")
    assert config.dataset.name == "aifb_baseline"
    assert config.dataset.rdf_path.name == "aifbfixed_complete.n3"
    assert config.project.artifacts_dir == Path("artifacts/aifb_baseline").resolve()
    assert config.model.initial_features == "node_id"
    assert config.model.hidden_dim == 64
    assert config.model.validation_fraction == 0.0

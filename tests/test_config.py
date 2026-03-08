"""Tests for config loading and validation."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topic_modeling.config.loader import _coerce, _set_nested, load_config
from topic_modeling.config.schema import ExperimentConfig


# --- _coerce ---

def test_coerce_int():
    assert _coerce("42") == 42

def test_coerce_float():
    assert _coerce("3.14") == pytest.approx(3.14)

def test_coerce_bool_true():
    assert _coerce("true") is True

def test_coerce_bool_false():
    assert _coerce("false") is False

def test_coerce_none():
    assert _coerce("null") is None

def test_coerce_string():
    assert _coerce("hello") == "hello"


# --- _set_nested ---

def test_set_nested_simple():
    d = {}
    _set_nested(d, "run.seed", "99")
    assert d == {"run": {"seed": 99}}

def test_set_nested_deep():
    d = {"model": {"params": {}}}
    _set_nested(d, "model.params.num_topics", "30")
    assert d["model"]["params"]["num_topics"] == 30

def test_set_nested_override_existing():
    d = {"run": {"seed": 42}}
    _set_nested(d, "run.seed", "7")
    assert d["run"]["seed"] == 7


# --- load_config ---

def test_load_baseline_bertopic(tmp_path):
    config_path = (
        Path(__file__).parent.parent / "configs/experiment/baseline_bertopic.yaml"
    )
    cfg = load_config(str(config_path))
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.model.backend == "bertopic"
    assert cfg.run.seed == 42

def test_load_config_with_override(tmp_path):
    config_path = (
        Path(__file__).parent.parent / "configs/experiment/baseline_bertopic.yaml"
    )
    cfg = load_config(str(config_path), overrides=["run.seed=99"])
    assert cfg.run.seed == 99

def test_load_config_invalid_override():
    config_path = (
        Path(__file__).parent.parent / "configs/experiment/baseline_bertopic.yaml"
    )
    with pytest.raises(ValueError, match="key=value"):
        load_config(str(config_path), overrides=["run.seed"])

"""Tests for ArtifactStore local persistence."""
import json
from pathlib import Path

import pandas as pd
import pytest

from topic_modeling.utils.artifacts import ArtifactStore


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(run_dir=tmp_path / "run_001", target="local")


def test_save_json(store, tmp_path):
    data = {"metric": 0.72, "count": 5}
    path = store.save_json(data, "metrics.json")
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded["metric"] == pytest.approx(0.72)
    assert loaded["count"] == 5


def test_save_jsonl(store):
    records = [{"id": 0, "value": "a"}, {"id": 1, "value": "b"}]
    path = store.save_jsonl(records, "output.jsonl")
    assert path.exists()
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["value"] == "a"


def test_save_text(store):
    path = store.save_text("hello world", "report.md")
    assert path.exists()
    assert path.read_text() == "hello world"


def test_save_csv(store):
    df = pd.DataFrame({"topic_id": [0, 1], "terms": ["foo", "bar"]})
    path = store.save_csv(df, "topics.csv")
    assert path.exists()
    loaded = pd.read_csv(path)
    assert list(loaded["topic_id"]) == [0, 1]


def test_nested_artifact_path(store):
    """Artifacts can be saved to subdirectories."""
    path = store.save_json({"x": 1}, "artifacts/model/meta.json")
    assert path.exists()
    assert path.parent.name == "model"


def test_run_dir_created_on_init(tmp_path):
    run_dir = tmp_path / "new_run"
    assert not run_dir.exists()
    ArtifactStore(run_dir=run_dir, target="local")
    assert run_dir.exists()

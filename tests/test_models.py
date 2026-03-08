"""Tests for model adapter interface contract."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topic_modeling.config.schema import ModelConfig
from topic_modeling.models import build_model


def _inject_bertopic_mocks(mock_instance: MagicMock) -> dict:
    """Inject fake bertopic/umap/hdbscan/sentence_transformers into sys.modules."""
    mock_bertopic_cls = MagicMock(return_value=mock_instance)

    mocks = {
        "bertopic": MagicMock(BERTopic=mock_bertopic_cls),
        "umap": MagicMock(UMAP=MagicMock()),
        "hdbscan": MagicMock(HDBSCAN=MagicMock()),
        "sentence_transformers": MagicMock(SentenceTransformer=MagicMock()),
    }
    for name, mod in mocks.items():
        sys.modules[name] = mod
    return mocks


def _inject_fastopic_mocks(mock_instance: MagicMock) -> dict:
    mock_fastopic_cls = MagicMock(return_value=mock_instance)
    mocks = {"fastopic": MagicMock(FASTopic=mock_fastopic_cls)}
    for name, mod in mocks.items():
        sys.modules[name] = mod
    return mocks


def _remove_mocks(names: list) -> None:
    for name in names:
        sys.modules.pop(name, None)


# ---------------------------------------------------------------------------


def test_build_model_unknown_backend():
    cfg = ModelConfig(backend="bertopic")
    cfg.backend = "unknown"  # type: ignore
    with pytest.raises(ValueError, match="Unknown model backend"):
        build_model(cfg)


def test_model_interface_bertopic():
    """Smoke test BERTopic adapter with mocked library."""
    mock_instance = MagicMock()
    mock_instance.get_topics.return_value = {
        0: [("word1", 0.9), ("word2", 0.8)],
        1: [("word3", 0.7), ("word4", 0.6)],
    }
    mock_instance.get_topic_info.return_value = MagicMock()
    mock_instance.get_representative_docs.return_value = ["doc1", "doc2"]
    mock_instance.fit_transform.return_value = ([0, 1, 0], None)
    mock_instance.transform.return_value = ([0, 1], None)

    injected = _inject_bertopic_mocks(mock_instance)
    try:
        from topic_modeling.models.bertopic_adapter import BERTopicAdapter

        cfg = ModelConfig(
            backend="bertopic",
            params={
                "embedding_model": "all-MiniLM-L6-v2",
                "umap_n_components": 2,
                "umap_n_neighbors": 5,
                "hdbscan_min_cluster_size": 2,
            },
        )

        adapter = BERTopicAdapter(cfg, seed=42)
        adapter._model = mock_instance

        texts = ["doc one text", "doc two text", "doc three text"]
        adapter.fit(texts)

        assert adapter.get_topic_count() == 2
        topics = adapter.get_topics()
        assert 0 in topics
        assert 1 in topics

        reps = adapter.get_representative_docs(0, n=2)
        assert isinstance(reps, list)
    finally:
        _remove_mocks(list(injected))
        sys.modules.pop("topic_modeling.models.bertopic_adapter", None)


def test_model_interface_fastopic():
    """Smoke test FASTopic adapter with mocked library."""
    import numpy as np

    mock_instance = MagicMock()
    theta = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    mock_instance.get_theta.return_value = theta
    mock_instance.get_top_words.return_value = [
        ["word1", "word2"],
        ["word3", "word4"],
        ["word5", "word6"],
    ]
    mock_instance.get_beta.return_value = np.random.rand(3, 100)

    injected = _inject_fastopic_mocks(mock_instance)
    try:
        from topic_modeling.models.fastopic_adapter import FASTopicAdapter

        cfg = ModelConfig(backend="fastopic", params={"num_topics": 3, "num_top_words": 5})
        adapter = FASTopicAdapter(cfg, seed=42)

        texts = ["first document text", "second document text", "third document"]
        adapter.fit(texts)

        assert adapter.get_topic_count() == 3

        reps = adapter.get_representative_docs(0, n=2)
        assert isinstance(reps, list)
        assert len(reps) <= 2
    finally:
        _remove_mocks(list(injected))
        sys.modules.pop("topic_modeling.models.fastopic_adapter", None)

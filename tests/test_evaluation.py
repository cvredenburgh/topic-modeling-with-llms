"""Tests for evaluation metrics."""
from unittest.mock import MagicMock

import numpy as np
import pytest

from topic_modeling.config.schema import EvaluationConfig
from topic_modeling.evaluation.metrics import (
    _diversity,
    _dist_entropy,
    _per_topic_diversity,
    _silhouette,
    evaluate,
)


def _make_model(topics: dict, rep_docs: list | None = None) -> MagicMock:
    model = MagicMock()
    model.get_topics.return_value = topics
    model.get_representative_docs.return_value = rep_docs or []
    return model


# --- diversity ---

def test_diversity_all_unique():
    topic_words = {
        0: ["word1", "word2", "word3"],
        1: ["word4", "word5", "word6"],
    }
    score = _diversity(topic_words, topn=3)
    assert score == pytest.approx(1.0)


def test_diversity_all_same():
    topic_words = {
        0: ["word1", "word1", "word1"],
        1: ["word1", "word1", "word1"],
    }
    score = _diversity(topic_words, topn=3)
    assert score == pytest.approx(1 / 6)


def test_diversity_empty():
    assert _diversity({}, topn=10) == 0.0


def test_diversity_partial_overlap():
    topic_words = {
        0: ["apple", "banana"],
        1: ["banana", "cherry"],
    }
    score = _diversity(topic_words, topn=2)
    # 4 total words, 3 unique
    assert score == pytest.approx(3 / 4)


# --- outlier ratio ---

def test_outlier_ratio_none():
    model = _make_model({0: [("w", 0.9)], 1: [("x", 0.8)]})
    cfg = EvaluationConfig(metrics=["outlier_ratio"])
    topic_ids = [0, 1, 0, 1]
    metrics = evaluate(model, ["a", "b", "c", "d"], topic_ids, cfg)
    assert metrics["outlier_ratio"] == 0.0
    assert metrics["outlier_count"] == 0


def test_outlier_ratio_some():
    model = _make_model({0: [("w", 0.9)], -1: []})
    cfg = EvaluationConfig(metrics=["outlier_ratio"])
    topic_ids = [0, -1, -1, 0]
    metrics = evaluate(model, ["a", "b", "c", "d"], topic_ids, cfg)
    assert metrics["outlier_ratio"] == pytest.approx(0.5)
    assert metrics["outlier_count"] == 2


def test_outlier_ratio_all():
    model = _make_model({-1: []})
    cfg = EvaluationConfig(metrics=["outlier_ratio"])
    topic_ids = [-1, -1, -1]
    metrics = evaluate(model, ["a", "b", "c"], topic_ids, cfg)
    assert metrics["outlier_ratio"] == pytest.approx(1.0)


# --- topic size stats ---

def test_topic_size_stats():
    model = _make_model({
        0: [("w1", 0.9)],
        1: [("w2", 0.8)],
    })
    cfg = EvaluationConfig(metrics=["topic_size_stats"])
    topic_ids = [0, 0, 0, 1, 1]
    metrics = evaluate(model, ["a"] * 5, topic_ids, cfg)
    assert metrics["topic_count"] == 2
    assert metrics["topic_size_mean"] == pytest.approx(2.5)
    assert metrics["topic_size_min"] == 2
    assert metrics["topic_size_max"] == 3


# --- combined metrics ---

def test_evaluate_combined():
    model = _make_model({
        0: [("alpha", 0.9), ("beta", 0.7)],
        1: [("gamma", 0.8), ("delta", 0.6)],
    })
    cfg = EvaluationConfig(metrics=["diversity", "outlier_ratio", "topic_size_stats"])
    topic_ids = [0, 0, 1, 1, -1]
    metrics = evaluate(model, ["doc"] * 5, topic_ids, cfg)
    assert "diversity" in metrics
    assert "outlier_ratio" in metrics
    assert "topic_count" in metrics
    assert metrics["diversity"] == pytest.approx(1.0)


# --- dist_entropy ---

def test_dist_entropy_uniform():
    # Equal sizes -> maximum entropy for 2 topics
    topic_ids = [0, 0, 1, 1]
    entropy = _dist_entropy(topic_ids)
    assert entropy > 0.0


def test_dist_entropy_skewed():
    # Very unequal sizes -> low entropy
    topic_ids = [0] * 99 + [1]
    entropy_skewed = _dist_entropy(topic_ids)
    topic_ids_equal = [0] * 50 + [1] * 50
    entropy_equal = _dist_entropy(topic_ids_equal)
    assert entropy_skewed < entropy_equal


def test_dist_entropy_outliers_excluded():
    # -1 (outliers) should not count toward entropy
    topic_ids = [0, 0, -1, -1, 1, 1]
    entropy = _dist_entropy(topic_ids)
    assert entropy > 0.0


# --- size_ratio_max_min ---

def test_size_ratio_max_min():
    model = _make_model({0: [("w", 0.9)], 1: [("x", 0.8)]})
    cfg = EvaluationConfig(metrics=["size_ratio_max_min"])
    topic_ids = [0, 0, 0, 1]  # sizes: {0: 3, 1: 1}
    metrics = evaluate(model, ["a"] * 4, topic_ids, cfg)
    assert metrics["size_ratio_max_min"] == pytest.approx(3.0)


# --- size_ratio_max_median ---

def test_size_ratio_max_median():
    model = _make_model({0: [("w", 0.9)], 1: [("x", 0.8)], 2: [("y", 0.7)]})
    cfg = EvaluationConfig(metrics=["size_ratio_max_median"])
    topic_ids = [0, 0, 0, 1, 2]  # sizes: {0:3, 1:1, 2:1}, median=1
    metrics = evaluate(model, ["a"] * 5, topic_ids, cfg)
    assert metrics["size_ratio_max_median"] == pytest.approx(3.0)


# --- per-topic diversity ---

def test_per_topic_diversity_disjoint():
    topic_words = {0: ["a", "b"], 1: ["c", "d"]}
    result = _per_topic_diversity(topic_words, topn=2)
    # Jaccard distance = 1 when sets are disjoint
    assert result["0"] == pytest.approx(1.0)
    assert result["1"] == pytest.approx(1.0)


def test_per_topic_diversity_identical():
    topic_words = {0: ["a", "b"], 1: ["a", "b"]}
    result = _per_topic_diversity(topic_words, topn=2)
    # Jaccard distance = 0 when sets are identical
    assert result["0"] == pytest.approx(0.0)


def test_per_topic_diversity_single_topic():
    topic_words = {0: ["a", "b"]}
    result = _per_topic_diversity(topic_words, topn=2)
    assert result["0"] == pytest.approx(0.0)


# --- silhouette (with mocked embeddings) ---

def test_silhouette_with_embeddings():
    model = _make_model({0: [("w", 0.9)], 1: [("x", 0.8)]})
    cfg = EvaluationConfig(metrics=["silhouette"])
    topic_ids = [0, 0, 0, 1, 1, 1]
    # Clearly separated 2-D embeddings
    embeddings = np.array([
        [0.0, 0.0], [0.1, 0.0], [0.0, 0.1],
        [5.0, 5.0], [5.1, 5.0], [5.0, 5.1],
    ])
    metrics = evaluate(model, ["a"] * 6, topic_ids, cfg, embeddings=embeddings)
    assert "silhouette" in metrics
    assert metrics["silhouette"] > 0.5  # well-separated clusters


def test_silhouette_skipped_when_no_embeddings():
    model = _make_model({0: [("w", 0.9)], 1: [("x", 0.8)]})
    cfg = EvaluationConfig(metrics=["silhouette"])
    topic_ids = [0, 0, 1, 1]
    metrics = evaluate(model, ["a"] * 4, topic_ids, cfg)  # no embeddings
    assert "silhouette" not in metrics


# --- per_topic_metrics in evaluate ---

def test_evaluate_per_topic_diversity():
    model = _make_model({
        0: [("apple", 0.9), ("banana", 0.7)],
        1: [("car", 0.8), ("bus", 0.6)],
    })
    cfg = EvaluationConfig(metrics=["per_topic_diversity"])
    topic_ids = [0, 0, 1, 1]
    metrics = evaluate(model, ["doc"] * 4, topic_ids, cfg)
    assert "per_topic_metrics" in metrics
    assert "diversity" in metrics["per_topic_metrics"]
    assert "0" in metrics["per_topic_metrics"]["diversity"]

"""Tests for topic hierarchy builder."""
import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from topic_modeling.analysis.hierarchy import (
    _cosine_distance,
    _jaccard_distance,
    build_topic_hierarchy,
)


def _make_mock_model(topics: dict) -> MagicMock:
    model = MagicMock()
    model.get_topics.return_value = topics
    return model


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def test_cosine_distance_identical():
    a = np.array([1.0, 0.5, 0.0])
    assert _cosine_distance(a, a) == pytest.approx(0.0, abs=1e-9)


def test_cosine_distance_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert _cosine_distance(a, b) == pytest.approx(1.0, abs=1e-9)


def test_cosine_distance_zero_vector():
    a = np.zeros(3)
    b = np.array([1.0, 2.0, 3.0])
    assert _cosine_distance(a, b) == 1.0


def test_jaccard_distance_identical():
    s = {"a", "b", "c"}
    assert _jaccard_distance(s, s) == pytest.approx(0.0, abs=1e-9)


def test_jaccard_distance_disjoint():
    assert _jaccard_distance({"a"}, {"b"}) == pytest.approx(1.0, abs=1e-9)


def test_jaccard_distance_partial():
    a = {"a", "b", "c"}
    b = {"b", "c", "d"}
    # |intersection|=2, |union|=4 → Jaccard sim=0.5 → dist=0.5
    assert _jaccard_distance(a, b) == pytest.approx(0.5, abs=1e-9)


def test_jaccard_distance_both_empty():
    assert _jaccard_distance(set(), set()) == 0.0


# ---------------------------------------------------------------------------
# build_topic_hierarchy
# ---------------------------------------------------------------------------

def _sample_topics():
    return {
        0: [("machine", 0.9), ("learning", 0.8), ("model", 0.7)],
        1: [("neural", 0.9), ("network", 0.8), ("deep", 0.7)],
        2: [("policy", 0.9), ("government", 0.8), ("law", 0.7)],
        -1: [("noise", 0.1)],  # should be ignored
    }


def test_hierarchy_output_schema():
    model = _make_mock_model(_sample_topics())
    df = build_topic_hierarchy(model, keyword_weight=0.7, n_keywords=3)

    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"parent_id", "child_id", "linkage_height", "similarity"}


def test_hierarchy_row_count():
    """For n topics, agglomerative clustering produces n-1 merges × 2 rows."""
    model = _make_mock_model(_sample_topics())
    df = build_topic_hierarchy(model, keyword_weight=0.7, n_keywords=3)
    n_real_topics = 3  # ids 0, 1, 2
    assert len(df) == (n_real_topics - 1) * 2


def test_hierarchy_similarity_in_range():
    model = _make_mock_model(_sample_topics())
    df = build_topic_hierarchy(model, keyword_weight=0.7, n_keywords=3)
    assert (df["similarity"] >= 0).all()
    assert (df["similarity"] <= 1).all()


def test_hierarchy_linkage_height_non_negative():
    model = _make_mock_model(_sample_topics())
    df = build_topic_hierarchy(model, keyword_weight=0.7, n_keywords=3)
    assert (df["linkage_height"] >= 0).all()


def test_hierarchy_children_are_known_topic_ids():
    """All child_id values at the leaf level should be original topic ids."""
    topics = _sample_topics()
    real_ids = {t for t in topics if t != -1}
    model = _make_mock_model(topics)
    df = build_topic_hierarchy(model, keyword_weight=0.7, n_keywords=3)

    # Leaf children — those child_ids that are in real topic set
    leaf_children = df[df["child_id"].isin(real_ids)]["child_id"].unique()
    assert set(leaf_children) == real_ids


def test_hierarchy_single_topic_returns_empty():
    model = _make_mock_model({0: [("word", 0.9)]})
    df = build_topic_hierarchy(model)
    assert df.empty


def test_hierarchy_only_outliers_returns_empty():
    model = _make_mock_model({-1: [("noise", 0.1)]})
    df = build_topic_hierarchy(model)
    assert df.empty


def test_hierarchy_two_topics():
    topics = {
        0: [("cat", 0.9), ("dog", 0.8)],
        1: [("sky", 0.9), ("cloud", 0.8)],
    }
    model = _make_mock_model(topics)
    df = build_topic_hierarchy(model, n_keywords=2)
    assert len(df) == 2  # one merge, two child rows


def test_hierarchy_keyword_weight_zero():
    """Pure Jaccard mode should still produce a valid DataFrame."""
    model = _make_mock_model(_sample_topics())
    df = build_topic_hierarchy(model, keyword_weight=0.0, n_keywords=3)
    assert not df.empty


def test_hierarchy_keyword_weight_one():
    """Pure cosine mode should still produce a valid DataFrame."""
    model = _make_mock_model(_sample_topics())
    df = build_topic_hierarchy(model, keyword_weight=1.0, n_keywords=3)
    assert not df.empty

"""Tests for pairwise topic association metrics."""
import pandas as pd
import pytest

from topic_modeling.analysis.associations import compute_topic_associations


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_assignments():
    df = compute_topic_associations([])
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_all_outliers():
    df = compute_topic_associations([-1, -1, -1])
    assert df.empty


def test_single_topic():
    df = compute_topic_associations([0, 0, 0, 0, 0])
    assert df.empty


def test_output_schema():
    assignments = list(range(10)) * 5  # topics 0-9, 50 docs
    df = compute_topic_associations(assignments, min_cooccurrence=1)
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"topic_a", "topic_b", "jaccard", "pmi", "lift"}


# ---------------------------------------------------------------------------
# Correctness checks
# ---------------------------------------------------------------------------

def _alternating(n: int) -> list[int]:
    """Sequence 0,1,0,1,... of length n — high adjacency between 0 and 1."""
    return [i % 2 for i in range(n)]


def test_alternating_topics_high_cooccurrence():
    """Topics 0 and 1 always appear next to each other."""
    assignments = _alternating(50)
    df = compute_topic_associations(assignments, min_cooccurrence=1)
    assert not df.empty
    row = df[(df["topic_a"] == 0) & (df["topic_b"] == 1)]
    assert not row.empty
    assert row.iloc[0]["lift"] > 1.0  # co-occur more than chance


def test_jaccard_between_zero_and_one():
    assignments = _alternating(50)
    df = compute_topic_associations(assignments, min_cooccurrence=1)
    assert (df["jaccard"] >= 0).all()
    assert (df["jaccard"] <= 1).all()


def test_pmi_finite():
    assignments = _alternating(50)
    df = compute_topic_associations(assignments, min_cooccurrence=1)
    assert df["pmi"].apply(lambda x: x == x).all()  # no NaN


def test_lift_positive():
    assignments = _alternating(50)
    df = compute_topic_associations(assignments, min_cooccurrence=1)
    assert (df["lift"] > 0).all()


def test_min_cooccurrence_filter():
    """High threshold should filter out rare pairs."""
    assignments = [0, 1] + [0] * 100  # only 1 adjacent pair (0,1)
    df_low = compute_topic_associations(assignments, min_cooccurrence=1)
    df_high = compute_topic_associations(assignments, min_cooccurrence=10)
    assert len(df_low) >= len(df_high)
    assert df_high.empty  # only 1 co-occurrence, below threshold 10


def test_outliers_excluded():
    """Outlier documents (-1) should not appear as topics."""
    assignments = [0, -1, 1, 0, -1, 1] * 10
    df = compute_topic_associations(assignments, min_cooccurrence=1)
    assert -1 not in df["topic_a"].values
    assert -1 not in df["topic_b"].values


def test_topic_a_less_than_topic_b():
    """topic_a should always be < topic_b (canonical ordering)."""
    assignments = list(range(5)) * 20
    df = compute_topic_associations(assignments, min_cooccurrence=1)
    if not df.empty:
        assert (df["topic_a"] < df["topic_b"]).all()


def test_many_topics_returns_dataframe():
    import random
    random.seed(0)
    assignments = [random.randint(0, 9) for _ in range(200)]
    df = compute_topic_associations(assignments, min_cooccurrence=1)
    assert isinstance(df, pd.DataFrame)

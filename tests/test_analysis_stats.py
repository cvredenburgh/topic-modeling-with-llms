"""Tests for statistical significance tests on topic distributions."""
import numpy as np
import pandas as pd
import pytest

from topic_modeling.analysis.stats import compare_topic_prevalence, test_topic_trend_significance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trend_df(counts_by_topic: dict[int, list[int]]) -> pd.DataFrame:
    """Build a synthetic trend DataFrame with given per-topic doc_count sequences."""
    n_periods = max(len(v) for v in counts_by_topic.values())
    periods = [f"2023-{i+1:02d}" for i in range(n_periods)]
    rows = []
    for tid, counts in counts_by_topic.items():
        for i, c in enumerate(counts):
            rows.append({"topic_id": tid, "period": periods[i], "doc_count": c,
                         "ci_low": 0.0, "ci_high": 1.0, "share": 0.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# test_topic_trend_significance
# ---------------------------------------------------------------------------

def test_trend_significance_output_schema():
    trend_df = _make_trend_df({0: [10, 10, 10]})
    df = test_topic_trend_significance(trend_df)
    assert set(df.columns) == {"topic_id", "chi2", "p_value", "significant"}


def test_trend_significance_empty():
    df = test_topic_trend_significance(pd.DataFrame(
        columns=["topic_id", "period", "doc_count", "ci_low", "ci_high", "share"]
    ))
    assert df.empty


def test_trend_significance_uniform_not_significant():
    """A perfectly uniform distribution should not be significant."""
    trend_df = _make_trend_df({0: [10, 10, 10, 10, 10]})
    df = test_topic_trend_significance(trend_df, alpha=0.05)
    row = df[df["topic_id"] == 0]
    assert not row.iloc[0]["significant"]


def test_trend_significance_skewed_is_significant():
    """Strongly skewed counts should yield a significant result."""
    trend_df = _make_trend_df({0: [100, 1, 1, 1, 1]})
    df = test_topic_trend_significance(trend_df, alpha=0.05)
    row = df[df["topic_id"] == 0]
    assert row.iloc[0]["significant"]


def test_trend_significance_p_value_in_range():
    trend_df = _make_trend_df({0: [10, 20, 5], 1: [8, 8, 8]})
    df = test_topic_trend_significance(trend_df)
    assert (df["p_value"] >= 0).all()
    assert (df["p_value"] <= 1).all()


def test_trend_significance_chi2_non_negative():
    trend_df = _make_trend_df({0: [10, 20, 30]})
    df = test_topic_trend_significance(trend_df)
    assert (df["chi2"] >= 0).all()


def test_trend_significance_zero_counts():
    trend_df = _make_trend_df({0: [0, 0, 0]})
    df = test_topic_trend_significance(trend_df, alpha=0.05)
    row = df[df["topic_id"] == 0]
    assert not row.iloc[0]["significant"]
    assert row.iloc[0]["p_value"] == pytest.approx(1.0)


def test_trend_significance_multiple_topics():
    trend_df = _make_trend_df({
        0: [100, 1, 1, 1, 1],  # strongly skewed → significant
        1: [10, 10, 10, 10, 10],  # uniform → not significant
    })
    df = test_topic_trend_significance(trend_df, alpha=0.05)
    assert df[df["topic_id"] == 0]["significant"].iloc[0]
    assert not df[df["topic_id"] == 1]["significant"].iloc[0]


# ---------------------------------------------------------------------------
# compare_topic_prevalence
# ---------------------------------------------------------------------------

def test_compare_output_schema():
    a = [0, 0, 1, 1, 2]
    b = [0, 1, 2, 2, 2]
    df = compare_topic_prevalence(a, b)
    assert set(df.columns) == {"topic_id", "count_a", "count_b", "chi2", "p_value", "significant"}


def test_compare_empty_both():
    df = compare_topic_prevalence([], [])
    assert df.empty


def test_compare_outliers_excluded():
    a = [0, -1, 1]
    b = [-1, 1, 2]
    df = compare_topic_prevalence(a, b)
    assert -1 not in df["topic_id"].values


def test_compare_counts_correct():
    a = [0, 0, 0, 1]
    b = [1, 1, 2]
    df = compare_topic_prevalence(a, b)
    row0 = df[df["topic_id"] == 0]
    row1 = df[df["topic_id"] == 1]
    row2 = df[df["topic_id"] == 2]
    assert row0.iloc[0]["count_a"] == 3
    assert row0.iloc[0]["count_b"] == 0
    assert row1.iloc[0]["count_a"] == 1
    assert row1.iloc[0]["count_b"] == 2
    assert row2.iloc[0]["count_a"] == 0
    assert row2.iloc[0]["count_b"] == 2


def test_compare_identical_distributions_not_significant():
    """Same topic distribution across both groups → should not be significant."""
    a = [0] * 50 + [1] * 50
    b = [0] * 50 + [1] * 50
    df = compare_topic_prevalence(a, b, alpha=0.05)
    assert not df.iloc[0]["significant"]


def test_compare_very_different_distributions_significant():
    """Group A only has topic 0, group B only has topic 1 → significant."""
    a = [0] * 100
    b = [1] * 100
    df = compare_topic_prevalence(a, b, alpha=0.05)
    assert df.iloc[0]["significant"]


def test_compare_p_value_in_range():
    a = [0, 1, 2, 0, 1]
    b = [2, 2, 0, 1, 0]
    df = compare_topic_prevalence(a, b)
    assert (df["p_value"] >= 0).all()
    assert (df["p_value"] <= 1).all()


def test_compare_all_topics_appear_in_output():
    """Union of topics from both groups should all appear in output."""
    a = [0, 1]
    b = [2, 3]
    df = compare_topic_prevalence(a, b)
    assert set(df["topic_id"].tolist()) == {0, 1, 2, 3}


def test_compare_single_group_empty():
    """One group being empty should still produce output for the other group's topics."""
    a = [0, 1, 0]
    b = []
    df = compare_topic_prevalence(a, b)
    assert not df.empty
    assert set(df["topic_id"].tolist()) == {0, 1}

"""Tests for temporal trend analysis and emerging topic detection."""
import pandas as pd
import pytest

from topic_modeling.analysis.trends import compute_topic_trends, detect_emerging_topics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_docs(n_months: int, topics_per_month: dict[int, list[int]]) -> tuple[list[str], list[int]]:
    """Build parallel (dates, assignments) lists.

    topics_per_month: {month_index: [topic_id, ...]}
    """
    from datetime import date
    dates, assignments = [], []
    base_year = 2023
    for month_idx in range(n_months):
        year = base_year + (month_idx // 12)
        month = (month_idx % 12) + 1
        d = date(year, month, 15).isoformat()
        for tid in topics_per_month.get(month_idx, []):
            dates.append(d)
            assignments.append(tid)
    return dates, assignments


# ---------------------------------------------------------------------------
# compute_topic_trends
# ---------------------------------------------------------------------------

def test_trends_output_schema():
    dates = ["2023-01-15", "2023-02-15", "2023-03-15"]
    assignments = [0, 1, 0]
    df = compute_topic_trends(dates, assignments, freq="ME")
    assert set(df.columns) == {"topic_id", "period", "doc_count", "ci_low", "ci_high", "share"}


def test_trends_empty_when_all_outliers():
    dates = ["2023-01-15", "2023-02-15"]
    assignments = [-1, -1]
    df = compute_topic_trends(dates, assignments)
    assert df.empty


def test_trends_doc_count_correct():
    dates = ["2023-01-15", "2023-01-20", "2023-01-25"]
    assignments = [0, 0, 1]
    df = compute_topic_trends(dates, assignments, freq="ME")
    jan = df[df["period"].str.startswith("2023-01")]
    count_0 = jan[jan["topic_id"] == 0]["doc_count"].sum()
    count_1 = jan[jan["topic_id"] == 1]["doc_count"].sum()
    assert count_0 == 2
    assert count_1 == 1


def test_trends_share_sums_to_one_per_period():
    """Shares across all topics within a period should sum to 1."""
    dates = ["2023-01-15"] * 5 + ["2023-02-15"] * 5
    assignments = [0, 1, 2, 0, 1] + [0, 0, 1, 2, 2]
    df = compute_topic_trends(dates, assignments, freq="ME")
    for period in df["period"].unique():
        period_share = df[df["period"] == period]["share"].sum()
        assert period_share == pytest.approx(1.0, abs=1e-6)


def test_trends_ci_bounds_valid():
    dates = ["2023-01-15"] * 10
    assignments = [i % 3 for i in range(10)]
    df = compute_topic_trends(dates, assignments, freq="ME")
    assert (df["ci_low"] >= 0).all()
    assert (df["ci_high"] <= 1).all()
    assert (df["ci_low"] <= df["ci_high"]).all()


def test_trends_multiple_periods():
    dates, assignments = _make_docs(
        n_months=4,
        topics_per_month={0: [0, 0, 1], 1: [0, 1, 1], 2: [2, 2, 2], 3: [0, 1, 2]},
    )
    df = compute_topic_trends(dates, assignments, freq="ME")
    assert df["period"].nunique() == 4


def test_trends_outliers_excluded():
    dates = ["2023-01-15", "2023-01-16", "2023-01-17"]
    assignments = [0, -1, 1]
    df = compute_topic_trends(dates, assignments, freq="ME")
    assert -1 not in df["topic_id"].values


def test_trends_quarterly_freq():
    dates = ["2023-01-15", "2023-04-15", "2023-07-15"]
    assignments = [0, 1, 2]
    df = compute_topic_trends(dates, assignments, freq="QE")
    assert df["period"].nunique() == 3


# ---------------------------------------------------------------------------
# detect_emerging_topics
# ---------------------------------------------------------------------------

def _make_trend_df(shares_by_topic: dict[int, list[float]], freq: str = "ME") -> pd.DataFrame:
    """Build a synthetic trend DataFrame with given per-topic share sequences."""
    from datetime import date
    rows = []
    base = date(2023, 1, 1)
    n_periods = max(len(v) for v in shares_by_topic.values())
    periods = [f"2023-{i+1:02d}" for i in range(n_periods)]
    total_docs = 100  # arbitrary
    for tid, shares in shares_by_topic.items():
        for i, share in enumerate(shares):
            rows.append({
                "topic_id": tid,
                "period": periods[i],
                "doc_count": int(share * total_docs),
                "ci_low": max(0, share - 0.02),
                "ci_high": min(1, share + 0.02),
                "share": share,
            })
    return pd.DataFrame(rows)


def test_emerging_output_schema():
    trend_df = _make_trend_df({0: [0.1, 0.2, 0.3]})
    df = detect_emerging_topics(trend_df)
    assert set(df.columns) == {"topic_id", "growth_rate", "emerging"}


def test_emerging_empty_trend():
    df = detect_emerging_topics(pd.DataFrame(columns=["topic_id", "period", "doc_count", "ci_low", "ci_high", "share"]))
    assert df.empty


def test_emerging_growing_topic_flagged():
    trend_df = _make_trend_df({0: [0.05, 0.10, 0.20]})  # 4× growth
    df = detect_emerging_topics(trend_df, window=3, min_growth_rate=0.5)
    assert bool(df[df["topic_id"] == 0]["emerging"].iloc[0]) is True


def test_emerging_stable_topic_not_flagged():
    trend_df = _make_trend_df({0: [0.20, 0.20, 0.20]})  # flat
    df = detect_emerging_topics(trend_df, window=3, min_growth_rate=0.5)
    assert not df[df["topic_id"] == 0]["emerging"].iloc[0]


def test_emerging_declining_topic_not_flagged():
    trend_df = _make_trend_df({0: [0.30, 0.20, 0.10]})  # declining
    df = detect_emerging_topics(trend_df, window=3, min_growth_rate=0.5)
    assert not df[df["topic_id"] == 0]["emerging"].iloc[0]


def test_emerging_multiple_topics():
    trend_df = _make_trend_df({
        0: [0.05, 0.10, 0.20],  # emerging
        1: [0.20, 0.20, 0.20],  # stable
        2: [0.30, 0.10, 0.05],  # declining
    })
    df = detect_emerging_topics(trend_df, window=3, min_growth_rate=0.5)
    assert df[df["topic_id"] == 0]["emerging"].iloc[0]
    assert not df[df["topic_id"] == 1]["emerging"].iloc[0]
    assert not df[df["topic_id"] == 2]["emerging"].iloc[0]


def test_emerging_window_larger_than_periods():
    """Should not crash when window > number of available periods."""
    trend_df = _make_trend_df({0: [0.1, 0.3]})
    df = detect_emerging_topics(trend_df, window=10)
    assert not df.empty


def test_emerging_from_zero_share_infinite_growth():
    """Topic going from 0 to non-zero should be flagged as emerging."""
    trend_df = _make_trend_df({0: [0.0, 0.0, 0.1]})
    df = detect_emerging_topics(trend_df, window=3, min_growth_rate=0.5)
    assert df[df["topic_id"] == 0]["emerging"].iloc[0]

"""Temporal topic trend analysis and emerging topic detection."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def compute_topic_trends(
    doc_dates: List[str],
    doc_topic_assignments: List[int],
    freq: str = "ME",
) -> pd.DataFrame:
    """Topic document counts per time bucket with 95% bootstrap CI.

    Parameters
    ----------
    doc_dates:
        ISO-format date strings parallel to ``doc_topic_assignments``.
    doc_topic_assignments:
        Topic id per document (-1 outliers are excluded).
    freq:
        Pandas period alias for bucketing (``"ME"`` = month-end,
        ``"QE"`` = quarter-end, ``"W"`` = week).

    Returns
    -------
    pd.DataFrame
        Columns: ``topic_id``, ``period``, ``doc_count``, ``ci_low``,
        ``ci_high``, ``share``.
    """
    df = pd.DataFrame({"date": pd.to_datetime(doc_dates), "topic_id": doc_topic_assignments})
    df = df[df["topic_id"] != -1].copy()
    if df.empty:
        return pd.DataFrame(columns=["topic_id", "period", "doc_count", "ci_low", "ci_high", "share"])

    df["period"] = df["date"].dt.to_period(freq)
    periods = sorted(df["period"].unique())
    topics = sorted(df["topic_id"].unique())

    rows = []
    rng = np.random.default_rng(42)

    for period in periods:
        period_mask = df["period"] == period
        period_df = df[period_mask]
        period_total = len(period_df)
        topic_counts = period_df["topic_id"].value_counts()

        for tid in topics:
            count = int(topic_counts.get(tid, 0))
            share = count / period_total if period_total > 0 else 0.0

            # Bootstrap 95% CI on share
            ci_low, ci_high = _bootstrap_ci(period_df["topic_id"].values, tid, rng)

            rows.append(
                {
                    "topic_id": tid,
                    "period": str(period),
                    "doc_count": count,
                    "ci_low": round(ci_low, 6),
                    "ci_high": round(ci_high, 6),
                    "share": round(share, 6),
                }
            )

    return pd.DataFrame(rows)


def detect_emerging_topics(
    trend_df: pd.DataFrame,
    window: int = 3,
    min_growth_rate: float = 0.5,
) -> pd.DataFrame:
    """Flag topics with accelerating share growth over the last N periods.

    Parameters
    ----------
    trend_df:
        Output of :func:`compute_topic_trends`.
    window:
        Number of most-recent periods to examine.
    min_growth_rate:
        Minimum fractional growth in share required to flag as emerging.
        E.g. 0.5 means share must grow ≥ 50% from first to last window period.

    Returns
    -------
    pd.DataFrame
        Columns: ``topic_id``, ``growth_rate``, ``emerging``.
    """
    if trend_df.empty:
        return pd.DataFrame(columns=["topic_id", "growth_rate", "emerging"])

    periods = sorted(trend_df["period"].unique())
    recent_periods = periods[-window:] if len(periods) >= window else periods

    rows = []
    for tid in sorted(trend_df["topic_id"].unique()):
        topic_df = trend_df[trend_df["topic_id"] == tid].set_index("period")
        shares = [float(topic_df.loc[p, "share"]) if p in topic_df.index else 0.0 for p in recent_periods]

        if len(shares) < 2:
            growth_rate = 0.0
        else:
            start = shares[0]
            end = shares[-1]
            if start == 0:
                growth_rate = float("inf") if end > 0 else 0.0
            else:
                growth_rate = (end - start) / start

        emerging = growth_rate >= min_growth_rate
        rows.append({"topic_id": tid, "growth_rate": round(growth_rate, 6), "emerging": emerging})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    labels: np.ndarray,
    topic_id: int,
    rng: np.random.Generator,
    n_bootstrap: int = 500,
    confidence: float = 0.95,
) -> tuple[float, float]:
    n = len(labels)
    if n == 0:
        return 0.0, 0.0

    shares = []
    for _ in range(n_bootstrap):
        sample = rng.choice(labels, size=n, replace=True)
        shares.append(np.mean(sample == topic_id))

    alpha = 1.0 - confidence
    lo = float(np.percentile(shares, 100 * alpha / 2))
    hi = float(np.percentile(shares, 100 * (1 - alpha / 2)))
    return lo, hi

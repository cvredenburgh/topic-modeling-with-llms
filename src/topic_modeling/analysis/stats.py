"""Statistical significance tests for topic distributions."""
from __future__ import annotations

from collections import Counter
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency  # type: ignore


def compute_topic_trend_significance(
    trend_df: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Chi-square test: is each topic's distribution over time non-uniform?

    The null hypothesis is that a topic's document counts are uniformly
    distributed across all time periods.  Rejection means the topic has
    statistically significant temporal variation.

    Parameters
    ----------
    trend_df:
        Output of :func:`compute_topic_trends`.
    alpha:
        Significance level.

    Returns
    -------
    pd.DataFrame
        Columns: ``topic_id``, ``chi2``, ``p_value``, ``significant``.
    """
    if trend_df.empty:
        return pd.DataFrame(columns=["topic_id", "chi2", "p_value", "significant"])

    rows = []
    for tid in sorted(trend_df["topic_id"].unique()):
        topic_df = trend_df[trend_df["topic_id"] == tid].sort_values("period")
        counts = topic_df["doc_count"].values.astype(float)

        if counts.sum() == 0 or len(counts) < 2:
            rows.append({"topic_id": tid, "chi2": 0.0, "p_value": 1.0, "significant": False})
            continue

        # Chi-square goodness-of-fit against uniform expectation
        expected = np.full_like(counts, counts.mean())
        # Use scipy's chisquare (1-sample)
        from scipy.stats import chisquare  # type: ignore
        stat, p_val = chisquare(counts, f_exp=expected)

        rows.append(
            {
                "topic_id": tid,
                "chi2": round(float(stat), 6),
                "p_value": round(float(p_val), 6),
                "significant": bool(p_val < alpha),
            }
        )

    return pd.DataFrame(rows)


def test_topic_trend_significance(
    trend_df: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Backward-compatible alias for trend significance testing.

    Several call sites and tests use the older ``test_*`` function name.
    Keep this thin wrapper so existing code continues to work.
    """
    return compute_topic_trend_significance(trend_df=trend_df, alpha=alpha)


test_topic_trend_significance.__test__ = False


def compare_topic_prevalence(
    assignments_a: List[int],
    assignments_b: List[int],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Chi-square test: do two document groups have different topic distributions?

    Builds a 2 × n_topics contingency table and tests whether the row
    (group) and column (topic) variables are independent.

    Parameters
    ----------
    assignments_a:
        Topic ids for group A (-1 outliers excluded).
    assignments_b:
        Topic ids for group B (-1 outliers excluded).
    alpha:
        Significance level.

    Returns
    -------
    pd.DataFrame
        Columns: ``topic_id``, ``count_a``, ``count_b``, ``chi2``,
        ``p_value``, ``significant``.
        The ``chi2`` and ``p_value`` columns are the *overall* test statistic
        repeated on each row for convenience; ``significant`` reflects that
        global test.
    """
    a_clean = [t for t in assignments_a if t != -1]
    b_clean = [t for t in assignments_b if t != -1]

    all_topics = sorted(set(a_clean) | set(b_clean))
    if not all_topics or (len(a_clean) == 0 and len(b_clean) == 0):
        return pd.DataFrame(columns=["topic_id", "count_a", "count_b", "chi2", "p_value", "significant"])

    count_a = Counter(a_clean)
    count_b = Counter(b_clean)

    contingency = np.array(
        [[count_a.get(t, 0), count_b.get(t, 0)] for t in all_topics],
        dtype=float,
    ).T  # shape (2, n_topics)

    # Need at least one non-zero cell in each row
    if contingency.sum() == 0:
        chi2_stat, p_val = 0.0, 1.0
    else:
        try:
            chi2_stat, p_val, _, _ = chi2_contingency(contingency)
            chi2_stat = float(chi2_stat)
            p_val = float(p_val)
        except ValueError:
            chi2_stat, p_val = 0.0, 1.0

    significant = bool(p_val < alpha)
    rows = [
        {
            "topic_id": t,
            "count_a": count_a.get(t, 0),
            "count_b": count_b.get(t, 0),
            "chi2": round(chi2_stat, 6),
            "p_value": round(p_val, 6),
            "significant": significant,
        }
        for t in all_topics
    ]
    return pd.DataFrame(rows)

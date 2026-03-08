"""Pairwise topic co-occurrence metrics: Jaccard, PMI, Lift."""
from __future__ import annotations

import math
from collections import Counter
from typing import List

import numpy as np
import pandas as pd


def compute_topic_associations(
    doc_topic_assignments: List[int],
    min_cooccurrence: int = 5,
) -> pd.DataFrame:
    """Pairwise Jaccard / PMI / Lift between topics.

    Each document is treated as a single observation belonging to exactly one
    topic.  Co-occurrence is therefore defined as two topics appearing in the
    same *sliding window* of documents — or, more simply, across the full
    corpus — using a binary doc-topic matrix approach where each topic is
    present once per document.

    Because each document has exactly one assigned topic the raw co-occurrence
    count between two distinct topics is always zero, which would make
    Jaccard/PMI ill-defined.  To be meaningful this function uses a
    **document-neighbourhood** model: two topics *co-occur* if they appear
    within `window=1` positions of each other in the assignment sequence.
    This captures which topics are discussed in adjacent documents (e.g. in a
    thread or time-ordered corpus).

    Parameters
    ----------
    doc_topic_assignments:
        Topic id per document in corpus order.  Outlier docs (-1) are excluded.
    min_cooccurrence:
        Minimum co-occurrence count to include a pair.

    Returns
    -------
    pd.DataFrame
        Columns: ``topic_a``, ``topic_b``, ``jaccard``, ``pmi``, ``lift``.
    """
    # Filter outliers
    assignments = [t for t in doc_topic_assignments if t != -1]
    n_docs = len(assignments)
    if n_docs < 2:
        return pd.DataFrame(columns=["topic_a", "topic_b", "jaccard", "pmi", "lift"])

    # Topic counts
    topic_counts: Counter = Counter(assignments)
    topics = sorted(topic_counts)
    n_topics = len(topics)
    if n_topics < 2:
        return pd.DataFrame(columns=["topic_a", "topic_b", "jaccard", "pmi", "lift"])

    # Adjacency-based co-occurrence: pair (assignments[i], assignments[i+1])
    cooc: Counter = Counter()
    for i in range(n_docs - 1):
        a, b = assignments[i], assignments[i + 1]
        if a != b:
            pair = (min(a, b), max(a, b))
            cooc[pair] += 1

    # Effective window count
    n_pairs = n_docs - 1  # number of adjacent pairs

    rows = []
    for i, ta in enumerate(topics):
        for tb in topics[i + 1 :]:
            pair = (ta, tb)
            c_ab = cooc.get(pair, 0)
            if c_ab < min_cooccurrence:
                continue

            p_ab = c_ab / n_pairs
            p_a = topic_counts[ta] / n_docs
            p_b = topic_counts[tb] / n_docs

            # Jaccard: |A∩B| / |A∪B| (approximated via probabilities)
            union = p_a + p_b - p_ab
            jaccard = p_ab / union if union > 0 else 0.0

            # PMI
            if p_a > 0 and p_b > 0 and p_ab > 0:
                pmi = math.log2(p_ab / (p_a * p_b))
            else:
                pmi = 0.0

            # Lift
            lift = p_ab / (p_a * p_b) if p_a > 0 and p_b > 0 else 0.0

            rows.append(
                {
                    "topic_a": ta,
                    "topic_b": tb,
                    "jaccard": round(jaccard, 6),
                    "pmi": round(pmi, 6),
                    "lift": round(lift, 6),
                }
            )

    return pd.DataFrame(rows)

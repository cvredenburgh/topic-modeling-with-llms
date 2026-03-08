"""Agglomerative topic hierarchy from keyword cosine + Jaccard distance."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import average, fcluster  # type: ignore
from scipy.spatial.distance import squareform  # type: ignore

if TYPE_CHECKING:
    from topic_modeling.models.base import TopicModelBase


def build_topic_hierarchy(
    model: "TopicModelBase",
    keyword_weight: float = 0.7,
    n_keywords: int = 10,
) -> pd.DataFrame:
    """Agglomerative topic hierarchy using keyword cosine + Jaccard distance.

    Parameters
    ----------
    model:
        Fitted topic model implementing ``get_topics()``.
    keyword_weight:
        Weight for cosine distance component; Jaccard weight = 1 - keyword_weight.
    n_keywords:
        Number of top keywords to consider per topic.

    Returns
    -------
    pd.DataFrame
        Columns: ``parent_id``, ``child_id``, ``linkage_height``, ``similarity``.
        Each row records the merge of two nodes in the dendrogram.
    """
    topics = model.get_topics()
    topic_ids = sorted(t for t in topics if t != -1)
    if len(topic_ids) < 2:
        return pd.DataFrame(columns=["parent_id", "child_id", "linkage_height", "similarity"])

    # Build keyword sets and score vectors per topic
    keyword_sets: list[set] = []
    score_vecs: list[dict[str, float]] = []
    for tid in topic_ids:
        word_scores = topics[tid][:n_keywords]
        keyword_sets.append({w for w, _ in word_scores})
        score_vecs.append({w: s for w, s in word_scores})

    # Global vocabulary across all topics
    vocab = sorted({w for kw in keyword_sets for w in kw})
    vocab_index = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)

    # Topic-keyword score matrix (topics × vocab)
    mat = np.zeros((len(topic_ids), V), dtype=float)
    for row, sv in enumerate(score_vecs):
        for w, s in sv.items():
            mat[row, vocab_index[w]] = s

    # Pairwise hybrid distance matrix
    n = len(topic_ids)
    dist_mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            cos_d = _cosine_distance(mat[i], mat[j])
            jac_d = _jaccard_distance(keyword_sets[i], keyword_sets[j])
            hybrid = keyword_weight * cos_d + (1.0 - keyword_weight) * jac_d
            dist_mat[i, j] = hybrid
            dist_mat[j, i] = hybrid

    condensed = squareform(dist_mat)
    linkage_matrix = average(condensed)

    # Convert scipy linkage matrix to parent-child rows
    # Linkage matrix shape: (n-1, 4) — [left, right, distance, count]
    rows = []
    node_map = {i: topic_ids[i] for i in range(n)}
    next_node_id = max(topic_ids) + 1
    for step, (left, right, dist, _) in enumerate(linkage_matrix):
        left_id = node_map[int(left)]
        right_id = node_map[int(right)]
        parent_id = next_node_id + step
        similarity = max(0.0, 1.0 - float(dist))
        rows.append(
            {
                "parent_id": parent_id,
                "child_id": left_id,
                "linkage_height": float(dist),
                "similarity": similarity,
            }
        )
        rows.append(
            {
                "parent_id": parent_id,
                "child_id": right_id,
                "linkage_height": float(dist),
                "similarity": similarity,
            }
        )
        node_map[n + step] = parent_id

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (norm_a * norm_b))


def _jaccard_distance(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return 1.0 - intersection / union

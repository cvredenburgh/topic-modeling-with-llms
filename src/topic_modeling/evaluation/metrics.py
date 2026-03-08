"""Topic model evaluation metrics."""
from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

from topic_modeling.config.schema import EvaluationConfig
from topic_modeling.models.base import TopicModelBase

logger = logging.getLogger(__name__)


def evaluate(
    model: TopicModelBase,
    texts: List[str],
    topic_ids: List[int],
    config: EvaluationConfig,
    embeddings: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute all configured evaluation metrics.

    Args:
        model:       Fitted topic model.
        texts:       Original document texts.
        topic_ids:   Per-document topic assignments.
        config:      Evaluation configuration.
        embeddings:  Optional UMAP-reduced embeddings for cluster quality metrics.
                     When None, silhouette and calinski_harabasz are skipped.
    """
    metrics: Dict[str, Any] = {}
    topics = model.get_topics()
    topic_words = {k: [w for w, _ in v] for k, v in topics.items() if k != -1}

    if "coherence" in config.metrics:
        metrics["coherence"] = _coherence(
            topic_words, texts, config.coherence_measure, config.topn
        )

    if "diversity" in config.metrics:
        metrics["diversity"] = _diversity(topic_words, config.topn)

    if "outlier_ratio" in config.metrics:
        n_outliers = sum(1 for t in topic_ids if t == -1)
        metrics["outlier_count"] = n_outliers
        metrics["outlier_ratio"] = n_outliers / max(len(topic_ids), 1)

    if "topic_size_stats" in config.metrics:
        sizes = list(Counter(t for t in topic_ids if t != -1).values())
        metrics["topic_count"] = len(topic_words)
        metrics["topic_size_mean"] = float(np.mean(sizes)) if sizes else 0.0
        metrics["topic_size_min"] = int(min(sizes)) if sizes else 0
        metrics["topic_size_max"] = int(max(sizes)) if sizes else 0

    # --- Cluster quality metrics (require UMAP embeddings) ---
    if embeddings is None and hasattr(model, "get_umap_embeddings"):
        embeddings = model.get_umap_embeddings()

    if embeddings is not None:
        non_outlier_mask = [i for i, t in enumerate(topic_ids) if t != -1]
        filtered_ids = [topic_ids[i] for i in non_outlier_mask]
        filtered_embs = embeddings[non_outlier_mask] if len(non_outlier_mask) > 0 else np.array([])
        n_clusters = len(set(filtered_ids))

        if "silhouette" in config.metrics:
            if n_clusters > 1 and len(filtered_embs) > n_clusters:
                metrics["silhouette"] = _silhouette(filtered_embs, filtered_ids)
            else:
                logger.warning("silhouette skipped: need >1 cluster and >n_clusters samples")

        if "calinski_harabasz" in config.metrics:
            if n_clusters > 1 and len(filtered_embs) > n_clusters:
                metrics["calinski_harabasz"] = _calinski_harabasz(filtered_embs, filtered_ids)
            else:
                logger.warning("calinski_harabasz skipped: need >1 cluster and >n_clusters samples")
    else:
        if "silhouette" in config.metrics or "calinski_harabasz" in config.metrics:
            logger.info("silhouette/calinski_harabasz skipped: no embeddings available")

    # --- Distribution metrics ---
    if "dist_entropy" in config.metrics:
        metrics["dist_entropy"] = _dist_entropy(topic_ids)

    if "size_ratio_max_min" in config.metrics:
        sizes = list(Counter(t for t in topic_ids if t != -1).values())
        if len(sizes) >= 2:
            metrics["size_ratio_max_min"] = max(sizes) / min(sizes)
        elif sizes:
            metrics["size_ratio_max_min"] = 1.0

    if "size_ratio_max_median" in config.metrics:
        sizes = list(Counter(t for t in topic_ids if t != -1).values())
        if sizes:
            median = float(np.median(sizes))
            metrics["size_ratio_max_median"] = max(sizes) / median if median > 0 else 1.0

    # --- Per-topic metrics ---
    per_topic_requested = (
        "per_topic_coherence" in config.metrics
        or "per_topic_diversity" in config.metrics
    )
    if per_topic_requested:
        per_topic: Dict[str, Any] = {}
        if "per_topic_coherence" in config.metrics:
            per_topic["coherence"] = _per_topic_coherence(
                topic_words, texts, config.coherence_measure, config.topn
            )
        if "per_topic_diversity" in config.metrics:
            per_topic["diversity"] = _per_topic_diversity(topic_words, config.topn)
        metrics["per_topic_metrics"] = per_topic

    scalar_metrics = {k: v for k, v in metrics.items() if k != "per_topic_metrics"}
    logger.info(f"Evaluation metrics: {scalar_metrics}")
    return metrics


# ---------------------------------------------------------------------------
# Existing metrics
# ---------------------------------------------------------------------------

def _coherence(
    topic_words: Dict[int, List[str]],
    texts: List[str],
    measure: str,
    topn: int,
) -> float:
    try:
        from gensim.corpora import Dictionary  # type: ignore
        from gensim.models.coherencemodel import CoherenceModel  # type: ignore

        tokenized = [t.lower().split() for t in texts]
        dictionary = Dictionary(tokenized)
        topics_list = [words[:topn] for words in topic_words.values()]
        if not topics_list:
            return 0.0

        cm = CoherenceModel(
            topics=topics_list,
            texts=tokenized,
            dictionary=dictionary,
            coherence=measure,
        )
        score = float(cm.get_coherence())
        logger.info(f"Coherence ({measure}): {score:.4f}")
        return score
    except Exception as exc:
        logger.warning(f"Coherence computation failed: {exc}")
        return float("nan")


def _diversity(topic_words: Dict[int, List[str]], topn: int) -> float:
    """Proportion of unique words across all topic top-word lists."""
    all_words = [w for words in topic_words.values() for w in words[:topn]]
    if not all_words:
        return 0.0
    return len(set(all_words)) / len(all_words)


# ---------------------------------------------------------------------------
# New cluster quality metrics
# ---------------------------------------------------------------------------

def _silhouette(embeddings: np.ndarray, labels: List[int]) -> float:
    from sklearn.metrics import silhouette_score  # type: ignore
    return float(silhouette_score(embeddings, labels))


def _calinski_harabasz(embeddings: np.ndarray, labels: List[int]) -> float:
    from sklearn.metrics import calinski_harabasz_score  # type: ignore
    return float(calinski_harabasz_score(embeddings, labels))


# ---------------------------------------------------------------------------
# New distribution metrics
# ---------------------------------------------------------------------------

def _dist_entropy(topic_ids: List[int]) -> float:
    """Shannon entropy of topic size distribution (higher = more balanced)."""
    from scipy.stats import entropy  # type: ignore

    counts = Counter(t for t in topic_ids if t != -1)
    if not counts:
        return 0.0
    sizes = np.array(list(counts.values()), dtype=float)
    return float(entropy(sizes / sizes.sum()))


# ---------------------------------------------------------------------------
# New per-topic metrics
# ---------------------------------------------------------------------------

def _per_topic_coherence(
    topic_words: Dict[int, List[str]],
    texts: List[str],
    measure: str,
    topn: int,
) -> Dict[str, float]:
    """Per-topic coherence scores via gensim."""
    try:
        from gensim.corpora import Dictionary  # type: ignore
        from gensim.models.coherencemodel import CoherenceModel  # type: ignore

        tokenized = [t.lower().split() for t in texts]
        dictionary = Dictionary(tokenized)
        result: Dict[str, float] = {}
        for tid, words in topic_words.items():
            try:
                cm = CoherenceModel(
                    topics=[words[:topn]],
                    texts=tokenized,
                    dictionary=dictionary,
                    coherence=measure,
                )
                result[str(tid)] = float(cm.get_coherence())
            except Exception:
                result[str(tid)] = float("nan")
        return result
    except Exception as exc:
        logger.warning(f"Per-topic coherence failed: {exc}")
        return {}


def _per_topic_diversity(
    topic_words: Dict[int, List[str]], topn: int
) -> Dict[str, float]:
    """Average Jaccard distance from each topic to all others."""
    topic_sets = {tid: set(words[:topn]) for tid, words in topic_words.items()}
    tids = list(topic_sets.keys())
    result: Dict[str, float] = {}
    for tid in tids:
        if len(tids) <= 1:
            result[str(tid)] = 0.0
            continue
        distances = []
        for other in tids:
            if other == tid:
                continue
            a, b = topic_sets[tid], topic_sets[other]
            union = len(a | b)
            distances.append(0.0 if union == 0 else 1.0 - len(a & b) / union)
        result[str(tid)] = float(np.mean(distances))
    return result

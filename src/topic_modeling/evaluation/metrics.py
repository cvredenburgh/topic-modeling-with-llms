"""Topic model evaluation metrics."""
from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List

import numpy as np

from topic_modeling.config.schema import EvaluationConfig
from topic_modeling.models.base import TopicModelBase

logger = logging.getLogger(__name__)


def evaluate(
    model: TopicModelBase,
    texts: List[str],
    topic_ids: List[int],
    config: EvaluationConfig,
) -> Dict[str, Any]:
    """Compute all configured evaluation metrics."""
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

    logger.info(f"Evaluation metrics: {metrics}")
    return metrics


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

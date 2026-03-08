"""FASTopic adapter implementing TopicModelBase."""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from topic_modeling.config.schema import ModelConfig
from topic_modeling.models.base import TopicModelBase

logger = logging.getLogger(__name__)


class FASTopicAdapter(TopicModelBase):
    """Wraps the FASTopic neural topic model.

    FASTopic API reference:
        model = FASTopic(num_topics=50)
        model.fit(texts)
        model.get_beta()           -> (n_topics, vocab_size) topic-word matrix
        model.get_theta()          -> (n_docs, n_topics) doc-topic matrix
        model.get_top_words(top_n) -> list[list[str]] top words per topic
    """

    def __init__(self, config: ModelConfig, seed: int = 42):
        self.config = config
        self.seed = seed
        self._model: Any = None
        self._fitted_texts: List[str] = []
        self._theta: Optional[np.ndarray] = None  # (n_docs, n_topics)

    def fit(self, texts: List[str]) -> "FASTopicAdapter":
        from fastopic import FASTopic  # type: ignore

        p = self.config.params
        num_topics = p.get("num_topics", 50)
        logger.info(f"Fitting FASTopic on {len(texts)} documents ({num_topics} topics)")

        self._model = FASTopic(num_topics=num_topics)
        self._model.fit(texts)
        self._fitted_texts = texts

        try:
            self._theta = np.array(self._model.get_theta())
        except Exception:
            self._theta = None

        logger.info("FASTopic fit complete")
        return self

    def transform(
        self, texts: List[str]
    ) -> Tuple[List[int], Optional[np.ndarray]]:
        """Return dominant topic per document.

        FASTopic does not support inference on unseen documents. If the texts
        match the fitted corpus the stored theta is returned; otherwise topic
        assignments default to -1 with a logged warning.
        """
        if self._theta is not None and texts == self._fitted_texts:
            topic_ids = np.argmax(self._theta, axis=1).tolist()
            return topic_ids, self._theta

        logger.warning(
            "FASTopic does not support transform on unseen documents. "
            "Call transform with the same texts passed to fit() to get "
            "training-set assignments."
        )
        return [-1] * len(texts), None

    def get_topics(self) -> Dict[int, List[Tuple[str, float]]]:
        """Return {topic_id: [(word, score), ...]} using beta rank positions.

        Words are returned in descending score order. Scores are taken from
        the beta matrix at the corresponding rank position so that word[i]
        receives the i-th highest weight for that topic.
        """
        beta = np.array(self._model.get_beta())  # (n_topics, vocab_size)
        top_n = self.config.params.get("num_top_words", 15)
        top_words_per_topic = self._model.get_top_words(top_n=top_n)

        topics: Dict[int, List[Tuple[str, float]]] = {}
        for t_id, words in enumerate(top_words_per_topic):
            # Rank indices for this topic in descending weight order
            ranked_indices = np.argsort(beta[t_id])[::-1][: len(words)]
            topics[t_id] = [
                (word, float(beta[t_id, idx]))
                for word, idx in zip(words, ranked_indices)
            ]
        return topics

    def get_topic_info(self) -> Any:
        top_n = self.config.params.get("num_top_words", 15)
        top_words_per_topic = self._model.get_top_words(top_n=top_n)
        return [
            {"topic_id": t_id, "top_words": words}
            for t_id, words in enumerate(top_words_per_topic)
        ]

    def get_document_topic_assignments(self) -> List[int]:
        """Return dominant topic_id per document from theta matrix."""
        if self._theta is not None:
            return np.argmax(self._theta, axis=1).tolist()
        return []

    def get_representative_docs(self, topic_id: int, n: int = 3) -> List[str]:
        if self._theta is None or not self._fitted_texts:
            return []
        scores = self._theta[:, topic_id]
        top_idx = np.argsort(scores)[::-1][:n]
        return [self._fitted_texts[i] for i in top_idx]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "fastopic_model.pkl", "wb") as f:
            pickle.dump(self._model, f)
        theta = self._theta if self._theta is not None else np.array([])
        np.save(path / "theta.npy", theta)
        logger.info(f"FASTopic model saved to {path}")

    def load(self, path: Path) -> "FASTopicAdapter":
        with open(path / "fastopic_model.pkl", "rb") as f:
            self._model = pickle.load(f)
        theta_path = path / "theta.npy"
        if theta_path.exists():
            arr = np.load(theta_path)
            self._theta = arr if arr.size else None
        logger.info(f"FASTopic model loaded from {path}")
        return self

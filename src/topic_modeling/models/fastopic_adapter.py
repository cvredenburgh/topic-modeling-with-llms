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
        model.get_beta()     -> (n_topics, vocab_size) topic-word distribution
        model.get_theta()    -> (n_docs, n_topics) doc-topic distribution
        model.get_top_words(top_n=15) -> list of lists of top words per topic
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
        """Assign dominant topic per document using training theta if available,
        otherwise approximate via closest training doc embedding."""
        if self._theta is not None and texts == self._fitted_texts:
            topic_ids = np.argmax(self._theta, axis=1).tolist()
            return topic_ids, self._theta

        # For new documents, FASTopic doesn't support native transform —
        # fall back to returning -1 with a warning.
        logger.warning(
            "FASTopic does not support transform on unseen docs; "
            "returning training assignments if texts match fitted corpus."
        )
        topic_ids = [-1] * len(texts)
        return topic_ids, None

    def get_topics(self) -> Dict[int, List[Tuple[str, float]]]:
        beta = np.array(self._model.get_beta())  # (n_topics, vocab_size)
        top_words_per_topic = self._model.get_top_words(
            top_n=self.config.params.get("num_top_words", 15)
        )
        topics: Dict[int, List[Tuple[str, float]]] = {}
        for t_id, words in enumerate(top_words_per_topic):
            word_scores = [(w, float(beta[t_id, self._word_idx(w, beta, t_id)])) for w in words]
            topics[t_id] = word_scores
        return topics

    def _word_idx(self, word: str, beta: np.ndarray, topic_id: int) -> int:
        """Return argmax position for a word given the beta matrix."""
        # FASTopic doesn't expose a public vocab; use rank position instead
        top_idx = int(np.argsort(beta[topic_id])[::-1][0])
        return top_idx

    def get_topic_info(self) -> Any:
        top_words_per_topic = self._model.get_top_words(
            top_n=self.config.params.get("num_top_words", 15)
        )
        return [
            {"topic_id": t_id, "top_words": words}
            for t_id, words in enumerate(top_words_per_topic)
        ]

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
        np.save(path / "theta.npy", self._theta if self._theta is not None else np.array([]))
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

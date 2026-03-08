"""BERTopic adapter implementing TopicModelBase."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from topic_modeling.config.schema import ModelConfig
from topic_modeling.models.base import TopicModelBase

logger = logging.getLogger(__name__)


class BERTopicAdapter(TopicModelBase):
    def __init__(self, config: ModelConfig, seed: int = 42):
        self.config = config
        self.seed = seed
        self._model: Any = None
        self._fitted_texts: List[str] = []
        self._build_model()

    def _build_model(self) -> None:
        from bertopic import BERTopic  # type: ignore
        from hdbscan import HDBSCAN  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
        from umap import UMAP  # type: ignore

        p = self.config.params

        umap_model = UMAP(
            n_components=p.get("umap_n_components", 5),
            n_neighbors=p.get("umap_n_neighbors", 15),
            min_dist=p.get("umap_min_dist", 0.0),
            metric=p.get("umap_metric", "cosine"),
            random_state=self.seed,
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=p.get("hdbscan_min_cluster_size", 15),
            min_samples=p.get("hdbscan_min_samples", 1),
            cluster_selection_method=p.get("hdbscan_cluster_selection_method", "eom"),
            prediction_data=True,
        )
        embedding_model = SentenceTransformer(
            p.get("embedding_model", "all-MiniLM-L6-v2")
        )

        self._model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            embedding_model=embedding_model,
            top_n_words=p.get("top_n_words", 10),
            nr_topics=p.get("nr_topics", None),
            verbose=False,
        )
        logger.info(f"BERTopic built with params: {p}")

    def fit(self, texts: List[str]) -> "BERTopicAdapter":
        logger.info(f"Fitting BERTopic on {len(texts)} documents")
        self._fitted_texts = texts
        self._model.fit_transform(texts)
        logger.info(f"BERTopic: discovered {self.get_topic_count()} topics")
        return self

    def transform(
        self, texts: List[str]
    ) -> Tuple[List[int], Optional[np.ndarray]]:
        topic_ids, probs = self._model.transform(texts)
        return list(topic_ids), probs

    def get_topics(self) -> Dict[int, List[Tuple[str, float]]]:
        return self._model.get_topics()

    def get_topic_info(self) -> Any:
        return self._model.get_topic_info()

    def get_representative_docs(self, topic_id: int, n: int = 3) -> List[str]:
        docs = self._model.get_representative_docs(topic_id) or []
        return docs[:n]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._model.save(str(path / "bertopic_model"), serialization="pickle")
        logger.info(f"BERTopic model saved to {path}")

    def load(self, path: Path) -> "BERTopicAdapter":
        from bertopic import BERTopic  # type: ignore

        self._model = BERTopic.load(str(path / "bertopic_model"))
        logger.info(f"BERTopic model loaded from {path}")
        return self

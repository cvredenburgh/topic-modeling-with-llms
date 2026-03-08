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
        self._umap_embeddings: Optional[np.ndarray] = None
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

        vectorizer_model = self._build_vectorizer(p)
        representation_model = self._build_representation_model(p)

        kwargs: Dict[str, Any] = dict(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            embedding_model=embedding_model,
            top_n_words=p.get("top_n_words", 10),
            nr_topics=p.get("nr_topics", None),
            verbose=False,
        )
        if vectorizer_model is not None:
            kwargs["vectorizer_model"] = vectorizer_model
        if representation_model is not None:
            kwargs["representation_model"] = representation_model

        self._model = BERTopic(**kwargs)
        logger.info(f"BERTopic built with params: {p}")

    def _build_vectorizer(self, p: Dict[str, Any]) -> Any:
        vectorizer_type = p.get("vectorizer_type")
        if not vectorizer_type:
            return None

        ngram_range = tuple(p.get("vectorizer_ngram_range", [1, 1]))
        stop_words = p.get("vectorizer_stop_words", None)

        if vectorizer_type == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
            return TfidfVectorizer(
                max_features=p.get("vectorizer_max_features", 10000),
                ngram_range=ngram_range,
                min_df=p.get("vectorizer_min_df", 1),
                max_df=p.get("vectorizer_max_df", 1.0),
                stop_words=stop_words if stop_words else None,
            )
        if vectorizer_type == "count":
            from sklearn.feature_extraction.text import CountVectorizer  # type: ignore
            return CountVectorizer(
                max_features=p.get("vectorizer_max_features", 10000),
                ngram_range=ngram_range,
                min_df=p.get("vectorizer_min_df", 1),
                max_df=p.get("vectorizer_max_df", 1.0),
                stop_words=stop_words if stop_words else None,
            )
        logger.warning(f"Unknown vectorizer_type {vectorizer_type!r}; using default")
        return None

    def _build_representation_model(self, p: Dict[str, Any]) -> Any:
        rep_name = p.get("representation_model")
        if not rep_name:
            return None

        if rep_name == "keybert":
            from bertopic.representation import KeyBERTInspired  # type: ignore
            return KeyBERTInspired()
        if rep_name == "mmr":
            from bertopic.representation import MaximalMarginalRelevance  # type: ignore
            return MaximalMarginalRelevance(
                diversity=p.get("representation_diversity", 0.3)
            )
        logger.warning(f"Unknown representation_model {rep_name!r}; using default c-TF-IDF")
        return None

    def fit(self, texts: List[str]) -> "BERTopicAdapter":
        logger.info(f"Fitting BERTopic on {len(texts)} documents")
        self._fitted_texts = texts
        topics, _ = self._model.fit_transform(texts)

        # Store UMAP-reduced embeddings for downstream cluster quality metrics
        self._umap_embeddings = getattr(
            getattr(self._model, "umap_model", None), "embedding_", None
        )

        if self.config.params.get("reduce_outliers", False) and topics is not None:
            try:
                topics = self._model.reduce_outliers(texts, topics, strategy="embeddings")
                self._model.update_topics(texts, topics=topics)
                logger.info("Outlier reduction applied")
            except Exception as exc:
                logger.warning(f"Outlier reduction failed: {exc}")

        logger.info(f"BERTopic: discovered {self.get_topic_count()} topics")
        return self

    def get_document_topic_assignments(self) -> List[int]:
        """Return topic_id per document from HDBSCAN labels after fit."""
        hdbscan_model = getattr(self._model, "hdbscan_model", None)
        if hdbscan_model is not None:
            labels = getattr(hdbscan_model, "labels_", None)
            if labels is not None:
                return list(map(int, labels))
        # Fallback: re-transform fitted texts
        if self._fitted_texts:
            ids, _ = self.transform(self._fitted_texts)
            return ids
        return []

    def get_umap_embeddings(self) -> Optional[np.ndarray]:
        """Return UMAP-reduced embeddings stored after fit (for cluster metrics)."""
        return self._umap_embeddings

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

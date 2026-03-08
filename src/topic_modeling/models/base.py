"""Abstract topic model interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class TopicModelBase(ABC):
    """Shared interface for BERTopic and FASTopic adapters."""

    @abstractmethod
    def fit(self, texts: List[str]) -> "TopicModelBase":
        """Fit on texts, store internal state."""

    @abstractmethod
    def transform(
        self, texts: List[str]
    ) -> Tuple[List[int], Optional[np.ndarray]]:
        """Return (topic_ids, probabilities) for each text."""

    @abstractmethod
    def get_topics(self) -> Dict[int, List[Tuple[str, float]]]:
        """Return {topic_id: [(word, score), ...]}. -1 = outlier bucket."""

    @abstractmethod
    def get_topic_info(self) -> Any:
        """Return a DataFrame or list of dicts with per-topic metadata."""

    @abstractmethod
    def get_representative_docs(self, topic_id: int, n: int = 3) -> List[str]:
        """Return up to n representative documents for topic_id."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist model artifacts to path."""

    @abstractmethod
    def load(self, path: Path) -> "TopicModelBase":
        """Load model artifacts from path."""

    def get_topic_count(self) -> int:
        """Number of real topics (excludes -1 outlier bucket)."""
        return sum(1 for k in self.get_topics() if k != -1)

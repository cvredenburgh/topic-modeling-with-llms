"""Model adapters — factory by backend name."""
from topic_modeling.config.schema import ModelConfig
from topic_modeling.models.base import TopicModelBase


def build_model(config: ModelConfig, seed: int = 42) -> TopicModelBase:
    """Return the correct adapter based on config.backend."""
    if config.backend == "bertopic":
        from topic_modeling.models.bertopic_adapter import BERTopicAdapter
        return BERTopicAdapter(config, seed=seed)
    if config.backend == "fastopic":
        from topic_modeling.models.fastopic_adapter import FASTopicAdapter
        return FASTopicAdapter(config, seed=seed)
    raise ValueError(f"Unknown model backend: {config.backend!r}")

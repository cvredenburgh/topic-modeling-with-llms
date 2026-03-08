"""Topic analysis utilities: hierarchy, associations, trends, and significance."""
from topic_modeling.analysis.associations import compute_topic_associations
from topic_modeling.analysis.hierarchy import build_topic_hierarchy
from topic_modeling.analysis.stats import compare_topic_prevalence, test_topic_trend_significance
from topic_modeling.analysis.trends import compute_topic_trends, detect_emerging_topics

__all__ = [
    "build_topic_hierarchy",
    "compute_topic_associations",
    "compute_topic_trends",
    "detect_emerging_topics",
    "test_topic_trend_significance",
    "compare_topic_prevalence",
]

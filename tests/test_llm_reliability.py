"""Unit tests for LLM reliability consensus helpers."""
import pytest

from topic_modeling.llm.reliability import summary_consensus, tag_consensus


def test_summary_consensus_empty():
    out = summary_consensus([], min_agreement=0.67)
    assert out["summary"] == ""
    assert out["reliability_score"] == 0.0
    assert out["reliability_consistent"] is False


def test_summary_consensus_single_is_consistent():
    out = summary_consensus(["Topic about shipping delays"], min_agreement=0.67)
    assert out["summary"] == "Topic about shipping delays"
    assert out["reliability_score"] == 1.0
    assert out["reliability_consistent"] is True


def test_tag_consensus_majority_tag():
    out = tag_consensus(
        [["shipping", "delivery delay"], ["shipping"], ["shipping", "price"]],
        min_agreement=0.67,
    )
    tags = {t["tag"]: t for t in out["tags"]}
    assert tags["shipping"]["consistent"] is True
    assert tags["shipping"]["agreement"] == pytest.approx(1.0)
    assert tags["price"]["consistent"] is False


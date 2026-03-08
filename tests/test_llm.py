"""Tests for LLM output parsing and client retry logic."""
import json
from unittest.mock import MagicMock, patch

import pytest

from topic_modeling.config.schema import LLMConfig
from topic_modeling.llm.summarizer import _parse_json, summarize_topics
from topic_modeling.llm.tagger import tag_topics


# --- JSON parsing ---

def test_parse_clean_json():
    raw = '{"summary": "This topic is about product quality."}'
    result = _parse_json(raw)
    assert result["summary"] == "This topic is about product quality."


def test_parse_json_with_markdown_fence():
    raw = '```json\n{"summary": "Quality."}\n```'
    result = _parse_json(raw)
    assert result["summary"] == "Quality."


def test_parse_json_invalid():
    with pytest.raises(json.JSONDecodeError):
        _parse_json("not json at all")


# --- LLM client retry ---

def test_llm_client_retries_on_failure():
    from topic_modeling.llm.client import LLMClient

    cfg = LLMConfig(max_retries=3, retry_delay_seconds=0.0)

    mock_client = MagicMock()
    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError("transient error")
        result = MagicMock()
        result.content = [MagicMock(text='{"summary": "ok"}')]
        return result

    mock_client.messages.create.side_effect = side_effect

    with patch.object(LLMClient, "_build_client", return_value=mock_client):
        client = LLMClient(cfg)
        response = client.complete("test prompt")
        assert response == '{"summary": "ok"}'
        assert call_count == 3


def test_llm_client_raises_after_max_retries():
    from topic_modeling.llm.client import LLMClient

    cfg = LLMConfig(max_retries=2, retry_delay_seconds=0.0)
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("always fails")

    with patch.object(LLMClient, "_build_client", return_value=mock_client):
        client = LLMClient(cfg)
        with pytest.raises(RuntimeError, match="failed after"):
            client.complete("test prompt")


def test_llm_client_rate_limiter_constructed():
    from topic_modeling.llm.client import LLMClient, _RateLimiter

    cfg = LLMConfig(tokens_per_minute=40000)
    with patch.object(LLMClient, "_build_client", return_value=MagicMock()):
        client = LLMClient(cfg)
        assert client._limiter is not None
        assert isinstance(client._limiter, _RateLimiter)


def test_llm_client_no_rate_limiter_by_default():
    from topic_modeling.llm.client import LLMClient

    cfg = LLMConfig()
    with patch.object(LLMClient, "_build_client", return_value=MagicMock()):
        client = LLMClient(cfg)
        assert client._limiter is None


def test_rate_limiter_acquire_tracks_usage():
    from topic_modeling.llm.client import _RateLimiter

    limiter = _RateLimiter(tpm=10000, safety_margin=1.0)
    limiter.acquire(100)
    assert limiter._used == 100
    limiter.acquire(200)
    assert limiter._used == 300


def test_llm_client_openai_provider_path():
    from topic_modeling.llm.client import LLMClient

    cfg = LLMConfig(provider="openai", model="gpt-4.1-mini", max_retries=1)
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"summary":"ok"}'))]
    )

    with patch.object(LLMClient, "_build_client", return_value=mock_client):
        client = LLMClient(cfg)
        out = client.complete("prompt")
        assert out == '{"summary":"ok"}'
        mock_client.chat.completions.create.assert_called_once()


def test_llm_client_gemini_provider_path():
    from topic_modeling.llm.client import LLMClient

    cfg = LLMConfig(provider="gemini", model="gemini-2.0-flash", max_retries=1)
    mock_client = MagicMock()
    mock_client.generate_content.return_value = MagicMock(text='{"summary":"ok"}')

    with patch.object(LLMClient, "_build_client", return_value=mock_client):
        client = LLMClient(cfg)
        out = client.complete("prompt")
        assert out == '{"summary":"ok"}'
        mock_client.generate_content.assert_called_once()


# --- Summarizer output structure ---

def test_summarize_topics_output_structure():
    cfg = LLMConfig(enabled=True, max_retries=1, retry_delay_seconds=0.0)

    mock_model = MagicMock()
    mock_model.get_topics.return_value = {
        0: [("word1", 0.9), ("word2", 0.8)],
        1: [("word3", 0.7), ("word4", 0.6)],
    }
    mock_model.get_representative_docs.return_value = ["doc one", "doc two"]

    mock_response = '{"summary": "This is a summary."}'

    with patch("topic_modeling.llm.summarizer.LLMClient") as MockClient:
        MockClient.return_value.complete.return_value = mock_response
        results = summarize_topics(mock_model, cfg)

    assert len(results) == 2
    for r in results:
        assert "topic_id" in r
        assert "keywords" in r
        assert "summary" in r
        assert "analysis_status" in r
        assert r["summary"] == "This is a summary."
        assert r["analysis_status"] == "success"
        assert "reliability_score" in r
        assert "reliability_consistent" in r


def test_summarize_topics_failed_status():
    cfg = LLMConfig(enabled=True, max_retries=1, retry_delay_seconds=0.0)

    mock_model = MagicMock()
    mock_model.get_topics.return_value = {0: [("word1", 0.9)]}
    mock_model.get_representative_docs.return_value = ["doc one"]

    with patch("topic_modeling.llm.summarizer.LLMClient") as MockClient:
        MockClient.return_value.complete.side_effect = RuntimeError("API error")
        results = summarize_topics(mock_model, cfg)

    assert results[0]["analysis_status"] == "failed"
    assert results[0]["summary"] == ""


def test_summarize_topics_reliability_consensus():
    cfg = LLMConfig(
        enabled=True,
        reliability_enabled=True,
        reliability_samples=3,
        reliability_min_agreement=0.6,
        max_retries=1,
        retry_delay_seconds=0.0,
    )
    mock_model = MagicMock()
    mock_model.get_topics.return_value = {0: [("word1", 0.9)]}
    mock_model.get_representative_docs.return_value = ["doc one"]

    with patch("topic_modeling.llm.summarizer.LLMClient") as MockClient:
        MockClient.return_value.complete.side_effect = [
            '{"summary": "Shipping delays and damaged packaging"}',
            '{"summary": "Shipping delays and damaged packaging"}',
            '{"summary": "Customers report delays and package damage"}',
        ]
        results = summarize_topics(mock_model, cfg)

    assert results[0]["analysis_status"] == "success"
    assert len(results[0]["candidate_summaries"]) == 3
    assert results[0]["reliability_score"] > 0.6
    assert results[0]["reliability_consistent"] is True


# --- Tagger output structure ---

def test_tag_topics_output_structure():
    cfg = LLMConfig(enabled=True, max_retries=1, retry_delay_seconds=0.0)

    mock_model = MagicMock()
    mock_model.get_topics.return_value = {
        0: [("quality", 0.9), ("durable", 0.8)],
    }
    mock_model.get_representative_docs.return_value = ["great product"]

    mock_response = '{"tags": [{"tag": "product quality", "consistent": true}, {"tag": "durability", "consistent": false}]}'

    with patch("topic_modeling.llm.tagger.LLMClient") as MockClient:
        MockClient.return_value.complete.return_value = mock_response
        results = tag_topics(mock_model, cfg)

    assert len(results) == 1
    r = results[0]
    assert r["analysis_status"] == "success"
    assert r["tags"] == [
        {"tag": "product quality", "consistent": True, "agreement": 1.0},
        {"tag": "durability", "consistent": True, "agreement": 1.0},
    ]


def test_tag_topics_consistency_flag():
    cfg = LLMConfig(enabled=True, max_retries=1, retry_delay_seconds=0.0)

    mock_model = MagicMock()
    mock_model.get_topics.return_value = {0: [("shipping", 0.9), ("delay", 0.8)]}
    mock_model.get_representative_docs.return_value = ["late delivery", "slow shipping"]

    mock_response = '{"tags": [{"tag": "shipping delays", "consistent": true}]}'

    with patch("topic_modeling.llm.tagger.LLMClient") as MockClient:
        MockClient.return_value.complete.return_value = mock_response
        results = tag_topics(mock_model, cfg)

    tag = results[0]["tags"][0]
    assert tag["tag"] == "shipping delays"
    assert tag["consistent"] is True


def test_tag_topics_backward_compat_str_tags():
    """_normalize_tags should accept old str-list format without error."""
    from topic_modeling.llm.tagger import _normalize_tags

    result = _normalize_tags(["product quality", "shipping"])
    assert result == [
        {"tag": "product quality", "consistent": False},
        {"tag": "shipping", "consistent": False},
    ]


def test_tag_topics_failed_status():
    cfg = LLMConfig(enabled=True, max_retries=1, retry_delay_seconds=0.0)

    mock_model = MagicMock()
    mock_model.get_topics.return_value = {0: [("quality", 0.9)]}
    mock_model.get_representative_docs.return_value = ["great product"]

    with patch("topic_modeling.llm.tagger.LLMClient") as MockClient:
        MockClient.return_value.complete.side_effect = RuntimeError("API error")
        results = tag_topics(mock_model, cfg)

    assert results[0]["analysis_status"] == "failed"
    assert results[0]["tags"] == []


def test_tag_topics_reliability_consensus():
    cfg = LLMConfig(
        enabled=True,
        reliability_enabled=True,
        reliability_samples=3,
        reliability_min_agreement=0.67,
        max_retries=1,
        retry_delay_seconds=0.0,
    )

    mock_model = MagicMock()
    mock_model.get_topics.return_value = {0: [("quality", 0.9)]}
    mock_model.get_representative_docs.return_value = ["great product"]

    with patch("topic_modeling.llm.tagger.LLMClient") as MockClient:
        MockClient.return_value.complete.side_effect = [
            '{"tags": [{"tag":"shipping","consistent": true},{"tag":"delivery delay","consistent": true}]}',
            '{"tags": [{"tag":"shipping","consistent": true}]}',
            '{"tags": [{"tag":"shipping","consistent": true},{"tag":"price","consistent": false}]}',
        ]
        results = tag_topics(mock_model, cfg)

    tags = {t["tag"]: t for t in results[0]["tags"]}
    assert results[0]["analysis_status"] == "success"
    assert "shipping" in tags
    assert tags["shipping"]["consistent"] is True
    assert tags["shipping"]["agreement"] == pytest.approx(1.0)
    assert results[0]["reliability_score"] > 0.0

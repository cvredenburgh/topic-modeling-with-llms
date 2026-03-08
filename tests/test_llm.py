"""Tests for LLM output parsing and client retry logic."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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

    mock_anthropic = MagicMock()
    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError("transient error")
        result = MagicMock()
        result.content = [MagicMock(text='{"summary": "ok"}')]
        return result

    mock_anthropic.messages.create.side_effect = side_effect

    with patch("topic_modeling.llm.client.anthropic.Anthropic", return_value=mock_anthropic):
        client = LLMClient(cfg)
        response = client.complete("test prompt")
        assert response == '{"summary": "ok"}'
        assert call_count == 3


def test_llm_client_raises_after_max_retries():
    from topic_modeling.llm.client import LLMClient

    cfg = LLMConfig(max_retries=2, retry_delay_seconds=0.0)
    mock_anthropic = MagicMock()
    mock_anthropic.messages.create.side_effect = RuntimeError("always fails")

    with patch("topic_modeling.llm.client.anthropic.Anthropic", return_value=mock_anthropic):
        client = LLMClient(cfg)
        with pytest.raises(RuntimeError, match="failed after"):
            client.complete("test prompt")


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
        assert r["summary"] == "This is a summary."


# --- Tagger output structure ---

def test_tag_topics_output_structure():
    cfg = LLMConfig(enabled=True, max_retries=1, retry_delay_seconds=0.0)

    mock_model = MagicMock()
    mock_model.get_topics.return_value = {
        0: [("quality", 0.9), ("durable", 0.8)],
    }
    mock_model.get_representative_docs.return_value = ["great product"]

    mock_response = '{"tags": ["product quality", "durability"]}'

    with patch("topic_modeling.llm.tagger.LLMClient") as MockClient:
        MockClient.return_value.complete.return_value = mock_response
        results = tag_topics(mock_model, cfg)

    assert len(results) == 1
    assert results[0]["tags"] == ["product quality", "durability"]

"""Tests for the preprocessing module."""
from unittest.mock import MagicMock, patch

import pytest

from topic_modeling.config.schema import PreprocessingConfig
from topic_modeling.data.schema import Document
from topic_modeling.preprocessing.cleaner import preprocess, _strip_html, _remove_stops


def _doc(text: str, doc_id: str = "0") -> Document:
    return Document(doc_id=doc_id, text=text)


def test_strip_html():
    doc = _doc("<p>Hello <b>world</b></p>")
    cleaned = _strip_html(doc)
    assert cleaned.text == "Hello world"


def test_drops_empty():
    docs = [_doc(""), _doc("   "), _doc("real content here is long enough")]
    cfg = PreprocessingConfig(min_length=10)
    out, stats = preprocess(docs, cfg)
    assert len(out) == 1
    assert stats["dropped"]["empty"] == 2


def test_drops_too_short():
    docs = [_doc("hi"), _doc("this is long enough text")]
    cfg = PreprocessingConfig(min_length=20)
    out, stats = preprocess(docs, cfg)
    assert len(out) == 1
    assert stats["dropped"]["too_short"] == 1


def test_drops_too_long():
    docs = [_doc("a" * 500), _doc("normal length text here")]
    cfg = PreprocessingConfig(min_length=5, max_length=100)
    out, stats = preprocess(docs, cfg)
    assert len(out) == 1
    assert stats["dropped"]["too_long"] == 1


def test_deduplication():
    text = "this is a duplicate document that is long enough"
    docs = [_doc(text, "1"), _doc(text, "2"), _doc("unique text here that is different", "3")]
    cfg = PreprocessingConfig(min_length=5, remove_duplicates=True)
    out, stats = preprocess(docs, cfg)
    assert len(out) == 2
    assert stats["dropped"]["duplicates"] == 1


def test_lowercase():
    docs = [_doc("HELLO WORLD this is a test document")]
    cfg = PreprocessingConfig(min_length=5, lowercase=True)
    out, _ = preprocess(docs, cfg)
    assert out[0].text == out[0].text.lower()


def test_stats_output():
    docs = [_doc("valid text that passes all filters and is long enough")] * 5
    cfg = PreprocessingConfig(min_length=10)
    _, stats = preprocess(docs, cfg)
    assert stats["total_input"] == 5
    assert stats["total_output"] == 1  # 4 duplicates removed
    assert "avg_length" in stats


# --- HTML entity decoding ---

def test_strip_html_decodes_entities():
    doc = _doc("Price &amp; quality are &lt;great&gt;")
    cleaned = _strip_html(doc)
    assert "&amp;" not in cleaned.text
    assert "& quality" in cleaned.text


def test_strip_html_removes_tags_and_decodes():
    doc = _doc("<p>Hello &amp; <b>world</b></p>")
    cleaned = _strip_html(doc)
    assert cleaned.text == "Hello & world"


# --- stop word removal ---

def test_remove_stops_removes_common_words():
    stops = {"the", "a", "is"}
    result = _remove_stops("the cat is a animal", stops)
    assert "the" not in result.split()
    assert "a" not in result.split()
    assert "cat" in result.split()


def test_remove_stops_case_insensitive():
    stops = {"the"}
    result = _remove_stops("The quick brown fox", stops)
    assert "The" not in result.split()
    assert "quick" in result.split()


def test_remove_stopwords_integration():
    docs = [_doc("the cat sat on the mat and it was very good for a long time")]
    cfg = PreprocessingConfig(min_length=5, remove_stopwords=True)
    with patch("topic_modeling.preprocessing.cleaner._load_stopwords", return_value={"the", "on", "and", "it", "was", "a", "for"}):
        out, _ = preprocess(docs, cfg)
    assert len(out) == 1
    text = out[0].text
    assert "the" not in text.split()


# --- lemmatization ---

def test_lemmatize_integration():
    docs = [_doc("the cats are running quickly through the forest today")]
    cfg = PreprocessingConfig(min_length=5, lemmatize=True)

    mock_token = lambda lemma: type("Tok", (), {"lemma_": lemma})()
    mock_nlp_doc = [mock_token(w) for w in ["the", "cat", "be", "run", "quickly", "through", "the", "forest", "today"]]
    mock_nlp = MagicMock(return_value=mock_nlp_doc)

    with patch("topic_modeling.preprocessing.cleaner._load_spacy", return_value=mock_nlp):
        out, _ = preprocess(docs, cfg)
    assert len(out) == 1
    assert "cat" in out[0].text
    assert "run" in out[0].text


def test_lemmatize_skipped_when_spacy_unavailable():
    docs = [_doc("the cats are running through the forest and it is a beautiful day")]
    cfg = PreprocessingConfig(min_length=5, lemmatize=True)
    with patch("topic_modeling.preprocessing.cleaner._load_spacy", return_value=None):
        out, _ = preprocess(docs, cfg)
    assert len(out) == 1
    # Text should be unchanged (lemmatize was skipped)
    assert "cats" in out[0].text

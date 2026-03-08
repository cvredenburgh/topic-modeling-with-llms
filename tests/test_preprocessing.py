"""Tests for the preprocessing module."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topic_modeling.config.schema import PreprocessingConfig
from topic_modeling.data.schema import Document
from topic_modeling.preprocessing.cleaner import preprocess, _strip_html


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

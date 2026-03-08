"""Text cleaning and filtering pipeline."""
from __future__ import annotations

import html
import logging
import re
from typing import Any, Callable, List, Set, Tuple

from topic_modeling.config.schema import PreprocessingConfig
from topic_modeling.data.schema import Document

logger = logging.getLogger(__name__)


def preprocess(
    docs: List[Document],
    config: PreprocessingConfig,
) -> Tuple[List[Document], dict]:
    """Apply cleaning and filtering steps.

    Returns:
        (filtered_docs, stats_dict)
    """
    stats: dict = {
        "total_input": len(docs),
        "dropped": {},
        "total_output": 0,
        "avg_length": 0.0,
    }

    if config.strip_html:
        docs = [_strip_html(d) for d in docs]

    if config.lowercase:
        docs = [Document(d.doc_id, d.text.lower(), d.metadata) for d in docs]

    if config.remove_stopwords:
        stops = _load_stopwords()
        docs = [Document(d.doc_id, _remove_stops(d.text, stops), d.metadata) for d in docs]

    if config.lemmatize:
        nlp = _load_spacy()
        if nlp is not None:
            docs = [Document(d.doc_id, _lemmatize(d.text, nlp), d.metadata) for d in docs]

    docs, stats["dropped"]["empty"] = _drop(docs, lambda d: bool(d.text and d.text.strip()))
    docs, stats["dropped"]["too_short"] = _drop(docs, lambda d: len(d.text) >= config.min_length)

    if config.max_length:
        docs, stats["dropped"]["too_long"] = _drop(
            docs, lambda d: len(d.text) <= config.max_length
        )

    if config.remove_duplicates:
        before = len(docs)
        seen: set = set()
        unique = []
        for d in docs:
            key = d.text.strip()
            if key not in seen:
                seen.add(key)
                unique.append(d)
        docs = unique
        stats["dropped"]["duplicates"] = before - len(docs)

    if config.language_filter:
        docs, stats["dropped"]["wrong_language"] = _filter_language(
            docs, config.language_filter
        )

    stats["total_output"] = len(docs)
    stats["avg_length"] = (
        sum(len(d.text) for d in docs) / len(docs) if docs else 0.0
    )

    logger.info(
        f"Preprocessing complete: {stats['total_input']} -> {stats['total_output']} docs  "
        f"dropped={stats['dropped']}"
    )
    return docs, stats


def _drop(
    docs: List[Document], keep: Callable[[Document], bool]
) -> Tuple[List[Document], int]:
    kept = [d for d in docs if keep(d)]
    return kept, len(docs) - len(kept)


def _strip_html(doc: Document) -> Document:
    text = html.unescape(doc.text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return Document(doc.doc_id, text, doc.metadata)


def _remove_stops(text: str, stops: Set[str]) -> str:
    tokens = text.split()
    return " ".join(t for t in tokens if t.lower() not in stops)


def _lemmatize(text: str, nlp: Any) -> str:
    doc = nlp(text)
    return " ".join(token.lemma_ for token in doc)


def _load_stopwords() -> Set[str]:
    try:
        from nltk.corpus import stopwords  # type: ignore
        import nltk  # type: ignore
        try:
            return set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            return set(stopwords.words("english"))
    except ImportError:
        logger.warning("nltk not installed; skipping stop word removal")
        return set()


def _load_spacy() -> Any:
    try:
        import spacy  # type: ignore
        return spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except ImportError:
        logger.warning("spacy not installed; skipping lemmatization")
        return None
    except OSError:
        logger.warning(
            "spacy model 'en_core_web_sm' not found; "
            "run: python -m spacy download en_core_web_sm"
        )
        return None


def _filter_language(
    docs: List[Document], lang: str
) -> Tuple[List[Document], int]:
    try:
        from langdetect import detect, LangDetectException  # type: ignore
    except ImportError:
        logger.warning("langdetect not installed; skipping language filter")
        return docs, 0

    kept = []
    for d in docs:
        try:
            if detect(d.text) == lang:
                kept.append(d)
        except Exception:
            pass
    return kept, len(docs) - len(kept)

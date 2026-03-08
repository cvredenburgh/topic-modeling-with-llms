"""Text cleaning and filtering pipeline."""
from __future__ import annotations

import logging
import re
from typing import List, Tuple

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
    docs: List[Document], keep: object
) -> Tuple[List[Document], int]:
    kept = [d for d in docs if keep(d)]
    return kept, len(docs) - len(kept)


def _strip_html(doc: Document) -> Document:
    text = re.sub(r"<[^>]+>", " ", doc.text)
    text = re.sub(r"\s+", " ", text).strip()
    return Document(doc.doc_id, text, doc.metadata)


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

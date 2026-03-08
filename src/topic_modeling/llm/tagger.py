"""LLM-based topic tagging."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from topic_modeling.config.schema import LLMConfig
from topic_modeling.llm.client import LLMClient
from topic_modeling.models.base import TopicModelBase

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "tag.txt"


def tag_topics(
    model: TopicModelBase,
    config: LLMConfig,
    domain_context: str = "ecommerce product reviews",
    top_n_terms: int = 10,
    top_n_docs: int = 3,
) -> List[Dict]:
    """Assign business/domain tags to every non-outlier topic.

    Returns:
        List of dicts: {topic_id, keywords, tags, raw_response, analysis_status}
        where tags is List[{"tag": str, "consistent": bool}]
    """
    client = LLMClient(config)
    prompt_template = _PROMPT_PATH.read_text()
    topics = model.get_topics()

    results = []
    topic_ids = sorted(k for k in topics if k != -1)

    for i in range(0, len(topic_ids), config.batch_size):
        batch = topic_ids[i : i + config.batch_size]
        for topic_id in batch:
            keywords = [w for w, _ in topics[topic_id][:top_n_terms]]
            rep_docs = model.get_representative_docs(topic_id, n=top_n_docs)

            prompt = prompt_template.format(
                keywords=", ".join(keywords),
                documents=_format_docs(rep_docs),
                domain_context=domain_context,
            )

            raw = ""
            tags: List[Dict[str, Any]] = []
            status = "failed"
            try:
                raw = client.complete(prompt)
                parsed = _parse_json(raw)
                tags = _normalize_tags(parsed.get("tags", []))
                status = "success"
            except Exception as exc:
                logger.warning(f"Tagging failed for topic {topic_id}: {exc}")

            results.append(
                {
                    "topic_id": topic_id,
                    "keywords": keywords,
                    "tags": tags,
                    "raw_response": raw,
                    "analysis_status": status,
                }
            )
            logger.info(f"Tagged topic {topic_id}: {tags}")

    return results


def _normalize_tags(raw_tags: list) -> List[Dict[str, Any]]:
    """Accept both old str format and new dict format, normalizing to dicts."""
    result = []
    for t in raw_tags:
        if isinstance(t, dict):
            result.append({"tag": str(t.get("tag", "")), "consistent": bool(t.get("consistent", False))})
        else:
            result.append({"tag": str(t), "consistent": False})
    return result


def _format_docs(docs: List[str]) -> str:
    return "\n".join(f"  [{i+1}] {doc[:300]}" for i, doc in enumerate(docs))


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)

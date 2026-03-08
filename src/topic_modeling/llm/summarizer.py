"""LLM-based topic summarization."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

from topic_modeling.config.schema import LLMConfig
from topic_modeling.llm.client import LLMClient
from topic_modeling.llm.reliability import summary_consensus
from topic_modeling.models.base import TopicModelBase

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "summarize.txt"


def summarize_topics(
    model: TopicModelBase,
    config: LLMConfig,
    top_n_terms: int = 10,
    top_n_docs: int = 3,
) -> List[Dict]:
    """Generate LLM summaries for every non-outlier topic.

    Returns:
        List of dicts with summary text, reliability metadata, and raw responses.
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
            )

            raw = ""
            raw_responses: List[str] = []
            sampled_summaries: List[str] = []
            summary = ""
            status = "failed"
            try:
                n_samples = config.reliability_samples if config.reliability_enabled else 1
                n_samples = max(1, n_samples)
                for _ in range(n_samples):
                    raw = client.complete(prompt)
                    raw_responses.append(raw)
                    parsed = _parse_json(raw)
                    sampled_summaries.append(str(parsed.get("summary", "")).strip())

                consensus = summary_consensus(
                    sampled_summaries,
                    min_agreement=config.reliability_min_agreement,
                )
                summary = str(consensus["summary"])
                status = "success"
            except Exception as exc:
                logger.warning(f"Summarization failed for topic {topic_id}: {exc}")
                consensus = summary_consensus(
                    sampled_summaries,
                    min_agreement=config.reliability_min_agreement,
                )

            results.append(
                {
                    "topic_id": topic_id,
                    "keywords": keywords,
                    "representative_docs": rep_docs,
                    "summary": summary,
                    "raw_response": raw,
                    "raw_responses": raw_responses,
                    "candidate_summaries": consensus["candidate_summaries"],
                    "reliability_score": consensus["reliability_score"],
                    "reliability_consistent": consensus["reliability_consistent"],
                    "analysis_status": status,
                }
            )
            logger.info(f"Summarized topic {topic_id}: {summary[:80]!r}...")

    return results


def _format_docs(docs: List[str]) -> str:
    return "\n".join(f"  [{i+1}] {doc[:300]}" for i, doc in enumerate(docs))


def _parse_json(text: str) -> dict:
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)

"""Generate run artifact bundles and markdown reports."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from topic_modeling.config.schema import ExperimentConfig, ReportingConfig
from topic_modeling.models.base import TopicModelBase
from topic_modeling.utils.artifacts import ArtifactStore

logger = logging.getLogger(__name__)


def generate_report(
    config: ExperimentConfig,
    store: ArtifactStore,
    model: TopicModelBase,
    topic_ids: List[int],
    metrics: Dict[str, Any],
    summaries: List[Dict],
    tags: List[Dict],
    preprocessing_stats: Dict[str, Any],
) -> None:
    """Save all run artifacts and a human-readable markdown summary."""
    rpt = config.reporting

    # --- topics.csv ---
    topics_rows = _build_topics_df(model, summaries, tags, rpt)
    store.save_csv(topics_rows, "topics.csv")

    # --- metrics.json ---
    store.save_json(metrics, "metrics.json")

    # --- topic_summaries.jsonl ---
    store.save_jsonl(summaries, "topic_summaries.jsonl")

    # --- topic_tags.jsonl ---
    store.save_jsonl(tags, "topic_tags.jsonl")

    # --- preprocessing_stats.json ---
    store.save_json(preprocessing_stats, "preprocessing_stats.json")

    # --- markdown report ---
    if "markdown" in rpt.formats:
        md = _build_markdown(config, metrics, topics_rows, preprocessing_stats)
        store.save_text(md, "report.md")

    logger.info("Report generation complete")


def _build_topics_df(
    model: TopicModelBase,
    summaries: List[Dict],
    tags: List[Dict],
    rpt: ReportingConfig,
) -> pd.DataFrame:
    summary_map = {r["topic_id"]: r for r in summaries}
    tag_map = {r["topic_id"]: r for r in tags}

    rows = []
    for topic_id, word_scores in model.get_topics().items():
        if topic_id == -1:
            continue
        top_terms = ", ".join(w for w, _ in word_scores[: rpt.top_n_terms])
        rep_docs = model.get_representative_docs(topic_id, n=rpt.top_n_docs)
        summary = summary_map.get(topic_id, {}).get("summary", "")
        topic_tags = "|".join(tag_map.get(topic_id, {}).get("tags", []))
        rows.append(
            {
                "topic_id": topic_id,
                "top_terms": top_terms,
                "summary": summary,
                "tags": topic_tags,
                "representative_doc_1": rep_docs[0] if len(rep_docs) > 0 else "",
                "representative_doc_2": rep_docs[1] if len(rep_docs) > 1 else "",
                "representative_doc_3": rep_docs[2] if len(rep_docs) > 2 else "",
            }
        )
    return pd.DataFrame(rows).sort_values("topic_id")


def _build_markdown(
    config: ExperimentConfig,
    metrics: Dict[str, Any],
    topics_df: pd.DataFrame,
    pre_stats: Dict[str, Any],
) -> str:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# Topic Modeling Run Report",
        f"",
        f"**Run:** `{config.run.name}`  ",
        f"**Backend:** `{config.model.backend}`  ",
        f"**Generated:** {now}  ",
        f"**Seed:** {config.run.seed}  ",
        f"",
        f"---",
        f"",
        f"## Dataset",
        f"- Source: `{config.dataset.source}`",
        f"- Name: `{config.dataset.name}`",
        f"- Split: `{config.dataset.split}`",
        f"- Input docs: {pre_stats.get('total_input', '?')}",
        f"- After preprocessing: {pre_stats.get('total_output', '?')}",
        f"- Avg doc length (chars): {pre_stats.get('avg_length', 0):.0f}",
        f"",
        f"## Evaluation Metrics",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
    ]
    for k, v in metrics.items():
        fmt = f"{v:.4f}" if isinstance(v, float) else str(v)
        lines.append(f"| {k} | {fmt} |")

    lines += [
        f"",
        f"## Topics ({len(topics_df)})",
        f"",
    ]

    for _, row in topics_df.iterrows():
        lines += [
            f"### Topic {int(row['topic_id'])}",
            f"**Top terms:** {row['top_terms']}  ",
            f"**Tags:** {row['tags']}  ",
            f"**Summary:** {row['summary']}  ",
            f"",
        ]

    return "\n".join(lines)

"""Generate run artifact bundles and markdown reports."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

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
    tuning_trials: Optional[List[Dict]] = None,
    analysis_results: Optional[Dict[str, Any]] = None,
) -> None:
    """Save all run artifacts and a human-readable markdown summary."""
    rpt = config.reporting

    # --- topics.csv ---
    topics_rows = _build_topics_df(model, summaries, tags, rpt)
    store.save_csv(topics_rows, "topics.csv")

    # --- metrics.json (flatten out per-topic sub-dict) ---
    per_topic = metrics.get("per_topic_metrics")
    store.save_json(metrics, "metrics.json")
    if per_topic:
        store.save_json(per_topic, "topic_metrics.json")

    # --- topic_summaries.jsonl ---
    store.save_jsonl(summaries, "topic_summaries.jsonl")

    # --- topic_tags.jsonl ---
    store.save_jsonl(tags, "topic_tags.jsonl")

    # --- preprocessing_stats.json ---
    store.save_json(preprocessing_stats, "preprocessing_stats.json")

    # --- analysis artifacts ---
    if analysis_results:
        if "hierarchy" in analysis_results:
            store.save_csv(analysis_results["hierarchy"], "hierarchy.csv")
        if "associations" in analysis_results:
            store.save_csv(analysis_results["associations"], "associations.csv")
        if "trends" in analysis_results:
            store.save_csv(analysis_results["trends"], "topic_trends.csv")
        if "emerging" in analysis_results:
            store.save_csv(analysis_results["emerging"], "emerging_topics.csv")
        if "trend_stats" in analysis_results:
            store.save_csv(analysis_results["trend_stats"], "trend_stats.csv")

    # --- markdown report ---
    if "markdown" in rpt.formats:
        md = _build_markdown(config, metrics, topics_rows, preprocessing_stats, analysis_results or {})
        store.save_text(md, "report.md")

    # --- figures ---
    if "figures" in rpt.formats and tuning_trials:
        from topic_modeling.reporting.figures import generate_figures
        metric_cols = rpt.figure_metric_cols or ["coherence", "diversity"]
        generate_figures(
            trials=tuning_trials,
            metric_cols=metric_cols,
            param_metric_pairs=rpt.figure_param_metric_pairs,
            output_dir=store.run_dir / "figures",
        )

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
        raw_tags = tag_map.get(topic_id, {}).get("tags", [])
        topic_tags = "|".join(
            t["tag"] if isinstance(t, dict) else t for t in raw_tags
        )
        consistent_tags = "|".join(
            t["tag"] for t in raw_tags if isinstance(t, dict) and t.get("consistent")
        )
        rows.append(
            {
                "topic_id": topic_id,
                "top_terms": top_terms,
                "summary": summary,
                "tags": topic_tags,
                "consistent_tags": consistent_tags,
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
    analysis_results: Dict[str, Any] = {},
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

    # --- Hierarchy section ---
    if "hierarchy" in analysis_results and not analysis_results["hierarchy"].empty:
        hier_df = analysis_results["hierarchy"]
        lines += ["---", "", "## Topic Hierarchy", ""]
        # Show top-level clusters: highest linkage_height rows (one per parent)
        top_merges = hier_df.drop_duplicates("parent_id").nlargest(5, "linkage_height")
        for _, r in top_merges.iterrows():
            children = hier_df[hier_df["parent_id"] == r["parent_id"]]["child_id"].tolist()
            lines.append(f"- **Cluster {int(r['parent_id'])}** (similarity={r['similarity']:.3f}): topics {children}")
        lines.append("")

    # --- Emerging topics section ---
    if "emerging" in analysis_results and not analysis_results["emerging"].empty:
        emerging = analysis_results["emerging"][analysis_results["emerging"]["emerging"]]
        if not emerging.empty:
            lines += ["---", "", "## Emerging Topics", ""]
            lines += [
                "| Topic ID | Growth Rate |",
                "|----------|-------------|",
            ]
            for _, r in emerging.iterrows():
                gr = r["growth_rate"]
                gr_str = f"{gr:.1%}" if gr != float("inf") else "∞"
                lines.append(f"| {int(r['topic_id'])} | {gr_str} |")
            lines.append("")

    return "\n".join(lines)

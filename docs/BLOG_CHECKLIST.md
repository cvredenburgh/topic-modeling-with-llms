# Blog Outline To Artifacts Checklist

Use this to ensure each section in `content/index.md` is backed by generated outputs.

## Topic Modeling Background

- Statistical foundations -> `report.md`, `metrics.json`, `topic_metrics.json`.
- Limits of topic modeling -> contrast raw `topics.csv` terms vs LLM-enriched fields.

## Popular APIs

- BERTopic run -> `baseline_bertopic` outputs.
- FASTopic run -> `baseline_fastopic` outputs.
- Comparison table -> build from both runs' `metrics.json` and runtime stats.

## Evaluation & Optimization

- Baseline metrics -> `metrics.json`.
- Tuning process -> `tuning_results.json` + figures.
- Best-parameter outcomes -> tuned run `report.md`.

## Hierarchical Trends & Relationships

- Topic hierarchy -> `hierarchy.csv`.
- Topic associations -> `associations.csv`.
- Temporal trends + emerging topics -> `topic_trends.csv`, `emerging_topics.csv`.
- Significance tests -> `trend_stats.csv`.

## LLM Topic Enrichment

- Summarization -> `topic_summaries.jsonl` + `topics.csv.summary`.
- Automated labeling -> `topic_tags.jsonl` + `topics.csv.tags`.
- Reliability layer -> `reliability_score` and `reliability_consistent` fields in LLM artifacts and `topics.csv`.

## From Signal To Action

- Segment/action framing -> use topic tags + trend significance to produce KPI-oriented narrative in notebook/blog.

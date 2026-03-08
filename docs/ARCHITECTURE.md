# Architecture

## Overview

The pipeline is a sequential, config-driven workflow. All runtime behavior is
controlled by YAML experiment configs; no constants are hardcoded in modules.

```
scripts/run_pipeline.py
    │
    ├─ config/loader.py          Load + validate ExperimentConfig (Pydantic)
    ├─ utils/seeds.py            Set global random seeds
    │
    ├─ data/loader.py            Load from HuggingFace or local file
    ├─ data/schema.py            Canonical Document dataclass
    │
    ├─ preprocessing/cleaner.py  Clean, filter, deduplicate
    │
    ├─ tuning/tuner.py           Optuna / grid / random search (optional)
    │
    ├─ models/
    │   ├─ base.py               Abstract TopicModelBase interface
    │   ├─ bertopic_adapter.py   BERTopic wrapper
    │   └─ fastopic_adapter.py   FASTopic wrapper
    │
    ├─ evaluation/metrics.py     Coherence, diversity, outlier ratio, size stats
    │
    ├─ llm/
    │   ├─ client.py             Anthropic API client with retry
    │   ├─ summarizer.py         Topic summarization
    │   ├─ tagger.py             Topic tagging
    │   └─ prompts/              Prompt templates (plain text, format strings)
    │
    ├─ reporting/reporter.py     topics.csv, metrics.json, report.md
    │
    └─ utils/artifacts.py        Local + GCS artifact persistence
```

## Key Design Decisions

### Config-first execution
All adjustable behaviour lives in YAML. The `--set key=value` CLI flag supports
runtime overrides without file edits.

### Pluggable backends
`models/__init__.py::build_model()` is the single factory. Switching between
BERTopic and FASTopic requires only changing `model.backend` in the config.

### Abstract interface
`TopicModelBase` defines: `fit()`, `transform()`, `get_topics()`,
`get_topic_info()`, `get_representative_docs()`, `save()`, `load()`.
Both adapters implement this contract identically.

### LLM post-processing
Prompt templates live in `llm/prompts/` as plain `.txt` files with Python
`str.format()` placeholders. The `LLMClient` handles retry with exponential
backoff. All raw and parsed LLM responses are saved to disk.

### Artifact persistence
`ArtifactStore` writes locally under `outputs/runs/<run_id>/` and optionally
mirrors to GCS when `runtime.target=gcp`.

## Runtime Targets

| Target | IO | Execution |
|--------|-----|-----------|
| `local` | Local filesystem | Direct Python process |
| `gcp`   | GCS bucket (via `google-cloud-storage`) | Vertex AI / Cloud Run job |

Switch via config: `--set runtime.target=gcp`

## Run Artifact Layout

```
outputs/runs/<run_id>/
  config.json               Full resolved config
  preprocessing_stats.json  Documents dropped per stage
  metrics.json              Coherence, diversity, outlier ratio
  topics.csv                Per-topic: terms, summary, tags, rep docs
  topic_summaries.jsonl     Raw + parsed LLM summarization output
  topic_tags.jsonl          Raw + parsed LLM tagging output
  tuning_results.json       Best params + all trial scores (if tuning enabled)
  report.md                 Human-readable run summary
  run_summary.json          Top-level run metadata
  artifacts/model/          Serialised topic model
```

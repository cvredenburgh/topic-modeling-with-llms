# Topic Modeling With LLMs

End-to-end pipeline for bottom-up topic discovery on real-world text corpora, with LLM-assisted post-processing to produce human-readable topic summaries and business labels.

Supports **BERTopic** and **FASTopic** as interchangeable backends, configurable hyperparameter tuning via Optuna, and both local and GCP execution — all controlled by YAML config with no code changes required.

---

## Features

- **Pluggable backends** — swap BERTopic and FASTopic via a single config field
- **Config-driven** — all pipeline behaviour controlled by YAML; runtime overrides via `--set`
- **Hyperparameter tuning** — Optuna, grid, or random search with configurable search spaces
- **LLM post-processing** — configurable provider (`anthropic`, `openai`, `gemini`, `grok`) for topic summaries and business tags
- **LLM reliability layer** — multi-sample consensus scoring for summaries/tags with agreement thresholds
- **Structured outputs** — `topics.csv`, `metrics.json`, `.jsonl` artifacts, and a markdown report per run
- **Reproducible** — global seed setting, cached interim data, saved model artifacts
- **GCP-ready** — config-only switch from local to GCP artifact storage

---

## Project Structure

```
topic-modeling-with-llms/
  configs/
    experiment/        # Top-level run configs (baseline & tuning, per backend)
    dataset/           # Dataset source configs
    llm/               # LLM provider/prompt configs
    runtime/           # local.yaml and gcp.yaml
  data/
    raw/ interim/ processed/
  outputs/runs/<run_id>/
    config.json  metrics.json  topics.csv
    topic_summaries.jsonl  topic_tags.jsonl
    report.md  run_summary.json
    artifacts/model/
  src/topic_modeling/
    config/            # Pydantic schema + YAML loader
    data/              # HuggingFace / local file ingestion
    preprocessing/     # Text cleaning and filtering
    models/            # BERTopic and FASTopic adapters
    tuning/            # Optuna / grid / random search
    evaluation/        # Coherence, diversity, outlier ratio
    llm/               # Anthropic client, summarizer, tagger, prompt templates
    pipelines/         # End-to-end Pipeline orchestrator
    reporting/         # Artifact generation and markdown report
    utils/             # Logging, seeds, artifact store
  scripts/
    run_pipeline.py    # Single CLI entrypoint
  tests/               # Unit tests (49 passing)
  docs/
    ARCHITECTURE.md
    EXPERIMENTS.md
    INSTRUCTIONS.md
```

---

## Quickstart

### 1. Environment

```bash
pyenv activate topic-modeling-env   # or: source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run a baseline experiment

```bash
# BERTopic on Amazon Beauty reviews (20k sample)
python scripts/run_pipeline.py --config configs/experiment/baseline_bertopic.yaml

# FASTopic
python scripts/run_pipeline.py --config configs/experiment/baseline_fastopic.yaml
```

### 4. Run with hyperparameter tuning

```bash
python scripts/run_pipeline.py --config configs/experiment/tune_bertopic.yaml
python scripts/run_pipeline.py --config configs/experiment/tune_fastopic.yaml
```

### 5. Override config values at runtime

```bash
python scripts/run_pipeline.py \
  --config configs/experiment/baseline_bertopic.yaml \
  --set run.seed=123 \
  --set dataset.sample_n=5000 \
  --set llm.enabled=false
```

### 6. Switch to GCP

```bash
# Edit configs/runtime/gcp.yaml with your project/bucket, then:
python scripts/run_pipeline.py \
  --config configs/experiment/baseline_bertopic.yaml \
  --set runtime.target=gcp
```

### 7. Switch LLM providers and enable reliability consensus

```bash
# OpenAI example
python scripts/run_pipeline.py \
  --config configs/experiment/baseline_bertopic.yaml \
  --set llm.provider=openai \
  --set llm.model=gpt-4.1-mini

# Grok example (OpenAI-compatible API)
python scripts/run_pipeline.py \
  --config configs/experiment/baseline_bertopic.yaml \
  --set llm.provider=grok \
  --set llm.model=grok-2-latest

# Reliability consensus (3 samples per topic)
python scripts/run_pipeline.py \
  --config configs/experiment/baseline_bertopic.yaml \
  --set llm.reliability_enabled=true \
  --set llm.reliability_samples=3 \
  --set llm.reliability_min_agreement=0.67
```

---

## Run Outputs

Each run writes artifacts under `outputs/runs/<run_id>/`:

| File | Contents |
|------|----------|
| `config.json` | Full resolved config |
| `preprocessing_stats.json` | Documents dropped per filter stage |
| `metrics.json` | Coherence, diversity, outlier ratio, topic size stats |
| `topics.csv` | Per-topic: top terms, LLM summary, tags, representative docs |
| `topic_summaries.jsonl` | Raw + parsed LLM summarization output |
| `topic_tags.jsonl` | Raw + parsed LLM tagging output |
| `tuning_results.json` | Best params + all trial scores (if tuning enabled) |
| `report.md` | Human-readable run summary |
| `run_summary.json` | Top-level run metadata |
| `artifacts/model/` | Serialized topic model |

---

## Tests

```bash
pytest tests/ -v
```

49 unit tests covering: config loading and overrides, preprocessing transforms, model adapter interface, evaluation metrics, LLM client retry + output parsing, artifact persistence, and pipeline stage orchestration.

---

## Configuration Reference

Top-level experiment configs reference these sections:

| Section | Key fields |
|---------|-----------|
| `run` | `name`, `seed`, `output_dir`, `log_level` |
| `runtime` | `target` (`local`/`gcp`), GCP settings |
| `dataset` | `source`, `name`, `subset`, `split`, `text_column`, `sample_n` |
| `preprocessing` | `min_length`, `max_length`, `remove_duplicates`, `language_filter` |
| `model` | `backend` (`bertopic`/`fastopic`), `params` |
| `tuning` | `enabled`, `method`, `n_trials`, `search_space` |
| `evaluation` | `metrics`, `coherence_measure`, `topn` |
| `llm` | `provider`, `model`, `api_key_env`, `api_base`, `reliability_*`, `batch_size`, `max_retries`, `enabled` |
| `reporting` | `formats`, `top_n_terms`, `top_n_docs`, `save_model` |

See `docs/ARCHITECTURE.md` for design details and `docs/EXPERIMENTS.md` for the experiment log.

---

## Dataset

Default: **Amazon Reviews 2023** (Beauty category, 20k sample) via HuggingFace Datasets.
Citation: McAuley Lab, UCSD — [https://amazon-reviews-2023.github.io](https://amazon-reviews-2023.github.io)

# Topic Modeling With LLMs - Build Instructions for Claude Code

## Purpose

Build a production-quality, portfolio-ready AI/data science project that demonstrates:

1. Bottom-up topic modeling on real-world text corpora.
2. Model optimization and hyperparameter tuning.
3. Pluggable support for either `BERTopic` or `FASTopic`.
4. LLM-assisted post-processing to:
   - summarize discovered topics;
   - tag topics with useful designations (business/domain labels).

The codebase should be modular, config-driven, reproducible, and easy to extend.

---

## High-Level Requirements

### Core goals

- Implement a complete end-to-end pipeline:
  - data ingestion;
  - text preprocessing;
  - embedding/topic modeling;
  - tuning and evaluation;
  - LLM topic summarization/tagging;
  - outputs and reporting.
- Keep architecture model-agnostic enough to swap `BERTopic` and `FASTopic`.
- Support two execution targets via config:
  - local development execution;
  - GCP execution (batch/job-style).
- Minimize notebook-only logic; prefer package modules + a CLI entrypoint.
- Use deterministic runs where possible (set random seeds).

### Project quality goals

- Strong separation of concerns by module.
- Config-first execution (YAML/TOML/JSON).
- Clear logging and artifact tracking.
- Re-runnable pipelines with saved intermediate artifacts.
- Documentation that explains how to run baseline and tuned experiments.

---

## Suggested Use Cases / Datasets

Start with one primary dataset, then optionally add a second domain for transferability.

### Candidate dataset domains

- Ecommerce product reviews (preferred first dataset).
- Public comments / policy feedback (government or civic sources).
- Additional optional corpus for robustness checks.

### Data source suggestions

- Hugging Face Datasets (recommended for easy loading/versioning).
- Public CSV/JSONL corpora with clear licensing.

### Dataset criteria

- Enough volume to reveal meaningful topic structure.
- Rich free-text fields (not only ratings/metadata).
- Clear license and citation metadata.

---

## Deliverable Architecture

Target a structure similar to:

```text
topic-modeling-with-llms/
  configs/
    experiment/
      baseline_bertopic.yaml
      baseline_fastopic.yaml
      tune_bertopic.yaml
      tune_fastopic.yaml
    dataset/
      ecommerce_reviews.yaml
      gov_comments.yaml
    llm/
      summarizer_default.yaml
      tags_default.yaml
    runtime/
      local.yaml
      gcp.yaml
  data/
    raw/
    interim/
    processed/
  outputs/
    runs/
      <run_id>/
        metrics.json
        topics.csv
        topic_summaries.jsonl
        topic_tags.jsonl
        artifacts/
  src/
    topic_modeling/
      __init__.py
      config/
      data/
      preprocessing/
      models/
      tuning/
      evaluation/
      llm/
      pipelines/
      reporting/
      utils/
  scripts/
    run_pipeline.py
  notebooks/
    publish/
      final-post.md
      final-post.html
  tests/
  docs/
    INSTRUCTIONS.md
    ARCHITECTURE.md
    EXPERIMENTS.md
```

Notes:
- `scripts/run_pipeline.py` is the single execution entrypoint.
- `src/topic_modeling/models/` contains interchangeable wrappers for BERTopic/FASTopic.
- Keep generated artifacts out of source modules.
- `notebooks/publish/` is for the final website-ready narrative artifact.

---

## Functional Requirements by Module

### 1) Data module

- Load data from configurable providers:
  - Hugging Face dataset loader and/or local files.
- Normalize schema into canonical fields:
  - `doc_id`, `text`, optional metadata (date, source, rating, category, etc.).
- Persist canonical dataset to `data/interim/` or `data/processed/`.

### 2) Preprocessing module

- Configurable text cleaning and filtering:
  - null/short text removal;
  - language filter (optional);
  - stopword and noise handling;
  - deduplication;
  - optional lemmatization/token normalization.
- Store preprocessing stats (documents dropped, avg length, etc.).

### 3) Modeling module (bottom-up topic modeling)

- Provide an abstract interface:
  - `fit()`, `transform()`, `get_topics()`, `save()`, `load()`.
- Implement adapters:
  - `BERTopicAdapter`
  - `FASTopicAdapter`
- All adapter behavior should be driven by config (not hardcoded constants).

### 4) Tuning module

- Hyperparameter search (grid/random/optuna-style; at least one method).
- Configurable search space for each model type.
- Record all trials and best configuration.
- Persist tuning summary and selected best params.

### 5) Evaluation module

- Track model quality metrics (as applicable):
  - coherence;
  - diversity;
  - topic size distribution;
  - outlier/unassigned ratio;
  - optional stability across seeds/subsamples.
- Output machine-readable metrics + human-readable summary table.

### 6) LLM module (topic post-processing)

Two required capabilities:

1. Topic summarization:
   - Given representative docs/keywords for each topic, produce concise summary text.

2. Topic tagging:
   - Assign practical labels/tags (for example: product quality, shipping, policy concern, pricing, service).

Requirements:
- Prompt templates stored separately from code.
- Model/provider configurable.
- Batched inference with retry/error handling.
- Structured outputs (JSON schema preferred).
- Save raw and normalized outputs for traceability.

### 7) Reporting module

- Generate run artifact bundle with:
  - top terms per topic;
  - representative docs;
  - metrics;
  - LLM summaries/tags.
- Produce a compact markdown report per run for easy review.

---

## Config-Driven Design

Execution should be controlled by one top-level experiment config that references others.

### Example config sections

- `run`: run name, seed, output directory, logging level.
- `runtime`: execution target (`local` or `gcp`) and target-specific settings.
- `dataset`: source, split, sampling rules, text column, filters.
- `preprocessing`: cleaning flags and thresholds.
- `model`: backend (`bertopic` or `fastopic`) and params.
- `tuning`: enabled flag, method, trials/search space.
- `evaluation`: metrics to compute.
- `llm`: provider/model, prompts, batch size, retry policy.
- `reporting`: output file formats and report generation options.

Do not scatter runtime constants across modules; everything adjustable should come from config.

### Runtime execution requirements

Support both execution modes from the same pipeline code:

1. Local mode:
   - run directly on developer machine;
   - local cache and outputs under project directories.

2. GCP mode:
   - package and execute as a reproducible cloud job;
   - use configurable project/region/bucket/image settings;
   - write artifacts to cloud storage (with optional local sync).

The selected mode must come from config, not code edits.

---

## Execution / Run File Requirements

Implement a single executable script:

- `scripts/run_pipeline.py`

Responsibilities:

1. Parse CLI args (config path, override flags, run id).
2. Load/validate config.
3. Resolve runtime target (`local` or `gcp`) and initialize IO paths.
4. Execute pipeline stages in order.
5. Save artifacts and metrics.
6. Emit clear logs and final run summary.

Nice-to-have CLI examples:

```bash
python scripts/run_pipeline.py --config configs/experiment/baseline_bertopic.yaml
python scripts/run_pipeline.py --config configs/experiment/tune_fastopic.yaml --set run.seed=123
python scripts/run_pipeline.py --config configs/experiment/baseline_bertopic.yaml --set runtime.target=gcp
```

---

## Engineering Standards

- Use type hints and dataclasses/pydantic for config validation.
- Write small, testable functions (avoid monolithic notebooks/scripts).
- Add unit tests for core logic:
  - config loading/validation;
  - preprocessing transforms;
  - model adapter interface behavior;
  - LLM output parsing/validation.
- Prefer explicit interfaces over implicit coupling.
- Add logging at each pipeline stage.

---

## Milestones (Suggested Build Plan)

1. Scaffold package structure + config loader + CLI.
2. Implement data ingestion + preprocessing baseline.
3. Add BERTopic adapter + baseline run.
4. Add FASTopic adapter with shared interface.
5. Implement tuning workflow and metrics logging.
6. Add LLM summarization + tagging with structured outputs.
7. Add reporting and run artifacts.
8. Add tests and documentation.

---

## Success Criteria

Project is complete when:

- A single command runs an end-to-end pipeline from raw text to labeled topics.
- Switching between BERTopic and FASTopic requires config change only.
- Tuning workflow identifies and stores a best configuration.
- LLM summaries and tags are generated and saved in structured form.
- Outputs are reproducible and understandable by another engineer/recruiter.
- The same pipeline can execute locally and on GCP via config-only runtime switch.

---

## Notebook and Publishing Deliverable

- Use `notebooks/` for exploratory analysis and final narrative assembly.
- Create a final publishable artifact in `notebooks/publish/` as `.md` or `.html`.
- The publishable artifact should be suitable for posting on `chrisvred.com`.
- Keep core pipeline logic in `src/` and `scripts/`; notebooks should orchestrate and explain, not own critical business logic.

---

## Optional Stretch Goals

- Add topic drift monitoring for temporal slices.
- Add experiment tracking integration (MLflow or equivalent).
- Add lightweight dashboard for topic exploration.
- Add prompt/version registry for LLM post-processing reproducibility.

# CLAUDE.md — Agent Instructions

This file tells Claude Code how to work in this repository.

## Project Overview

End-to-end topic modeling pipeline (BERTopic / FASTopic) with LLM post-processing via the Anthropic API. Config-driven, reproducible, supports local and GCP execution.

## Environment Setup

```bash
pyenv activate topic-modeling-env   # Python 3.11.6
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...
```

**Always run scripts and tests from the project root** (`topic-modeling-with-llms/`). Relative paths (e.g. `data/interim/`, `outputs/runs/`, `configs/`) are resolved from CWD.

## Running Tests

```bash
pytest tests/ -v
```

All 49 tests must pass before committing. Tests use mocks for external dependencies (BERTopic, FASTopic, Anthropic API) — no network calls or real model loads required.

## Running the Pipeline

```bash
python scripts/run_pipeline.py --config configs/experiment/baseline_bertopic.yaml
python scripts/run_pipeline.py --config configs/experiment/baseline_bertopic.yaml --set llm.enabled=false
```

## Repository Layout

```
src/topic_modeling/   Package source — all importable modules
scripts/              CLI entrypoints only (run_pipeline.py)
configs/              YAML experiment/dataset/llm/runtime configs
tests/                Unit tests (pytest)
data/                 Raw, interim, processed data (gitignored except .gitkeep)
outputs/              Run artifacts (gitignored)
docs/                 ARCHITECTURE.md, EXPERIMENTS.md, INSTRUCTIONS.md
```

## Code Conventions

- **Config changes only** to switch backends, datasets, or execution targets — never hardcode in source modules.
- **Pydantic v2** for all config schemas (`src/topic_modeling/config/schema.py`).
- **Lazy imports** for heavy dependencies (bertopic, fastopic, sentence-transformers, gensim) — import inside functions, not at module level.
- **Abstract interface** `TopicModelBase` (`models/base.py`) must be implemented by all adapters. Do not add backend-specific logic outside the adapter files.
- **ArtifactStore** (`utils/artifacts.py`) is the only way to write run outputs — never write files directly from pipeline stages.
- Prompt templates live in `src/topic_modeling/llm/prompts/` as plain `.txt` files.

## Adding a New Model Backend

1. Create `src/topic_modeling/models/<name>_adapter.py` implementing `TopicModelBase`.
2. Register it in `src/topic_modeling/models/__init__.py::build_model()`.
3. Add a config entry to `configs/experiment/baseline_<name>.yaml`.
4. Add adapter tests in `tests/test_models.py` following the existing mock pattern.

## Adding a New Config Field

1. Add field to the relevant Pydantic class in `config/schema.py`.
2. Update any relevant YAML configs under `configs/`.
3. Add a test in `tests/test_config.py` if the field has validation logic.

## Key Files

| File | Purpose |
|------|---------|
| `src/topic_modeling/config/schema.py` | All Pydantic config models |
| `src/topic_modeling/config/loader.py` | YAML load + `--set` override logic |
| `src/topic_modeling/models/base.py` | Abstract adapter interface |
| `src/topic_modeling/pipelines/pipeline.py` | Pipeline orchestration |
| `scripts/run_pipeline.py` | CLI entrypoint |
| `configs/experiment/baseline_bertopic.yaml` | Primary reference config |

## GCP Execution

Set `runtime.target=gcp` and fill in `configs/runtime/gcp.yaml` with your project, region, bucket, and image. The pipeline will mirror all artifacts to GCS after writing locally. Install GCP extras: `pip install -e ".[gcp]"`.

## What Not To Do

- Do not add logic to `scripts/run_pipeline.py` beyond CLI arg parsing and calling `Pipeline.run()`.
- Do not import `bertopic`, `fastopic`, `sentence_transformers`, or `gensim` at module level.
- Do not write output files outside of `ArtifactStore`.
- Do not add notebook-only business logic — keep `notebooks/` for narrative and exploration only.

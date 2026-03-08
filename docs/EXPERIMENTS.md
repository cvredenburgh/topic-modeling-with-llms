# Experiments

## How to Run

### Prerequisites

```bash
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-...
```

### Baseline BERTopic

```bash
python scripts/run_pipeline.py --config configs/experiment/baseline_bertopic.yaml
```

### Baseline FASTopic

```bash
python scripts/run_pipeline.py --config configs/experiment/baseline_fastopic.yaml
```

### BERTopic with Optuna tuning (30 trials)

```bash
python scripts/run_pipeline.py --config configs/experiment/tune_bertopic.yaml
```

### Override seed or runtime target

```bash
python scripts/run_pipeline.py \
  --config configs/experiment/baseline_bertopic.yaml \
  --set run.seed=123 \
  --set runtime.target=gcp
```

### Disable LLM post-processing (faster iteration)

```bash
python scripts/run_pipeline.py \
  --config configs/experiment/baseline_bertopic.yaml \
  --set llm.enabled=false
```

---

## Experiment Log

| Run ID | Backend | Dataset | n_docs | Topics | Coherence | Notes |
|--------|---------|---------|--------|--------|-----------|-------|
| TBD    | BERTopic | Amazon Beauty | 20k | — | — | Baseline |
| TBD    | FASTopic | Amazon Beauty | 20k | 50 | — | Baseline |
| TBD    | BERTopic | Amazon Beauty | 20k | — | — | Tuned (30 trials) |

---

## Metric Definitions

| Metric | Description |
|--------|-------------|
| `coherence` | Gensim C_V coherence — higher is better (typical range 0.4–0.7) |
| `diversity` | Fraction of unique words across all top-word lists — higher = less redundancy |
| `outlier_ratio` | Fraction of docs assigned to topic -1 (HDBSCAN noise cluster) |
| `topic_size_mean` | Average number of documents per topic |

---

## Tuning Search Space (BERTopic)

| Parameter | Type | Range |
|-----------|------|-------|
| `umap_n_components` | int | 3–15 |
| `umap_n_neighbors` | int | 5–50 |
| `hdbscan_min_cluster_size` | int | 5–50 |
| `hdbscan_min_samples` | int | 1–10 |

## Tuning Search Space (FASTopic)

| Parameter | Type | Range |
|-----------|------|-------|
| `num_topics` | int | 20–100 |

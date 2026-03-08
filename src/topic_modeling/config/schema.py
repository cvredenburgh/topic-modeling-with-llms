"""Pydantic config schema for all pipeline sections."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    name: str
    seed: int = 42
    output_dir: str = "outputs/runs"
    log_level: str = "INFO"


class RuntimeConfig(BaseModel):
    target: Literal["local", "gcp"] = "local"
    gcp_project: Optional[str] = None
    gcp_region: Optional[str] = None
    gcp_bucket: Optional[str] = None
    gcp_image: Optional[str] = None


class DatasetConfig(BaseModel):
    source: Literal["huggingface", "local"] = "huggingface"
    name: Optional[str] = None
    subset: Optional[str] = None
    split: str = "train"
    text_column: str = "text"
    id_column: Optional[str] = None
    metadata_columns: List[str] = Field(default_factory=list)
    sample_n: Optional[int] = None
    sample_frac: Optional[float] = None
    local_path: Optional[str] = None
    filters: Dict[str, Any] = Field(default_factory=dict)


class PreprocessingConfig(BaseModel):
    min_length: int = 20
    max_length: Optional[int] = None
    language_filter: Optional[str] = None
    remove_duplicates: bool = True
    lowercase: bool = False
    strip_html: bool = True
    remove_stopwords: bool = False
    lemmatize: bool = False


class ModelConfig(BaseModel):
    backend: Literal["bertopic", "fastopic"] = "bertopic"
    params: Dict[str, Any] = Field(default_factory=dict)


class TuningConfig(BaseModel):
    enabled: bool = False
    method: Literal["grid", "random", "optuna"] = "optuna"
    n_trials: int = 20
    timeout_seconds: Optional[int] = None
    metric: str = "coherence"
    search_space: Dict[str, Any] = Field(default_factory=dict)


class EvaluationConfig(BaseModel):
    metrics: List[str] = Field(
        default=["coherence", "diversity", "outlier_ratio", "topic_size_stats"]
    )
    coherence_measure: str = "c_v"
    topn: int = 10


class LLMConfig(BaseModel):
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-6"
    batch_size: int = 10
    max_retries: int = 3
    retry_delay_seconds: float = 2.0
    max_tokens: int = 1024
    temperature: float = 0.0
    enabled: bool = True


class ReportingConfig(BaseModel):
    formats: List[str] = Field(default=["markdown", "json"])
    top_n_terms: int = 10
    top_n_docs: int = 3
    save_model: bool = True


class ExperimentConfig(BaseModel):
    run: RunConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    dataset: DatasetConfig
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    model: ModelConfig
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)

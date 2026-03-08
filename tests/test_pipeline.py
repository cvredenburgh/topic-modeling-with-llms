"""Unit tests for Pipeline stage orchestration."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from topic_modeling.config.schema import (
    DatasetConfig,
    EvaluationConfig,
    ExperimentConfig,
    LLMConfig,
    ModelConfig,
    PreprocessingConfig,
    ReportingConfig,
    RunConfig,
    RuntimeConfig,
    TuningConfig,
)
from topic_modeling.pipelines.pipeline import Pipeline, _make_run_id
from topic_modeling.tuning.scoring import composite_score, estimate_bounds


def _minimal_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        run=RunConfig(name="test_run", output_dir=str(tmp_path / "runs")),
        runtime=RuntimeConfig(target="local"),
        dataset=DatasetConfig(source="local", local_path="dummy.csv", text_column="text"),
        preprocessing=PreprocessingConfig(min_length=5),
        model=ModelConfig(backend="bertopic"),
        tuning=TuningConfig(enabled=False),
        evaluation=EvaluationConfig(metrics=["diversity", "outlier_ratio"]),
        llm=LLMConfig(enabled=False),
        reporting=ReportingConfig(save_model=False),
    )


def test_make_run_id_format():
    run_id = _make_run_id("my_run")
    assert run_id.startswith("my_run_")
    # timestamp portion: YYYYMMDD_HHMMSS = 15 chars
    assert len(run_id) == len("my_run_") + 15


def test_pipeline_init(tmp_path):
    cfg = _minimal_config(tmp_path)
    pipeline = Pipeline(cfg, run_id="test_001")
    assert pipeline.run_id == "test_001"
    assert pipeline.run_dir == Path(cfg.run.output_dir) / "test_001"


def test_pipeline_run_stages_called(tmp_path):
    """Verify pipeline calls each stage in order with mocked implementations."""
    cfg = _minimal_config(tmp_path)
    pipeline = Pipeline(cfg, run_id="test_stages")

    mock_docs = [MagicMock(text="this is a test document") for _ in range(5)]
    mock_model = MagicMock()
    mock_model.get_topic_count.return_value = 3
    mock_model.get_topics.return_value = {
        0: [("word1", 0.9)],
        1: [("word2", 0.8)],
        2: [("word3", 0.7)],
    }
    mock_model.get_representative_docs.return_value = []

    with (
        patch.object(pipeline, "_stage_data", return_value=mock_docs) as mock_data,
        patch.object(
            pipeline,
            "_stage_preprocess",
            return_value=(mock_docs, {"total_input": 5, "total_output": 5, "dropped": {}, "avg_length": 20.0}),
        ) as mock_pre,
        patch.object(
            pipeline,
            "_stage_model",
            return_value=(mock_model, [0, 1, 2, 0, 1]),
        ) as mock_model_stage,
        patch.object(
            pipeline,
            "_stage_evaluate",
            return_value={"diversity": 1.0, "outlier_ratio": 0.0},
        ) as mock_eval,
        patch.object(pipeline, "_stage_llm", return_value=([], [])) as mock_llm,
        patch.object(pipeline, "_stage_report") as mock_report,
    ):
        summary = pipeline.run()

    mock_data.assert_called_once()
    mock_pre.assert_called_once()
    mock_model_stage.assert_called_once()
    mock_eval.assert_called_once()
    mock_llm.assert_called_once()
    mock_report.assert_called_once()

    assert summary["run_id"] == "test_stages"
    assert summary["topic_count"] == 3
    assert "metrics" in summary


def test_pipeline_llm_skipped_when_disabled(tmp_path):
    cfg = _minimal_config(tmp_path)
    cfg.llm.enabled = False
    pipeline = Pipeline(cfg, run_id="test_no_llm")

    summaries, tags = pipeline._stage_llm(MagicMock())
    assert summaries == []
    assert tags == []


def test_pipeline_artifacts_written(tmp_path):
    """Verify run_summary.json and config.json are written."""
    cfg = _minimal_config(tmp_path)
    pipeline = Pipeline(cfg, run_id="test_artifacts")

    mock_docs = [MagicMock(text="long enough text here") for _ in range(3)]
    mock_model = MagicMock()
    mock_model.get_topic_count.return_value = 2
    mock_model.get_topics.return_value = {0: [("w", 0.9)], 1: [("x", 0.8)]}
    mock_model.get_representative_docs.return_value = []

    with (
        patch.object(pipeline, "_stage_data", return_value=mock_docs),
        patch.object(
            pipeline,
            "_stage_preprocess",
            return_value=(mock_docs, {"total_input": 3, "total_output": 3, "dropped": {}, "avg_length": 21.0}),
        ),
        patch.object(pipeline, "_stage_model", return_value=(mock_model, [0, 1, 0])),
        patch.object(pipeline, "_stage_evaluate", return_value={"diversity": 1.0}),
        patch.object(pipeline, "_stage_llm", return_value=([], [])),
        patch.object(pipeline, "_stage_report"),
    ):
        pipeline.run()

    run_dir = pipeline.run_dir
    assert (run_dir / "run_summary.json").exists()
    assert (run_dir / "config.json").exists()


# --- composite scoring ---

def test_composite_score_all_weights():
    metrics = {"coherence": 0.6, "diversity": 0.8, "outlier_ratio": 0.1}
    weights = {"coherence": 1.0, "diversity": 1.0, "outlier_ratio": 1.0}
    hib = {"coherence": True, "diversity": True, "outlier_ratio": False}
    bounds = {
        "coherence": [0.0, 1.0],
        "diversity": [0.0, 1.0],
        "outlier_ratio": [0.0, 1.0],
    }
    score = composite_score(metrics, weights, hib, bounds)
    # coherence=0.6 norm, diversity=0.8 norm, outlier_ratio=1-0.1=0.9 norm
    # avg = (0.6 + 0.8 + 0.9) / 3
    assert score == pytest.approx((0.6 + 0.8 + 0.9) / 3.0)


def test_composite_score_no_bounds_neutral():
    metrics = {"coherence": 0.5}
    weights = {"coherence": 1.0}
    hib = {"coherence": True}
    score = composite_score(metrics, weights, hib, bounds=None)
    assert score == pytest.approx(0.5)


def test_composite_score_missing_metric_skipped():
    metrics = {"coherence": 0.7}
    weights = {"coherence": 1.0, "silhouette": 2.0}
    hib = {"coherence": True, "silhouette": True}
    bounds = {"coherence": [0.0, 1.0]}
    score = composite_score(metrics, weights, hib, bounds)
    # only coherence contributes; silhouette missing
    assert score == pytest.approx(0.7)


def test_composite_score_empty_weights():
    score = composite_score({"coherence": 0.5}, {}, {})
    assert score == 0.0


def test_estimate_bounds_basic():
    history = [
        {"coherence": 0.3, "diversity": 0.8},
        {"coherence": 0.7, "diversity": 0.6},
        {"coherence": 0.5, "diversity": 0.9},
    ]
    bounds = estimate_bounds(history, ["coherence", "diversity"])
    assert bounds["coherence"] == [0.3, 0.7]
    assert bounds["diversity"] == [0.6, 0.9]


def test_estimate_bounds_single_trial_excluded():
    # Only 1 value → no bounds entry
    history = [{"coherence": 0.5}]
    bounds = estimate_bounds(history, ["coherence"])
    assert "coherence" not in bounds


def test_estimate_bounds_skips_nan():
    import math
    history = [
        {"coherence": 0.3},
        {"coherence": float("nan")},
        {"coherence": 0.7},
    ]
    bounds = estimate_bounds(history, ["coherence"])
    assert bounds["coherence"] == [0.3, 0.7]


def test_tuning_config_composite_fields():
    cfg = TuningConfig(
        metric="composite",
        metric_weights={"coherence": 2.0, "diversity": 1.0},
        higher_is_better={"coherence": True, "diversity": True},
    )
    assert cfg.metric == "composite"
    assert cfg.metric_weights["coherence"] == 2.0
    assert cfg.higher_is_better["diversity"] is True

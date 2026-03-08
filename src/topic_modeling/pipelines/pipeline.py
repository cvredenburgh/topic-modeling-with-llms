"""End-to-end topic modeling pipeline."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from topic_modeling.config.schema import ExperimentConfig
from topic_modeling.data.loader import load_dataset, load_canonical, persist_canonical
from topic_modeling.data.schema import Document
from topic_modeling.evaluation.metrics import evaluate
from topic_modeling.llm.summarizer import summarize_topics
from topic_modeling.llm.tagger import tag_topics
from topic_modeling.models import build_model
from topic_modeling.models.base import TopicModelBase
from topic_modeling.preprocessing.cleaner import preprocess
from topic_modeling.reporting.reporter import generate_report
from topic_modeling.tuning.tuner import tune
from topic_modeling.utils.artifacts import ArtifactStore
from topic_modeling.utils.seeds import set_seeds

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrates all pipeline stages from data ingestion to reporting."""

    def __init__(self, config: ExperimentConfig, run_id: Optional[str] = None):
        self.config = config
        self.run_id = run_id or _make_run_id(config.run.name)
        self.run_dir = Path(config.run.output_dir) / self.run_id
        self.store = ArtifactStore(
            run_dir=self.run_dir,
            target=config.runtime.target,
            gcp=config.runtime.model_dump() if config.runtime.target == "gcp" else {},
        )

    def run(self) -> Dict[str, Any]:
        cfg = self.config
        set_seeds(cfg.run.seed)

        logger.info(f"=== Pipeline start: {self.run_id} ===")
        self.store.save_json(cfg.model_dump(), "config.json")

        # Stage 1: Data ingestion
        docs = self._stage_data()

        # Stage 2: Preprocessing
        docs, pre_stats = self._stage_preprocess(docs)

        texts = [d.text for d in docs]

        # Stage 3: Modeling (or tuning then modeling)
        model, topic_ids = self._stage_model(texts)

        # Stage 4: Evaluation
        metrics = self._stage_evaluate(model, texts, topic_ids)

        # Stage 5: LLM post-processing
        summaries, tags = self._stage_llm(model)

        # Stage 6: Reporting
        self._stage_report(model, topic_ids, metrics, summaries, tags, pre_stats)

        # Save final run summary
        summary = {
            "run_id": self.run_id,
            "backend": cfg.model.backend,
            "topic_count": model.get_topic_count(),
            "metrics": metrics,
            "run_dir": str(self.run_dir),
        }
        self.store.save_json(summary, "run_summary.json")
        logger.info(f"=== Pipeline complete: {self.run_id} ===")
        logger.info(f"Artifacts: {self.run_dir}")
        return summary

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    def _stage_data(self) -> List[Document]:
        logger.info("--- Stage: Data ingestion ---")
        interim_path = Path("data/interim") / f"{self.config.run.name}_canonical.parquet"

        if interim_path.exists():
            logger.info(f"Loading cached canonical dataset: {interim_path}")
            return load_canonical(interim_path)

        docs = load_dataset(self.config.dataset, seed=self.config.run.seed)
        interim_path.parent.mkdir(parents=True, exist_ok=True)
        persist_canonical(docs, interim_path)
        return docs

    def _stage_preprocess(
        self, docs: List[Document]
    ) -> Tuple[List[Document], dict]:
        logger.info("--- Stage: Preprocessing ---")
        docs, stats = preprocess(docs, self.config.preprocessing)
        self.store.save_json(stats, "preprocessing_stats.json")
        return docs, stats

    def _stage_model(
        self, texts: List[str]
    ) -> Tuple[TopicModelBase, List[int]]:
        cfg = self.config

        if cfg.tuning.enabled:
            logger.info("--- Stage: Hyperparameter tuning ---")
            model_cfg, trials = self._run_tuning(texts)
            self.store.save_json(
                {"best_params": model_cfg.params, "trials": trials},
                "tuning_results.json",
            )
        else:
            model_cfg = cfg.model

        logger.info("--- Stage: Model fitting ---")
        model = build_model(model_cfg, seed=cfg.run.seed)
        model.fit(texts)

        if cfg.reporting.save_model:
            model.save(self.run_dir / "artifacts" / "model")

        topic_ids, _ = model.transform(texts)
        return model, topic_ids

    def _run_tuning(self, texts: List[str]):
        from topic_modeling.evaluation.metrics import evaluate as eval_fn
        from topic_modeling.config.schema import ModelConfig

        cfg = self.config

        def score_fn(model_cfg: ModelConfig, texts: List[str]) -> float:
            m = build_model(model_cfg, seed=cfg.run.seed)
            m.fit(texts)
            t_ids, _ = m.transform(texts)
            metrics = eval_fn(m, texts, t_ids, cfg.evaluation)
            return float(metrics.get(cfg.tuning.metric, 0) or 0)

        best_params, trials = tune(
            texts=texts,
            base_model_config=cfg.model,
            tuning_config=cfg.tuning,
            evaluate_fn=score_fn,
            seed=cfg.run.seed,
        )
        best_config = ModelConfig(backend=cfg.model.backend, params=best_params)
        return best_config, trials

    def _stage_evaluate(
        self,
        model: TopicModelBase,
        texts: List[str],
        topic_ids: List[int],
    ) -> Dict[str, Any]:
        logger.info("--- Stage: Evaluation ---")
        metrics = evaluate(model, texts, topic_ids, self.config.evaluation)
        self.store.save_json(metrics, "metrics.json")
        return metrics

    def _stage_llm(
        self, model: TopicModelBase
    ) -> Tuple[List[Dict], List[Dict]]:
        if not self.config.llm.enabled:
            logger.info("LLM post-processing disabled — skipping")
            return [], []

        logger.info("--- Stage: LLM summarization ---")
        summaries = summarize_topics(
            model,
            self.config.llm,
            top_n_terms=self.config.reporting.top_n_terms,
            top_n_docs=self.config.reporting.top_n_docs,
        )

        logger.info("--- Stage: LLM tagging ---")
        domain = self.config.dataset.name or "text corpus"
        tags = tag_topics(
            model,
            self.config.llm,
            domain_context=domain,
            top_n_terms=self.config.reporting.top_n_terms,
            top_n_docs=self.config.reporting.top_n_docs,
        )
        return summaries, tags

    def _stage_report(
        self,
        model: TopicModelBase,
        topic_ids: List[int],
        metrics: Dict[str, Any],
        summaries: List[Dict],
        tags: List[Dict],
        pre_stats: Dict[str, Any],
    ) -> None:
        logger.info("--- Stage: Reporting ---")
        generate_report(
            config=self.config,
            store=self.store,
            model=model,
            topic_ids=topic_ids,
            metrics=metrics,
            summaries=summaries,
            tags=tags,
            preprocessing_stats=pre_stats,
        )


def _make_run_id(name: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{name}_{ts}"

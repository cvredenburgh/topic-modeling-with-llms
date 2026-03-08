"""Hyperparameter tuning: Optuna, grid, and random search."""
from __future__ import annotations

import itertools
import logging
import random
from typing import Any, Callable, Dict, List, Tuple

from topic_modeling.config.schema import ModelConfig, TuningConfig

logger = logging.getLogger(__name__)

TrialResult = Dict[str, Any]  # {"params": {...}, "score": float}


def tune(
    texts: List[str],
    base_model_config: ModelConfig,
    tuning_config: TuningConfig,
    evaluate_fn: Callable[[ModelConfig, List[str]], float],
    seed: int = 42,
) -> Tuple[Dict[str, Any], List[TrialResult]]:
    """Run hyperparameter search.

    Args:
        texts:             Training corpus.
        base_model_config: Default model config (params are starting point).
        tuning_config:     Tuning settings (method, n_trials, search_space).
        evaluate_fn:       Callable(config, texts) -> scalar score to maximise.
        seed:              Random seed.

    Returns:
        (best_params_dict, all_trial_results)
    """
    method = tuning_config.method
    logger.info(f"Starting {method} tuning ({tuning_config.n_trials} trials)")

    if method == "optuna":
        return _tune_optuna(texts, base_model_config, tuning_config, evaluate_fn, seed)
    if method in ("grid", "random"):
        return _tune_exhaustive(texts, base_model_config, tuning_config, evaluate_fn, seed)
    raise ValueError(f"Unknown tuning method: {method!r}")


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def _tune_optuna(
    texts: List[str],
    base_config: ModelConfig,
    cfg: TuningConfig,
    evaluate_fn: Callable,
    seed: int,
) -> Tuple[Dict[str, Any], List[TrialResult]]:
    import optuna  # type: ignore

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    trials_log: List[TrialResult] = []
    search_space = cfg.search_space

    def objective(trial: Any) -> float:
        params = dict(base_config.params)
        for name, space in search_space.items():
            kind = space.get("type", "categorical")
            if kind == "int":
                params[name] = trial.suggest_int(name, space["low"], space["high"])
            elif kind == "float":
                params[name] = trial.suggest_float(
                    name, space["low"], space["high"], log=space.get("log", False)
                )
            elif kind == "categorical":
                params[name] = trial.suggest_categorical(name, space["choices"])
            else:
                raise ValueError(f"Unknown search space type: {kind!r}")

        trial_config = ModelConfig(backend=base_config.backend, params=params)
        try:
            score = evaluate_fn(trial_config, texts)
        except Exception as exc:
            logger.warning(f"Trial failed: {exc}")
            score = float("-inf")

        trials_log.append({"params": params, "score": score})
        logger.info(f"Trial {len(trials_log)}: score={score:.4f}  params={params}")
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(
        objective,
        n_trials=cfg.n_trials,
        timeout=cfg.timeout_seconds,
    )

    best_params = dict(base_config.params)
    best_params.update(study.best_params)
    logger.info(f"Best score={study.best_value:.4f}  best_params={best_params}")
    return best_params, trials_log


# ---------------------------------------------------------------------------
# Grid / Random
# ---------------------------------------------------------------------------

def _tune_exhaustive(
    texts: List[str],
    base_config: ModelConfig,
    cfg: TuningConfig,
    evaluate_fn: Callable,
    seed: int,
) -> Tuple[Dict[str, Any], List[TrialResult]]:
    search_space = cfg.search_space
    param_names = list(search_space.keys())
    param_choices = [search_space[k].get("choices", []) for k in param_names]
    all_combos = list(itertools.product(*param_choices))

    if cfg.method == "random":
        rng = random.Random(seed)
        rng.shuffle(all_combos)
        all_combos = all_combos[: cfg.n_trials]

    best_score = float("-inf")
    best_params = dict(base_config.params)
    trials_log: List[TrialResult] = []

    for combo in all_combos:
        params = dict(base_config.params)
        params.update(dict(zip(param_names, combo)))
        trial_config = ModelConfig(backend=base_config.backend, params=params)

        try:
            score = evaluate_fn(trial_config, texts)
        except Exception as exc:
            logger.warning(f"Trial failed: {exc}")
            score = float("-inf")

        trials_log.append({"params": params, "score": score})
        logger.info(f"Trial {len(trials_log)}: score={score:.4f}  params={params}")

        if score > best_score:
            best_score = score
            best_params = params

    logger.info(f"Best score={best_score:.4f}  best_params={best_params}")
    return best_params, trials_log

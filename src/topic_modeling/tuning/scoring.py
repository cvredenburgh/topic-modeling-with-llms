"""Composite weighted scoring for hyperparameter tuning trials."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np


def composite_score(
    metrics: Dict[str, Any],
    weights: Dict[str, float],
    higher_is_better: Dict[str, bool],
    bounds: Optional[Dict[str, List[float]]] = None,
) -> float:
    """Compute a weighted composite score from a metrics dict.

    Steps:
        1. Min-max normalize each metric to [0, 1] using ``bounds``.
           If bounds are unavailable for a metric, use 0.5 (neutral).
        2. Invert lower-is-better metrics: score = 1 - normalized.
        3. Return the weighted average across all available metrics.

    Args:
        metrics:          Dict of metric name -> scalar value.
        weights:          Per-metric weights (positive floats).
        higher_is_better: True when a higher raw value is better.
        bounds:           Optional {metric: [min, max]} for normalization.

    Returns:
        Composite score in [0, 1].
    """
    if not weights:
        return 0.0

    total_weight = 0.0
    total_score = 0.0

    for name, weight in weights.items():
        value = metrics.get(name)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isnan(value):
            continue

        # Min-max normalize
        if bounds and name in bounds:
            lo, hi = bounds[name]
            normalized = (value - lo) / (hi - lo) if hi > lo else 0.5
        else:
            normalized = 0.5

        normalized = max(0.0, min(1.0, normalized))

        # Invert if lower is better
        if not higher_is_better.get(name, True):
            normalized = 1.0 - normalized

        total_score += weight * normalized
        total_weight += weight

    return total_score / total_weight if total_weight > 0 else 0.0


def estimate_bounds(
    trial_metrics_list: List[Dict[str, Any]],
    metric_names: Iterable[str],
) -> Dict[str, List[float]]:
    """Estimate [min, max] bounds per metric from accumulated trial history.

    Args:
        trial_metrics_list: List of metrics dicts from prior trials.
        metric_names:       Metric names to compute bounds for.

    Returns:
        Dict mapping metric name to [min_val, max_val].
        Only includes metrics where at least 2 finite values exist.
    """
    bounds: Dict[str, List[float]] = {}
    for name in metric_names:
        values = []
        for m in trial_metrics_list:
            v = m.get(name)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if not np.isnan(fv):
                values.append(fv)
        if len(values) >= 2:
            bounds[name] = [min(values), max(values)]
    return bounds

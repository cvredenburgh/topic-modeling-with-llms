"""Visualization for topic modeling tuning results."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def generate_figures(
    trials: List[Dict[str, Any]],
    metric_cols: List[str],
    param_metric_pairs: List[List[str]],
    output_dir: Path,
    top_n: int = 5,
) -> None:
    """Generate all configured figures from tuning trial data.

    Produces:
        - radar_chart.png     : top-N trials compared across metrics
        - metric_heatmap.png  : cross-metric correlation heatmap
        - <param>_vs_<metric>.png : param-metric scatter/box plots

    Args:
        trials:              List of trial dicts with ``params`` and ``score`` keys.
        metric_cols:         Metric columns to include in radar/heatmap.
        param_metric_pairs:  [[param, metric], ...] pairs for param-metric plots.
        output_dir:          Directory to write PNG files.
        top_n:               Number of top trials to show in radar chart.
    """
    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
        import seaborn as sns  # type: ignore
    except ImportError as exc:
        logger.warning(f"Visualization dependencies unavailable ({exc}); skipping figures")
        return

    if not trials:
        logger.warning("No trial data available for figures")
        return

    df = _trials_to_df(trials)
    if df.empty:
        return

    os.makedirs(output_dir, exist_ok=True)

    available_metrics = [c for c in metric_cols if c in df.columns]
    if available_metrics:
        _radar_chart(df, available_metrics, top_n, output_dir, plt, np)
        if len(df) >= 3:
            _metric_heatmap(df, available_metrics, output_dir, plt, sns)

    for pair in param_metric_pairs:
        if len(pair) == 2:
            param_col, metric_col = pair[0], pair[1]
            if param_col in df.columns and metric_col in df.columns:
                _param_metric_plot(df, param_col, metric_col, output_dir, plt)

    plt.close("all")
    logger.info(f"Figures saved to {output_dir}")


def _trials_to_df(trials: List[Dict[str, Any]]) -> Any:
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        return None

    rows = []
    for t in trials:
        row = dict(t.get("params", {}))
        row["score"] = t.get("score")
        rows.append(row)
    return pd.DataFrame(rows)


def _radar_chart(df, metric_cols, top_n, output_dir, plt, np) -> None:
    cols = [c for c in metric_cols if c in df.columns]
    if len(cols) < 3:
        return

    top_df = (
        df.nlargest(min(top_n, len(df)), "score")
        if "score" in df.columns
        else df.head(top_n)
    )

    # Normalize each metric column to [0, 1]
    normed = top_df[cols].copy()
    for c in cols:
        mn, mx = normed[c].min(), normed[c].max()
        normed[c] = (normed[c] - mn) / (mx - mn) if mx > mn else 0.5

    N = len(cols)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for i, (_, row) in enumerate(normed.iterrows()):
        values = row[cols].tolist() + [row[cols[0]]]
        ax.plot(angles, values, label=f"Trial {i + 1}")
        ax.fill(angles, values, alpha=0.1)

    ax.set_thetagrids(np.degrees(angles[:-1]), cols)
    ax.set_title("Top Trials — Metric Radar Chart")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    plt.tight_layout()
    plt.savefig(str(output_dir / "radar_chart.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)


def _metric_heatmap(df, metric_cols, output_dir, plt, sns) -> None:
    cols = [c for c in metric_cols if c in df.columns]
    if len(cols) < 2:
        return

    corr = df[cols].corr()
    size = max(6, len(cols))
    fig, ax = plt.subplots(figsize=(size, size - 1))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, vmin=-1, vmax=1)
    ax.set_title("Metric Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(str(output_dir / "metric_heatmap.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)


def _param_metric_plot(df, param_col, metric_col, output_dir, plt) -> None:
    plot_df = df[[param_col, metric_col]].dropna()
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    if plot_df[param_col].dtype == object or plot_df[param_col].nunique() <= 10:
        plot_df.boxplot(column=metric_col, by=param_col, ax=ax)
        ax.set_title(f"{metric_col} by {param_col}")
        plt.suptitle("")
    else:
        ax.scatter(plot_df[param_col], plot_df[metric_col], alpha=0.6)
        ax.set_xlabel(param_col)
        ax.set_ylabel(metric_col)
        ax.set_title(f"{metric_col} vs {param_col}")

    plt.tight_layout()
    safe_name = f"{param_col}_vs_{metric_col}".replace("/", "_").replace(" ", "_")
    plt.savefig(str(output_dir / f"{safe_name}.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)

#!/usr/bin/env python
"""Single CLI entrypoint for the topic modeling pipeline.

Usage examples:
    python scripts/run_pipeline.py --config configs/experiment/baseline_bertopic.yaml
    python scripts/run_pipeline.py --config configs/experiment/tune_fastopic.yaml --set run.seed=123
    python scripts/run_pipeline.py --config configs/experiment/baseline_bertopic.yaml --set runtime.target=gcp
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src/ is on the path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topic_modeling.config.loader import load_config
from topic_modeling.pipelines.pipeline import Pipeline
from topic_modeling.utils.logging_utils import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the topic modeling pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to experiment YAML config.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, e.g. --set run.seed=42  (repeatable).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        metavar="ID",
        help="Optional explicit run ID (default: auto-generated from name + timestamp).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config, overrides=args.overrides)
    configure_logging(config.run.log_level)

    pipeline = Pipeline(config, run_id=args.run_id)
    summary = pipeline.run()

    print("\n" + "=" * 60)
    print("Run complete.")
    print(json.dumps(summary, indent=2, default=str))
    print("=" * 60)


if __name__ == "__main__":
    main()

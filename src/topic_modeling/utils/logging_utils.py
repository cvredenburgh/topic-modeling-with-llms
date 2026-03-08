"""Logging setup for the pipeline."""
import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    # Quiet noisy libraries
    for noisy in ("transformers", "datasets", "huggingface_hub", "torch", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

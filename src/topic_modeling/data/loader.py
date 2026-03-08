"""Data ingestion from HuggingFace or local files."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd

from topic_modeling.config.schema import DatasetConfig
from topic_modeling.data.schema import Document

logger = logging.getLogger(__name__)


def load_dataset(config: DatasetConfig, seed: int = 42) -> List[Document]:
    """Load and return a list of canonical Documents."""
    if config.source == "huggingface":
        return _load_huggingface(config, seed)
    if config.source == "local":
        return _load_local(config, seed)
    raise ValueError(f"Unknown data source: {config.source!r}")


def _load_huggingface(config: DatasetConfig, seed: int) -> List[Document]:
    from datasets import load_dataset as hf_load  # type: ignore

    logger.info(
        f"Loading HuggingFace dataset: {config.name!r}  "
        f"subset={config.subset!r}  split={config.split!r}"
    )
    kwargs: dict = {}
    if config.subset:
        kwargs["name"] = config.subset
    ds = hf_load(config.name, split=config.split, **kwargs)
    df = ds.to_pandas()
    return _finalize(df, config, seed)


def _load_local(config: DatasetConfig, seed: int) -> List[Document]:
    path = Path(config.local_path)
    logger.info(f"Loading local file: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif suffix == ".json":
        df = pd.read_json(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported local file type: {suffix}")
    return _finalize(df, config, seed)


def _finalize(df: pd.DataFrame, config: DatasetConfig, seed: int) -> List[Document]:
    if config.sample_n and config.sample_n < len(df):
        df = df.sample(n=config.sample_n, random_state=seed)
    elif config.sample_frac:
        df = df.sample(frac=config.sample_frac, random_state=seed)

    df = df.reset_index(drop=True)

    docs: List[Document] = []
    for i, row in df.iterrows():
        text = str(row.get(config.text_column, "") or "")
        if config.id_column and config.id_column in row:
            doc_id = str(row[config.id_column])
        else:
            doc_id = str(i)
        metadata = {
            col: row[col]
            for col in config.metadata_columns
            if col in row.index
        }
        docs.append(Document(doc_id=doc_id, text=text, metadata=metadata))

    logger.info(f"Loaded {len(docs)} documents")
    return docs


def persist_canonical(docs: List[Document], path: Path) -> None:
    """Save canonical documents to parquet for reuse."""
    records = [
        {"doc_id": d.doc_id, "text": d.text, **d.metadata} for d in docs
    ]
    pd.DataFrame(records).to_parquet(path, index=False)
    logger.info(f"Canonical dataset saved: {path}")


def load_canonical(path: Path) -> List[Document]:
    df = pd.read_parquet(path)
    docs = []
    for _, row in df.iterrows():
        doc_id = str(row["doc_id"])
        text = str(row["text"])
        meta = {k: v for k, v in row.items() if k not in ("doc_id", "text")}
        docs.append(Document(doc_id=doc_id, text=text, metadata=meta))
    return docs

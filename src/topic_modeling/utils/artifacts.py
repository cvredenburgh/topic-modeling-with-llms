"""Artifact persistence — local and GCP."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ArtifactStore:
    """Write run artifacts locally; optionally mirror to GCS."""

    def __init__(self, run_dir: Path, target: str = "local", gcp: dict | None = None):
        self.run_dir = run_dir
        self.target = target
        self.gcp = gcp or {}
        run_dir.mkdir(parents=True, exist_ok=True)

    def save_json(self, data: Any, name: str) -> Path:
        path = self.run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Artifact saved: {path}")
        if self.target == "gcp":
            self._upload(path, name)
        return path

    def save_jsonl(self, records: list[dict], name: str) -> Path:
        path = self.run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec, default=str) + "\n")
        logger.info(f"Artifact saved: {path}")
        if self.target == "gcp":
            self._upload(path, name)
        return path

    def save_text(self, text: str, name: str) -> Path:
        path = self.run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        logger.info(f"Artifact saved: {path}")
        if self.target == "gcp":
            self._upload(path, name)
        return path

    def save_csv(self, df: Any, name: str) -> Path:
        path = self.run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Artifact saved: {path}")
        if self.target == "gcp":
            self._upload(path, name)
        return path

    def _upload(self, local_path: Path, name: str) -> None:
        try:
            from google.cloud import storage  # type: ignore

            bucket_name = self.gcp.get("gcp_bucket")
            project = self.gcp.get("gcp_project")
            prefix = f"runs/{self.run_dir.name}"
            client = storage.Client(project=project)
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(f"{prefix}/{name}")
            blob.upload_from_filename(str(local_path))
            logger.info(f"Uploaded to gs://{bucket_name}/{prefix}/{name}")
        except Exception as exc:
            logger.warning(f"GCS upload failed ({name}): {exc}")

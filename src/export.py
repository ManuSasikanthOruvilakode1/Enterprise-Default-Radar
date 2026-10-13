"""Persistence for the trained risk model and its metadata."""

import json
import os
from typing import Any

import joblib


class ModelPersistence:
    """Saves and loads model artifacts. Only load artifacts you created yourself: joblib files can run code."""

    @staticmethod
    def save_pipeline(pipeline_obj: Any, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        joblib.dump(pipeline_obj, filepath)
        metadata = getattr(pipeline_obj, "metadata", None)
        if metadata is not None:
            with open(os.path.splitext(filepath)[0] + "_metadata.json", "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2, default=str)

    @staticmethod
    def load_pipeline(filepath: str) -> Any:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model artifact not found at {filepath}. Run `uv run python scripts/run_pipeline.py` first.")
        return joblib.load(filepath)

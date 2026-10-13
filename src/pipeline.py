"""Deployable risk model: feature pipeline + estimator, probability calibrator and decision threshold."""

from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .calibrate import ProbabilityCalibrator
from .schema import INPUT_COLUMNS, NUMERIC_COLUMNS, BusinessProfile

FLAGGED_LABEL = "Elevated risk - flag for review"
NOT_FLAGGED_LABEL = "Lower risk - no review flagged"


class RiskModel:
    """Scores raw business records with the fitted pipeline, calibration and cost-based threshold."""

    def __init__(
        self,
        model_pipeline: Pipeline,
        calibrator: ProbabilityCalibrator,
        threshold: float,
        training_ranges: Dict[str, List[float]],
        background_sample: pd.DataFrame,
        metadata: Dict[str, Any],
    ):
        self.model_pipeline = model_pipeline
        self.calibrator = calibrator
        self.threshold = float(threshold)
        self.training_ranges = training_ranges
        self.background_sample = background_sample
        self.metadata = metadata

    def raw_probability(self, df_raw: pd.DataFrame) -> np.ndarray:
        return self.model_pipeline.predict_proba(df_raw[INPUT_COLUMNS])[:, 1]

    def predict_proba(self, df_raw: pd.DataFrame) -> np.ndarray:
        """Calibrated insolvency probabilities."""
        return self.calibrator.transform(self.raw_probability(df_raw))

    def predict(self, df_raw: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(df_raw) >= self.threshold).astype(int)

    def range_warnings(self, profile: BusinessProfile) -> List[str]:
        """Valid inputs that fall outside the training data range (the model is extrapolating there)."""
        warnings = []
        for col in NUMERIC_COLUMNS:
            low, high = self.training_ranges[col]
            value = getattr(profile, col)
            if value < low or value > high:
                warnings.append(f"{col} = {value:,} is outside the training range [{low:,}, {high:,}]")
        return warnings

    def score_profile(self, profile: Union[BusinessProfile, Dict[str, Any]]) -> Dict[str, Any]:
        """Validates and scores one business record."""
        if not isinstance(profile, BusinessProfile):
            profile = BusinessProfile(**profile)
        frame = pd.DataFrame([profile.model_dump()])
        probability = float(self.predict_proba(frame)[0])
        flagged = probability >= self.threshold
        return {
            "probability": probability,
            "raw_model_score": float(self.raw_probability(frame)[0]),
            "threshold": self.threshold,
            "flagged": flagged,
            "category": FLAGGED_LABEL if flagged else NOT_FLAGGED_LABEL,
            "warnings": self.range_warnings(profile),
        }

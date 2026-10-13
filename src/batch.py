"""Batch scoring: validate and score many business records at once.

Invalid rows are not dropped or guessed; they are returned with an explanation and no score.
"""

import numpy as np
import pandas as pd
from pydantic import ValidationError

from .pipeline import FLAGGED_LABEL, NOT_FLAGGED_LABEL, RiskModel
from .schema import INPUT_COLUMNS, BusinessProfile

OUTPUT_COLUMNS = ["insolvency_probability", "flagged", "risk_category", "warnings", "validation_error"]


def _describe(error: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(p) for p in item['loc']) or 'record'}: {item['msg']}" for item in error.errors())


def score_frame(risk_model: RiskModel, df: pd.DataFrame) -> pd.DataFrame:
    """Returns `df` with probability, decision, warnings and validation error columns appended."""
    missing = [col for col in INPUT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    clashing = [col for col in OUTPUT_COLUMNS if col in df.columns]
    if clashing:
        raise ValueError(f"Input already contains output columns: {clashing}")

    records = df[INPUT_COLUMNS].astype(object).where(df[INPUT_COLUMNS].notna(), None).to_dict(orient="records")
    profiles, errors = [], []
    for record in records:
        try:
            profiles.append(BusinessProfile(**record))
            errors.append("")
        except ValidationError as error:
            profiles.append(None)
            errors.append(_describe(error))

    valid = [i for i, p in enumerate(profiles) if p is not None]
    probabilities = np.full(len(df), np.nan)
    if valid:
        frame = pd.DataFrame([profiles[i].model_dump() for i in valid])
        probabilities[valid] = risk_model.predict_proba(frame)

    out = df.copy()
    out["insolvency_probability"] = probabilities
    out["flagged"] = [None if p is None else bool(prob >= risk_model.threshold) for p, prob in zip(profiles, probabilities, strict=True)]
    out["risk_category"] = [
        "" if p is None else (FLAGGED_LABEL if prob >= risk_model.threshold else NOT_FLAGGED_LABEL)
        for p, prob in zip(profiles, probabilities, strict=True)
    ]
    out["warnings"] = ["" if p is None else "; ".join(risk_model.range_warnings(p)) for p in profiles]
    out["validation_error"] = errors
    return out

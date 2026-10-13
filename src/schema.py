"""Shared column definitions and validated input schema for a single business record."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ID_COLUMN = "business_id"
TARGET_COLUMN = "insolvency_risk"

STATES = ("NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT")
INDUSTRIES = (
    "Construction",
    "Retail Trade",
    "Manufacturing",
    "Professional Services",
    "Mining",
    "Accommodation & Food",
    "Health Care",
)

CATEGORICAL_COLUMNS = ["state", "industry"]
NUMERIC_COLUMNS = [
    "employee_count",
    "annual_turnover_aud",
    "years_operating",
    "r_and_d_expenditure_aud",
    "debt_to_equity_ratio",
    "current_ratio",
    "profit_margin",
]
INPUT_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


class BusinessProfile(BaseModel):
    """One business record as accepted by the scoring model and the demo app.

    Bounds reject values that are impossible or clearly malformed. Values that are
    valid but outside the training data range are reported as warnings at scoring time.
    """

    model_config = ConfigDict(extra="forbid")

    state: Literal[STATES]
    industry: Literal[INDUSTRIES]
    employee_count: int = Field(ge=1, le=100_000)
    annual_turnover_aud: float = Field(gt=0, le=1e11)
    years_operating: int = Field(ge=0, le=200)
    r_and_d_expenditure_aud: float = Field(ge=0, le=1e11)
    debt_to_equity_ratio: float = Field(ge=0, le=100)
    current_ratio: float = Field(gt=0, le=100)
    profit_margin: float = Field(ge=-1, le=1)

    @model_validator(mode="after")
    def check_r_and_d_not_above_turnover(self) -> "BusinessProfile":
        if self.r_and_d_expenditure_aud > self.annual_turnover_aud:
            raise ValueError("r_and_d_expenditure_aud cannot exceed annual_turnover_aud")
        return self

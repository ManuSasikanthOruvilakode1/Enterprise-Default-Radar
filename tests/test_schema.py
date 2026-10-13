"""Input validation tests for single business records (used by the app)."""

import pytest
from pydantic import ValidationError

from src.schema import BusinessProfile


def test_valid_profile_accepted(valid_profile):
    assert BusinessProfile(**valid_profile).state == "NSW"


@pytest.mark.parametrize("field,value", [
    ("state", "XYZ"),
    ("industry", "Space Travel"),
    ("employee_count", 0),
    ("annual_turnover_aud", -5.0),
    ("years_operating", -1),
    ("debt_to_equity_ratio", -0.1),
    ("current_ratio", 0.0),
    ("profit_margin", 1.5),
    ("employee_count", "twelve"),
    ("current_ratio", None),
])
def test_invalid_values_rejected(valid_profile, field, value):
    with pytest.raises(ValidationError):
        BusinessProfile(**{**valid_profile, field: value})


@pytest.mark.parametrize("field", ["state", "current_ratio", "profit_margin"])
def test_missing_fields_rejected(valid_profile, field):
    profile = dict(valid_profile)
    del profile[field]
    with pytest.raises(ValidationError):
        BusinessProfile(**profile)


def test_r_and_d_above_turnover_rejected(valid_profile):
    with pytest.raises(ValidationError, match="cannot exceed"):
        BusinessProfile(**{**valid_profile, "r_and_d_expenditure_aud": 2_000_000.0})


def test_unexpected_fields_rejected(valid_profile):
    with pytest.raises(ValidationError):
        BusinessProfile(**valid_profile, owner_name="someone")

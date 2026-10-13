"""Synthetic business outcomes generator.

The data is fully synthetic. Marginal distributions are loosely shaped after public
economic statistics (state and industry mix, firm size, financial ratios), and the
target is drawn from a known logistic model. No real company records are used.
"""

import numpy as np
import pandas as pd

from .schema import INDUSTRIES, STATES

STATE_PROBABILITIES = [0.32, 0.26, 0.20, 0.11, 0.06, 0.02, 0.02, 0.01]
INDUSTRY_PROBABILITIES = [0.20, 0.18, 0.15, 0.22, 0.05, 0.12, 0.08]


def true_insolvency_logit(
    debt_to_equity: np.ndarray,
    current_ratio: np.ndarray,
    profit_margin: np.ndarray,
    years_operating: np.ndarray,
    industry: np.ndarray,
) -> np.ndarray:
    """Ground-truth log-odds used to draw the target (roughly a 6% positive rate).

    State, employee count, turnover and R&D spend deliberately have no effect.
    """
    return (
        -2.8
        + 0.65 * (debt_to_equity - 1.5)
        - 0.90 * (current_ratio - 1.5)
        - 2.50 * profit_margin
        - 0.03 * years_operating
        + np.where(industry == "Construction", 0.45, 0.0)
        + np.where(industry == "Accommodation & Food", 0.35, 0.0)
    )


def generate_business_dataset(n_samples: int = 2500, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    state_col = np.random.choice(STATES, size=n_samples, p=STATE_PROBABILITIES)
    industry_col = np.random.choice(INDUSTRIES, size=n_samples, p=INDUSTRY_PROBABILITIES)

    employee_count = np.random.lognormal(mean=2.5, sigma=1.0, size=n_samples).astype(int) + 1
    employee_count = np.clip(employee_count, 1, 350)

    turnover = employee_count * np.random.uniform(75000, 220000, size=n_samples) + np.random.normal(0, 20000, size=n_samples)
    turnover = np.clip(turnover, 80000, 50000000)

    years_operating = np.random.exponential(scale=8.0, size=n_samples).astype(int) + 1
    years_operating = np.clip(years_operating, 1, 45)

    r_and_d = np.where(
        np.random.rand(n_samples) > 0.4,
        turnover * np.random.uniform(0.01, 0.12, size=n_samples),
        0.0,
    )

    debt_to_equity = np.random.gamma(shape=2.0, scale=0.8, size=n_samples)
    debt_to_equity = np.clip(debt_to_equity, 0.1, 6.0)

    current_ratio = np.random.normal(loc=1.6, scale=0.6, size=n_samples)
    current_ratio = np.clip(current_ratio, 0.2, 4.5)

    profit_margin = np.random.normal(loc=0.08, scale=0.10, size=n_samples)
    profit_margin = np.clip(profit_margin, -0.40, 0.45)

    z = true_insolvency_logit(debt_to_equity, current_ratio, profit_margin, years_operating, industry_col)
    p_insolvency = 1.0 / (1.0 + np.exp(-z))
    insolvency_risk = (np.random.rand(n_samples) < p_insolvency).astype(int)

    return pd.DataFrame({
        "business_id": [f"BIZ_{i:06d}" for i in range(1, n_samples + 1)],
        "state": state_col,
        "industry": industry_col,
        "employee_count": employee_count,
        "annual_turnover_aud": np.round(turnover, 2),
        "years_operating": years_operating,
        "r_and_d_expenditure_aud": np.round(r_and_d, 2),
        "debt_to_equity_ratio": np.round(debt_to_equity, 3),
        "current_ratio": np.round(current_ratio, 3),
        "profit_margin": np.round(profit_margin, 4),
        "insolvency_risk": insolvency_risk,
    })


def oracle_probabilities(df: pd.DataFrame) -> np.ndarray:
    """True data-generating probabilities, recomputed from the (rounded) stored features.

    Used only as a performance ceiling reference in reports, never as a model input.
    """
    z = true_insolvency_logit(
        df["debt_to_equity_ratio"].to_numpy(),
        df["current_ratio"].to_numpy(),
        df["profit_margin"].to_numpy(),
        df["years_operating"].to_numpy(),
        df["industry"].to_numpy(),
    )
    return 1.0 / (1.0 + np.exp(-z))

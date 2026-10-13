"""Interactive demo: score one business profile with the trained risk model.

Run from the project root:  uv run streamlit run app/streamlit_app.py
Nothing entered here is stored or logged by the app.
"""

import os
import sys

import pandas as pd
import streamlit as st
from pydantic import ValidationError

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.export import ModelPersistence
from src.interpret import ExplanationUnavailable, ModelExplainer
from src.schema import INDUSTRIES, STATES, BusinessProfile

MODEL_PATH = os.path.join(BASE_DIR, "models", "risk_model.joblib")

DISCLAIMER = (
    "**Experimental demo trained on synthetic data.** The model has never seen real company records. "
    "Its output is not financial advice, a credit assessment or a real-world insolvency prediction. "
    "Do not enter confidential business information."
)


@st.cache_resource
def load_model():
    return ModelPersistence.load_pipeline(MODEL_PATH)


@st.cache_resource
def load_explainer(_risk_model):
    try:
        return ModelExplainer(_risk_model.model_pipeline, _risk_model.background_sample)
    except ExplanationUnavailable:
        return None


def contribution_chart(contributions: pd.DataFrame):
    """Horizontal bars sorted by size: red pushes towards higher risk, blue towards lower risk."""
    import matplotlib.pyplot as plt

    ordered = contributions.iloc[::-1]
    colors = ["#c0392b" if v > 0 else "#2e6fba" for v in ordered["shap_value"]]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(ordered) + 1))
    ax.barh(ordered["feature"], ordered["shap_value"], color=colors)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Contribution to model log-odds")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def format_validation_errors(error: ValidationError) -> list:
    messages = []
    for item in error.errors():
        field = ".".join(str(part) for part in item["loc"]) or "input"
        messages.append(f"{field}: {item['msg']}")
    return messages


def main() -> None:
    st.set_page_config(page_title="Enterprise Default Radar", layout="centered")
    st.title("Enterprise Default Radar")
    st.caption("Business insolvency risk scoring demo")
    st.warning(DISCLAIMER)

    if not os.path.exists(MODEL_PATH):
        st.error("No trained model found. Run `uv run python scripts/run_pipeline.py` from the project root first.")
        st.stop()
    risk_model = load_model()
    meta = risk_model.metadata

    with st.form("business_profile"):
        col_left, col_right = st.columns(2)
        with col_left:
            state = st.selectbox("State / territory", STATES)
            industry = st.selectbox("Industry", INDUSTRIES)
            employee_count = st.number_input("Employees", min_value=0, value=12, step=1)
            years_operating = st.number_input("Years operating", min_value=0, value=6, step=1)
            annual_turnover = st.number_input("Annual turnover (AUD)", min_value=0.0, value=1_500_000.0, step=50_000.0)
        with col_right:
            r_and_d = st.number_input("R&D expenditure (AUD)", min_value=0.0, value=50_000.0, step=10_000.0)
            debt_to_equity = st.number_input("Debt-to-equity ratio", min_value=0.0, value=1.5, step=0.1, format="%.3f")
            current_ratio = st.number_input("Current ratio", min_value=0.0, value=1.5, step=0.1, format="%.3f")
            profit_margin_pct = st.number_input("Profit margin (%)", min_value=-100.0, max_value=100.0, value=5.0, step=1.0)
        submitted = st.form_submit_button("Score business")

    if not submitted:
        st.info(
            f"Model: `{meta['selected_model']}` with `{meta['calibration_method']}` calibration. "
            f"Decision threshold {risk_model.threshold:.2f}, chosen to minimise an illustrative cost of "
            f"${meta['cost_assumptions']['false_positive_cost']:.0f} per unnecessary review and "
            f"${meta['cost_assumptions']['false_negative_cost']:.0f} per missed insolvency."
        )
        return

    try:
        profile = BusinessProfile(
            state=state,
            industry=industry,
            employee_count=int(employee_count),
            annual_turnover_aud=float(annual_turnover),
            years_operating=int(years_operating),
            r_and_d_expenditure_aud=float(r_and_d),
            debt_to_equity_ratio=float(debt_to_equity),
            current_ratio=float(current_ratio),
            profit_margin=float(profit_margin_pct) / 100.0,
        )
    except ValidationError as error:
        st.error("Please correct the inputs:\n\n" + "\n".join(f"- {m}" for m in format_validation_errors(error)))
        return

    result = risk_model.score_profile(profile)
    metric_col, category_col = st.columns(2)
    metric_col.metric("Estimated insolvency probability", f"{result['probability']:.1%}")
    category_col.metric("Decision", "Flag for review" if result["flagged"] else "No flag")
    st.write(f"**{result['category']}** (flag if probability >= {result['threshold']:.2f}).")
    for warning in result["warnings"]:
        st.warning(f"Outside training data: {warning}. The estimate is an extrapolation and less reliable.")

    st.subheader("Why this score?")
    explainer = load_explainer(risk_model)
    if explainer is None:
        st.info("Install the optional `advanced` dependencies to see SHAP explanations.")
        return
    contributions, _ = explainer.explain_record(pd.DataFrame([profile.model_dump()]), top_n=8)
    st.pyplot(contribution_chart(contributions))
    st.caption(
        "SHAP contributions to the model's log-odds: positive values push towards higher risk, negative values towards "
        "lower risk. They show how this model uses the inputs, not what causes insolvency."
    )


if __name__ == "__main__":
    main()

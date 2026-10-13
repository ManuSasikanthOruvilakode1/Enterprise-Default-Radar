"""Headless smoke tests for the Streamlit app (need a trained model artifact)."""

import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PATH = os.path.join(ROOT, "app", "streamlit_app.py")
MODEL_PATH = os.path.join(ROOT, "models", "risk_model.joblib")

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
pytestmark = pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Run scripts/run_pipeline.py to create the model first")


def run_app():
    return streamlit_testing.AppTest.from_file(APP_PATH, default_timeout=60).run()


def test_app_shows_disclaimer_and_form():
    app = run_app()
    assert not app.exception
    assert any("synthetic data" in w.value for w in app.warning)
    assert len(app.number_input) == 7


def test_app_scores_valid_input():
    app = run_app()
    app.button[0].click().run()
    assert not app.exception
    labels = [m.label for m in app.metric]
    assert "Estimated insolvency probability" in labels


def test_app_reports_invalid_input():
    app = run_app()
    app.number_input[0].set_value(0)  # zero employees is rejected by the schema
    app.button[0].click().run()
    assert not app.exception
    assert any("employee_count" in e.value for e in app.error)
    assert not app.metric

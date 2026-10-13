"""Score a CSV of businesses with the trained risk model.

Usage:
    uv run python scripts/score_csv.py input.csv [--output scored.csv] [--model models/risk_model.joblib]

The output keeps every input column and adds: insolvency_probability, flagged, risk_category,
warnings (values outside the training range) and validation_error (rows that could not be scored).
"""

import argparse
import os
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.batch import score_frame
from src.export import ModelPersistence


def main() -> int:
    parser = argparse.ArgumentParser(description="Score a CSV of businesses with the trained risk model.")
    parser.add_argument("input", help="CSV with the model's input columns")
    parser.add_argument("--output", help="Output CSV (default: <input>_scored.csv)")
    parser.add_argument("--model", default=os.path.join(BASE_DIR, "models", "risk_model.joblib"))
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 1
    output = args.output or os.path.splitext(args.input)[0] + "_scored.csv"

    risk_model = ModelPersistence.load_pipeline(args.model)
    try:
        scored = score_frame(risk_model, pd.read_csv(args.input))
    except ValueError as error:
        print(f"Cannot score {args.input}: {error}", file=sys.stderr)
        return 1
    scored.to_csv(output, index=False)

    invalid = int((scored["validation_error"] != "").sum())
    flagged = int(scored["flagged"].fillna(False).astype(bool).sum())
    print(f"Scored {len(scored) - invalid} of {len(scored)} rows ({invalid} invalid); {flagged} flagged for review "
          f"at threshold {risk_model.threshold:.2f}.")
    print(f"Wrote {output}")
    print("Note: the model is trained on synthetic data; scores are not real-world insolvency assessments.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

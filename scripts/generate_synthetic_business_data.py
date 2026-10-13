"""Generate the synthetic business outcomes dataset used by the pipeline."""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_generation import generate_business_dataset

if __name__ == "__main__":
    out_dir = os.path.join(BASE_DIR, "data", "raw")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "business_outcomes.csv")
    df = generate_business_dataset(n_samples=2500, seed=42)
    df.to_csv(out_csv, index=False, lineterminator="\n")
    print(f"Generated synthetic business dataset with {len(df)} rows and {len(df.columns)} columns.")
    print(f"Insolvency positive rate: {df['insolvency_risk'].mean() * 100:.2f}%")
    print(f"Saved to: {out_csv}")

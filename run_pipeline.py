"""
run_pipeline.py

Runs the full pipeline end-to-end:
  1. Generate synthetic data (skip this step once you swap in the real Kaggle CSV)
  2. EDA
  3. Segmentation
  4. Churn risk modeling

After this finishes, launch the dashboard with:
    streamlit run dashboard.py
"""

import subprocess
import sys
import os

STEPS = [
    ("Generating synthetic data", "generate_synthetic_data.py"),
    ("Running EDA", "eda.py"),
    ("Running segmentation", "segmentation.py"),
    ("Training churn risk model", "churn_model.py"),
]


def main():
    # Skip data generation if the real dataset has already been placed
    use_real_data = os.path.exists("data/Churn_Modelling.csv") and "--force-synthetic" not in sys.argv

    for label, script in STEPS:
        if script == "generate_synthetic_data.py" and use_real_data:
            print(f"\n>>> Skipping synthetic data generation — data/Churn_Modelling.csv already exists")
            continue
        print(f"\n{'=' * 70}\n>>> {label} ({script})\n{'=' * 70}")
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"Step failed: {script}")
            sys.exit(1)

    print("\n" + "=" * 70)
    print("Pipeline complete! Now run: streamlit run dashboard.py")
    print("=" * 70)


if __name__ == "__main__":
    main()

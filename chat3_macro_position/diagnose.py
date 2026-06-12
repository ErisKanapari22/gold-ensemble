# chat3_macro_position/diagnose.py
# Diagnostic script — checks whether engineered macro features
# actually correlate with the DIRECTIONAL/NEUTRAL label.
# Run: python diagnose.py

import numpy as np
import pandas as pd

import config
from data import generate_labels, FEATURE_COLS


def main():
    df = pd.read_csv(config.DATA_CACHE_PATH, parse_dates=["Date"])

    labels = generate_labels(df)
    df = df.iloc[: len(df) - config.FORWARD_DAYS].reset_index(drop=True)
    labels = labels[: len(df)]

    print(f"Total rows: {len(df)}")
    print(f"DIRECTIONAL: {(labels == 0).sum()}  NEUTRAL: {(labels == 1).sum()}\n")

    print(f"{'Feature':<20} {'StdDev':>10} {'CorrWithLabel':>15} {'MeanDir':>10} {'MeanNeutral':>12}")
    print("-" * 70)

    for col in FEATURE_COLS:
        vals = df[col].values.astype(float)
        std = vals.std()

        # Correlation between feature value and label (0=DIRECTIONAL, 1=NEUTRAL)
        # Negative correlation = higher feature value -> more likely DIRECTIONAL
        if std > 0:
            corr = np.corrcoef(vals, labels)[0, 1]
        else:
            corr = 0.0

        mean_dir = vals[labels == 0].mean()
        mean_neu = vals[labels == 1].mean()

        print(f"{col:<20} {std:>10.4f} {corr:>15.4f} {mean_dir:>10.4f} {mean_neu:>12.4f}")

    # ── Check how "blocky" the labels are (consecutive runs) ───────────────────
    print("\n--- Label run-length analysis ---")
    runs = []
    current = labels[0]
    run_len = 1
    for l in labels[1:]:
        if l == current:
            run_len += 1
        else:
            runs.append((current, run_len))
            current = l
            run_len = 1
    runs.append((current, run_len))

    run_lengths = [r[1] for r in runs]
    print(f"Number of runs: {len(runs)}")
    print(f"Avg run length: {np.mean(run_lengths):.1f}")
    print(f"Max run length: {np.max(run_lengths)}")
    print(f"Min run length: {np.min(run_lengths)}")


if __name__ == "__main__":
    main()
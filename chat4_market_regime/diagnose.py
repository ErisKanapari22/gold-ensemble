# diagnose.py
# Run this BEFORE train.py.
#
# Performs:
#   1. Feature <-> label correlation analysis
#   2. Label run-length / autocorrelation analysis
#   3. Class balance report
#   4. Architecture recommendation (logistic regression vs MLP w/ hidden layer)
#
# Usage:
#   python diagnose.py

import numpy as np
import pandas as pd

import config
from data import fetch_data, engineer_features, generate_labels


# ── Run-length analysis ───────────────────────────────────────────────────────

def run_length_analysis(labels: np.ndarray):
    """
    Returns a list of (label_value, run_length) tuples describing
    consecutive blocks of identical labels.
    """
    runs = []
    current_label = labels[0]
    current_len = 1
    for l in labels[1:]:
        if l == current_label:
            current_len += 1
        else:
            runs.append((current_label, current_len))
            current_label = l
            current_len = 1
    runs.append((current_label, current_len))
    return runs


def print_run_length_stats(runs):
    print("\n── Label run-length analysis ──────────────────────────────")
    for cls, name in enumerate(config.CLASS_NAMES):
        lengths = [r[1] for r in runs if r[0] == cls]
        if not lengths:
            print(f"  {name:>12}: no runs found")
            continue
        lengths_arr = np.array(lengths)
        print(
            f"  {name:>12}: n_runs={len(lengths_arr):4d}  "
            f"mean={lengths_arr.mean():6.1f}  "
            f"median={np.median(lengths_arr):6.1f}  "
            f"max={lengths_arr.max():4d}  "
            f"p90={np.percentile(lengths_arr, 90):6.1f}"
        )

    all_lengths = np.array([r[1] for r in runs])
    print(f"\n  Overall: n_runs={len(runs)}  mean_run={all_lengths.mean():.1f}  max_run={all_lengths.max()}")
    if all_lengths.mean() > 20:
        print(
            "  WARNING: average run length is long (>20 bars). Regimes persist for "
            "many candles, so a single chronological val split may land mostly inside "
            "one regime block. Consider a larger val set or block-based CV if val "
            "metrics look unstable across runs."
        )


# ── Correlation analysis ──────────────────────────────────────────────────────

def correlation_analysis(feature_matrix: np.ndarray, labels: np.ndarray):
    """
    Pearson correlation between each feature and the binary label
    (0=DIRECTIONAL, 1=NEUTRAL). Negative correlation -> feature is HIGHER
    when regime is DIRECTIONAL.
    """
    print("\n── Feature <-> label correlation (label: 0=DIRECTIONAL, 1=NEUTRAL) ──")
    correlations = {}
    for i, feat in enumerate(config.FEATURE_COLS):
        col = feature_matrix[:, i]
        if np.std(col) == 0:
            corr = 0.0
        else:
            corr = float(np.corrcoef(col, labels)[0, 1])
        correlations[feat] = corr
        print(f"  {feat:>18}: {corr:+.3f}")
    return correlations


# ── Class balance ─────────────────────────────────────────────────────────────

def class_balance(labels: np.ndarray):
    print("\n── Class balance ───────────────────────────────────────────")
    n_total = len(labels)
    for cls, name in enumerate(config.CLASS_NAMES):
        count = int((labels == cls).sum())
        print(f"  {name:>12}: {count:6d}  ({count / n_total * 100:.1f}%)")


# ── Architecture recommendation ──────────────────────────────────────────────

def recommend_architecture(correlations: dict):
    print("\n── Architecture recommendation ─────────────────────────────")
    max_abs_corr = max(abs(c) for c in correlations.values())
    strongest_feat = max(correlations, key=lambda k: abs(correlations[k]))

    print(f"  Strongest single-feature correlation: {strongest_feat} ({correlations[strongest_feat]:+.3f})")

    if max_abs_corr >= 0.5:
        print(
            "  -> Strong linear signal detected (|corr| >= 0.5). This is EXPECTED here "
            "since the label is derived directly from ADX14.\n"
            "  -> RECOMMENDATION: keep USE_HIDDEN_LAYER = False (logistic regression + "
            "BatchNorm). The model should learn a smoothed/refined version of the ADX "
            "threshold boundary, lightly adjusted by the other features."
        )
    elif max_abs_corr >= 0.25:
        print(
            "  -> Moderate linear signal (0.25 <= |corr| < 0.5).\n"
            "  -> RECOMMENDATION: try USE_HIDDEN_LAYER = False first. If val "
            "DIRECTIONAL F1 plateaus very low after training, switch to "
            "USE_HIDDEN_LAYER = True (small MLP)."
        )
    else:
        print(
            "  -> Weak individual feature correlations (|corr| < 0.25). Signal may be "
            "non-linear / combination-based.\n"
            "  -> RECOMMENDATION: set USE_HIDDEN_LAYER = True (BatchNorm -> Linear(10->16) "
            "-> ReLU -> Dropout -> Linear(16->2))."
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    raw = fetch_data(use_cache=True)
    df = engineer_features(raw)
    labels, directions = generate_labels(df, config.ADX_THRESHOLD)

    feature_matrix = df[config.FEATURE_COLS].values.astype(np.float32)

    print(f"Total samples after feature engineering: {len(labels)}")
    print(f"ADX_THRESHOLD = {config.ADX_THRESHOLD}")

    class_balance(labels)
    correlations = correlation_analysis(feature_matrix, labels)

    runs = run_length_analysis(labels)
    print_run_length_stats(runs)

    recommend_architecture(correlations)

    print(
        "\nNext steps:\n"
        "  1. Review the class balance above. If one class is < 10%, the "
        "WeightedRandomSampler in data.py will handle it, but consider whether "
        "ADX_THRESHOLD needs adjusting.\n"
        "  2. Apply the architecture recommendation to config.USE_HIDDEN_LAYER.\n"
        "  3. If run-lengths are very long, consider increasing the val set size "
        "(lower TRAIN_RATIO) before running train.py.\n"
        "  4. Run: python train.py"
    )


if __name__ == "__main__":
    main()
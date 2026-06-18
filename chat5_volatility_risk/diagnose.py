# diagnose.py
# Run BEFORE train.py. Reveals feature-label structure, label autocorrelation,
# and recommends the correct architecture and threshold settings.
#
# Usage: python diagnose.py

import numpy as np
import pandas as pd

import config
from data import build_features_and_labels


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_run_lengths(labels: np.ndarray):
    """Return list of (class_id, run_length) tuples for consecutive label runs."""
    runs, current, count = [], labels[0], 1
    for v in labels[1:]:
        if v == current:
            count += 1
        else:
            runs.append((int(current), count))
            current, count = v, 1
    runs.append((int(current), count))
    return runs


def bar(value: float, width: int = 30) -> str:
    return "#" * int(abs(value) * width)


# ── Main ──────────────────────────────────────────────────────────────────────

def diagnose():
    print("=" * 62)
    print("  Chat 5 - Volatility & Risk Model  |  DIAGNOSTICS")
    print("=" * 62)

    print("\nLoading features and labels ...")
    features, labels, df = build_features_and_labels()
    n_total = len(labels)

    # ── 1. Class balance ──────────────────────────────────────────────────────
    print(f"\n{'-' * 62}")
    print("1. CLASS BALANCE")
    print(f"{'-' * 62}")
    for cls, name in enumerate(config.CLASS_NAMES):
        count = int((labels == cls).sum())
        print(f"  {name:>12}: {count:5d}  ({count / n_total * 100:.1f}%)")

    # ── 2. Feature ↔ Label Pearson correlation ────────────────────────────────
    print(f"\n{'-' * 62}")
    print("2. FEATURE <-> LABEL PEARSON CORRELATION")
    print(f"   (label: 0=DIRECTIONAL, 1=NEUTRAL  |  negative r -> feature^ = more DIRECTIONAL)")
    print(f"{'-' * 62}")

    correlations = {}
    for i, col in enumerate(config.FEATURE_COLS):
        r = float(np.corrcoef(features[:, i], labels)[0, 1])
        correlations[col] = r

    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, r in sorted_corr:
        sign = "+" if r >= 0 else "-"
        print(f"  {feat:>18}: {sign}{abs(r):.4f}  {bar(r)}")

    top_abs_corr = abs(sorted_corr[0][1])

    # ── 3. Label run-length analysis ──────────────────────────────────────────
    print(f"\n{'-' * 62}")
    print("3. LABEL RUN-LENGTH ANALYSIS")
    print(f"   (long runs = label autocorrelation = unstable chronological val split)")
    print(f"{'-' * 62}")

    runs = compute_run_lengths(labels)
    dir_runs  = [r for cls, r in runs if cls == 0]
    neut_runs = [r for cls, r in runs if cls == 1]

    print(f"  Total label runs: {len(runs)}")
    if dir_runs:
        print(f"  DIRECTIONAL:  {len(dir_runs):4d} runs  |  mean={np.mean(dir_runs):.1f}  "
              f"median={int(np.median(dir_runs))}  max={max(dir_runs)}")
    if neut_runs:
        print(f"  NEUTRAL:      {len(neut_runs):4d} runs  |  mean={np.mean(neut_runs):.1f}  "
              f"median={int(np.median(neut_runs))}  max={max(neut_runs)}")

    max_run = max(
        max(dir_runs)  if dir_runs  else 0,
        max(neut_runs) if neut_runs else 0,
    )
    if max_run > 48:
        print("\n  [!] Very long runs detected (> 48 bars = 2 trading days).")
        print("      Val metrics may be noisy. Chat 7 should assign lower confidence")
        print("      weight to this specialist during extended vol regimes.")
    elif max_run > 24:
        print("\n  [!] Moderate label runs (> 24 bars). Chronological split is OK")
        print("      but expect some instability in per-epoch val F1.")
    else:
        print("  [ok] Run-lengths are short -- chronological split looks stable.")

    # ── 4. VOL_THRESHOLD sensitivity ─────────────────────────────────────────
    print(f"\n{'-' * 62}")
    print("4. VOL_THRESHOLD SENSITIVITY")
    print(f"   (RETURN_THRESHOLD fixed at {config.RETURN_THRESHOLD}  |  both conditions must hold)")
    print(f"{'-' * 62}")

    atr_ratio_vals = features[:, config.FEATURE_COLS.index("atr_ratio")]
    return_5_vals  = features[:, config.FEATURE_COLS.index("return_5")]

    print(f"  {'Threshold':>10}  {'DIRECTIONAL bars':>18}  {'%':>6}")
    for thresh in [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]:
        mask  = (atr_ratio_vals > thresh) & (np.abs(return_5_vals) > config.RETURN_THRESHOLD)
        count = int(mask.sum())
        pct   = count / n_total * 100
        arrow = "  <- current" if abs(thresh - config.VOL_THRESHOLD) < 0.01 else ""
        print(f"  {thresh:>10.1f}  {count:>18d}  {pct:>5.1f}%{arrow}")

    # ── 5. Architecture recommendation ───────────────────────────────────────
    print(f"\n{'-' * 62}")
    print("5. ARCHITECTURE RECOMMENDATION")
    print(f"{'-' * 62}")

    if top_abs_corr >= 0.25:
        arch = "LINEAR  ->  BatchNorm1d -> Linear(12 -> 2)"
        advice = (
            f"Top feature correlation is {top_abs_corr:.3f} - signal is linearly separable.\n"
            "  Keep  USE_HIDDEN_LAYER = False  in config.py.\n"
            "  If DIRECTIONAL F1 < 0.35 after 50 epochs, then try USE_HIDDEN_LAYER = True."
        )
    elif top_abs_corr >= 0.10:
        arch = "LINEAR first  (escalate to MLP if F1 < 0.35)"
        advice = (
            f"Moderate correlation {top_abs_corr:.3f}. Start linear (USE_HIDDEN_LAYER = False).\n"
            "  If DIRECTIONAL F1 stalls below 0.35, set USE_HIDDEN_LAYER = True and retrain."
        )
    else:
        arch = "MLP  ->  BatchNorm1d -> Linear -> ReLU -> Dropout -> Linear(-> 2)"
        advice = (
            f"Low correlations (top {top_abs_corr:.3f}) - likely non-linear structure.\n"
            "  Set  USE_HIDDEN_LAYER = True  in config.py before running train.py."
        )

    print(f"  Recommended: {arch}")
    print(f"  Rationale:   {advice}")

    print(f"\n{'=' * 62}")
    print("  Diagnostics complete. Adjust config.py if needed, then run train.py.")
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    diagnose()
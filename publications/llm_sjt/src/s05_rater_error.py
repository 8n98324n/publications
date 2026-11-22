# -*- coding: utf-8 -*-
"""s05_rater_error.py: Two-way ANOVA rater-error mean squares and bootstrap ratios (§3.2.1.6, Table D2a)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_stats import SEED, RAW_DIR, TAB_DIR, icc3k_anova

DIMS = ["realism", "relevance", "typicality", "arousal"]


def run(n_boot: int = 2000) -> pd.DataFrame:
    wide = pd.read_csv(RAW_DIR / "expert_scenario_ratings.csv")
    rng = np.random.default_rng(SEED)
    rows = []
    for dim in DIMS:
        X_kg = wide[[f"item{i}_{dim}" for i in range(1, 31)]].values.T.astype(float)
        X_bs = wide[[f"item{i}_{dim}" for i in range(31, 61)]].values.T.astype(float)
        a_kg = icc3k_anova(X_kg)
        a_bs = icc3k_anova(X_bs)
        ratio = a_kg["ms_error"] / a_bs["ms_error"]
        b_ratios = np.empty(n_boot)
        for b in range(n_boot):
            i1 = rng.integers(0, 30, size=30)
            i2 = rng.integers(0, 30, size=30)
            b_ratios[b] = icc3k_anova(X_kg[i1])["ms_error"] / max(1e-9, icc3k_anova(X_bs[i2])["ms_error"])
        rows.append({
            "dimension": dim,
            "ms_item_kg": a_kg["ms_item"],
            "ms_item_llm": a_bs["ms_item"],
            "ms_error_kg": a_kg["ms_error"],
            "ms_error_llm": a_bs["ms_error"],
            "mse_ratio": ratio,
            "ratio_ci_low": float(np.percentile(b_ratios, 2.5)),
            "ratio_ci_high": float(np.percentile(b_ratios, 97.5)),
            "error_sd_kg": a_kg["error_sd"],
            "error_sd_llm": a_bs["error_sd"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(TAB_DIR / "table_D2a_rater_error.csv", index=False)
    return df


if __name__ == "__main__":
    print(run().to_string(index=False))

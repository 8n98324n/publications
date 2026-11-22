# -*- coding: utf-8 -*-
"""s11_icc_ceiling_simulation.py: Parametric and discrete empirical ceiling simulations for ICC(3,k) (§3.2.1.5, App. F.4)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_stats import SEED, RAW_DIR, TAB_DIR, icc3k_anova


def run(n_param: int = 2000, n_emp: int = 10000) -> dict:
    rng = np.random.default_rng(SEED)

    # Simulation 1: Normal-rounding under heterogeneous (sigma=0.70) vs ceiling-compressed (sigma=0.22) items
    icc_a = np.empty(n_param)
    icc_b = np.empty(n_param)
    ceil_b = np.empty(n_param)
    for r in range(n_param):
        true_a = rng.normal(3.8, 0.70, size=(30, 1))
        X_a = np.clip(np.rint(true_a + rng.normal(0.0, 0.50, size=(30, 5))), 1, 5)
        icc_a[r] = icc3k_anova(X_a)["icc3k"]

        true_b = rng.normal(4.5, 0.22, size=(30, 1))
        X_b = np.clip(np.rint(true_b + rng.normal(0.0, 0.50, size=(30, 5))), 1, 5)
        icc_b[r] = icc3k_anova(X_b)["icc3k"]
        ceil_b[r] = np.mean(X_b >= 4)

    # Simulation 2: 10,000 i.i.d. draws from observed base LLM empirical rating distribution (zero true item variance)
    opt_long = pd.read_csv(RAW_DIR / "expert_option_spss_long.csv")
    bs_scores = opt_long[opt_long["group"] == "base_LLM"]["score"].values
    probs = np.array([np.mean(bs_scores == k) for k in range(1, 6)], dtype=float)
    probs /= probs.sum()

    rng2 = np.random.default_rng(SEED + 11)
    emp_iccs = np.empty(n_emp)
    for r in range(n_emp):
        X_null = rng2.choice([1, 2, 3, 4, 5], size=(30, 5), p=probs)
        emp_iccs[r] = icc3k_anova(X_null)["icc3k"]

    res = {
        "param_condA_mean_icc": float(np.mean(icc_a)),
        "param_condB_mean_icc": float(np.mean(icc_b)),
        "param_condB_ceil45_pct": float(np.mean(ceil_b) * 100.0),
        "emp_base_probs": probs.tolist(),
        "emp_ceil45_pct": float((probs[3] + probs[4]) * 100.0),
        "emp_neg_icc_pct": float(np.mean(emp_iccs < 0) * 100.0),
        "emp_median_icc": float(np.median(emp_iccs)),
        "emp_iqr_low": float(np.percentile(emp_iccs, 25)),
        "emp_iqr_high": float(np.percentile(emp_iccs, 75)),
        "emp_ci95_low": float(np.percentile(emp_iccs, 2.5)),
        "emp_ci95_high": float(np.percentile(emp_iccs, 97.5)),
    }
    pd.DataFrame([res]).to_csv(TAB_DIR / "s11_icc_ceiling_simulations.csv", index=False)
    return res


if __name__ == "__main__":
    r = run()
    print(f"Sim1: .868->{r['param_condA_mean_icc']:.3f}, .353->{r['param_condB_mean_icc']:.3f} | Sim2 %neg={r['emp_neg_icc_pct']:.1f}%")

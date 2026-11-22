# -*- coding: utf-8 -*-
"""s07_reliability.py: Cronbach's alpha, 1-factor ML McDonald's omega, paired bootstrap, and polarity item analysis (§3.3.3, F.7)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_stats import SEED, DERIVED_DIR, TAB_DIR, cronbach_alpha, mcdonald_omega_ml
import s00_prepare_data


def _citc_vec(M: np.ndarray) -> np.ndarray:
    tot = M.sum(axis=1)
    out = np.empty(M.shape[1])
    for j in range(M.shape[1]):
        rest = tot - M[:, j]
        out[j] = float(stats.pearsonr(M[:, j], rest).statistic)
    return out


def _raw_mean_interitem_r(M: np.ndarray) -> float:
    R = np.corrcoef(M, rowvar=False)
    k = R.shape[0]
    iu = np.triu_indices(k, k=1)
    return float(np.mean(R[iu]))


def run(n_boot: int = 2000) -> dict:
    scores_path = DERIVED_DIR / "formal_item_scores.csv"
    if not scores_path.exists():
        s00_prepare_data.run()
    df = pd.read_csv(scores_path)
    rng = np.random.default_rng(SEED)

    kg_cols = [f"LLM_KG{i}" for i in range(1, 41)]
    bs_cols = [f"LLM{i}" for i in range(1, 41)]
    res_cols = [f"5x5resilience{i}" for i in range(1, 26)]

    M_kg = df[kg_cols].values.astype(float)
    M_bs = df[bs_cols].values.astype(float)
    M_res = df[res_cols].values.astype(float)
    n = M_kg.shape[0]

    alpha_kg = cronbach_alpha(M_kg)
    alpha_bs = cronbach_alpha(M_bs)
    omega_kg = mcdonald_omega_ml(M_kg)
    omega_bs = mcdonald_omega_ml(M_bs)

    b_akg = np.empty(n_boot)
    b_abs = np.empty(n_boot)
    b_okg = np.empty(n_boot)
    b_obs = np.empty(n_boot)
    b_da_ml = np.empty(n_boot)
    b_da_ll = np.empty(n_boot)

    M_kg_ml, M_kg_ll = M_kg[:, 0::2], M_kg[:, 1::2]
    M_bs_ml, M_bs_ll = M_bs[:, 0::2], M_bs[:, 1::2]

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bk = M_kg[idx]
        bb = M_bs[idx]
        b_akg[b] = cronbach_alpha(bk)
        b_abs[b] = cronbach_alpha(bb)
        b_okg[b] = mcdonald_omega_ml(bk)
        b_obs[b] = mcdonald_omega_ml(bb)
        b_da_ml[b] = cronbach_alpha(bk[:, 0::2]) - cronbach_alpha(bb[:, 0::2])
        b_da_ll[b] = cronbach_alpha(bk[:, 1::2]) - cronbach_alpha(bb[:, 1::2])

    b_da_40 = b_akg - b_abs
    p_da_40 = float(2.0 * min(np.mean(b_da_40 <= 0), np.mean(b_da_40 >= 0)))
    p_da_ml = float(2.0 * min(np.mean(b_da_ml <= 0), np.mean(b_da_ml >= 0)))
    p_da_ll = float(2.0 * min(np.mean(b_da_ll <= 0), np.mean(b_da_ll >= 0)))

    rbar_imp_kg = alpha_kg / (40.0 - alpha_kg * 39.0)
    rbar_imp_bs = alpha_bs / (40.0 - alpha_bs * 39.0)
    sb20_kg = (20.0 * rbar_imp_kg) / (1.0 + 19.0 * rbar_imp_kg)
    sb20_bs = (20.0 * rbar_imp_bs) / (1.0 + 19.0 * rbar_imp_bs)

    M_kg_comp20 = M_kg_ml + M_kg_ll
    M_bs_comp20 = M_bs_ml + M_bs_ll
    alpha_comp20_kg = cronbach_alpha(M_kg_comp20)
    alpha_comp20_bs = cronbach_alpha(M_bs_comp20)
    within_scen_r_kg = float(np.mean([stats.pearsonr(M_kg_ml[:, s], M_kg_ll[:, s]).statistic for s in range(20)]))
    within_scen_r_bs = float(np.mean([stats.pearsonr(M_bs_ml[:, s], M_bs_ll[:, s]).statistic for s in range(20)]))

    alpha_kg_ml = cronbach_alpha(M_kg_ml)
    alpha_bs_ml = cronbach_alpha(M_bs_ml)
    alpha_kg_ll = cronbach_alpha(M_kg_ll)
    alpha_bs_ll = cronbach_alpha(M_bs_ll)

    citc_kg_40 = _citc_vec(M_kg)
    citc_bs_40 = _citc_vec(M_bs)
    citc_kg_ml = _citc_vec(M_kg_ml)
    citc_bs_ml = _citc_vec(M_bs_ml)
    citc_kg_ll = _citc_vec(M_kg_ll)
    citc_bs_ll = _citc_vec(M_bs_ll)

    mw_citc = stats.mannwhitneyu(citc_kg_40, citc_bs_40, alternative="two-sided")
    mw_diff = stats.mannwhitneyu(M_kg.mean(axis=0), M_bs.mean(axis=0), alternative="two-sided")

    r_ml_ll_kg = stats.pearsonr(M_kg_ml.sum(axis=1), M_kg_ll.sum(axis=1))
    r_ml_ll_bs = stats.pearsonr(M_bs_ml.sum(axis=1), M_bs_ll.sum(axis=1))

    summary_df = pd.DataFrame([
        {
            "condition": "KG-LLM",
            "alpha_40": alpha_kg, "alpha_ci_low": np.percentile(b_akg, 2.5), "alpha_ci_high": np.percentile(b_akg, 97.5),
            "omega_40": omega_kg, "omega_ci_low": np.percentile(b_okg, 2.5), "omega_ci_high": np.percentile(b_okg, 97.5),
            "alpha_ml20": alpha_kg_ml, "alpha_ll20": alpha_kg_ll, "sb_proj_20": sb20_kg,
            "alpha_scen_comp20": alpha_comp20_kg, "within_scen_ml_ll_r": within_scen_r_kg,
            "rbar_implied": rbar_imp_kg, "rbar_raw": _raw_mean_interitem_r(M_kg),
            "median_citc": float(np.median(citc_kg_40)), "n_citc_lt_30": int(np.sum(citc_kg_40 < 0.30)),
        },
        {
            "condition": "base_LLM",
            "alpha_40": alpha_bs, "alpha_ci_low": np.percentile(b_abs, 2.5), "alpha_ci_high": np.percentile(b_abs, 97.5),
            "omega_40": omega_bs, "omega_ci_low": np.percentile(b_obs, 2.5), "omega_ci_high": np.percentile(b_obs, 97.5),
            "alpha_ml20": alpha_bs_ml, "alpha_ll20": alpha_bs_ll, "sb_proj_20": sb20_bs,
            "alpha_scen_comp20": alpha_comp20_bs, "within_scen_ml_ll_r": within_scen_r_bs,
            "rbar_implied": rbar_imp_bs, "rbar_raw": _raw_mean_interitem_r(M_bs),
            "median_citc": float(np.median(citc_bs_40)), "n_citc_lt_30": int(np.sum(citc_bs_40 < 0.30)),
        },
    ])
    summary_df.to_csv(TAB_DIR / "s07_reliability_summary.csv", index=False)

    return {
        "summary_table": summary_df,
        "delta_alpha_40": alpha_kg - alpha_bs,
        "delta_alpha_ci_low": float(np.percentile(b_da_40, 2.5)),
        "delta_alpha_ci_high": float(np.percentile(b_da_40, 97.5)),
        "p_delta_alpha_40": p_da_40,
        "p_delta_alpha_ml20": p_da_ml,
        "p_delta_alpha_ll20": p_da_ll,
        "rbar_5x5rs": _raw_mean_interitem_r(M_res),
        "rbar_ml_kg": _raw_mean_interitem_r(M_kg_ml),
        "rbar_ml_base": _raw_mean_interitem_r(M_bs_ml),
        "rbar_ll_kg": _raw_mean_interitem_r(M_kg_ll),
        "rbar_ll_base": _raw_mean_interitem_r(M_bs_ll),
        "mean_citc_ml_kg": float(citc_kg_ml.mean()),
        "mean_citc_ml_base": float(citc_bs_ml.mean()),
        "mean_citc_ll_kg": float(citc_kg_ll.mean()),
        "mean_citc_ll_base": float(citc_bs_ll.mean()),
        "mw_citc_p": float(mw_citc.pvalue),
        "mw_difficulty_p": float(mw_diff.pvalue),
        "r_ml_ll_kg": float(r_ml_ll_kg.statistic),
        "p_ml_ll_kg": float(r_ml_ll_kg.pvalue),
        "r_ml_ll_base": float(r_ml_ll_bs.statistic),
        "p_ml_ll_base": float(r_ml_ll_bs.pvalue),
    }


if __name__ == "__main__":
    res = run()
    print(res["summary_table"].to_string(index=False))

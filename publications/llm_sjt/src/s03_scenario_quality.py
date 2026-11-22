# -*- coding: utf-8 -*-
"""s03_scenario_quality.py: Expert evaluation of 30 KG-LLM vs. 30 base LLM scenarios and citation-count OLS (H3; §3.2.1, App. D.3, F.3)."""
from __future__ import annotations
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_stats import (
    SEED, RAW_DIR, TAB_DIR,
    icc3k_anova, gwet_ac2_ordinal, welch_t_hedges_g, tost_welch, bh_fdr,
)
from scipy import stats

DIMS = ["realism", "relevance", "typicality", "arousal"]


def run(n_boot: int = 2000) -> dict:
    wide = pd.read_csv(RAW_DIR / "expert_scenario_ratings.csv")
    scen_texts = pd.read_csv(RAW_DIR / "scenario_texts.csv")
    item_agg = pd.read_csv(RAW_DIR / "item_level_expert_aggregates.csv")
    rng = np.random.default_rng(SEED)

    rows = []
    kg_mats = {}
    bs_mats = {}

    for dim in DIMS:
        cols_kg = [f"item{i}_{dim}" for i in range(1, 31)]
        cols_bs = [f"item{i}_{dim}" for i in range(31, 61)]
        X_kg = wide[cols_kg].values.T.astype(float)  # 30 x 5
        X_bs = wide[cols_bs].values.T.astype(float)  # 30 x 5
        kg_mats[dim] = X_kg
        bs_mats[dim] = X_bs

        icc_kg = icc3k_anova(X_kg)
        icc_bs = icc3k_anova(X_bs)
        ac2_kg = gwet_ac2_ordinal(X_kg)
        ac2_bs = gwet_ac2_ordinal(X_bs)

        b_icc_diff = np.empty(n_boot)
        b_ac2_kg = np.empty(n_boot)
        b_ac2_bs = np.empty(n_boot)
        b_ac2_diff = np.empty(n_boot)
        for b in range(n_boot):
            i1 = rng.integers(0, 30, size=30)
            i2 = rng.integers(0, 30, size=30)
            b_icc_diff[b] = icc3k_anova(X_kg[i1])["icc3k"] - icc3k_anova(X_bs[i2])["icc3k"]
            ak = gwet_ac2_ordinal(X_kg[i1])
            ab = gwet_ac2_ordinal(X_bs[i2])
            b_ac2_kg[b] = ak
            b_ac2_bs[b] = ab
            b_ac2_diff[b] = ak - ab

        p_icc_diff = float(2.0 * min(np.mean(b_icc_diff <= 0), np.mean(b_icc_diff >= 0)))
        m_kg = X_kg.mean(axis=1)
        m_bs = X_bs.mean(axis=1)
        wt = welch_t_hedges_g(m_kg, m_bs)
        tost = tost_welch(m_kg, m_bs, eq_d=0.5)

        ceil_kg = float(np.mean(X_kg >= 4) * 100.0)
        ceil_bs = float(np.mean(X_bs >= 4) * 100.0)

        rows.append({
            "dimension": dim,
            "kg_mean": wt["m1"], "kg_sd": wt["sd1"],
            "base_mean": wt["m2"], "base_sd": wt["sd2"],
            "welch_t": wt["t"], "welch_df": wt["df"], "welch_p": wt["p"],
            "hedges_g": wt["hedges_g"], "p_tost_05sd": tost["p_tost"],
            "icc_kg": icc_kg["icc3k"], "icc_kg_lo": icc_kg["ci_low"], "icc_kg_hi": icc_kg["ci_high"],
            "icc_base": icc_bs["icc3k"], "icc_base_lo": icc_bs["ci_low"], "icc_base_hi": icc_bs["ci_high"],
            "delta_icc": icc_kg["icc3k"] - icc_bs["icc3k"],
            "delta_icc_lo": float(np.percentile(b_icc_diff, 2.5)),
            "delta_icc_hi": float(np.percentile(b_icc_diff, 97.5)),
            "p_delta_icc": min(1.0, p_icc_diff),
            "ac2_kg": ac2_kg, "ac2_kg_lo": float(np.percentile(b_ac2_kg, 2.5)), "ac2_kg_hi": float(np.percentile(b_ac2_kg, 97.5)),
            "ac2_base": ac2_bs, "ac2_base_lo": float(np.percentile(b_ac2_bs, 2.5)), "ac2_base_hi": float(np.percentile(b_ac2_bs, 97.5)),
            "delta_ac2": ac2_kg - ac2_bs,
            "delta_ac2_lo": float(np.percentile(b_ac2_diff, 2.5)),
            "delta_ac2_hi": float(np.percentile(b_ac2_diff, 97.5)),
            "ceiling_4_5_kg_pct": ceil_kg,
            "ceiling_4_5_base_pct": ceil_bs,
        })

    df_scen = pd.DataFrame(rows)
    df_scen["welch_p_bh"] = bh_fdr(df_scen["welch_p"].values)
    df_scen["welch_p_bonf"] = np.minimum(1.0, df_scen["welch_p"].values * len(df_scen))
    df_scen.to_csv(TAB_DIR / "s03_scenario_quality.csv", index=False)

    # Citation-count analyses within 30 KG-LLM items (§3.2.1.3 & Figure 4b-c)
    kg_sub = scen_texts[scen_texts["group"] == "KG-LLM"].sort_values("within_group_id").copy()
    n_triples = np.array([len(re.findall(r"\d+", str(s))) for s in kg_sub["evidence_triples"]], dtype=int)
    scen_len = np.array([len(str(s).strip()) for s in kg_sub["scenario_text"]], dtype=float)
    serial_pos = np.array([((i - 1) % 6) + 1 for i in kg_sub["within_group_id"]], dtype=float)
    themes = kg_sub["stress_theme"].values

    kg_scen_mean = item_agg[item_agg["group"] == "KG-LLM"].sort_values("within_group_id")["scenario_mean"].values
    kg_opt_mean = item_agg[item_agg["group"] == "KG-LLM"].sort_values("within_group_id")["option_mean"].values
    bs_scen_mean = item_agg[item_agg["group"] == "base_LLM"].sort_values("within_group_id")["scenario_mean"].values
    bs_opt_mean = item_agg[item_agg["group"] == "base_LLM"].sort_values("within_group_id")["option_mean"].values
    kg_arousal = kg_mats["arousal"].mean(axis=1)

    r_scen, p_r_scen = stats.pearsonr(n_triples, kg_scen_mean)
    r_arous, p_r_arous = stats.pearsonr(n_triples, kg_arousal)

    mask_3 = n_triples == 3
    mask_57 = n_triples >= 5
    wt_3_vs_bs_scen = welch_t_hedges_g(kg_scen_mean[mask_3], bs_scen_mean)
    tost_3_vs_bs_scen = tost_welch(kg_scen_mean[mask_3], bs_scen_mean, eq_d=0.5)
    wt_3_vs_bs_opt = welch_t_hedges_g(kg_opt_mean[mask_3], bs_opt_mean)
    wt_57_vs_bs_scen = welch_t_hedges_g(kg_scen_mean[mask_57], bs_scen_mean)

    # Controlled OLS: scenario_mean ~ n_triples + scen_len + serial_pos + C(theme) (df = 22)
    ols_df = pd.DataFrame({
        "scen_mean": kg_scen_mean,
        "opt_mean": kg_opt_mean,
        "n_triples": n_triples.astype(float),
        "scen_len": scen_len,
        "serial_pos": serial_pos,
        "theme": themes,
    })
    X_design = pd.get_dummies(ols_df[["n_triples", "scen_len", "serial_pos", "theme"]], drop_first=True, dtype=float)
    X_design = sm.add_constant(X_design)
    fit_scen = sm.OLS(ols_df["scen_mean"], X_design).fit()
    fit_opt = sm.OLS(ols_df["opt_mean"], X_design).fit()

    cit_summary = {
        "n_triples_mean": float(n_triples.mean()),
        "n_triples_sd": float(n_triples.std(ddof=1)),
        "count_3": int(mask_3.sum()),
        "count_4": int((n_triples == 4).sum()),
        "count_5_7": int(mask_57.sum()),
        "r_ntriples_scen": float(r_scen),
        "p_ntriples_scen": float(p_r_scen),
        "r_ntriples_arousal": float(r_arous),
        "p_ntriples_arousal": float(p_r_arous),
        "wt_3_vs_base_scen": wt_3_vs_bs_scen,
        "tost_3_vs_base_scen_p": tost_3_vs_bs_scen["p_tost"],
        "wt_3_vs_base_opt": wt_3_vs_bs_opt,
        "wt_57_vs_base_scen": wt_57_vs_bs_scen,
        "ols_b_ntriples_scen": float(fit_scen.params["n_triples"]),
        "ols_se_ntriples_scen": float(fit_scen.bse["n_triples"]),
        "ols_t_ntriples_scen": float(fit_scen.tvalues["n_triples"]),
        "ols_p_ntriples_scen": float(fit_scen.pvalues["n_triples"]),
        "ols_df_resid": int(fit_scen.df_resid),
        "ols_p_ntriples_opt": float(fit_opt.pvalues["n_triples"]),
    }
    return {"scenario_table": df_scen, "citation_analysis": cit_summary}


if __name__ == "__main__":
    out = run()
    print(out["scenario_table"][["dimension", "kg_mean", "base_mean", "welch_t", "welch_p", "hedges_g", "ac2_kg", "ac2_base"]].to_string(index=False))

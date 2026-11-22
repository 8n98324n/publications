# -*- coding: utf-8 -*-
"""s06_mixed_model.py: Cross-classified MixedLM, LRTs, REML variance decomposition, and G-study (§3.2.3, App. D.8.1, F.6)."""
from __future__ import annotations
import math
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_stats import DERIVED_DIR, TAB_DIR
import s00_prepare_data


def run() -> dict:
    long_path = DERIVED_DIR / "scenario_ratings_long.csv"
    if not long_path.exists():
        s00_prepare_data.run()
    df = pd.read_csv(long_path)
    df["condition"] = pd.Categorical(df["condition"], categories=["base_LLM", "KG-LLM"])
    df["dimension"] = pd.Categorical(df["dimension"], categories=["realism", "relevance", "typicality", "arousal"])
    df["grp_one"] = 1

    vc = {
        "rater": "0 + C(rater_code)",
        "item": "0 + C(item_num)",
        "theme": "0 + C(stress_theme)",
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_dim_ml = smf.mixedlm("rating ~ C(dimension)", df, groups="grp_one", vc_formula=vc).fit(reml=False)
        m_add_ml = smf.mixedlm("rating ~ C(condition) + C(dimension)", df, groups="grp_one", vc_formula=vc).fit(reml=False)
        m_full_ml = smf.mixedlm("rating ~ C(condition) * C(dimension)", df, groups="grp_one", vc_formula=vc).fit(reml=False)
        m_add_reml = smf.mixedlm("rating ~ C(condition) + C(dimension)", df, groups="grp_one", vc_formula=vc).fit(reml=True)
        m_full_reml = smf.mixedlm("rating ~ C(condition) * C(dimension)", df, groups="grp_one", vc_formula=vc).fit(reml=True)

    chi2_cond = float(2.0 * (m_add_ml.llf - m_dim_ml.llf))
    p_cond = float(stats.chi2.sf(chi2_cond, df=1))
    chi2_int = float(2.0 * (m_full_ml.llf - m_add_ml.llf))
    p_int = float(stats.chi2.sf(chi2_int, df=3))

    v_dict = dict(zip(m_add_reml.model.exog_vc.names, m_add_reml.vcomp))
    var_rater = float(v_dict["rater"])
    var_item = float(v_dict["item"])
    var_theme = float(v_dict["theme"])
    var_resid = float(m_add_reml.scale)
    var_tot = var_rater + var_item + var_theme + var_resid

    shares = {
        "residual_pct": var_resid / var_tot * 100.0,
        "rater_pct": var_rater / var_tot * 100.0,
        "item_pct": var_item / var_tot * 100.0,
        "theme_pct": var_theme / var_tot * 100.0,
    }

    def g_coef(n_r: int) -> float:
        return var_item / (var_item + var_resid / float(n_r))

    def n_for_g(g_star: float) -> int:
        return int(math.ceil((g_star / (1.0 - g_star)) * (var_resid / var_item)))

    fe = m_full_reml.fe_params
    cov_fe = m_full_reml.cov_params().loc[fe.index, fe.index]
    mm_rows = []
    idx_map = {name: i for i, name in enumerate(fe.index)}
    for cond in ["KG-LLM", "base_LLM"]:
        for dim in ["realism", "relevance", "typicality", "arousal"]:
            vec = np.zeros(len(fe))
            vec[idx_map["Intercept"]] = 1.0
            if cond == "KG-LLM":
                vec[idx_map["C(condition)[T.KG-LLM]"]] = 1.0
            if dim != "realism":
                vec[idx_map[f"C(dimension)[T.{dim}]"]] = 1.0
                if cond == "KG-LLM":
                    vec[idx_map[f"C(condition)[T.KG-LLM]:C(dimension)[T.{dim}]"]] = 1.0
            est = float(vec @ fe.values)
            se = float(math.sqrt(vec @ cov_fe.values @ vec))
            mm_rows.append({"condition": cond, "dimension": dim, "marginal_mean": est, "se": se})

    df_mm = pd.DataFrame(mm_rows)
    df_mm.to_csv(TAB_DIR / "s06_mixed_model_marginal_means.csv", index=False)

    return {
        "b_kg_additive": float(m_add_reml.fe_params["C(condition)[T.KG-LLM]"]),
        "se_kg_additive": float(m_add_reml.bse_fe["C(condition)[T.KG-LLM]"]),
        "b_kg_realism_ref": float(m_full_reml.fe_params["C(condition)[T.KG-LLM]"]),
        "se_kg_realism_ref": float(m_full_reml.bse_fe["C(condition)[T.KG-LLM]"]),
        "lrt_chi2_cond": chi2_cond,
        "lrt_p_cond": p_cond,
        "lrt_chi2_interaction": chi2_int,
        "lrt_p_interaction": p_int,
        "variance_shares": shares,
        "g_5": g_coef(5),
        "g_10": g_coef(10),
        "g_20": g_coef(20),
        "raters_for_g70": n_for_g(0.70),
        "raters_for_g80": n_for_g(0.80),
        "marginal_means": df_mm,
    }


if __name__ == "__main__":
    r = run()
    print(f"LRT cond chi2={r['lrt_chi2_cond']:.2f}, shares={r['variance_shares']}, G(5)={r['g_5']:.3f}")

# -*- coding: utf-8 -*-
"""s12_hierarchical_regressions.py: Standardized OLS regressions for Tables D2, D3, D4, and D5 (§3.3.5-§3.3.6, App. F.8)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_stats import DERIVED_DIR, TAB_DIR
import s00_prepare_data


def _z(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / x.std(ddof=1)


def _fit_std(y, X_dict: dict):
    yz = _z(y)
    X_df = pd.DataFrame({k: _z(v) for k, v in X_dict.items()})
    X_df = sm.add_constant(X_df)
    m = sm.OLS(yz, X_df).fit()
    return m


def _f_change(m_prev, m_curr):
    dr2 = float(m_curr.rsquared - m_prev.rsquared)
    df_num = int(m_curr.df_model - m_prev.df_model)
    df_den = int(m_curr.df_resid)
    f_val = (dr2 / df_num) / ((1.0 - float(m_curr.rsquared)) / df_den)
    p_val = float(stats.f.sf(f_val, df_num, df_den))
    return dr2, f_val, p_val


def run() -> dict:
    if not (DERIVED_DIR / "formal_scale_totals.csv").exists():
        s00_prepare_data.run()
    t = pd.read_csv(DERIVED_DIR / "formal_scale_totals.csv")

    pa, na = t["PA_total"].values, t["NA_total"].values
    rs = t["RS_total"].values
    kg, bs = t["SJT_KG_total"].values, t["SJT_LLM_total"].values
    criteria = {
        "Depression": t["Depression"].values,
        "Anxiety": t["Anxiety"].values,
        "Stress": t["Stress"].values,
    }

    # Table D2: Regressing SJT-KG, SJT-LLM, and 5x5RS onto PA + NA
    d2_rows = []
    for target_nm, y in [("SJT_KG_LLM", kg), ("SJT_LLM", bs), ("5x5RS", rs)]:
        m = _fit_std(y, {"PA": pa, "NA": na})
        d2_rows.append({
            "outcome": target_nm,
            "beta_PA": float(m.params["PA"]), "p_PA": float(m.pvalues["PA"]),
            "beta_NA": float(m.params["NA"]), "p_NA": float(m.pvalues["NA"]),
            "r2": float(m.rsquared), "adj_r2": float(m.rsquared_adj), "p_model": float(m.f_pvalue),
        })
    df_d2 = pd.DataFrame(d2_rows)
    df_d2.to_csv(TAB_DIR / "table_D2_shared_variance_affect.csv", index=False)

    # Table D3: Step 1 (5x5RS) -> Step 2 (+ SJT-LLM) -> Step 3 (+ SJT-KG-LLM)
    d3_rows = []
    for cnm, y in criteria.items():
        m1 = _fit_std(y, {"RS": rs})
        m2 = _fit_std(y, {"RS": rs, "SJT_LLM": bs})
        m3 = _fit_std(y, {"RS": rs, "SJT_LLM": bs, "SJT_KG": kg})
        dr2_2, f2, p2 = _f_change(m1, m2)
        dr2_3, f3, p3 = _f_change(m2, m3)
        d3_rows.append({
            "criterion": cnm,
            "s1_beta_rs": float(m1.params["RS"]), "s1_r2": float(m1.rsquared), "s1_adj_r2": float(m1.rsquared_adj),
            "s2_beta_rs": float(m2.params["RS"]), "s2_beta_llm": float(m2.params["SJT_LLM"]), "s2_p_llm": float(m2.pvalues["SJT_LLM"]),
            "s2_r2": float(m2.rsquared), "s2_adj_r2": float(m2.rsquared_adj), "s2_dr2": dr2_2, "s2_p_dr2": p2,
            "s3_beta_rs": float(m3.params["RS"]), "s3_beta_llm": float(m3.params["SJT_LLM"]), "s3_beta_kg": float(m3.params["SJT_KG"]),
            "s3_r2": float(m3.rsquared), "s3_adj_r2": float(m3.rsquared_adj), "s3_dr2": dr2_3, "s3_p_dr2": p3,
        })
    df_d3 = pd.DataFrame(d3_rows)
    df_d3.to_csv(TAB_DIR / "table_D3_hierarchical_over_5x5rs.csv", index=False)

    # Table D4: Step 1 (PA + NA) -> Step 2 (+ one resilience measure at a time)
    d4_rows = []
    for cnm, y in criteria.items():
        m0 = _fit_std(y, {"PA": pa, "NA": na})
        for rnm, rv in [("SJT_KG_LLM", kg), ("SJT_LLM", bs), ("5x5RS", rs)]:
            m1 = _fit_std(y, {"PA": pa, "NA": na, "RES": rv})
            dr2, fval, pval = _f_change(m0, m1)
            d4_rows.append({
                "criterion": cnm, "measure": rnm,
                "beta_PA": float(m1.params["PA"]), "p_PA": float(m1.pvalues["PA"]),
                "beta_NA": float(m1.params["NA"]), "p_NA": float(m1.pvalues["NA"]),
                "beta_measure": float(m1.params["RES"]), "p_measure": float(m1.pvalues["RES"]),
                "r2": float(m1.rsquared), "adj_r2": float(m1.rsquared_adj),
                "dr2": dr2, "p_dr2": pval,
            })
    df_d4 = pd.DataFrame(d4_rows)
    df_d4.to_csv(TAB_DIR / "table_D4_incremental_over_pana.csv", index=False)

    # Table D5: Step 1 (PA + NA + 5x5RS) -> Step 2 (+ SJT-LLM) -> Step 3 (+ SJT-KG-LLM)
    d5_rows = []
    for cnm, y in criteria.items():
        m1 = _fit_std(y, {"PA": pa, "NA": na, "RS": rs})
        m2 = _fit_std(y, {"PA": pa, "NA": na, "RS": rs, "SJT_LLM": bs})
        m3 = _fit_std(y, {"PA": pa, "NA": na, "RS": rs, "SJT_LLM": bs, "SJT_KG": kg})
        dr2_2, _, p2 = _f_change(m1, m2)
        dr2_3, _, p3 = _f_change(m2, m3)
        d5_rows.append({
            "criterion": cnm,
            "s1_r2": float(m1.rsquared), "s1_adj_r2": float(m1.rsquared_adj),
            "s2_beta_llm": float(m2.params["SJT_LLM"]), "s2_p_llm": float(m2.pvalues["SJT_LLM"]),
            "s2_r2": float(m2.rsquared), "s2_adj_r2": float(m2.rsquared_adj), "s2_dr2": dr2_2, "s2_p_dr2": p2,
            "s3_beta_llm": float(m3.params["SJT_LLM"]), "s3_beta_kg": float(m3.params["SJT_KG"]),
            "s3_r2": float(m3.rsquared), "s3_adj_r2": float(m3.rsquared_adj), "s3_dr2": dr2_3, "s3_p_dr2": p3,
        })
    df_d5 = pd.DataFrame(d5_rows)
    df_d5.to_csv(TAB_DIR / "table_D5_hierarchical_full.csv", index=False)

    return {"table_d2": df_d2, "table_d3": df_d3, "table_d4": df_d4, "table_d5": df_d5}


if __name__ == "__main__":
    res = run()
    print(res["table_d3"][["criterion", "s2_dr2", "s2_p_dr2", "s3_dr2", "s3_p_dr2"]].to_string(index=False))

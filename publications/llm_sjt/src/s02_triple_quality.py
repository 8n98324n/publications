# -*- coding: utf-8 -*-
"""s02_triple_quality.py: Expert evaluation of 100 matched ReBEL vs. LLM triple pairs (H2; §3.1.1, App. D.1-D.2, F.2)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_stats import SEED, RAW_DIR, TAB_DIR, icc3k_anova, gwet_ac2_ordinal, paired_t_dz, bh_fdr

DIMS = ["correctness", "relevance", "completeness", "clarity_conciseness"]


def run(n_boot: int = 2000) -> dict:
    df = pd.read_csv(RAW_DIR / "triple_expert_ratings.csv")
    rng = np.random.default_rng(SEED)

    agr_rows = []
    comp_rows = []

    for dim in DIMS:
        sub = df[df["dimension"] == dim]
        piv_llm = sub[sub["model"] == "LLM"].pivot(index="sentence_id", columns="expert_code", values="rating").sort_index()
        piv_reb = sub[sub["model"] == "ReBEL"].pivot(index="sentence_id", columns="expert_code", values="rating").sort_index()
        X_llm = piv_llm.values
        X_reb = piv_reb.values

        for model_name, X in [("LLM", X_llm), ("ReBEL", X_reb)]:
            icc_res = icc3k_anova(X)
            ac2_val = gwet_ac2_ordinal(X)
            boot_ac2 = np.empty(n_boot)
            n_s = X.shape[0]
            for b in range(n_boot):
                idx = rng.integers(0, n_s, size=n_s)
                boot_ac2[b] = gwet_ac2_ordinal(X[idx])
            agr_rows.append({
                "model": model_name,
                "dimension": dim,
                "icc3k": icc_res["icc3k"],
                "icc_ci_low": icc_res["ci_low"],
                "icc_ci_high": icc_res["ci_high"],
                "F": icc_res["F"],
                "df1": icc_res["df1"],
                "df2": icc_res["df2"],
                "p_icc": icc_res["p"],
                "ac2": ac2_val,
                "ac2_ci_low": float(np.percentile(boot_ac2, 2.5)),
                "ac2_ci_high": float(np.percentile(boot_ac2, 97.5)),
            })

        m_llm = X_llm.mean(axis=1)
        m_reb = X_reb.mean(axis=1)
        pt = paired_t_dz(m_llm, m_reb)
        win_rate = float(np.mean(m_llm > m_reb) * 100.0)
        comp_rows.append({
            "dimension": dim,
            "n_pairs": len(m_llm),
            "llm_mean": pt["m1"],
            "llm_sd": pt["sd1"],
            "rebel_mean": pt["m2"],
            "rebel_sd": pt["sd2"],
            "mean_diff": pt["mean_diff"],
            "t": pt["t"],
            "df": pt["df"],
            "p": pt["p"],
            "cohen_dz": pt["dz"],
            "llm_win_pct": win_rate,
        })

    df_agr = pd.DataFrame(agr_rows)
    df_comp = pd.DataFrame(comp_rows)
    df_comp["p_bh"] = bh_fdr(df_comp["p"].values)
    df_comp["p_bonferroni"] = np.minimum(1.0, df_comp["p"].values * len(df_comp))

    df_agr.to_csv(TAB_DIR / "s02_triple_agreement.csv", index=False)
    df_comp.to_csv(TAB_DIR / "s02_triple_paired_tests.csv", index=False)
    return {"agreement": df_agr, "paired_tests": df_comp}


if __name__ == "__main__":
    res = run()
    print(res["paired_tests"].to_string(index=False))

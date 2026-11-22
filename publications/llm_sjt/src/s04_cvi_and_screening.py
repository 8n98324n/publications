# -*- coding: utf-8 -*-
"""s04_cvi_and_screening.py: I-CVI, S-CVI/Ave, 70-rating screening + jackknife, option ANOVA & rationale citations (§3.2.1.4, §3.2.2, F.5)."""
from __future__ import annotations
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_stats import RAW_DIR, TAB_DIR, icc3k_anova, gwet_ac2_ordinal, welch_t_hedges_g

SCEN_DIMS = ["realism", "relevance", "typicality", "arousal"]
STRATEGIES = ["Adaptability", "Emotion_Regulation", "Optimism", "Self_Efficacy", "Social_Support"]
OPT_DIMS = ["Typicality", "Distinctiveness"]


def run() -> dict:
    wide = pd.read_csv(RAW_DIR / "expert_scenario_ratings.csv")
    opt_long = pd.read_csv(RAW_DIR / "expert_option_spss_long.csv")
    agg = pd.read_csv(RAW_DIR / "item_level_expert_aggregates.csv")
    scen_texts = pd.read_csv(RAW_DIR / "scenario_texts.csv")

    # 1. Scenario I-CVI and S-CVI/Ave (§3.2.1.4, Table D6a)
    scvi_rows = []
    icvi_kg_mat = np.zeros((30, 4))
    icvi_bs_mat = np.zeros((30, 4))
    for d_idx, dim in enumerate(SCEN_DIMS):
        X_kg = wide[[f"item{i}_{dim}" for i in range(1, 31)]].values.T
        X_bs = wide[[f"item{i}_{dim}" for i in range(31, 61)]].values.T
        ic_kg = np.mean(X_kg >= 4, axis=1)
        ic_bs = np.mean(X_bs >= 4, axis=1)
        icvi_kg_mat[:, d_idx] = ic_kg
        icvi_bs_mat[:, d_idx] = ic_bs
        scvi_rows.append({
            "dimension": dim,
            "scvi_ave_kg": float(ic_kg.mean()),
            "scvi_ave_base": float(ic_bs.mean()),
        })
    scvi_rows.append({
        "dimension": "overall",
        "scvi_ave_kg": float(icvi_kg_mat.mean()),
        "scvi_ave_base": float(icvi_bs_mat.mean()),
    })
    df_scvi = pd.DataFrame(scvi_rows)
    df_scvi.to_csv(TAB_DIR / "s04_scenario_cvi.csv", index=False)

    item_icvi_kg = icvi_kg_mat.mean(axis=1)
    item_icvi_bs = icvi_bs_mat.mean(axis=1)
    mw_u, mw_p = stats.mannwhitneyu(item_icvi_kg, item_icvi_bs, alternative="two-sided")

    # 2. Item-level comparisons & 30-to-20 screening selectivity (Table D6b)
    kg_agg = agg[agg["group"] == "KG-LLM"].sort_values("item_num")
    bs_agg = agg[agg["group"] == "base_LLM"].sort_values("item_num")

    wt_scen_60 = welch_t_hedges_g(kg_agg["scenario_mean"], bs_agg["scenario_mean"])
    wt_opt_60 = welch_t_hedges_g(kg_agg["option_mean"], bs_agg["option_mean"])
    wt_comp_60 = welch_t_hedges_g(kg_agg["Total_Mean"], bs_agg["Total_Mean"])

    wt_kg_ret_scen = welch_t_hedges_g(
        kg_agg[kg_agg["retained"] == 1]["scenario_mean"],
        kg_agg[kg_agg["retained"] == 0]["scenario_mean"],
    )
    wt_kg_ret_opt = welch_t_hedges_g(
        kg_agg[kg_agg["retained"] == 1]["option_mean"],
        kg_agg[kg_agg["retained"] == 0]["option_mean"],
    )
    wt_bs_ret_scen = welch_t_hedges_g(
        bs_agg[bs_agg["retained"] == 1]["scenario_mean"],
        bs_agg[bs_agg["retained"] == 0]["scenario_mean"],
    )
    wt_bs_ret_opt = welch_t_hedges_g(
        bs_agg[bs_agg["retained"] == 1]["option_mean"],
        bs_agg[bs_agg["retained"] == 0]["option_mean"],
    )

    # Leave-one-rater-out jackknife overlap on 70-rating Total_Score (§2.1.4.3)
    raters = list(wide["rater_code"])
    full_ret_set = set(agg[agg["retained"] == 1]["item_num"])
    jk_kg, jk_bs, jk_all = [], [], []
    for drop_r in raters:
        kept_scen = wide[wide["rater_code"] != drop_r]
        kept_opt = opt_long[opt_long["expert_code"] != drop_r]
        opt_by_item = kept_opt.groupby("item")["score"].apply(np.array).to_dict()
        jk_rows = []
        for it in range(1, 61):
            s_vals = kept_scen[[f"item{it}_{d}" for d in SCEN_DIMS]].values.ravel().astype(float)
            o_vals = opt_by_item[it].astype(float)
            v56 = np.concatenate([s_vals, o_vals])
            tm = float(v56.mean())
            tsd = float(v56.std(ddof=0))
            jk_rows.append({
                "item_num": it,
                "group": "KG-LLM" if it <= 30 else "base_LLM",
                "stress_theme": agg.loc[agg["item_num"] == it, "stress_theme"].values[0],
                "score": tm - 0.5 * tsd,
            })
        jdf = pd.DataFrame(jk_rows)
        sel_jk = set()
        for grp in ["KG-LLM", "base_LLM"]:
            for th in jdf["stress_theme"].unique():
                top4 = jdf[(jdf["group"] == grp) & (jdf["stress_theme"] == th)].sort_values(
                    ["score", "item_num"], ascending=[False, True]
                ).head(4)
                sel_jk.update(top4["item_num"].tolist())
        ov_kg = len({i for i in sel_jk if i <= 30} & {i for i in full_ret_set if i <= 30}) / 20.0
        ov_bs = len({i for i in sel_jk if i > 30} & {i for i in full_ret_set if i > 30}) / 20.0
        jk_kg.append(ov_kg); jk_bs.append(ov_bs); jk_all.append(0.5 * (ov_kg + ov_bs))

    # 3. Per-rater option quality Table D2b (3,000 ratings; §3.2.2.1)
    opt_table_rows = []
    for strat in STRATEGIES:
        for odim in OPT_DIMS:
            sub = opt_long[(opt_long["strategy"] == strat) & (opt_long["dimension"] == odim)]
            piv = sub.pivot(index="item", columns="expert_code", values="score").sort_index()
            X_kg = piv.loc[1:30].values.astype(float)
            X_bs = piv.loc[31:60].values.astype(float)
            a_kg = icc3k_anova(X_kg)
            a_bs = icc3k_anova(X_bs)
            wt = welch_t_hedges_g(X_kg.mean(axis=1), X_bs.mean(axis=1))
            opt_table_rows.append({
                "strategy": strat,
                "dimension": odim,
                "ms_error_kg": a_kg["ms_error"],
                "ms_error_base": a_bs["ms_error"],
                "mse_ratio": a_kg["ms_error"] / a_bs["ms_error"],
                "icc_kg": a_kg["icc3k"],
                "icc_base": a_bs["icc3k"],
                "ac2_kg": gwet_ac2_ordinal(X_kg),
                "ac2_base": gwet_ac2_ordinal(X_bs),
                "kg_mean": wt["m1"],
                "kg_sd": wt["sd1"],
                "base_mean": wt["m2"],
                "base_sd": wt["sd2"],
                "welch_t": wt["t"],
                "welch_p": wt["p"],
                "hedges_g": wt["hedges_g"],
                "scvi_kg": float(np.mean(X_kg >= 4)),
                "scvi_base": float(np.mean(X_bs >= 4)),
                "base_ceil_45_pct": float(np.mean(X_bs >= 4) * 100.0),
            })

    # Composite option typicality and distinctiveness across 5 strategies
    for odim in OPT_DIMS:
        sub = opt_long[opt_long["dimension"] == odim]
        item_m = sub.groupby("item")["score"].mean()
        item_scvi = sub.assign(ge4=sub["score"] >= 4).groupby("item")["ge4"].mean()
        wt = welch_t_hedges_g(item_m.loc[1:30].values, item_m.loc[31:60].values)
        opt_table_rows.append({
            "strategy": "Composite",
            "dimension": odim,
            "ms_error_kg": np.nan, "ms_error_base": np.nan, "mse_ratio": np.nan,
            "icc_kg": np.nan, "icc_base": np.nan, "ac2_kg": np.nan, "ac2_base": np.nan,
            "kg_mean": wt["m1"], "kg_sd": wt["sd1"],
            "base_mean": wt["m2"], "base_sd": wt["sd2"],
            "welch_t": wt["t"], "welch_p": wt["p"], "hedges_g": wt["hedges_g"],
            "scvi_kg": float(item_scvi.loc[1:30].mean()),
            "scvi_base": float(item_scvi.loc[31:60].mean()),
            "base_ceil_45_pct": np.nan,
        })
    df_opt_d2b = pd.DataFrame(opt_table_rows)
    df_opt_d2b.to_csv(TAB_DIR / "table_D2b_option_quality.csv", index=False)

    # 4. Clause-level rationale citation rates by strategy (Appendix G.2 & Figure 4d)
    kg_texts = scen_texts[scen_texts["group"] == "KG-LLM"].sort_values("within_group_id")
    letter_to_strat = dict(zip(["A", "B", "C", "D", "E"], STRATEGIES))
    cit_counts = {s: 0 for s in STRATEGIES}
    for rat in kg_texts["reasoning_cn"]:
        clauses = re.split(r"[。；;！!\n]+", str(rat))
        for L, sname in letter_to_strat.items():
            if any(
                re.search(rf"(选项\s*{L}|（{L}）|\({L}\)|(?<![A-Za-z]){L}(?![A-Za-z]))", cl)
                and re.search(r"\[[0-9,\s]+\]", cl)
                for cl in clauses
            ):
                cit_counts[sname] += 1
    rationale_cit_pct = {s: cit_counts[s] / 30.0 * 100.0 for s in STRATEGIES}

    return {
        "scvi_table": df_scvi,
        "icvi_mannwhitney_u": float(mw_u),
        "icvi_mannwhitney_p": float(mw_p),
        "wt_scen_60": wt_scen_60,
        "wt_opt_60": wt_opt_60,
        "wt_comp_60": wt_comp_60,
        "kg_ret20_total_score_mean": float(kg_agg[kg_agg["retained"] == 1]["Total_Score"].mean()),
        "kg_ret20_total_mean": float(kg_agg[kg_agg["retained"] == 1]["Total_Mean"].mean()),
        "bs_ret20_total_score_mean": float(bs_agg[bs_agg["retained"] == 1]["Total_Score"].mean()),
        "bs_ret20_total_mean": float(bs_agg[bs_agg["retained"] == 1]["Total_Mean"].mean()),
        "wt_kg_ret_scen": wt_kg_ret_scen,
        "wt_kg_ret_opt": wt_kg_ret_opt,
        "wt_bs_ret_scen": wt_bs_ret_scen,
        "wt_bs_ret_opt": wt_bs_ret_opt,
        "jackknife_overlap_overall_pct": float(np.mean(jk_all) * 100.0),
        "jackknife_overlap_kg_pct": float(np.mean(jk_kg) * 100.0),
        "jackknife_overlap_base_pct": float(np.mean(jk_bs) * 100.0),
        "option_table_d2b": df_opt_d2b,
        "rationale_citation_pct": rationale_cit_pct,
    }


if __name__ == "__main__":
    res = run()
    print(f"Mann-Whitney U={res['icvi_mannwhitney_u']}, Jackknife={res['jackknife_overlap_overall_pct']:.1f}%")

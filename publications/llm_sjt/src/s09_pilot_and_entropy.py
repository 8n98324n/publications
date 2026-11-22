# -*- coding: utf-8 -*-
"""s09_pilot_and_entropy.py: Pilot consensus, item provenance, pilot-to-formal TVD, choice profiles & entropy (§3.3.1, §3.3.3.3, F.7, F.9)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_stats import (
    SEED, RAW_DIR, DERIVED_DIR, OUT_DIR, TAB_DIR,
    welch_t_hedges_g, paired_t_dz, shannon_entropy_norm,
)
import s00_prepare_data

STRATEGIES = ["Adaptability", "Emotion_Regulation", "Optimism", "Self_Efficacy", "Social_Support"]


def run(n_null_reps: int = 2000) -> dict:
    if not (DERIVED_DIR / "formal_scale_totals.csv").exists():
        s00_prepare_data.run()

    pilot_a = pd.read_csv(RAW_DIR / "pilot_A_kg_llm_responses.csv")
    pilot_b = pd.read_csv(RAW_DIR / "pilot_B_llm_responses.csv")
    key_df = pd.read_csv(RAW_DIR / "scoring_key.csv")
    agg_df = pd.read_csv(RAW_DIR / "item_level_expert_aggregates.csv")
    formal_raw = pd.read_csv(RAW_DIR / "formal_administration_responses.csv")
    chosen_strat = pd.read_csv(DERIVED_DIR / "formal_chosen_strategies.csv")
    totals = pd.read_csv(DERIVED_DIR / "formal_scale_totals.csv")
    rng = np.random.default_rng(SEED)

    # 1. Item provenance mapping (outputs/s09_item_provenance.csv; Appendix F.9)
    prov = (
        key_df[["formal_item_id", "formal_col", "condition", "source_scenario", "stress_theme", "judgment_type"]]
        .drop_duplicates()
        .sort_values("formal_item_id")
        .reset_index(drop=True)
    )
    prov.to_csv(OUT_DIR / "s09_item_provenance.csv", index=False)

    # Verify exact match with top-4-per-theme screened scenarios
    screen_kg = sorted(agg_df[(agg_df["group"] == "KG-LLM") & (agg_df["retained"] == 1)]["within_group_id"].tolist())
    screen_bs = sorted(agg_df[(agg_df["group"] == "base_LLM") & (agg_df["retained"] == 1)]["within_group_id"].tolist())
    formal_kg_scen = sorted(prov[prov["condition"] == "KG-LLM"]["source_scenario"].unique().tolist())
    formal_bs_scen = sorted(prov[prov["condition"] == "base_LLM"]["source_scenario"].unique().tolist())
    provenance_exact_match = (screen_kg == formal_kg_scen) and (screen_bs == formal_bs_scen)

    # 2. Pilot 30 x 5 proportion matrices (Figures D1 & D2)
    def pilot_prop_mats(pdf: pd.DataFrame):
        ml_mat = np.zeros((30, 5))
        ll_mat = np.zeros((30, 5))
        for s in range(1, 31):
            for o in range(1, 6):
                ml_mat[s - 1, o - 1] = np.mean(pdf[f"S{s}_ML"].values == o)
                ll_mat[s - 1, o - 1] = np.mean(pdf[f"S{s}_LL"].values == o)
        return ml_mat, ll_mat

    pa_ml, pa_ll = pilot_prop_mats(pilot_a)
    pb_ml, pb_ll = pilot_prop_mats(pilot_b)

    # 3. Item-level formal option entropy & pilot-to-formal TVD (Table D6e, Figure 10b)
    n_f = len(formal_raw)
    item_ent_kg, item_ent_bs = [], []
    tvd_ml_kg, tvd_ll_kg = [], []
    tvd_ml_bs, tvd_ll_bs = [], []
    pooled_probs_kg, pooled_probs_bs = [], []

    for i in range(1, 81):
        sub_k = key_df[key_df["formal_item_id"] == i].sort_values("formal_option_pos")
        col = sub_k["formal_col"].iloc[0]
        cond = sub_k["condition"].iloc[0]
        jt = sub_k["judgment_type"].iloc[0]
        p_vec = sub_k["pilot_proportion"].values
        p_cnt = sub_k["pilot_count"].values
        n_p = 50 if cond == "KG-LLM" else 52

        ch = formal_raw[col].values
        f_cnt = np.array([np.sum(ch == o) for o in range(1, 6)], dtype=float)
        f_vec = f_cnt / float(n_f)

        ent = shannon_entropy_norm(f_vec)
        tvd = 0.5 * float(np.sum(np.abs(p_vec - f_vec)))
        pooled_p = (p_cnt + f_cnt) / float(n_p + n_f)

        if cond == "KG-LLM":
            item_ent_kg.append(ent)
            pooled_probs_kg.append(pooled_p)
            (tvd_ml_kg if jt == "ML" else tvd_ll_kg).append(tvd)
        else:
            item_ent_bs.append(ent)
            pooled_probs_bs.append(pooled_p)
            (tvd_ml_bs if jt == "ML" else tvd_ll_bs).append(tvd)

    mw_item_ent = stats.mannwhitneyu(item_ent_kg, item_ent_bs, alternative="two-sided")
    wt_item_ent = welch_t_hedges_g(item_ent_kg, item_ent_bs)
    wt_tvd_ml = welch_t_hedges_g(tvd_ml_kg, tvd_ml_bs)
    wt_tvd_ll = welch_t_hedges_g(tvd_ll_kg, tvd_ll_bs)

    # Multinomial finite-sample null TVD benchmark
    null_tvd_kg = []
    null_tvd_bs = []
    for pvec in pooled_probs_kg:
        d1 = rng.multinomial(50, pvec, size=n_null_reps) / 50.0
        d2 = rng.multinomial(n_f, pvec, size=n_null_reps) / float(n_f)
        null_tvd_kg.append(float(np.mean(0.5 * np.sum(np.abs(d1 - d2), axis=1))))
    for pvec in pooled_probs_bs:
        d1 = rng.multinomial(52, pvec, size=n_null_reps) / 52.0
        d2 = rng.multinomial(n_f, pvec, size=n_null_reps) / float(n_f)
        null_tvd_bs.append(float(np.mean(0.5 * np.sum(np.abs(d1 - d2), axis=1))))

    # 4. Coping-strategy choice shares & within-person ML strategy entropy (Table D6f, Figure 11b)
    ml_cols_all = [f"LLM_KG{i}" for i in range(1, 41, 2)] + [f"LLM{i}" for i in range(1, 41, 2)]
    ll_cols_all = [f"LLM_KG{i}" for i in range(2, 41, 2)] + [f"LLM{i}" for i in range(2, 41, 2)]
    ml_cols_kg = [f"LLM_KG{i}" for i in range(1, 41, 2)]
    ml_cols_bs = [f"LLM{i}" for i in range(1, 41, 2)]

    ml_mat_str = chosen_strat[ml_cols_all].values
    ll_mat_str = chosen_strat[ll_cols_all].values

    person_ml_shares = {s: np.mean(ml_mat_str == s, axis=1) * 100.0 for s in STRATEGIES}
    person_ll_shares = {s: np.mean(ll_mat_str == s, axis=1) * 100.0 for s in STRATEGIES}

    choice_table = pd.DataFrame([
        {
            "strategy": s,
            "ml_mean_pct": float(np.mean(person_ml_shares[s])),
            "ml_sd_pct": float(np.std(person_ml_shares[s], ddof=1)),
            "ll_mean_pct": float(np.mean(person_ll_shares[s])),
            "ll_sd_pct": float(np.std(person_ll_shares[s], ddof=1)),
            "r_ml_5x5rs": float(stats.pearsonr(person_ml_shares[s], totals["RS_total"]).statistic),
            "r_ml_pa": float(stats.pearsonr(person_ml_shares[s], totals["PA_total"]).statistic),
            "r_ml_dep": float(stats.pearsonr(person_ml_shares[s], totals["Depression"]).statistic),
            "r_ll_5x5rs": float(stats.pearsonr(person_ll_shares[s], totals["RS_total"]).statistic),
        }
        for s in STRATEGIES
    ])
    choice_table.to_csv(TAB_DIR / "table_D6f_coping_choices.csv", index=False)

    r_ll_ss_subscale = float(stats.pearsonr(person_ll_shares["Social_Support"], totals["RS_SocialSupport"]).statistic)

    # Person-level strategy entropy across 20 ML scenarios
    pers_ent_kg = np.array([
        shannon_entropy_norm([np.mean(row == s) for s in STRATEGIES])
        for row in chosen_strat[ml_cols_kg].values
    ])
    pers_ent_bs = np.array([
        shannon_entropy_norm([np.mean(row == s) for s in STRATEGIES])
        for row in chosen_strat[ml_cols_bs].values
    ])
    pt_pers_ent = paired_t_dz(pers_ent_kg, pers_ent_bs)

    return {
        "provenance_exact_match": provenance_exact_match,
        "screened_kg_scenarios": screen_kg,
        "screened_base_scenarios": screen_bs,
        "pilot_a_ml": pa_ml, "pilot_a_ll": pa_ll,
        "pilot_b_ml": pb_ml, "pilot_b_ll": pb_ll,
        "pilot_a_ml_modal_mean": float(pa_ml.max(axis=1).mean()),
        "pilot_a_ll_modal_mean": float(pa_ll.max(axis=1).mean()),
        "pilot_b_ml_modal_mean": float(pb_ml.max(axis=1).mean()),
        "pilot_b_ll_modal_mean": float(pb_ll.max(axis=1).mean()),
        "item_entropy_kg_mean": float(np.mean(item_ent_kg)),
        "item_entropy_kg_sd": float(np.std(item_ent_kg, ddof=1)),
        "item_entropy_base_mean": float(np.mean(item_ent_bs)),
        "item_entropy_base_sd": float(np.std(item_ent_bs, ddof=1)),
        "item_entropy_mw_u": float(mw_item_ent.statistic),
        "item_entropy_mw_p": float(mw_item_ent.pvalue),
        "item_entropy_welch": wt_item_ent,
        "tvd_ml_kg": np.array(tvd_ml_kg),
        "tvd_ml_base": np.array(tvd_ml_bs),
        "tvd_ll_kg": np.array(tvd_ll_kg),
        "tvd_ll_base": np.array(tvd_ll_bs),
        "wt_tvd_ml": wt_tvd_ml,
        "wt_tvd_ll": wt_tvd_ll,
        "null_tvd_kg_mean": float(np.mean(null_tvd_kg)),
        "null_tvd_base_mean": float(np.mean(null_tvd_bs)),
        "choice_table": choice_table,
        "r_ll_ss_subscale": r_ll_ss_subscale,
        "person_ml_entropy_paired": pt_pers_ent,
    }


if __name__ == "__main__":
    res = run()
    print("Provenance match:", res["provenance_exact_match"], "| TVD ML g:", round(res["wt_tvd_ml"]["hedges_g"], 2))

# -*- coding: utf-8 -*-
"""s00_prepare_data.py: Reshape raw datasets, verify pilot keys, score SJT and criterion scales (Appendix F.0)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_stats import RAW_DIR, DERIVED_DIR

REVERSE_5X5 = {3, 5, 6, 9, 10, 11, 12, 14, 15, 18, 22, 23, 25}


def run() -> dict:
    # 1. Reshape expert scenario ratings (5 x 240 -> 1,200 long rows)
    scen_wide = pd.read_csv(RAW_DIR / "expert_scenario_ratings.csv")
    scen_texts = pd.read_csv(RAW_DIR / "scenario_texts.csv")
    theme_map = dict(zip(scen_texts["item_num"], scen_texts["stress_theme"]))

    long_rows = []
    dims = ["realism", "relevance", "typicality", "arousal"]
    for _, r in scen_wide.iterrows():
        rc = r["rater_code"]
        for it in range(1, 61):
            grp = "KG-LLM" if it <= 30 else "base_LLM"
            for d in dims:
                long_rows.append({
                    "rater_code": rc,
                    "item_num": it,
                    "within_group_id": ((it - 1) % 30) + 1,
                    "condition": grp,
                    "stress_theme": theme_map[it],
                    "dimension": d,
                    "rating": float(r[f"item{it}_{d}"]),
                })
    df_scen_long = pd.DataFrame(long_rows)
    df_scen_long.to_csv(DERIVED_DIR / "scenario_ratings_long.csv", index=False)

    # 2. Verify pilot proportions against scoring_key.csv
    pilot_a = pd.read_csv(RAW_DIR / "pilot_A_kg_llm_responses.csv")
    pilot_b = pd.read_csv(RAW_DIR / "pilot_B_llm_responses.csv")
    key_df = pd.read_csv(RAW_DIR / "scoring_key.csv")

    recomp_props = []
    for _, kr in key_df.iterrows():
        sc = int(kr["source_scenario"])
        jt = kr["judgment_type"]
        p_pos = int(kr["pilot_option_pos"])
        col = f"S{sc}_{jt}"
        pdf = pilot_a if kr["condition"] == "KG-LLM" else pilot_b
        prop = float(np.mean(pdf[col].values == p_pos))
        recomp_props.append(prop)
    max_key_err = float(np.max(np.abs(np.array(recomp_props) - key_df["pilot_proportion"].values)))

    # 3. Score formal SJT items & criterion scales (N = 150)
    formal_raw = pd.read_csv(RAW_DIR / "formal_administration_responses.csv")
    key_lookup = {
        (r["formal_col"], int(r["formal_option_pos"])): float(r["pilot_proportion"])
        for _, r in key_df.iterrows()
    }
    strat_lookup = {
        (r["formal_col"], int(r["formal_option_pos"])): r["coping_strategy"]
        for _, r in key_df.iterrows()
    }

    sjt_cols_kg = [f"LLM_KG{i}" for i in range(1, 41)]
    sjt_cols_bs = [f"LLM{i}" for i in range(1, 41)]
    all_sjt_cols = sjt_cols_kg + sjt_cols_bs

    scored_data = {
        "participant_code": formal_raw["participant_code"],
        "gender": formal_raw["gender"],
        "age_bracket": formal_raw["age_bracket"],
        "duration_sec": formal_raw["duration_sec"],
    }
    strat_data = {"participant_code": formal_raw["participant_code"]}

    for col in all_sjt_cols:
        choices = formal_raw[col].astype(int).values
        scored_data[col] = [key_lookup[(col, c)] for c in choices]
        strat_data[col] = [strat_lookup[(col, c)] for c in choices]

    # Reverse-key 5x5RS items and compute 5 facets + total
    res_scored = {}
    for i in range(1, 26):
        c = f"5x5resilience{i}"
        v = formal_raw[c].astype(float).values
        res_scored[c] = (6.0 - v) if i in REVERSE_5X5 else v
        scored_data[c] = res_scored[c]

    pa_cols = [c for c in formal_raw.columns if c.startswith("PA")]
    na_cols = [c for c in formal_raw.columns if c.startswith("NA")]
    d_cols = [c for c in formal_raw.columns if c.startswith("D") and c[1:].isdigit()]
    a_cols = [c for c in formal_raw.columns if c.startswith("A") and c[1:].isdigit()]
    s_cols = [c for c in formal_raw.columns if c.startswith("S") and c[1:].isdigit()]

    for col in pa_cols + na_cols + d_cols + a_cols + s_cols:
        scored_data[col] = formal_raw[col].astype(float).values

    df_item_scores = pd.DataFrame(scored_data)
    df_item_scores.to_csv(DERIVED_DIR / "formal_item_scores.csv", index=False)
    pd.DataFrame(strat_data).to_csv(DERIVED_DIR / "formal_chosen_strategies.csv", index=False)

    # Scale totals (DASS-21 doubled to standard 0-42 range per Appendix F.0)
    totals = pd.DataFrame({
        "participant_code": formal_raw["participant_code"],
        "gender": formal_raw["gender"],
        "age_bracket": formal_raw["age_bracket"],
        "duration_sec": formal_raw["duration_sec"],
        "SJT_KG_total": df_item_scores[sjt_cols_kg].sum(axis=1),
        "SJT_LLM_total": df_item_scores[sjt_cols_bs].sum(axis=1),
        "SJT_KG_ML": df_item_scores[sjt_cols_kg[0::2]].sum(axis=1),
        "SJT_KG_LL": df_item_scores[sjt_cols_kg[1::2]].sum(axis=1),
        "SJT_LLM_ML": df_item_scores[sjt_cols_bs[0::2]].sum(axis=1),
        "SJT_LLM_LL": df_item_scores[sjt_cols_bs[1::2]].sum(axis=1),
        "RS_total": sum(res_scored[f"5x5resilience{i}"] for i in range(1, 26)),
        "RS_Adaptability": sum(res_scored[f"5x5resilience{i}"] for i in range(1, 6)),
        "RS_EmotionReg": sum(res_scored[f"5x5resilience{i}"] for i in range(6, 11)),
        "RS_Optimism": sum(res_scored[f"5x5resilience{i}"] for i in range(11, 16)),
        "RS_SelfEfficacy": sum(res_scored[f"5x5resilience{i}"] for i in range(16, 21)),
        "RS_SocialSupport": sum(res_scored[f"5x5resilience{i}"] for i in range(21, 26)),
        "PA_total": df_item_scores[pa_cols].sum(axis=1),
        "NA_total": df_item_scores[na_cols].sum(axis=1),
        "Depression": 2.0 * df_item_scores[d_cols].sum(axis=1),
        "Anxiety": 2.0 * df_item_scores[a_cols].sum(axis=1),
        "Stress": 2.0 * df_item_scores[s_cols].sum(axis=1),
    })
    totals.to_csv(DERIVED_DIR / "formal_scale_totals.csv", index=False)

    summary = {
        "n_scenario_ratings_long": len(df_scen_long),
        "n_pilot_a": len(pilot_a),
        "n_pilot_b": len(pilot_b),
        "pilot_a_median_min": float(np.median(pilot_a["duration_sec"]) / 60.0),
        "pilot_b_median_min": float(np.median(pilot_b["duration_sec"]) / 60.0),
        "n_formal": len(totals),
        "formal_female_n": int((totals["gender"] == "Female").sum()),
        "formal_male_n": int((totals["gender"] == "Male").sum()),
        "formal_mean_min": float(np.mean(totals["duration_sec"]) / 60.0),
        "formal_median_min": float(np.median(totals["duration_sec"]) / 60.0),
        "max_scoring_key_diff": max_key_err,
    }
    return summary


if __name__ == "__main__":
    res = run()
    print("s00_prepare_data complete:", res)

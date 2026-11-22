# -*- coding: utf-8 -*-
"""s10_scenario_length.py: Scenario and option character length comparisons (§4.2, Appendix F.3)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_stats import RAW_DIR, TAB_DIR, welch_t_hedges_g


def run() -> dict:
    df = pd.read_csv(RAW_DIR / "scenario_texts.csv")
    df["scen_chars"] = df["scenario_text"].astype(str).str.strip().str.len()
    opt_cols = ["option_A", "option_B", "option_C", "option_D", "option_E"]
    df["opt_mean_chars"] = np.mean([df[c].astype(str).str.strip().str.len() for c in opt_cols], axis=0)

    kg = df[df["group"] == "KG-LLM"]
    bs = df[df["group"] == "base_LLM"]
    wt_scen = welch_t_hedges_g(kg["scen_chars"].values, bs["scen_chars"].values)
    wt_opt = welch_t_hedges_g(kg["opt_mean_chars"].values, bs["opt_mean_chars"].values)

    out_df = pd.DataFrame([
        {"feature": "scenario_char_len", **wt_scen},
        {"feature": "option_mean_char_len", **wt_opt},
    ])
    out_df.to_csv(TAB_DIR / "s10_length_comparison.csv", index=False)
    return {"scen_len": wt_scen, "opt_len": wt_opt, "table": out_df}


if __name__ == "__main__":
    print(run()["table"].to_string(index=False))

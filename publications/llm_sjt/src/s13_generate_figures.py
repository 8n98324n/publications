# -*- coding: utf-8 -*-
"""s13_generate_figures.py: Generates all 19 publication-grade 300-DPI figures (Figures 1-13, C1-C2, D1-D4)."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_stats import RAW_DIR, FIG_DIR
import s00_prepare_data
import s01_kg_network
import s02_triple_quality
import s03_scenario_quality
import s04_cvi_and_screening
import s05_rater_error
import s06_mixed_model
import s07_reliability
import s08_validity
import s09_pilot_and_entropy
import fig_part1
import fig_part2


FIGURE_FILES = [
    ("Figure 1", "fig01_pipeline_overview.png"),
    ("Figure 2", "fig02_triple_evaluation.png"),
    ("Figure 3", "fig03_kg_core_subgraph.png"),
    ("Figure 4", "fig04_kg_content_and_citations.png"),
    ("Figure 5", "fig05_scenario_evaluation.png"),
    ("Figure 6", "fig06_cvi_error_and_screening.png"),
    ("Figure 7", "fig07_screening_and_options.png"),
    ("Figure 8", "fig08_mixed_effects_model.png"),
    ("Figure 9", "fig09_internal_consistency.png"),
    ("Figure 10", "fig10_polarity_and_key_drift.png"),
    ("Figure 11", "fig11_facets_and_coping_choices.png"),
    ("Figure 12", "fig12_criterion_correlations.png"),
    ("Figure 13", "fig13_moderation_and_hybrids.png"),
    ("Figure C1", "figC1_kg_degree_distribution.png"),
    ("Figure C2", "figC2_core_nodes_and_predicates.png"),
    ("Figure D1", "figD1_pilot_key_kg_llm.png"),
    ("Figure D2", "figD2_pilot_key_base_llm.png"),
    ("Figure D3", "figD3_alternative_scoring_keys.png"),
    ("Figure D4", "figD4_cross_stage_associations.png"),
]


def run(results_cache: dict | None = None) -> dict[str, Path]:
    if results_cache is None:
        s00_prepare_data.run()
        results_cache = {
            "s01": s01_kg_network.run(),
            "s02": s02_triple_quality.run(),
            "s03": s03_scenario_quality.run(),
            "s04": s04_cvi_and_screening.run(),
            "s05": s05_rater_error.run(),
            "s06": s06_mixed_model.run(),
            "s07": s07_reliability.run(),
            "s08": s08_validity.run(),
            "s09": s09_pilot_and_entropy.run(),
        }

    r1 = results_cache["s01"]
    r2 = results_cache["s02"]
    r3 = results_cache["s03"]
    r4 = results_cache["s04"]
    r5 = results_cache["s05"]
    r6 = results_cache["s06"]
    r7 = results_cache["s07"]
    r8 = results_cache["s08"]
    r9 = results_cache["s09"]

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = {lbl: FIG_DIR / fname for lbl, fname in FIGURE_FILES}

    fig_part1.plot_fig01(r1, paths["Figure 1"])
    fig_part1.plot_fig02(r2, paths["Figure 2"])
    fig_part1.plot_fig03(RAW_DIR, paths["Figure 3"])
    fig_part1.plot_fig04(r1, r3, r4, RAW_DIR, paths["Figure 4"])
    fig_part1.plot_fig05(r3, paths["Figure 5"])
    fig_part1.plot_fig06(r4, r5, r8, RAW_DIR, paths["Figure 6"])
    fig_part1.plot_fig07(r4, paths["Figure 7"])
    fig_part1.plot_fig08(r6, paths["Figure 8"])

    fig_part2.plot_fig09(r7, paths["Figure 9"])
    fig_part2.plot_fig10(r7, r9, paths["Figure 10"])
    fig_part2.plot_fig11(r8, r9, paths["Figure 11"])
    fig_part2.plot_fig12(r8, paths["Figure 12"])
    fig_part2.plot_fig13(r8, paths["Figure 13"])
    fig_part2.plot_figC1(RAW_DIR, r1, paths["Figure C1"])
    fig_part2.plot_figC2(r1, paths["Figure C2"])
    fig_part2.plot_figD1(r9, paths["Figure D1"])
    fig_part2.plot_figD2(r9, paths["Figure D2"])
    fig_part2.plot_figD3(r8, paths["Figure D3"])
    fig_part2.plot_figD4(r8, paths["Figure D4"])

    return paths


if __name__ == "__main__":
    p = run()
    print(f"Generated {len(p)} figures in {FIG_DIR}")

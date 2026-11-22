#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all.py: Master CLI entry point for reproducing all data preparation,
statistical analyses (s00-s12), 19 300-DPI figures (s13), and the 156-value
verification report (outputs/verification_report.md).
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

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
import s10_scenario_length
import s11_icc_ceiling_simulation
import s12_hierarchical_regressions
import s13_generate_figures
import verify_reported_values


def main() -> dict:
    t0 = time.time()
    print("=" * 72)
    print("Reproducing All Experiments & Figures: Resilience KG-LLM SJT")
    print("=" * 72)

    steps = [
        ("s00", "Data preparation & scale scoring", s00_prepare_data.run),
        ("s01", "Stage 1-2 corpus, KG topology, & ontology audit", s01_kg_network.run),
        ("s02", "Stage 2 triple quality evaluation (H2)", s02_triple_quality.run),
        ("s03", "Stage 3-4 scenario quality & citation-count OLS (H3)", s03_scenario_quality.run),
        ("s04", "Content validity, 30-to-20 screening, & option ANOVA", s04_cvi_and_screening.run),
        ("s05", "Two-way ANOVA rater-error mean squares", s05_rater_error.run),
        ("s06", "Cross-classified mixed-effects model & G-study", s06_mixed_model.run),
        ("s07", "Internal consistency (alpha, ML omega) & item analysis", s07_reliability.run),
        ("s08", "Validity, key robustness, 2x2 moderation, & hybrid CV", s08_validity.run),
        ("s09", "Pilot consensus, provenance, TVD, & choice entropy", s09_pilot_and_entropy.run),
        ("s10", "Scenario & option length comparisons", s10_scenario_length.run),
        ("s11", "Monte Carlo ICC ceiling simulations", s11_icc_ceiling_simulation.run),
        ("s12", "Standardized hierarchical OLS regressions", s12_hierarchical_regressions.run),
    ]

    results = {}
    for key, desc, fn in steps:
        ts = time.time()
        print(f"[{key}] Running {desc} ...", end=" ", flush=True)
        results[key] = fn()
        print(f"done ({time.time() - ts:.1f}s)")

    print("[s13] Generating all 19 300-DPI figures ...", end=" ", flush=True)
    ts = time.time()
    fig_paths = s13_generate_figures.run(results)
    results["s13"] = fig_paths
    print(f"done ({len(fig_paths)} figures, {time.time() - ts:.1f}s)")

    print("[verify] Checking all 156 reported manuscript values ...", end=" ", flush=True)
    df_ver = verify_reported_values.verify(results)
    results["verification"] = df_ver
    n_pass = int((df_ver["status"] == "PASS").sum())
    print(f"{n_pass}/{len(df_ver)} PASS")

    print("=" * 72)
    print(f"Pipeline finished in {time.time() - t0:.1f}s.")
    print(f"  - Figures directory : {ROOT / 'outputs' / 'figures'}")
    print(f"  - Tables directory  : {ROOT / 'outputs' / 'tables'}")
    print(f"  - Verification MD   : {ROOT / 'outputs' / 'verification_report.md'}")
    print("=" * 72)
    return results


if __name__ == "__main__":
    main()

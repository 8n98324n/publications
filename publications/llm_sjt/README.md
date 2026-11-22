# `resilience-sjt-repro` — Reproducible Code & De-Identified Data Repository

**Article**: *Evaluating Knowledge Graph Augmentation for Large Language Model–Generated Situational Judgment Tests of Resilience*  
**Authors**: Ivan Liu, Huiyu Li, Lufeng Lin, Zixin Yang  
**Random Seed**: `20240501`

This repository reproduces **100% of the empirical experiments, psychometric analyses, robustness simulations, summary tables, 156 reported benchmark values, and all 19 publication-grade 300-DPI figures** (`Figures 1–13`, `Figures C1–C2`, and `Figures D1–D4`) directly from de-identified raw datasets (`data/raw/`).

---

## 1. Quick Start — Single-Notebook Entry Point (`reproduce_figures.ipynb`)

All experiments and all 19 figures can be reproduced end-to-end from **a single Jupyter Notebook (`reproduce_figures.ipynb`)**:

```bash
git clone <repository-url>
cd resilience-sjt-repro
python -m venv .venv
# Windows: .venv\Scripts\activate  |  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter lab reproduce_figures.ipynb
```

Open `reproduce_figures.ipynb` and click **Run -> Restart Kernel and Run All Cells** (or execute headlessly via CLI):

```bash
jupyter nbconvert --to notebook --execute --inplace reproduce_figures.ipynb
```

### Alternative CLI Batch Entry Point (`run_all.py`)
You can also run the entire pipeline from the terminal with a single command:

```bash
python run_all.py
```

---

## 2. Repository Structure

```text
resilience-sjt-repro/
├── reproduce_figures.ipynb          # ★ Unified entry-point notebook: runs ALL experiments, checks, & 19 figures
├── run_all.py                       # CLI batch runner (executes s00-s13 + verification_report.md)
├── requirements.txt                 # Python >= 3.9 package dependencies
├── LICENSE                          # MIT License
├── data/
│   ├── raw/                         # De-identified empirical datasets (zero PII; see Table E1)
│   └── derived/                     # Clean analysis matrices generated automatically by s00_prepare_data.py
├── expected/
│   └── reported_values.csv          # 156 reported manuscript values and numerical tolerances
├── src/
│   ├── utils_stats.py               # Shared ANOVA/ICC(3,k), Gwet's AC2, ML omega, Steiger Z, TOST, & FDR routines
│   ├── s00_prepare_data.py          # Data reshaping, consensus scoring keys, & scale scoring (Appendix F.0)
│   ├── s01_kg_network.py            # Corpus summary, ReBEL-LLM overlap, KG topology, & ontology audit (§3.1, App. C)
│   ├── s02_triple_quality.py        # H2 triple agreement & paired quality comparisons (§3.1.1, App. D.1-D.2)
│   ├── s03_scenario_quality.py      # H3 scenario agreement, Welch/TOST, & citation-count OLS (§3.2.1, App. D.3)
│   ├── s04_cvi_and_screening.py     # CVI, 70-rating screening + jackknife, & option quality ANOVA (§3.2.2, Table D2b)
│   ├── s05_rater_error.py           # Two-way ANOVA rater-error mean squares & bootstrap ratios (Table D2a)
│   ├── s06_mixed_model.py           # Cross-classified MixedLM, LRTs, REML variance shares, & G-study (§3.2.3)
│   ├── s07_reliability.py           # H1/H4 Cronbach's alpha, 1-factor ML omega, & polarity item analysis (§3.3.3)
│   ├── s08_validity.py              # H1/H4 validity, Double Bootstrap, 2-Fold Cross-Fitting, 2x2 Moderation, Hybrid CV, DualHit, & Cross-Stage Null Sim (§3.3.4-3.3.6)
│   ├── s09_pilot_and_entropy.py     # Pilot consensus, item provenance, ML/LL TVD, & Shannon entropy (§3.3.1, §3.3.4.1)
│   ├── s10_scenario_length.py       # Scenario & option Chinese character length comparisons (§4.2)
│   ├── s11_icc_ceiling_simulation.py# Normal-rounding & discrete empirical ceiling ICC simulations (§3.2.1.5)
│   ├── s12_hierarchical_regressions.py # Standardized hierarchical OLS regressions (Tables D2-D5)
│   ├── s13_generate_figures.py      # Generates all 19 300-DPI figures (via fig_part1.py & fig_part2.py)
│   └── verify_reported_values.py    # Automated verification against expected/reported_values.csv
└── outputs/
    ├── figures/                     # All 19 300-DPI PNG figures (Figures 1-13, C1-C2, D1-D4)
    ├── tables/                      # Recomputed CSV statistical tables
    └── verification_report.md       # Automated verification report across all 156 benchmark values
```

---

## 3. Complete Inventory of Reproduced Figures (`outputs/figures/`)

| Figure | File | Manuscript Section | Content |
| :--- | :--- | :--- | :--- |
| **Figure 1** | `fig01_pipeline_overview.png` | §2.1 | Four-stage SJT generation and validation pipeline mapped to `H1–H4` |
| **Figure 2** | `fig02_triple_evaluation.png` | §3.1.1 | `H2`: ReBEL vs. LLM triple `ICC(3,k)`, Gwet's `AC2`, and paired quality ratings |
| **Figure 3** | `fig03_kg_core_subgraph.png` | §3.1.2 | Largest connected component (88 nodes) among top-120 PageRank KG concepts |
| **Figure 4** | `fig04_kg_content_and_citations.png` | §3.1.3, §3.2 | KG ontology granularity audit, cited triple counts, and rationale citation gaps |
| **Figure 5** | `fig05_scenario_evaluation.png` | §3.2.1 | `H3`: Scenario `ICC(3,k)` and between-condition quality ratings (`30` per condition) |
| **Figure 6** | `fig06_cvi_error_and_screening.png` | §3.2.1–3.2.2 | `S-CVI/Ave`, `MS_Error`, cross-stage correlations, item ratings, and 60-item screening |
| **Figure 7** | `fig07_screening_and_options.png` | §3.2.2 | 30-to-20 screening selectivity, option quality across 5 strategies, and option `MS_Error` |
| **Figure 8** | `fig08_mixed_effects_model.png` | §3.2.3 | Cross-classified `MixedLM` REML variance shares and marginal means |
| **Figure 9** | `fig09_internal_consistency.png` | §3.3.3.1 | `H1/H4`: Cronbach's `α` and McDonald's `ω` with 2,000-resample bootstrap 95% CIs |
| **Figure 10** | `fig10_polarity_and_key_drift.png` | §3.3.3.2–3.3.3.3 | Polarity reliability (`All`, `ML`, `LL`, `SB`) and pilot-to-formal ML `TVD` |
| **Figure 11** | `fig11_facets_and_coping_choices.png` | §3.3.4 | Facet-level convergent validity (`5×5RS` facets & `PA`) and coping choice shares |
| **Figure 12** | `fig12_criterion_correlations.png` | §3.3.6 | Criterion validity profiles and Steiger's `Z` contrasts (`N = 150`) |
| **Figure 13** | `fig13_moderation_and_hybrids.png` | §3.3.6.1–3.3.6.2 | Exploratory `LL` validity, `Domain × Polarity` moderation, 1,000-split Hybrid CV, & `ΔR²` |
| **Figure C1** | `figC1_kg_degree_distribution.png` | Appendix C | Log–log degree frequency and complementary CDF of the 3,756-node resilience KG |
| **Figure C2** | `figC2_core_nodes_and_predicates.png` | Appendix C | Top-15 PageRank concepts and Top-10 relation predicates |
| **Figure D1** | `figD1_pilot_key_kg_llm.png` | Appendix D.7 | KG-LLM pilot response distributions (`N = 50`, `S1–S30 × A–E`) |
| **Figure D2** | `figD2_pilot_key_base_llm.png` | Appendix D.7 | Base LLM pilot response distributions (`N = 52`, `S1–S30 × A–E`) |
| **Figure D3** | `figD3_alternative_scoring_keys.png` | Appendix D.8.5 | Sensitivity across Pilot Proportion, Formal LOO, Borda Rank, and Modal Binary keys |
| **Figure D4** | `figD4_cross_stage_associations.png` | Appendix D.8.6 | Cross-stage predictors of formal item discrimination (`cITC`) and validity (`r_{j,5×5RS}`) |

---

## 4. Ethics & Data De-Identification

All datasets in `data/raw/` have been strictly de-identified in accordance with Institutional Review Board approval (`BNU202603230116`, Faculty of Psychology, Beijing Normal University). Platform exports containing participant names, phone numbers, student IDs, IP addresses, geolocation coordinates, and submission timestamps are excluded; sequential alphanumeric codes (`R1–R4`, `E1–E5`, `PA01–PA50`, `PB01–PB52`, `F001–F150`) identify raters and respondents.

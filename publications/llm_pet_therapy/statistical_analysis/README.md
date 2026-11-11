# Statistical, Machine Learning & Multimodal LLM Analyses (`statistical_analysis/`)

This directory contains all reproducible statistical, machine-learning, and multimodal large language model (MLLM) evaluation scripts and Jupyter notebooks for **Study 1** (`Data Analysis 1`: Self-Report Psychological & Physiological Intervention Analysis) and **Study 2** (`Data Analysis 2`: 19-Feature Multimodal Statistical, Machine Learning & MLLM Experiments 1–4 Analysis). All scripts operate exclusively on the de-identified tabular datasets in `../data/` and `../df_results_exp_*_diff.csv`.

## Notebooks
1. `01_study1_self_report_analysis.ipynb` (and standalone script `study1_self_report_analysis.py`):
   - Scale reliability (`Cronbach's alpha` at Pre-test and Post-test)
   - STAI-Total (20 items), STAI-AA (Absence of Calmness & Security, 10 items), and STAI-AP (Presence of Acute Tension, 10 items) descriptives and Jacobson-Truax Reliable Change Index (`RCI`)
   - 2×2 Factorial ANCOVA, Additive Model, Exposure Time-Dose ANCOVA (`10 min = 1.0`, `5 min = 0.5`), Diminishing-Returns Dose Model, Planned Contrasts against Active Observation, and Non-Fearful Subgroup (`n = 84`)
   - Two One-Sided Tests (`TOST`) equivalence testing, JZS Bayes Factor (`BF01`), and outlier robustness diagnostics (`Huber M-estimation`, `Winsorization`, `HC3`)
   - Single Mediation, `PROCESS Model 6` Serial Mediation (`Feeding -> Situational Mattering -> General Self-Efficacy -> Anxiety Reduction`), and Dual-Pathway Dissociation (`Petting -> PANAS-PA` vs. `Feeding -> GSE`)
2. `02_study2_multimodal_statistical_analysis.ipynb` (and standalone script `study2_multimodal_statistics.py`):
   - 2×2 Factorial ANCOVA (OLS & HC3 robust SEs), Exposure Time-Dose ANCOVA, Direct Interaction vs. Observation contrasts, and Net Single-Component contrasts across all 19 facial, acoustic, thermal, and cardiovascular features
   - Cross-channel alignment (`Pearson r`, `Spearman rho`, and baseline-adjusted ANCOVA residual correlations) between multimodal feature changes and Study 1's psychological outcomes (`STAI-Total`, `STAI-AA`, `STAI-AP`, `Situational Mattering`, `GSE`, `PANAS-PA`)
3. `03_study2_multimodal_machine_learning.ipynb` (and standalone scripts `study2_multimodal_ml_benchmark.py`, `model_benchmark.py`, `incremental_benchmark.py`, `factor_permutation_tests.py`, `run_study2_baseline_pipeline.py`):
   - 10×5 Repeated Stratified Cross-Validation on the `n = 92` prediction cohort (`../data/cv/split_membership.csv`)
   - Primary focus: Cross-validated prediction of State Anxiety (`STAI-Total`, `STAI-AA`, `STAI-AP`, and `RCI`) and Study 1 psychometric targets under three scenarios (`Scenario A`: Pure Multimodal Passive Sensing; `Scenario B`: Condition-Blind Pre-Test + Passive Multimodal Decoded; `Scenario C`: Design-Informed Two-Stage Huber + Multimodal Residual)
   - Feature explainability (selection frequency, standardized shrinkage weights, and permutation $\Delta\text{MAE}$) and downstream surrogate endpoint validation on experimental groups (`Petting`, `Feeding`, `Direct Interaction vs. Observation`)
4. `04_study2_mllm_analysis.ipynb`:
   - Primary focus: Psychometric criterion validity of Multimodal Large Language Model (`Gemini 3.5 Flash-Lite`) predictions across Experiments 1–4 (`Video Only`, `Video + Audio [No Speech Text]`, `Video + Audio + Speech Context`, and `Video + Audio + Speech + Physio [HR & Thermal]`) at Pre-test, Post-test, and `Pre - Post` improvement
   - Explainability analysis: Quantitative cue-to-score attribution across all 736 reasoning chains (`explanation_pre` and `explanation_post` in `Exp 1–4`)
   - Downstream theoretical meaning for experimental groups: Testing $2\times 2$ Factorial ANCOVA and Exposure Time-Dose (`10 min = 1.0`, `5 min = 0.5`) surrogate endpoint effects on blind MLLM-predicted state anxiety scores

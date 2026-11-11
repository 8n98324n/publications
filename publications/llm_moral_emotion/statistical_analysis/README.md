# statistical_analysis — one-click reproduction of the statistics in the manuscript

These notebooks recompute, from the committed data files in `../final/`, the statistics reported in the
Results chapter and in Appendices A, D and E of *Addressing the Psychological Small-Sample Problem with
Multimodal Large Language Models: Evidence from Multiple Moral Emotion Recognition Study*. The folder
mirrors Appendix E (Detailed Computational Procedures of the Analysis Methods): notebook `0K_EK_*`
implements section E.K, and the section headings inside each notebook name the subsections they
reproduce. Notebook `00` builds the label tables that every other notebook reads.

| Notebook | Manuscript sections | Items reproduced |
|---|---|---|
| `00_sample_and_labels.ipynb` | Table A1 note; E.1.1.1; C.2.4; C.3 | valid-*N* sample and exclusion list, sample-mean discretization, late fusion of the machine-learning votes with one acoustic vote, observer consensus labels (Exp 14–15) |
| `01_E1_significance_of_classification.ipynb` | E.1; Table A1; Appendix Figures E1–E2; D.2.2, Tables D1–D2 | accuracy, macro-F1 and κ with bootstrap CIs, permutation and FDR *p*, two-dimensional accuracy (Table A1); per-class recall and the confusion-matrix counts of E.1.3; single-modality Random Forest accuracy, axis accuracy and recall (E.1.4); per-model still-frame statistics of Exp 17 (Tables D1–D2) |
| `02_E2_construct_validity_induction_design.ipynb` | E.2; Table A2; Appendix Figure E3 | Type II 2 × 2 ANOVAs with partial η² and 90% CI, incongruent effects and interactions |
| `03_E3_criterion_validity_self_report.ipynb` | E.3; Table A3 | Spearman ρ with the self-reports and bootstrap CIs; condition-controlled partial correlations; design dependence |
| `04_E4_convergent_validity_observers.ipynb` | E.4.2–E.4.6; Exp 14–15 | ICC(A,1), ICC(A,k), ICC(C,1) with CIs and per-item ICCs; manipulation check (Mann–Whitney *U*, *d*, *g*, cell means, 2 × 2 ANOVA); observer–model Spearman ρ (audiovisual and silent); individual and pooled human raters versus the model; human benchmark by rater |
| `05_E5_performance_comparisons_gee.ipynb` | E.5 | participant-clustered GEE on Exp 1–13 (ORs vs. baseline, omnibus Wald test, nine planned and six cross-approach contrasts with Holm correction within each family, bootstrap Δaccuracy, improved/worsened counts, per-axis models) |
| `06_E6_discretization_sensitivity.ipynb` | Supplementary (not reported in the manuscript) | accuracy under the mean, midpoint, median and label-informed thresholds, reassignment share, scores at the bracketing integers, macro-AUC |
| `07_E7_interpretability_explanations.ipynb` | E.6; Results 3.3; Figure 8; Appendix Figure E5 | 26-cue feature–class associations with bootstrap CIs, cue importance, worked example and Fisher tests, Figure 8A routing counts and χ² tests, valence separation by channel, cue counts |

Experiment numbers are those of the manuscript. Exp 16 (the supplementary vision + audio + physiology
control, Appendix D.1) is stored as `../final/experiment_14.csv` and appears in notebooks 00–04 and 06;
it is not a level of the GEE in notebook 05. Exp 17 (Appendix D.2) enters notebook 01 only, through its
saved confusion matrices.

## How to run

```bash
cd psychai_publications/publications/llm_moral_emotion/statistical_analysis
pip install -r requirements.txt                                      # numpy, pandas, scipy, scikit-learn, statsmodels, jupyter
jupyter nbconvert --to notebook --execute --inplace 0*.ipynb        # or open and run each notebook in order
```

Run the notebooks in numerical order: `00` writes `outputs/labels_all.csv`, the per-participant label
table that every later notebook reads. `common.py` holds the loaders, constants and statistical
functions shared by all notebooks; the docstring of each function names the Appendix E section it
implements. Every random quantity uses `numpy.random.default_rng(42)`, with 10,000 bootstrap resamples
and 10,000 label permutations unless a section states otherwise (κ intervals and the feature–class
association intervals: 2,000 resamples; the human-versus-model comparison of E.4.5: 2,000 resamples), so
the outputs are deterministic.

## Verification

Each notebook ends with a Verification section. Where a frozen copy exists in `../final/reproducibility/`
(the files from which the manuscript values were taken), the recomputed table is aligned with it and the
maximum absolute difference per column is printed. Point estimates, test statistics, *p* values, odds
ratios, ANOVA and ICC values reproduce to machine precision. Only bootstrap interval bounds and
permutation *p* values can differ, because the resampling order of the original run is not always
recoverable: by at most .011 for the accuracy and κ intervals (notebook 01), .011 for the self-report
correlation intervals (notebook 03), and .04 for the 2,000-resample feature–class association intervals
(notebook 07); permutation *p* values differ by at most .004. The FDR-adjusted *p* values are the one exception to the frozen-file comparison: the `p_fdr` column of `significance_frozen.csv` was computed by applying Benjamini–Hochberg to all 16 permutation *p* values (observer passes included); this reproduces its 14 model-configuration rows (and the Exp 15 row), but not its Exp 14 row (.150 vs .131), whose origin is not recoverable, whereas the manuscript (Table A1, Appendix E.1.2.5) applies Benjamini–Hochberg to the 14 model configurations (Exp 1–13 and 16) and tests the observer passes (Exp 14–15) outside that family. Notebook 01 computes the 14-member family and verifies it against the same procedure applied to the frozen permutation *p* values, which gives the Table A1 values (for example Exp 1 .024, Exp 11 .014, Exp 12 .020); the recomputed values differ from that reference by at most .003. The GEE (notebook 05, including its
bootstrap columns), the observer statistics (notebook 04, including the Spearman intervals) and the
discretization table (notebook 06) reproduce their frozen files exactly.

Three groups of reported values have no frozen file and are checked against the values printed in the
manuscript, at the printed precision: the confusion-matrix counts of Appendix E.1.3 and the Table D1
entries (notebook 01), the human-versus-model comparison and scale-use descriptives of Appendix E.4.5 (notebook 04), and the
directional physiological results of Results 3.3, Figure 8 and Appendix E.6.4–E.6.5 (notebook 07).

## What is and is not recomputed

- The model inferences (Exp 1–9, 16, 17) and the Random Forest predictions (Exp 10–13) are not re-run;
  the committed `experiment_*.csv`, `machine_learning.csv` and `exp17_18_confusion_matrices.json` are
  the authoritative records (see the package README and `REPRODUCIBILITY_GAP_REPORT.md`).
- The language-model coding of the Exp 4 explanations is not re-run. Notebook `07` starts from the two
  released coding files: `exp4_feature_coding.csv` (first pass; its five textual and eight nonverbal
  cues are used, its seven undirected physiological codes are superseded) and
  `exp4_physio_direction_coding.csv` (second pass; thirteen directional physiological cues). Their
  prompts are `exp4_coding_prompt.txt` and `exp4_physio_direction_prompt.txt`. The frozen files
  `exp4_feature_lift_predicted.csv`, `exp4_feature_importance_predicted.csv` and
  `exp4_valence_by_channel_predicted.csv` hold the first-pass results; notebook 07 uses them to verify the
  thirteen retained textual and nonverbal cues only.
- The observer ratings enter through the cleaned long table and composites
  (`observer_ratings_long_clean.csv`, `observer_composite_scores_clean.csv`); the cleaning rules are
  stated in Appendix E.4.1.
- Figures are not regenerated here; the tables written to `outputs/` are the inputs to the figures.

## Outputs

All results are written to `outputs/` as CSV files named after the quantity they hold:
`labels_all.csv`, `sample_summary.csv`, `exclusion_list.csv` (00); `significance.csv`,
`perclass_recall.csv`, `ml_modality_axis.csv`, `ml_modality_recall.csv`, `exp17_per_model.csv` (01);
`anova.csv` (02); `selfreport_validity.csv` (03); `observer_statistics.csv`, `observer_item_icc.csv`,
`observer_anova.csv`, `observer_model_spearman.csv`, `observer_human_vs_model.csv` (04); `gee.csv`,
`gee_contrasts.csv` (05); `threshold_sensitivity.csv` (06); `exp4_feature_lift.csv`,
`exp4_feature_importance.csv`, `exp4_valence_by_channel.csv` (07).

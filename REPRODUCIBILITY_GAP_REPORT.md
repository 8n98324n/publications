# Reproducibility & Gap Report
## psychai / psychai_publications release — MLLM moral-emotion study

This report accompanies the release of the two repositories. It records (1) what the package contains,
(2) what reproduces and what cannot, and why, (3) known gaps, and (4) the changes made in each release.
Experiment numbers are those of the manuscript unless a file name is quoted; the file-name mapping is in
the package README (section Experiment numbering).

---

## 1. Package contents

- `psychai/` — the study-agnostic library: 25 Python modules under `src/psychai/` (LLM client and model
  registry, feature extraction and interpretation per modality, feature engineering, classifiers and
  evaluation, data helpers, cloud storage, charts).
- `psychai_publications/publications/llm_moral_emotion/` — the study code:
  - entry-point notebooks in `final/`: `experiments_1_9.ipynb` (Exp 1–9 and the Exp 16 control),
    `experiment_10_13.ipynb` (Exp 10–13), `experiment_17_18.ipynb` (Exp 17, plus an unreported run);
  - eight feature-extraction notebooks in `feature_extraction/`;
  - eight reproduction notebooks and `common.py` in `statistical_analysis/` (Appendix E);
  - study modules in `utilities/`, `emotion_recognition/` and `env/configs/`;
  - data in `final/`: the per-modality feature tables `df_1.csv` … `df_7.csv`, the unreported MLLM-explanation features
    `df_8.csv` and the fusion output `df_fuse.csv`, the
    model outputs `experiment_1.csv` … `experiment_14.csv`, the Random Forest predictions
    `machine_learning.csv` and the regenerated fusion files `experiment_10_5fold.csv` …
    `experiment_13_5fold.csv`, the still-frame confusion matrices `exp17_18_confusion_matrices.json`,
    and the explanation-coding files (`exp4_coding_prompt.txt`, `exp4_coding_metadata.json`,
    `exp4_feature_coding.csv`, `exp4_physio_direction_prompt.txt`, `exp4_physio_direction_coding.csv`);
  - frozen statistical outputs in `final/reproducibility/`.

Removed before release: two dated `client_backup_*.py`, `configs_20260304.py`, `configs_lazy.py`,
`utilities_delete.py` and a duplicate `utilities/feature_interpreter.py`. Every Python file carries the
header `Copyright (c) 2026 Ivan Liu. Licensed under CC BY 4.0` and parses cleanly.

All comments, docstrings and user-facing strings are in English. Chinese text remains only in the data
(the model explanations and the Exp 4 `individual_prompt` column of `experiment_*.csv`, in the original
language of the study) and in the four questionnaire-item strings that `2_2_speech_text_to_feature.ipynb`
uses to strip the interviewer's prompts from the transcripts. Full-width (East Asian) punctuation marks (parentheses, comma, colon, equals sign, semicolon) remain
in executed code where it is functional: the output parsers of `experiment_helper.py`,
`fine_tuning_helper.py` and `experiment_17_18.ipynb` normalise it in model responses, and
`feature_interpreter.py` joins the feature descriptions inserted into the Exp 4 prompt with it. The executed
model prompts are kept verbatim.

---

## 2. Reproducibility

| Component | Reproducible? | Detail |
|---|---|---|
| Statistical analyses (Results; Appendices A, D, E) | Yes | The eight notebooks in `statistical_analysis/` recompute the reported statistics from the committed files and verify them against `final/reproducibility/` or, where no frozen file exists, against the values printed in the manuscript (see `statistical_analysis/README.md`). |
| Exp 10–13 late fusion | Yes, exactly | Majority voting over `machine_learning.csv` with one acoustic vote (the run with correlation pruning; Appendix C.2.4) reproduces every fused prediction in `experiment_10_5fold.csv` … `experiment_13_5fold.csv` (101/101, 103/103, 103/103, 103/103) and the reported accuracies .30, .36, .35, .40. Fusing over both acoustic runs reproduces the superseded files `experiment_10.csv` … `experiment_13.csv` (.30, .30, .35, .40). |
| Exp 10–13 per-modality classification | Partial | The classifier pipeline (Random Forest + SVM-RFE + GroupKFold, `EmotionClassificationTask.run_analysis`) runs end to end on the per-modality feature tables, and per-modality accuracy is comparable, but per-subject predictions are not bit-identical across environments, because the Random Forest, SVM-RFE and grouped cross-validation steps depend on library versions and threading. The committed `machine_learning.csv` is the authoritative record. |
| Exp 1–9 and 16 (Gemini MLLM) | No, by design | These call the hosted `gemini-2.5-flash` model; results cannot be regenerated without a key, and a hosted model at a later version would not reproduce the frozen predictions bit for bit. The committed `experiment_1.csv` … `experiment_9.csv` and `experiment_14.csv` (= Exp 16) are the authoritative records. |
| Exp 17 (six MLLMs, still frames) | No, by design | Same reason, across six providers. `experiment_17_18.ipynb` ships with outputs cleared; its confusion matrices and check values are in `exp17_18_confusion_matrices.json` (key `confusion_matrices_exp18` = Exp 17). |
| Exp 14–15 (human observers) | Ratings released, rating not repeatable | The cleaned ratings (`observer_ratings_long_clean.csv`, `observer_composite_scores_clean.csv`) are released; every observer statistic is recomputed from them (notebook `04_E4_…`). |
| Explanation coding (Exp 4) | Codes released, coding not re-run | Both coding passes called a language model with a fixed prompt; the prompts and the complete per-participant codes are released, and every association statistic is recomputed from them (notebook `07_E7_…`). |
| Raw feature extraction | No | The upstream extraction from the video, audio, thermal and physiological recordings is not runnable here, because the recordings are not distributed. The feature tables are the reproducible starting point. |

---

## 3. Known gaps and caveats

1. Second explanation-coding pass. `exp4_physio_direction_coding.csv` was produced with the fixed prompt
   `exp4_physio_direction_prompt.txt`, sending each explanation (truncated to its first 6,000 characters)
   to a language model and parsing the returned JSON; responses that came back truncated were re-requested
   with a larger output budget. The identifier of the coding model was not recorded, so, unlike the first
   pass (`exp4_coding_metadata.json`), the second pass has no model metadata. The codes have not been
   checked by hand.
2. Superseded first-pass physiological codes. The seven `phys_*` columns of `exp4_feature_coding.csv`
   (no direction or region) are retained in the file for transparency but are not used by any reported
   analysis. The frozen files `exp4_feature_lift_predicted.csv`, `exp4_feature_importance_predicted.csv`
   and `exp4_valence_by_channel_predicted.csv` were computed with the first-pass schema; their textual
   and nonverbal rows remain valid and are used for verification, their physiological rows are not.
3. Fine-tuning (Exp 8). No tuning-job metadata beyond the code exists. The helper's target mapping assigns
   the Control target to rows without a group name, so two unlabeled participants entered the training
   folds (disclosed in the manuscript). The released notebook evaluates Exp 8 with the sample-mean rule
   (`threshold_type="mean"`), which gives the reported .66; the helper still offers
   `truncated_mean_50`, which gives .69 on the same scores and is not used for any reported value.
4. Feature-extraction notebooks. `1_speech_acoustics.ipynb`, `2_2_speech_text_to_feature.ipynb` and
   `3_face_expression.ipynb` import the extractors under an older module path
   (`psychai.<modality>.feature_extraction.feature_extraction`); the library ships them as
   `psychai.<modality>.feature_extractor.feature_extractor`. The notebooks are retained as executed and
   need this import adjusted before they can run against the released library (they also need the raw
   recordings).
5. `experiment_17_18.ipynb` contains a second still-frame run on the post-viewing recordings (labelled
   Experiment 17 in the notebook) that the manuscript does not report; both runs use the same prompt
   sentence describing the frames as taken while the participant watched the film.

---

## 4. Environment

The entry-point notebooks were run with the pinned versions in `env/requirements/requirements.txt`
(numpy 2.2.6, pandas 2.2.3, scikit-learn 1.6.1, scipy 1.13.0, xgboost 2.1.2, lightgbm 4.5.0, plus plotly and
python-dotenv). The statistical notebooks were run with `statistical_analysis/requirements.txt`
(numpy 2.4.6, pandas 2.3.3, scipy 1.17.1, scikit-learn 1.9.0, statsmodels 0.14.6). The fusion result
is version-insensitive; the per-modality ML step is version-sensitive as noted.

---

## 5. Release history

### October 2026, first release

- Class label `ADMR` renamed to `INSP` (Inspiration) in all code and in the derived `Group Name` column;
  the numeric `Group` key is unchanged.
- All four-class results use the sample-mean threshold; `experiments_1_9.ipynb` calls
  `threshold_type="mean"` throughout.
- `final/machine_learning.csv` contains two acoustic prediction rows per participant (two runs differing
  only in the correlation-pruning flag); the reported Exp 11–13 use a single acoustic vote, and the
  regenerated fusion files are `final/experiment_1x_5fold.csv`.
- The explanation-coding file used by early drafts was lost; `exp4_feature_coding.csv` is a complete
  recoding of all 100 Exp 4 explanations with the saved prompt and schema.
- `statistical_analysis/` added, with frozen outputs in `final/reproducibility/`.

### October 2026, update aligned with the final manuscript

- Statistical notebooks renamed to follow Appendix E (`01_E1_…` to `07_E7_…`; previously `01_D1_…` to
  `08_D8_…`, named after the earlier Appendix D sections).
- Removed the ensemble, stacking and selective-prediction analyses, which the manuscript no longer
  reports: notebook `08_D8_exploratory_ensembles_selective_prediction.ipynb`, its outputs
  (`selective_prediction.csv`, `stacking_grid.csv`) and `final/reproducibility/selective_prediction_frozen.csv`.
- GEE (notebook `05_E5_…`) refitted on the 13 primary configurations (Exp 1–13; 103 participants, 1,305
  observations) with nine planned and six cross-approach contrasts, Holm-corrected within each family;
  `gee_frozen.csv` and `gee_contrasts_frozen.csv` replaced by the frozen outputs of that model. The
  supplementary control Exp 16 is no longer a level of the model.
- Explanation analysis (notebook `07_E7_…`) moved to the 26-cue schema: `exp4_physio_direction_coding.csv`
  and `exp4_physio_direction_prompt.txt` added to `final/`.
- Observer notebook (`04_E4_…`): the model-as-fourth-rater ICC replaced by the comparison of individual
  and pooled human raters with the model (Appendix E.4.5); per-item ICCs, the observer 2 × 2 ANOVA and
  the silent-pass correlations added.
- Single-modality machine-learning statistics (Appendix E.1.4) moved from the interpretability notebook
  to `01_E1_…`.
- FDR (notebook `01_E1_…`): Benjamini–Hochberg over the 14 model configurations (Exp 1–13 and 16), with
  the observer passes (Exp 14–15) outside the family, as in Table A1. The `p_fdr` column of
  `final/reproducibility/significance_frozen.csv` was computed over all 16 permutation *p* values (it
  reproduces the 14 family rows but not the Exp 14 observer row) and is retained unchanged;
  the notebook verifies its FDR values against the 14-member family applied to the frozen permutation
  *p* values.
- Section, table and figure references in every README, docstring, comment and notebook brought in line
  with the manuscript (Appendix D = Exp 16–17; Appendix E = procedures; Appendix F = source code index).

### October 2026, audit against the final manuscript (Appendix F index)

- All eight statistical notebooks re-executed; every verification section passes against
  `final/reproducibility/` (details in the released audit log). The re-executed `statistical_analysis/outputs/`
  files are byte-identical to the previous release.
- Table and figure numbers updated to the final manuscript: the still-frame statistics of Exp 17 are
  Tables D1–D2 (previously cited as Table A4); the discretization-sensitivity analysis (notebook 06) is a supplementary analysis that the final manuscript no longer reports (previously Methods 2.4.6 / Results 3.2.6 / Appendix E.6 / Table A5); Appendix E.7 (interpretability) is now Appendix E.6; the explanation figure is Figure 8 (previously Figure 9);
  the former "Appendix E.2.1" pointer now reads Table A2.
- Notebook 04: the E.4.5 scale-use descriptives are computed on the 97 participants of the Sympathy
  comparison, as the manuscript states (model 4.25, SD 3.43; raters 1.55–2.24, SD 0.93–1.45); the previous
  release computed them on every available score (4.12, SD 3.42).
- Notebook 07: the verification constant for the share of non-Elevation explanations citing "moved" is .10
  (7 of 71), as printed in Results 3.3; all 27 manuscript checks now match.
- Full-width punctuation removed from two notebook markdown cells and one docstring; the remaining
  non-ASCII characters are listed in Section 1.

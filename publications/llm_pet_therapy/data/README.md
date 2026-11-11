# Data Directory Reference (Study 1 & Study 2: Human-Pet Interaction Intervention)

All files in this directory are de-identified, UTF-8 encoded CSV or JSON files. All participant names, raw video/audio recordings, and personally identifiable information (PII) have been completely removed. Empty cells in CSV files represent missing values (not zero), and no cross-participant imputation is applied. When loading CSV files, read `participant_id` as a string (`str`) to preserve leading zeros (`"004"`–`"115"`).

| File / Directory | Description and Usage |
|---|---|
| `study1_self_report.csv` | De-identified Study 1 (`Data Analysis 1`) item-level and scale-level dataset ($N = 100$ main study cohort with `in_study1_main100=True`, plus 1 protocol-incomplete sensitivity row `EXCL_001` with `in_study1_main100=False`). Includes pre-test and post-test item responses (`Q1`–`Q10`), scale composites, demographic covariates, and pet ownership/fear indicators |
| `participants.csv` | Unified participant table ($N = 100$) with experimental condition, binary 2×2 factors (`Petting`, `Feeding`), Total State Anxiety (`STAI_pre`, `STAI_post`, `STAI_improvement`), the two 10-item STAI psychometric subscales from Study 1 (`STAI_AA_pre`, `STAI_AA_post`, `STAI_AA_improvement` for Anxiety-Absent / Calmness & Security Absence; `STAI_AP_pre`, `STAI_AP_post`, `STAI_AP_improvement` for Anxiety-Present / Acute Tension), and joint prediction scope indicator (`in_prediction_scope`) |
| `feature_definitions.csv` | English definitions, categories (`Facial`, `Acoustic`, `Thermal`, `Cardiovascular`), measurement units, and sample scopes for the 19 multimodal and physiological features |
| `scopes.json` | Sample scope definitions: `study_main100` ($N = 100$ main study cohort), `specified_avrr92` ($n = 92$ joint audiovisual + RR prediction cohort), and `independent_thermal79` ($n = 79$ independent thermal imaging cohort) |
| `features_long.csv` | Long-format table ($100 \times 19 \times 2 = 3800$ rows) containing pre-test and post-test stage values and eligibility flags |
| `features_paired.csv` | Paired table ($100 \times 19 = 1900$ rows) containing qualified stage values, paired eligibility flags, pre-to-post deltas, and missingness notes |
| `features_pre.csv` / `features_post.csv` / `features_delta.csv` | Wide-format tables ($N = 100$ rows) for pre-test, post-test, and delta ($\text{Post} - \text{Pre}$) features with standardized English column names and full numeric precision |
| `rr_variants.csv` | RR-interval heart rate and HRV stage values and eligibility flags across sensitivity window specifications (full response, strict 5% rule, first 60s, and standard 10% mean HR rule) |
| `smile_repeats.csv` | Repeated integer ratings (0–4 scale) across two independent runs for the two smile intensity metrics, used for ICC and ordinal sensitivity analyses |
| `cv/` | Pre-registered 10×5 repeated stratified cross-validation split definitions (`split_membership.csv`), random seeds (`seeds.csv`), permutation indices (`permutation_indices.csv`), and baseline reference parameters |

## Notes on Coding and Units
- **Experimental Conditions (`condition`)**: Coded as `observation` (Observation Control: `Petting=0, Feeding=0`), `petting` (Petting Only: `Petting=1, Feeding=0`), `feeding` (Feeding Only: `Petting=0, Feeding=1`), and `combined` (Combined Petting + Feeding: `Petting=1, Feeding=1`).
- **STAI State Anxiety Subscales (Study 1 & Study 2 Alignment)**:
  - `STAI_AA` (**Anxiety-Absent / Absence of Calmness & Security Subscale**, range 10–40): Sum of the 10 reverse-scored positively worded items (`Q3T1, 2, 5, 8, 10, 11, 15, 16, 19, 20`: feeling calm, secure, at ease, comfortable, relaxed, self-confident, satisfied, composed). Higher values indicate greater absence of calmness/security (higher anxiety); lower values indicate greater calmness and psychological comfort.
  - `STAI_AP` (**Anxiety-Present / Presence of Acute Tension Subscale**, range 10–40): Sum of the 10 directly scored negatively worded items (`Q3T3, 4, 6, 7, 9, 12, 13, 14, 17, 18`: feeling tense, strained, upset, frightened, nervous, jittery, indecisive, worried, confused/over-excited). Higher values indicate stronger acute somatic/autonomic tension.
  - `STAI` (**Total State Anxiety**, range 20–80): $\text{STAI} = \text{STAI\_AA} + \text{STAI\_AP}$.
- **Prediction Scope (`in_prediction_scope`)**: Indicates membership in the $n = 92$ joint multimodal prediction cohort (`specified_avrr92`).
- **Change Scores**: $\text{Improvement} = \text{Pre} - \text{Post}$ for `STAI`, `STAI_AA`, and `STAI_AP` (positive values indicate anxiety reduction); $\text{delta\_feature} = \text{post\_value} - \text{pre\_value}$.

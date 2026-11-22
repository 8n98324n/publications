# Pet the Dog, Feed the Soul: Investigating the Psychological Mechanisms of Animal-Assisted Interventions for State Anxiety

Companion reproducibility repository for the manuscript *"Pet the Dog, Feed the Soul: Investigating the Psychological Mechanisms of Animal-Assisted Interventions for State Anxiety"* (manuscript v1.36).

## Quick Start (Single Entry Point)

All analyses are organized around a single Jupyter notebook entry point: **[`reproduce_all_analyses.ipynb`](reproduce_all_analyses.ipynb)**.

```bash
python -m venv .venv
# Activate .venv for your platform, then:
python -m pip install -r requirements.txt
python -m jupyter lab reproduce_all_analyses.ipynb
```

Select **Run All** in `reproduce_all_analyses.ipynb` (or run `python scripts/reproduce.py` from the repository root) to execute the offline tabular verification pipeline and write outputs to `generated/`.

## Repository Structure

```text
pet-the-dog-feed-the-soul/
├── reproduce_all_analyses.ipynb   # Single entry-point notebook for all analyses
├── README.md                      # Repository documentation & manuscript mapping
├── requirements.txt               # Python dependencies
├── .gitignore
├── scripts/
│   └── reproduce.py               # Offline tabular analysis & figure verification pipeline
├── data/                          # De-identified participant-level & feature-level tables
│   ├── participants.csv           # Analysis 1 sample (n = 100) and Analysis 2 scope (n = 92)
│   ├── study1_self_report.csv     # Full self-report measures
│   ├── features_paired.csv        # 19 multimodal features (pretest & posttest)
│   ├── model_predictions/         # Frozen zero-shot MLLM predictions (Experiments 1–4)
│   └── cv/                        # Cross-validation splits, seeds, and permutation indices
├── statistical_analysis/          # Modular Analysis 1 & Analysis 2 statistical and ML scripts/notebooks
├── feature_extraction/            # Upstream audio, facial AU, thermal, and HRV extraction modules
├── chart/                         # Publication figure generation scripts and notebook
├── utilities/                     # Shared helper modules
├── mllm.ipynb                     # Upstream zero-shot MLLM inference notebook (requires raw video/API)
└── manuscript_snapshot/           # Archived manuscript figure assets and SHA-256 checksums
```

## Manuscript-to-Code Coverage (v1.36)

| Manuscript Item | Coverage in This Repository |
|---|---|
| **Figures 1–2** | Design and procedure diagrams (`chart/`, `manuscript_snapshot/`). |
| **Figure 3 & Appendix Tables A1–A2** | Recalculated STAI total, STAI-AA, and STAI-AP descriptives, reliable relief rates ($\Delta\text{STAI} \ge 7.33$), and baseline-adjusted exposure-dose ANCOVAs (`scripts/reproduce.py`, `statistical_analysis/`). |
| **Figure 4 & Appendix Table A3** | Feeling needed dose effects, PROCESS Model 6 serial mediation, parallel mediation, and 11-variable mediator screen (`statistical_analysis/study1_self_report_analysis.py`, `manuscript_snapshot/`). |
| **Figure 5** | Individual-difference moderation analyses (`statistical_analysis/study1_self_report_analysis.py`, `manuscript_snapshot/`). |
| **Figure 6 & Appendix Tables A4–A5; Supplementary Tables S3–S4** | Recalculated exposure-dose and departure-contrast effects across all 19 multimodal features with Benjamini–Hochberg FDR control (`scripts/reproduce.py`, `statistical_analysis/study2_multimodal_statistics.py`). |
| **Figure 7 & Appendix Table A6** | Baseline-adjusted residual correlations across 19 features × 6 psychological outcomes (114 tests with BH-FDR control; `scripts/reproduce.py`, `statistical_analysis/study2_multimodal_statistics.py`). |
| **Figure 8 & Appendix Table A7 (Panel D)** | Nested 5-fold cross-validation (10 repetitions) for feeding and petting component classification and permutation feature importance (`statistical_analysis/03_study2_multimodal_machine_learning.ipynb`, `data/cv/`). |
| **Supplementary Figures S1–S2 & Appendix Table A7 (Panel A)** | Nested grouped cross-validation for contemporaneous state anxiety regression with univariate feature selection (`statistical_analysis/03_study2_multimodal_machine_learning.ipynb`). |
| **Supplementary Figure S3 & Appendix Table A7 (Panels B–C)** | Offline replay of zero-shot MLLM predictions (Experiments 1–4: MAE, $R^2$, pooled Pearson $r$) from `data/model_predictions/` (`scripts/reproduce.py`, `statistical_analysis/04_study2_mllm_analysis.ipynb`). |
| **Appendices B, C, D, E & Supplementary Tables S1–S2** | Technical feature-extraction specifications, smile-coding repeatability (`data/smile_repeats.csv`), and self-report psychometric definitions (`feature_extraction/`, `statistical_analysis/config/`). |

## Notes on Data Privacy & Upstream Modules

- No raw audiovisual recordings, raw thermal videos, API credentials, or personally identifiable information are included in this repository.
- The offline entry point (`reproduce_all_analyses.ipynb`) runs entirely on the de-identified tabular inputs in `data/` without requiring external API keys.
- Upstream extraction modules (`feature_extraction/`) and raw MLLM inference (`mllm.ipynb`) are provided for methodological transparency and require restricted raw recordings and local API configuration to execute.
- Before making the repository public on GitHub, ensure institutional ethics/team clearance for participant-level tables and add your chosen license file (e.g., MIT for code, CC-BY-4.0 for data).

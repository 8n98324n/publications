# Deconstructing Human-Dog Interaction: Psychological Mechanisms, Multimodal Biobehavioral Signatures, and Multimodal LLM Evaluation

This repository contains the complete, de-identified supplementary code and tabular datasets for reproducing all analyses reported in the manuscript:

1. **Study 1 (`Data Analysis 1` — Self-Report Psychological & Physiological Intervention Analysis)**:
   - Located in [`statistical_analysis/01_study1_self_report_analysis.ipynb`](statistical_analysis/01_study1_self_report_analysis.ipynb) and [`statistical_analysis/study1_self_report_analysis.py`](statistical_analysis/study1_self_report_analysis.py).
   - Evaluates the 2×2 factorial intervention (`Petting × Feeding`, $N = 100$) across `STAI-Total` (20 items), `STAI-AA` (Absence of Calmness & Security, 10 items), `STAI-AP` (Presence of Acute Tension, 10 items), Jacobson-Truax Reliable Change Index (`RCI`), Exposure Time-Dose ANCOVA (`10 min = 1.0`, `5 min = 0.5`), `PROCESS Model 6` serial mediation (`Feeding -> Situational Mattering -> General Self-Efficacy -> Anxiety Reduction`), and dual-pathway dissociation (`Petting -> Positive Affect` vs. `Feeding -> General Self-Efficacy`).
2. **Study 2 (`Data Analysis 2` — 19-Feature Multimodal Extraction, Statistical Analysis, and 10×5 Cross-Validated Machine Learning)**:
   - **Feature Extraction Pipeline** ([`feature_extraction/`](feature_extraction/)): Extracts the 19 pre-specified facial (`Qwen3.5-Omni-Plus` smile prevalence/intensity, `OpenFace 2.2.0` AU05/AU07/AU12), non-semantic acoustic (`openSMILE 3.0.1` & `Silero VAD` F0 median/IQR, silent interval ratio, periodic vocalization ratio, local jitter, HNR), facial thermal (`TIStudio`/`TFAN` nasal tip and bilateral cheek temperatures), and cardiovascular (`NeuroKit2` HR, RMSSD, SDNN, and brachial SBP/DBP) features from raw recordings.
   - **Multimodal Statistical Analysis** ([`statistical_analysis/02_study2_multimodal_statistical_analysis.ipynb`](statistical_analysis/02_study2_multimodal_statistical_analysis.ipynb)): 2×2 Factorial ANCOVA, Time-Dose ANCOVA, and cross-channel correlations with Study 1 targets.
   - **Multimodal Machine Learning & Factorial Decoding** ([`statistical_analysis/03_study2_multimodal_machine_learning.ipynb`](statistical_analysis/03_study2_multimodal_machine_learning.ipynb)): 10×5 repeated stratified cross-validation on the $n = 92$ prediction cohort across 4-condition decoding, binary factor decoding, and Study 1 psychometric outcome prediction.
3. **Study 2 (`Data Analysis 2` — Two-Step Pre/Post Multimodal Large Language Model [MLLM] Modality Ablation & Physiological Integration)**:
   - Located in [`mllm.ipynb`](mllm.ipynb) and [`statistical_analysis/04_study2_mllm_analysis.ipynb`](statistical_analysis/04_study2_mllm_analysis.ipynb) with helper modules in [`utilities/`](utilities/).
   - Evaluates `Gemini 3.5 Flash-Lite` across four progressive modality stages using a two-step independent Pre-test and Post-test prediction protocol (`STAI-AA`, `STAI-AP`, and `STAI-Total`) followed by `Pre - Post` improvement subtraction:
     - **Experiment 1 (`Video Only`)**: Silent video frames only.
     - **Experiment 2 (`Video + Audio, Non-Semantic Vocal Acoustics Only`)**: Visual cues + non-semantic vocal prosody (spoken word meaning strictly prohibited).
     - **Experiment 3 (`Video + Audio + Speech Context`)**: Full audiovisual + spoken semantic context.
     - **Experiment 4 (`Video + Audio + Speech Context + Physio`)**: Augments Experiment 3 with concurrent Heart Rate / HRV (`HeartRateInterpreter`: `mean_hr`, `rmssd`, `sdnn`, `sbp`, `dbp`) and Facial Thermal Imaging (`ThermalInterpreter`: `nose_tip_temperature`, `left_cheek_temperature`, `right_cheek_temperature`) via `InterpreterBuilder`.
4. **Publication Figures ([`chart/`](chart/))**:
   - Located in [`chart/manuscript_figures.ipynb`](chart/manuscript_figures.ipynb) and [`chart/generate_publication_figures.py`](chart/generate_publication_figures.py).

---

## Privacy & De-Identification Statement

All raw multimedia recordings (`*.mp4`, `*.wav`, thermal matrices) and original spreadsheets containing participant names or personally identifiable information (PII) have been strictly excluded from this repository in accordance with institutional ethics and participant privacy protections. Only de-identified tabular datasets indexed by anonymous 3-digit participant IDs (`004`–`115`) are included in [`data/`](data/).

---

## Repository Structure

```text
llm_pet_therapy/
├── mllm.ipynb                          # Study 2: Multimodal LLM (Experiments 1-4) Pre/Post evaluation notebook
├── data/                               # De-identified tabular datasets (N=100 main sample, n=92 multimodal cohort)
│   ├── study1_self_report.csv          # De-identified Study 1 item-level & scale-level dataset
│   ├── participants.csv                # Unified participant metadata & STAI subscales (STAI-Total, STAI-AA, STAI-AP)
│   ├── feature_definitions.csv         # Definitions and units for the 19 multimodal features
│   ├── features_pre.csv                # Pre-intervention values of the 19 multimodal features
│   ├── features_post.csv               # Post-intervention values of the 19 multimodal features
│   ├── features_delta.csv              # Post-minus-Pre change scores of the 19 multimodal features
│   └── cv/                             # Pre-registered 10x5 repeated stratified CV split definitions
├── feature_extraction/                 # Study 2: Multimodal feature extraction pipeline (Facial, Acoustic, Thermal, HRV)
├── statistical_analysis/               # Study 1 & Study 2: Statistical & ML notebooks, scripts, and output CSVs
│   ├── 01_study1_self_report_analysis.ipynb
│   ├── 02_study2_multimodal_statistical_analysis.ipynb
│   ├── 03_study2_multimodal_machine_learning.ipynb
│   ├── 04_study2_mllm_analysis.ipynb
│   └── outputs/                        # Reproducible CSV tables generated by the statistical, ML, & MLLM pipelines
├── chart/                              # High-resolution publication figure generation scripts & notebook
│   ├── manuscript_figures.ipynb
│   ├── generate_publication_figures.py
│   └── figures/
├── utilities/                          # Helper modules for mllm.ipynb (data loader, Gemini API helper, analysis helper)
└── env/                                # Configuration directory (.env is git-ignored)
```

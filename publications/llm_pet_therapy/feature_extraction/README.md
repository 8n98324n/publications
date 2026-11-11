# Multimodal Behavioral and Physiological Feature Extraction Pipeline (Study 2)

This directory contains the signal-processing and feature-extraction modules used to derive the 19 multimodal behavioral and physiological features analyzed in Study 2 (`Data Analysis 2`). To protect participant privacy and identity, raw identifiable video recordings (`.mp4`), audio recordings (`.wav`), and raw thermal frame matrices are excluded from this repository; the extracted, de-identified feature tables are provided in `../data/`.

## Overview of Extracted Modalities and Modules

| Modality | Extracted Features | Module(s) |
|---|---|---|
| **1. Visible Smile Coding** | `smile_prevalence_score` (0–4), `smile_intensity_score` (0–4) | `smile.py`, `smile_batch.py`, `config/prompts/smile_coding.txt` |
| **2. Facial Action Units (OpenFace)** | `au05_prevalence`, `au07_prevalence`, `au12_prevalence` | `au.py`, `tools_io.py` |
| **3. Speech Activity & Duration (VAD / Whisper)** | `response_duration_s`, `speaking_time_ratio`, `auto_silent_interval_proportion` | `vad.py`, `acoustic_intervals.py` |
| **4. Vocal Acoustics (openSMILE GeMAPS & Praat/Parselmouth)** | `f0_median_st`, `f0_iqr_st`, `auto_periodic_vocalization_proportion`, `jitter_local_pct`, `hnr` | `acoustic.py`, `audio_energy.py`, `jitter_diagnostics.py`, `jitter_support.py`, `vendor/opensmile/` |
| **5. Facial Infrared Thermography (TFAN 478-Landmark ROI)** | `nose_tip_temperature`, `left_cheek_temperature`, `right_cheek_temperature` | `thermal.py` |
| **6. Heart Rate & Heart Rate Variability (Polar H10 RR Intervals)** | `mean_hr`, `rmssd`, `sdnn` | `rr_trace.py`, `cardiovascular.py` |
| **7. Arterial Blood Pressure (Oscillometric Cuff)** | `sbp`, `dbp` | `cardiovascular.py` |
| **Pipeline Assembly & Quality Control** | Stage-level eligibility flags, paired deltas (`Post - Pre`), and schema validation | `assembly.py`, `process.py`, `diagnostic_schema.json` |

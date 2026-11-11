# psychai_publications

Study code for published work built on the [`psychai`](../psychai) library. Each
study occupies one folder under `publications/`.

```
publications/
  llm_moral_emotion/    Addressing the psychological small-sample problem with
                        multimodal large language models
```

## publications/llm_moral_emotion

Code and pre-computed feature tables for the study recognising moral emotions
(Elevation, Sympathy, Inspiration, Control) from multimodal recordings, using a
multimodal large language model and, for comparison, conventional
machine-learning classifiers and human observers.

### Entry points

- `final/experiments_1_9.ipynb` — the MLLM pipeline (manuscript Exp 1–9):
  baseline, multimodal expansion, inference-time prompting, fine-tuning, agentic
  reasoning, and the supplementary physiology-without-speech-semantics control
  (manuscript Exp 16, labelled Experiment 14 in the notebook). Calls the Gemini API.
- `final/experiment_10_13.ipynb` — the conventional-ML pipeline (manuscript
  Exp 10–13): per-modality Random Forest classification plus majority-vote late
  fusion on the pre-computed feature tables.
- `final/experiment_17_18.ipynb` — the supplementary cross-model comparison
  (manuscript Exp 17, Appendix D.2): six multimodal models rating still frames
  sampled from the stimulus-viewing recordings. The notebook also holds a run on
  the post-viewing recordings that the manuscript does not report. Calls several
  provider APIs.
- `feature_extraction/` — the eight notebooks that extract the per-modality
  features released as `df_1.csv` … `df_7.csv` from the raw recordings
  (`2_1_speech_audio_to_text.ipynb` transcribes the audio that
  `2_2_speech_text_to_feature.ipynb` scores). The recordings are not distributed.
- `statistical_analysis/` — eight notebooks that recompute the reported
  statistics from the committed files (Appendix E of the manuscript; see below).

### Layout

```
publications/llm_moral_emotion/
  emotion_recognition/emotion_classification_tasks.py   ML analysis engine
  utilities/                                            study-specific helpers
  env/configs/configs.py                                paths, model, instances
  env/requirements/requirements.txt                     pinned dependencies
  final/                                                notebooks and tables
  final/reproducibility/                                frozen statistical outputs
  feature_extraction/                                   per-modality notebooks
  statistical_analysis/                                 reproduction notebooks (Appendix E)
```

`env/configs/configs.py` resolves both repositories from one project root:
it expects `<root>/github/psychai/src` and
`<root>/github/psychai_publications/publications/llm_moral_emotion`. Set
`path_project_root` in your `.env` accordingly.

### Experiment numbering

The file names use the numbering in force when the analyses were run, which the
manuscript later revised. The code was deliberately left unrenumbered, so the
two differ:

| Code | Manuscript | Configuration |
|---|---|---|
| `experiment_1.csv` … `experiment_9.csv` | Exp 1–9 | unchanged |
| `experiment_10_5fold.csv` … `experiment_13_5fold.csv` | Exp 10–13 | reported late-fusion predictions (one acoustic vote) |
| `experiment_10.csv` … `experiment_13.csv` | Exp 10–13, superseded | original fusion output (two acoustic votes; Appendix C.2.4) |
| `experiment_14.csv` | **Exp 16** | vision + audio + physiology, no speech semantics |
| — | Exp 14, Exp 15 | human observer passes (rated outside these notebooks) |
| `experiment_17_18.ipynb`, run "Experiment 18" | **Exp 17** | six MLLMs on eight still frames from the stimulus-viewing (step-23) recordings |
| `experiment_17_18.ipynb`, run "Experiment 17" | not reported | the same models on the post-viewing (step-25) recordings; retained for transparency |
| `exp17_18_confusion_matrices.json`, keys `confusion_matrices_exp18` / `exp18` | **Exp 17** | the still-frame confusion matrices and check values of the reported run |
| `exp17_18_confusion_matrices.json`, keys `confusion_matrices_exp17` / `exp17` | not reported | the post-viewing run |

### Class labels and the discretization rule

- The four induction cells are coded `ELVT`, `SYMP`, `INSP`, `CTRL`
  (Elevation, Sympathy, Inspiration, Control). Earlier versions of the code and
  data used `ADMR` for the third cell; the label was renamed to `INSP` in the
  released code and in the derived `Group Name` column of every CSV. The numeric
  `Group` column (1–4) is the primary key and is unchanged.
- Every reported four-class result splits each continuous score at the
  **sample mean** (`threshold_type="mean"` in
  `data_analysis_helper.calculate_and_show_accuracy`). The helper also offers
  `median`, `truncated_mean_50`, `truncated_mean_80`, `q1`, and `q3`; these are
  provided for sensitivity analysis only and are not used for any value in the
  manuscript.

### Reproducibility

- The **late-fusion** results reproduce exactly from `final/machine_learning.csv`
  via the majority-voting rule in the notebook.
- The **per-modality classification** runs end to end from the `df_*.csv` tables,
  but per-subject predictions are not bit-identical across environments, because
  the Random Forest, SVM-RFE, and grouped cross-validation steps depend on
  library versions.
- The **model experiments** cannot be regenerated without provider API keys, and
  a hosted model does not reproduce frozen predictions bit for bit. The committed
  `experiment_*.csv` files are the authoritative records.
- `final/experiment_17_18.ipynb` ships with its outputs cleared, as the other
  notebooks do. Its confusion matrices are therefore recorded separately in
  `final/exp17_18_confusion_matrices.json`, together with the accuracy, macro-F1,
  and per-class recall values each matrix was checked against.

- The explanation analysis (Methods 2.5, Results 3.3, Appendix E.6) uses a
  26-cue schema coded in two passes over the 100 Exp 4 explanations.
  `final/exp4_coding_prompt.txt`, `final/exp4_coding_metadata.json`, and
  `final/exp4_feature_coding.csv` are the fixed prompt, the coding-model metadata,
  and the per-participant codes of the first pass, which supplies the five textual
  and eight nonverbal cues; its seven physiological codes carry no direction and
  are superseded. `final/exp4_physio_direction_prompt.txt` and
  `final/exp4_physio_direction_coding.csv` are the fixed prompt and the
  per-participant codes of the second pass, which supplies the thirteen
  directional physiological cues. The model identifier of the second pass was not
  recorded.
- `final/experiment_10_5fold.csv` … `experiment_13_5fold.csv` are the regenerated
  late-fusion predictions with a single acoustic vote (correlation pruning on);
  `final/machine_learning.csv` is the original per-modality prediction file,
  which still holds both acoustic runs.
- `final/reproducibility/` holds the frozen analysis outputs from which every
  statistic in the manuscript was taken: valid-N sample (`frozen_sample_snapshot.csv`,
  `exclusion_list.csv`), per-participant labels for all configurations
  (`labels_all_frozen.csv`), significance battery (`significance_frozen.csv`),
  per-class recall, the participant-clustered GEE and its contrasts
  (`gee_frozen.csv`, `gee_contrasts_frozen.csv`; Exp 1–13, nine planned and six
  cross-approach contrasts), the 2 × 2 ANOVAs (`anova_frozen.csv`),
  discretization sensitivity, the cleaned observer panel and its statistics, the
  self-report criterion correlations, the single-modality machine-learning
  statistics, and the first-pass explanation-coding associations. All were computed with
  `numpy.random.default_rng(42)`, 10,000 bootstrap resamples, and 10,000 label
  permutations.

- `statistical_analysis/` holds eight notebooks that recompute the statistics in
  the Results chapter and Appendices A, D and E from the committed files
  (`00_sample_and_labels.ipynb` plus one notebook per Appendix E section,
  `01_E1_…` to `07_E7_…`, run in numerical order; `statistical_analysis/README.md`
  gives the map and the verification protocol). Each notebook ends by comparing its
  output with the frozen files in `final/reproducibility/` or, where no frozen
  file exists, with the values printed in the manuscript.

See `REPRODUCIBILITY_GAP_REPORT.md` for the full account.

### Configuration

Copy `.env.example` to `.env` and fill in your own credentials. The `.env` file
is never committed, and no key appears in any source file.

### Data files

The `experiment_*.csv` and `df_*.csv` tables carry the model scores,
explanations, and condition labels needed to reproduce the reported analyses. Two
kinds of field were removed or redacted before release:

- Removed: local file-system provenance pointing at the undistributed raw
  recordings (`Video Path`, `Audio Folder`, `Output Folder`, `Merged File Path`,
  `Compressed Video Path`, `path_video_to_upload`, `Base Name`, `File Size`) and
  the recording timestamp fields parsed from the raw file names (`Month`, `Day`,
  `Hour`, `Minute`, `Second`). No reported analysis reads these fields.
- Redacted to placeholders, keeping the column so the upload and fine-tuning
  helpers still run: `url_bucket_file` and `tuning_file_uri`
  (`gs://BUCKET_NAME/videos/<video>.mp4`), `uploaded_file_video_merged` and
  `uploaded_file_video_only` (`<uploaded_file_handle>`).

Participants are identified only by serial number (`Participant SN` / `user_id`).
Model explanations remain in the original language of the study.

## License

Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Copyright (c) 2026 Ivan Liu. Each source file carries the same notice in its header.

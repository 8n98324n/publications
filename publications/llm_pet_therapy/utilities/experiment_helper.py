# Copyright (c) 2026 Ivan Liu.
# Licensed under CC BY 4.0.
"""Experiment driver for Pet Therapy Multimodal LLM Analysis (Study 2):
- Supports separate two-step upload and prediction of Pre-test (experiment_step="pre")
  and Post-test (experiment_step="post") videos, followed by computing the Pre-minus-Post
  subtraction (Improvement = Pre - Post).
- Evaluates the two psychometric subscales of the Spielberger State-Trait Anxiety Inventory (STAI-S):
    * stai_aa: STAI Anxiety-Absent (Absence of Calmness & Security) subscale (10 items, range 10-40)
    * stai_ap: STAI Anxiety-Present (Presence of Acute Tension) subscale (10 items, range 10-40)
    * stai_total = stai_aa + stai_ap (Total State Anxiety, 20 items, range 20-80)
- Caches intermediate video uploads and LLM outputs to pickle files.
"""

import os
import re
import pickle
import traceback
import threading
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import pandas as pd
import numpy as np


class ExperimentHelper:
    """
    ExperimentHelper manages multimodal LLM experiment pipelines for the Pet Therapy study.
    """

    def __init__(
        self,
        experiment_name,
        data_loader,
        df_helper,
        video_helper,
        llm_client,
        path_data_experiment,
        path_ffmpeg_bin="ffmpeg",
        path_ffprobe_bin="ffprobe",
    ):
        self.experiment_name = experiment_name
        self.data_loader = data_loader
        self.df_helper = df_helper
        self.video_helper = video_helper
        self.llm_client = llm_client
        self.path_data_experiment = path_data_experiment
        self.path_ffmpeg_bin = path_ffmpeg_bin
        self.path_ffprobe_bin = path_ffprobe_bin
        self.experiment_defined = False
        self.df_video_info = None

    def get_llm_client(self):
        return self.llm_client

    def set_llm_client(self, llm_client):
        self.llm_client = llm_client

    def copy(self):
        return ExperimentHelper(
            experiment_name=self.experiment_name,
            data_loader=self.data_loader,
            df_helper=self.df_helper,
            video_helper=self.video_helper,
            llm_client=self.llm_client,
            path_data_experiment=self.path_data_experiment,
            path_ffmpeg_bin=self.path_ffmpeg_bin,
            path_ffprobe_bin=self.path_ffprobe_bin,
        )

    def _normalize_llm_text(self, text: str) -> str:
        if text is None:
            return ""
        s = str(text)
        # Normalize full-width punctuation characters to standard ASCII equivalents
        s = (
            s.replace("\uff08", "(")
            .replace("\uff09", ")")
            .replace("\uff0c", ",")
            .replace("\uff1a", ":")
            .replace("\uff1d", "=")
        )
        return s.strip()

    def parse_stai_subscale_scores(self, text: str) -> dict:
        """
        Parse (stai_aa, stai_ap) = (X, Y) and explanation Z from raw LLM response text.
        Returns dict with keys: stai_aa, stai_ap, stai_total, explanation.
        """
        raw_str = self._normalize_llm_text(text)
        if not raw_str:
            return {"stai_aa": np.nan, "stai_ap": np.nan, "stai_total": np.nan, "explanation": ""}

        pattern_stai_2 = re.compile(
            r"stai[_\-\s]*aa\s*,\s*stai[_\-\s]*ap.*?\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)",
            re.IGNORECASE | re.DOTALL,
        )
        pattern_generic_2 = re.compile(
            r"\([^()\n]+\)\s*=\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)",
            re.IGNORECASE,
        )
        match = pattern_stai_2.search(raw_str) or pattern_generic_2.search(raw_str)
        if match:
            try:
                aa_val = float(match.group(1))
                ap_val = float(match.group(2))
                explanation = raw_str[match.end() :].lstrip("\n").strip()
                return {
                    "stai_aa": aa_val,
                    "stai_ap": ap_val,
                    "stai_total": aa_val + ap_val,
                    "explanation": explanation,
                }
            except Exception:
                pass
        return {"stai_aa": np.nan, "stai_ap": np.nan, "stai_total": np.nan, "explanation": ""}

    def segment_llm_response(self, df: pd.DataFrame, column: str = "LLM_Result") -> pd.DataFrame:
        """
        Extract the two STAI psychometric component scores (stai_aa, stai_ap),
        automatically compute stai_total = stai_aa + stai_ap, and extract the explanation
        from the LLM response.

        Expected strict output format:
          (stai_aa, stai_ap) = (X, Y)
          Z
        where:
          - X is the predicted STAI-AA score (Anxiety-Absent / Absence of Calmness & Security, integer 10-40)
          - Y is the predicted STAI-AP score (Anxiety-Present / Presence of Acute Tension, integer 10-40)
          - Z is the brief analytical explanation
        """
        df = df.copy()
        for col in ["stai_aa", "stai_ap", "stai_total"]:
            if col not in df.columns:
                df[col] = np.nan
        if "explanation" not in df.columns:
            df["explanation"] = ""

        # Primary pattern: (stai_aa, stai_ap) = (X, Y)
        pattern_stai_2 = re.compile(
            r"stai[_\-\s]*aa\s*,\s*stai[_\-\s]*ap.*?\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)",
            re.IGNORECASE | re.DOTALL,
        )
        # Fallback generic 2-tuple pattern: (...) = (X, Y)
        pattern_generic_2 = re.compile(
            r"\([^()\n]+\)\s*=\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)",
            re.IGNORECASE,
        )

        for i, raw in enumerate(df[column].fillna("").values):
            raw_str = self._normalize_llm_text(str(raw))
            if not raw_str:
                continue

            match = pattern_stai_2.search(raw_str) or pattern_generic_2.search(raw_str)
            if match:
                try:
                    aa_val = float(match.group(1))
                    ap_val = float(match.group(2))
                    total_val = aa_val + ap_val

                    df.iloc[i, df.columns.get_loc("stai_aa")] = aa_val
                    df.iloc[i, df.columns.get_loc("stai_ap")] = ap_val
                    df.iloc[i, df.columns.get_loc("stai_total")] = total_val

                    explanation = raw_str[match.end() :].lstrip("\n").strip()
                    if explanation:
                        df.iloc[i, df.columns.get_loc("explanation")] = explanation
                except Exception:
                    pass

        return df

    def compute_pre_post_difference(
        self,
        df_pre: pd.DataFrame,
        df_post: pd.DataFrame,
        path_pickle_diff: str = None,
    ) -> pd.DataFrame:
        """
        Combine independent Pre-test (df_pre) and Post-test (df_post) MLLM predictions
        and compute the subtracted improvement scores (Improvement = Pre - Post):
          - stai_aa_imp_pred    = stai_aa_pre_pred - stai_aa_post_pred
          - stai_ap_imp_pred    = stai_ap_pre_pred - stai_ap_post_pred
          - stai_total_imp_pred = stai_total_pre_pred - stai_total_post_pred

        This matches Study 1's exact formula:
          STAI_improvement = STAI_pre - STAI_post
        where positive values indicate anxiety reduction (improvement) after the intervention.
        """
        if df_pre is None or df_post is None:
            raise ValueError("Both df_pre and df_post must be provided to compute Pre - Post differences.")

        pre_seg = self.segment_llm_response(df_pre.copy())
        post_seg = self.segment_llm_response(df_post.copy())

        pre_cols_rename = {
            "stai_aa": "stai_aa_pre_pred",
            "stai_ap": "stai_ap_pre_pred",
            "stai_total": "stai_total_pre_pred",
            "LLM_Result": "LLM_Result_pre",
            "explanation": "explanation_pre",
        }
        post_cols_rename = {
            "stai_aa": "stai_aa_post_pred",
            "stai_ap": "stai_ap_post_pred",
            "stai_total": "stai_total_post_pred",
            "LLM_Result": "LLM_Result_post",
            "explanation": "explanation_post",
        }

        meta_cols = [
            c
            for c in [
                "Participant SN",
                "user_id",
                "condition",
                "Group",
                "Group Name",
                "Petting",
                "Feeding",
                "STAI_pre",
                "STAI_post",
                "STAI_improvement",
                "STAI_AA_pre",
                "STAI_AA_post",
                "STAI_AA_improvement",
                "STAI_AP_pre",
                "STAI_AP_post",
                "STAI_AP_improvement",
                "in_prediction_scope",
            ]
            if c in pre_seg.columns
        ]

        df_left = pre_seg[meta_cols + list(pre_cols_rename.keys())].rename(
            columns=pre_cols_rename
        )
        df_right = post_seg[["Participant SN"] + list(post_cols_rename.keys())].rename(
            columns=post_cols_rename
        )

        df_diff = pd.merge(df_left, df_right, on="Participant SN", how="inner")

        for col in [
            "stai_aa_pre_pred",
            "stai_ap_pre_pred",
            "stai_total_pre_pred",
            "stai_aa_post_pred",
            "stai_ap_post_pred",
            "stai_total_post_pred",
        ]:
            df_diff[col] = pd.to_numeric(df_diff[col], errors="coerce")

        # Subtract Post-test from Pre-test (Improvement = Pre - Post)
        df_diff["stai_aa_imp_pred"] = (
            df_diff["stai_aa_pre_pred"] - df_diff["stai_aa_post_pred"]
        )
        df_diff["stai_ap_imp_pred"] = (
            df_diff["stai_ap_pre_pred"] - df_diff["stai_ap_post_pred"]
        )
        df_diff["stai_total_imp_pred"] = (
            df_diff["stai_total_pre_pred"] - df_diff["stai_total_post_pred"]
        )

        # Populate primary (stai_aa, stai_ap, stai_total) columns with the Pre - Post difference
        df_diff["stai_aa"] = df_diff["stai_aa_imp_pred"]
        df_diff["stai_ap"] = df_diff["stai_ap_imp_pred"]
        df_diff["stai_total"] = df_diff["stai_total_imp_pred"]

        if path_pickle_diff:
            os.makedirs(os.path.dirname(path_pickle_diff), exist_ok=True)
            df_diff.to_pickle(path_pickle_diff)
            print(f"[INFO] Subtracted Pre-Post difference DataFrame saved to: {path_pickle_diff}")

        return df_diff

    def _is_valid_uploaded_file(self, uploaded_obj) -> bool:
        """Check if a Gemini File object is still active and not expired."""
        if uploaded_obj is None or pd.isna(uploaded_obj):
            return False
        if isinstance(uploaded_obj, str):
            return False
        if hasattr(uploaded_obj, "expiration_time") and uploaded_obj.expiration_time is not None:
            try:
                return datetime.now(timezone.utc) < uploaded_obj.expiration_time
            except Exception:
                return False
        return hasattr(uploaded_obj, "uri")

    def _compress_and_upload_one_file(
        self,
        source_path_str: str,
        cached_upload_obj,
        cached_compressed_path_str: str,
        enable_compression: bool,
        temp_dir: Path,
        temp_suffix: str,
        scale_width: int,
        fps: int,
        crf: int,
        sn: str,
        stage_label: str,
        verbose: bool,
    ):
        """Compress a single video (pre or post) if needed and upload to Gemini."""
        if not source_path_str or not os.path.exists(source_path_str):
            return None, "", False

        src_path = Path(source_path_str)
        compressed_path = temp_dir / f"{src_path.stem}{temp_suffix}"

        # Reuse valid Gemini upload if available
        if self._is_valid_uploaded_file(cached_upload_obj):
            if verbose:
                print(f"[Participant {sn} - {stage_label}] Video already uploaded and active.")
            return cached_upload_obj, str(compressed_path), False

        path_to_upload = src_path
        if enable_compression:
            if not compressed_path.exists() or compressed_path.stat().st_size < 1024:
                self.video_helper.compress_video(
                    input_path=str(src_path),
                    output_path=str(compressed_path),
                    path_ffmpeg_bin=self.path_ffmpeg_bin,
                    scale_width=scale_width,
                    fps=fps,
                    crf=crf,
                )
            path_to_upload = compressed_path

            if verbose and compressed_path.exists():
                orig_mb = os.path.getsize(src_path) / (1024 * 1024)
                comp_mb = os.path.getsize(compressed_path) / (1024 * 1024)
                print(
                    f"[Participant {sn} - {stage_label}] {orig_mb:.2f} MB -> {comp_mb:.2f} MB ({compressed_path.name})"
                )

        uploaded_obj = None
        try:
            if path_to_upload.exists():
                uploaded_obj = self.llm_client.upload_file_to_gemini(
                    current_file=str(path_to_upload)
                )
                if isinstance(uploaded_obj, str) or not hasattr(uploaded_obj, "uri"):
                    uploaded_obj = None
        except Exception as e:
            if verbose:
                print(f"[Participant {sn} - {stage_label}] Upload error: {e}")
            uploaded_obj = None

        return uploaded_obj, str(path_to_upload), True

    def process_single_video_info(
        self,
        idx,
        row,
        df,
        video_only,
        use_pre_and_post,
        enable_compression,
        temp_suffix,
        temp_dir,
        scale_width,
        fps,
        crf,
        verbose,
    ):
        """Process video compression and upload for one participant row (single step 'pre' or 'post', or 'pre_post')."""
        try:
            sn = str(row["Participant SN"]).zfill(3)
            step_label = str(row.get("Experiment Step", "VIDEO")).strip().upper()
            temp_dir = Path(temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)

            if video_only:
                pre_src = row.get("Pre Video Path", "")
                if step_label == "PRE":
                    main_src = row.get("Video Path", "") or pre_src
                else:
                    main_src = row.get("Video Path", "") or row.get("Post Video Path", "")
                col_pre_up = "uploaded_file_pre_video_only"
                col_main_up = "uploaded_file_video_only"
            else:
                pre_src = row.get("Pre Merged File Path", "")
                if step_label == "PRE":
                    main_src = row.get("Merged File Path", "") or pre_src
                else:
                    main_src = row.get("Merged File Path", "") or row.get("Post Merged File Path", "")
                col_pre_up = "uploaded_file_pre_video_merged"
                col_main_up = "uploaded_file_video_merged"

            pre_up_obj = None
            pre_comp_path = ""
            pre_newly_uploaded = False

            if use_pre_and_post and pre_src:
                pre_up_obj, pre_comp_path, pre_newly_uploaded = self._compress_and_upload_one_file(
                    source_path_str=pre_src,
                    cached_upload_obj=row.get(col_pre_up, None),
                    cached_compressed_path_str=row.get("Pre Compressed Video Path", ""),
                    enable_compression=enable_compression,
                    temp_dir=temp_dir,
                    temp_suffix=temp_suffix,
                    scale_width=scale_width,
                    fps=fps,
                    crf=crf,
                    sn=sn,
                    stage_label="PRE",
                    verbose=verbose,
                )

            main_stage_label = step_label if step_label in ("PRE", "POST") else "POST"
            post_up_obj, post_comp_path, post_newly_uploaded = self._compress_and_upload_one_file(
                source_path_str=main_src,
                cached_upload_obj=row.get(col_main_up, None),
                cached_compressed_path_str=row.get("Compressed Video Path", ""),
                enable_compression=enable_compression,
                temp_dir=temp_dir,
                temp_suffix=temp_suffix,
                scale_width=scale_width,
                fps=fps,
                crf=crf,
                sn=sn,
                stage_label=main_stage_label,
                verbose=verbose,
            )

            return (
                idx,
                pre_up_obj,
                pre_comp_path,
                post_up_obj,
                post_comp_path,
                None,
                (pre_newly_uploaded or post_newly_uploaded),
            )

        except Exception:
            err_msg = traceback.format_exc()
            print("[Error in process_single_video_info]\n" + err_msg)
            return idx, None, "", None, "", err_msg, False

    def prepare_video_info(
        self,
        df_video_info,
        enable_compression=True,
        video_only=False,
        use_pre_and_post=False,
        scale_width=640,
        fps=30,
        crf=32,
        temp_dir="",
        temp_suffix="_compressed.mp4",
        max_workers=4,
        verbose=False,
    ):
        """Compress and upload videos in parallel with automatic pickle checkpointing."""
        temp_dir = Path(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        for col in [
            "uploaded_file_pre_video_only",
            "uploaded_file_video_only",
            "uploaded_file_pre_video_merged",
            "uploaded_file_video_merged",
            "Pre Compressed Video Path",
            "Compressed Video Path",
            "LLM_Result",
        ]:
            if col not in df_video_info.columns:
                df_video_info[col] = None

        col_pre_up = (
            "uploaded_file_pre_video_only"
            if video_only
            else "uploaded_file_pre_video_merged"
        )
        col_post_up = (
            "uploaded_file_video_only"
            if video_only
            else "uploaded_file_video_merged"
        )

        pickle_lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.process_single_video_info,
                    idx,
                    row,
                    df_video_info,
                    video_only,
                    use_pre_and_post,
                    enable_compression,
                    temp_suffix,
                    temp_dir,
                    scale_width,
                    fps,
                    crf,
                    verbose,
                ): idx
                for idx, row in df_video_info.iterrows()
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    (
                        row_idx,
                        pre_up_obj,
                        pre_comp_path,
                        post_up_obj,
                        post_comp_path,
                        err,
                        newly_uploaded,
                    ) = future.result()

                    if err is not None:
                        print(f"[Error] Row {idx} encountered an error:\n{err}")
                        continue

                    if row_idx is not None:
                        if pre_up_obj is not None:
                            df_video_info.iat[
                                row_idx, df_video_info.columns.get_loc(col_pre_up)
                            ] = pre_up_obj
                            df_video_info.at[row_idx, "Pre Compressed Video Path"] = str(
                                pre_comp_path
                            )
                        if post_up_obj is not None:
                            df_video_info.iat[
                                row_idx, df_video_info.columns.get_loc(col_post_up)
                            ] = post_up_obj
                            df_video_info.at[row_idx, "Compressed Video Path"] = str(
                                post_comp_path
                            )

                        if newly_uploaded and self.path_pickle_df_video:
                            with pickle_lock:
                                self._save_video_pickle_safely(df_video_info)
                            if verbose:
                                print(
                                    f"[AutoSaved] Row {row_idx} uploaded and saved to {self.path_pickle_df_video}"
                                )
                except Exception:
                    print(
                        f"[Error] Future encountered an unexpected exception:\n{traceback.format_exc()}"
                    )

        # Verify that at least one video in the active batch was validly prepared/uploaded
        valid_uploads = sum(
            1
            for _, r in df_video_info.iterrows()
            if self._is_valid_uploaded_file(r.get(col_post_up, None))
        )
        if len(df_video_info) > 0 and valid_uploads == 0:
            raise RuntimeError(
                f"[CRITICAL] 0/{len(df_video_info)} videos were compressed/uploaded successfully in prepare_video_info(). "
                "Halting before LLM inference so empty (0-video) requests are not sent. Check FFmpeg / Gemini API logs above."
            )

        return df_video_info

    def _save_video_pickle_safely(self, df_video_info: pd.DataFrame):
        """Save df_video_info to self.path_pickle_df_video without overwriting unselected participants."""
        if not self.path_pickle_df_video:
            return
        os.makedirs(os.path.dirname(self.path_pickle_df_video), exist_ok=True)
        if os.path.exists(self.path_pickle_df_video):
            try:
                df_full = pd.read_pickle(self.path_pickle_df_video)
                if len(df_full) > len(df_video_info):
                    for col in df_video_info.columns:
                        if col not in df_full.columns:
                            df_full[col] = None
                    for _, r in df_video_info.iterrows():
                        sn = r["Participant SN"]
                        idxs = df_full.index[df_full["Participant SN"] == sn].tolist()
                        for i_full in idxs:
                            for col in [
                                "uploaded_file_pre_video_only",
                                "uploaded_file_video_only",
                                "uploaded_file_pre_video_merged",
                                "uploaded_file_video_merged",
                                "Pre Compressed Video Path",
                                "Compressed Video Path",
                            ]:
                                if col in df_video_info.columns:
                                    df_full.iat[i_full, df_full.columns.get_loc(col)] = r[col]
                    df_full.to_pickle(self.path_pickle_df_video)
                    return
            except Exception:
                pass
        df_video_info.to_pickle(self.path_pickle_df_video)

    def check_llm_result(self, result) -> bool:
        if (
            result is None
            or str(result).strip() == ""
            or str(result).strip().lower() in ("none", "nan", "[no valid response from model]", "[unexpected error]")
        ):
            return False
        parsed = self.parse_stai_subscale_scores(str(result))
        aa = parsed.get("stai_aa", np.nan)
        ap = parsed.get("stai_ap", np.nan)
        if pd.isna(aa) or pd.isna(ap):
            return False
        if float(aa) < 10 or float(aa) > 40 or float(ap) < 10 or float(ap) > 40:
            return False
        return True

    def process_single_participant(
        self,
        participant_sn,
        sub_df,
        enable_llm=True,
        video_only=False,
        use_pre_and_post=False,
        over_ride_existing=False,
        default_prompt_text=None,
        verbose=False,
        attach_video=True,
    ):
        """Run LLM inference for a single participant on the active video step (Pre or Post)."""
        sub_df = sub_df.loc[:, ~sub_df.columns.duplicated()]
        llm_values = sub_df["LLM_Result"].fillna("").astype(str)

        if any(self.check_llm_result(v) for v in llm_values) and not over_ride_existing:
            if verbose:
                print(
                    f"[Participant {participant_sn}] Valid existing LLM_Result detected. Skipping processing."
                )
            valid_v = next(v for v in llm_values if self.check_llm_result(v))
            return participant_sn, valid_v

        result = ""
        for _, row in sub_df.iterrows():
            if not enable_llm:
                continue

            prompt_to_use = default_prompt_text
            if "individual_prompt" in sub_df.columns:
                prompts_series = (
                    sub_df["individual_prompt"].fillna("").astype(str).str.strip()
                )
                non_empty = prompts_series[prompts_series != ""]
                if len(non_empty) > 0:
                    prompt_to_use = non_empty.iloc[0]

            uploaded_files = []
            if attach_video:
                if video_only:
                    pre_file = row.get("uploaded_file_pre_video_only", None)
                    main_file = row.get("uploaded_file_video_only", None)
                else:
                    pre_file = row.get("uploaded_file_pre_video_merged", None)
                    main_file = row.get("uploaded_file_video_merged", None)

                if use_pre_and_post and self._is_valid_uploaded_file(pre_file):
                    uploaded_files.append(pre_file)
                if self._is_valid_uploaded_file(main_file):
                    uploaded_files.append(main_file)

                for add_video in self.additional_video:
                    if add_video is not None:
                        uploaded_files.append(add_video)

                expected_videos = 2 if use_pre_and_post else 1
                if len(uploaded_files) < expected_videos:
                    print(
                        f"[ERROR] Participant {participant_sn}: Expected {expected_videos} uploaded video(s), "
                        f"but got {len(uploaded_files)}. Skipping LLM call so invalid response is not cached."
                    )
                    return participant_sn, ""

            if verbose:
                step_tag = str(row.get("Experiment Step", "video")).upper()
                print(
                    f"[Thread] Participant {participant_sn} ({step_tag}) sending request to LLM ({len(uploaded_files)} video(s), prompt len={len(prompt_to_use or '')})"
                )

            result = self.llm_client.chat(
                prompt=prompt_to_use,
                template_used=False,
                stream=False,
                verbose=verbose,
                extra_info=uploaded_files,
            )

        return participant_sn, result

    def get_prompt_text_template(
        self,
        template_experiment: str,
        template_cot: str = None,
        template_knowledge_injection: str = None,
        template_tools: str = None,
        tools_results: list = None,
        stai_pre_val=None,
    ) -> str:
        """Replace [[COT]], [[KNOWLEDGE]], [[TOOLS]], and [[STAI_PRE]] placeholders."""
        prompt_text = template_experiment

        if (
            template_tools is not None
            and tools_results is not None
            and len(tools_results) > 0
        ):
            try:
                tools_text = template_tools.format(*tools_results)
            except Exception:
                tools_text = "\n".join(str(r) for r in tools_results)
        else:
            tools_text = ""

        if "[[TOOLS]]" in prompt_text:
            prompt_text = prompt_text.replace("[[TOOLS]]", tools_text)

        cot_text = template_cot if template_cot is not None else ""
        if "[[COT]]" in prompt_text:
            prompt_text = prompt_text.replace("[[COT]]", cot_text)

        knowledge_text = (
            template_knowledge_injection
            if template_knowledge_injection is not None
            else ""
        )
        if "[[KNOWLEDGE]]" in prompt_text:
            prompt_text = prompt_text.replace("[[KNOWLEDGE]]", knowledge_text)

        if "[[STAI_PRE]]" in prompt_text:
            stai_str = (
                f"Participant's baseline pre-test state anxiety score (STAI_pre, range 20-80): {int(stai_pre_val)}"
                if stai_pre_val is not None and not pd.isna(stai_pre_val)
                else ""
            )
            prompt_text = prompt_text.replace("[[STAI_PRE]]", stai_str)

        return prompt_text

    def get_user_features_from_df(self, df, participant_sn: int) -> dict:
        if df is None or df.empty:
            return {}
        mask = df["user_id"].astype(int) == int(participant_sn)
        if mask.any():
            return df.loc[mask].iloc[0].to_dict()
        return {}

    def update_df_with_individual_prompts(
        self,
        df_input: pd.DataFrame,
        df_features: pd.DataFrame,
        tools: list,
        template_main: str,
        template_cot: str,
        template_knowledge_injection: str,
        template_tools: str,
        include_stai_pre_in_prompt: bool = False,
    ) -> pd.DataFrame:
        df_extended = df_input.copy()
        if "individual_prompt" not in df_extended.columns:
            df_extended["individual_prompt"] = ""

        for idx, row in df_extended.iterrows():
            participant_sn = int(row["Participant SN"])
            features = self.get_user_features_from_df(df_features, participant_sn)

            interpreter_results = []
            for tool in tools or []:
                if callable(tool):
                    interpreter_results.append(tool(features))
                elif hasattr(tool, "upload_and_get_text_output"):
                    interpreter_results.append(tool.upload_and_get_text_output(features))

            stai_pre_val = row.get("STAI_pre", features.get("STAI_pre", None)) if include_stai_pre_in_prompt else None

            prompt = self.get_prompt_text_template(
                template_experiment=template_main,
                template_cot=template_cot,
                template_knowledge_injection=template_knowledge_injection,
                template_tools=template_tools,
                tools_results=interpreter_results,
                stai_pre_val=stai_pre_val,
            )
            df_extended.at[idx, "individual_prompt"] = prompt

        return df_extended

    def define_experiment(
        self,
        participant_sn_range,
        path_pickle_df_experiment_results,
        video_only: bool,
        experiment_step="pre",
        interpreters: list = None,
        template_main: str = "",
        template_cot: str = "",
        template_knowledge_injection: str = "",
        template_tools: str = "",
        path_pickle_df_video: str = "",
        path_compressed_video: str = "",
        attach_video: bool = True,
        use_pre_and_post: bool = False,
        include_stai_pre_in_prompt: bool = False,
        additional_video: list = None,
        df_video_info=None,
        verbose: bool = True,
    ):
        """
        Define experiment configuration for a specific step:
          - experiment_step="pre"  (Step 1: upload and evaluate Pre-test video independently)
          - experiment_step="post" (Step 2: upload and evaluate Post-test video independently)
        """
        self.participant_sn_range = [int(x) for x in participant_sn_range]
        self.path_pickle_df_video = path_pickle_df_video
        self.path_pickle_df_experiment_results = path_pickle_df_experiment_results
        self.video_only = video_only
        self.experiment_step = str(experiment_step).strip().lower()
        self.interpreters = interpreters or []
        self.template_main = template_main
        self.template_cot = template_cot
        self.template_knowledge_injection = template_knowledge_injection
        self.template_tools = template_tools
        self.attach_video = attach_video
        self.use_pre_and_post = use_pre_and_post
        self.include_stai_pre_in_prompt = include_stai_pre_in_prompt
        self.additional_video = additional_video or []
        self.verbose = verbose
        self.path_compressed_video = path_compressed_video
        self.experiment_defined = True

        self.df_users, self.df_features, self.df_fdef = (
            self.data_loader.load_participants_and_features(only_prediction_scope=False)
        )

        if df_video_info is None:
            self.df_video_info = self.data_loader.merge_and_get_video_with_audio(
                path_data_experiment=self.path_data_experiment,
                df_user_info=self.df_users,
                participant_sn_range=self.participant_sn_range,
                experiment_name=self.experiment_name,
                experiment_step=self.experiment_step,
                path_pickle=self.path_pickle_df_video,
                path_ffmpeg_bin=self.path_ffmpeg_bin,
                path_ffprobe_bin=self.path_ffprobe_bin,
            )
        else:
            self.df_video_info = df_video_info.copy()

        if "LLM_Result" not in self.df_video_info.columns:
            self.df_video_info["LLM_Result"] = ""

    def get_df_video_path(self) -> pd.DataFrame:
        if self.path_pickle_df_video and os.path.exists(self.path_pickle_df_video):
            try:
                df_cached = pd.read_pickle(self.path_pickle_df_video)
                if df_cached is not None and len(df_cached) > 0:
                    self.df_video_info = df_cached
                    return self.df_video_info
            except Exception:
                pass
        if self.df_video_info is not None and len(self.df_video_info) > 0:
            return self.df_video_info
        self.df_video_info = self.data_loader.merge_and_get_video_with_audio(
            path_data_experiment=self.path_data_experiment,
            df_user_info=self.df_users,
            participant_sn_range=self.participant_sn_range,
            experiment_name=self.experiment_name,
            experiment_step=self.experiment_step,
            path_pickle=self.path_pickle_df_video,
            path_ffmpeg_bin=self.path_ffmpeg_bin,
            path_ffprobe_bin=self.path_ffprobe_bin,
        )
        return self.df_video_info

    def get_df_experiment_results(self) -> pd.DataFrame:
        if not self.path_pickle_df_experiment_results:
            return None
        if not os.path.exists(self.path_pickle_df_experiment_results):
            return None
        df_res = pd.read_pickle(self.path_pickle_df_experiment_results)
        return self.segment_llm_response(df_res)

    def run_llm_on_files_multithread(
        self,
        df_experiment_results,
        enable_llm=True,
        video_only=False,
        use_pre_and_post=False,
        max_workers=3,
        task_timeout=180,
        over_ride_existing=False,
        default_prompt_text=None,
        verbose=False,
        attach_video=True,
    ):
        if df_experiment_results is None or df_experiment_results.empty:
            print("[Error]: DataFrame is empty. LLM inference cannot proceed.")
            return None

        df_selected = df_experiment_results[
            pd.to_numeric(df_experiment_results["Participant SN"], errors="coerce").isin(
                self.participant_sn_range
            )
        ].copy()

        grouped = df_selected.groupby("Participant SN")
        pickle_lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for participant_sn, sub_df in grouped:
                future = executor.submit(
                    self.process_single_participant,
                    participant_sn,
                    sub_df,
                    enable_llm,
                    video_only,
                    use_pre_and_post,
                    over_ride_existing,
                    default_prompt_text,
                    verbose,
                    attach_video,
                )
                future.participant_sn = participant_sn
                futures.append(future)

            for future in as_completed(futures):
                sn = future.participant_sn
                try:
                    participant_sn, result = future.result(timeout=task_timeout)
                    if self.check_llm_result(result):
                        df_selected.loc[
                            df_selected["Participant SN"] == participant_sn, "LLM_Result"
                        ] = result
                        df_selected = self.segment_llm_response(df_selected)

                        df_experiment_results.loc[
                            df_experiment_results["Participant SN"] == participant_sn,
                            "LLM_Result",
                        ] = result
                        df_experiment_results = self.segment_llm_response(
                            df_experiment_results
                        )

                        with pickle_lock:
                            if self.path_pickle_df_experiment_results:
                                os.makedirs(
                                    os.path.dirname(self.path_pickle_df_experiment_results),
                                    exist_ok=True,
                                )
                                df_experiment_results.to_pickle(
                                    self.path_pickle_df_experiment_results
                                )
                        if verbose:
                            print(
                                f"[AUTOSAVE-PARTICIPANT] {participant_sn} saved to {self.path_pickle_df_experiment_results}"
                            )
                    else:
                        if verbose:
                            print(
                                f"[NO RESULT] Participant {sn} did not produce a valid LLM output."
                            )
                except TimeoutError:
                    if verbose:
                        print(f"[TIMEOUT] Participant {sn} exceeded {task_timeout}s.")
                except Exception as e:
                    if verbose:
                        print(f"[Error]: Participant {sn} encountered an exception: {e}")

        return df_selected

    def run_exp_pet_therapy(
        self,
        max_workers=3,
        over_ride_existing=False,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Main runner for Pet Therapy MLLM experiments (analogous to run_exp_moral_elevation).
        """
        print(f"=== run_exp_pet_therapy ({self.experiment_step.upper()}) start ===")
        setting_fps = 30
        setting_crf = 32
        setting_width = 640
        setting_compression = True

        self.df_video_info = self.get_df_video_path()
        self.df_video_info = self.prepare_video_info(
            df_video_info=self.df_video_info,
            enable_compression=setting_compression,
            video_only=self.video_only,
            use_pre_and_post=self.use_pre_and_post,
            scale_width=setting_width,
            fps=setting_fps,
            crf=setting_crf,
            temp_dir=self.path_compressed_video,
            max_workers=max_workers,
            verbose=verbose,
        )

        df_experiment_results = None
        if self.path_pickle_df_experiment_results and not over_ride_existing:
            try:
                df_experiment_results = self.get_df_experiment_results()
            except Exception:
                df_experiment_results = None

        if df_experiment_results is None:
            df_experiment_results = self.update_df_with_individual_prompts(
                df_input=self.df_video_info,
                df_features=self.df_features,
                tools=self.interpreters,
                template_main=self.template_main,
                template_cot=self.template_cot,
                template_knowledge_injection=self.template_knowledge_injection,
                template_tools=self.template_tools,
                include_stai_pre_in_prompt=self.include_stai_pre_in_prompt,
            )
        else:
            # Refresh uploaded file references from df_video_info into cached df_experiment_results
            upload_cols = [
                c
                for c in [
                    "uploaded_file_pre_video_only",
                    "uploaded_file_video_only",
                    "uploaded_file_pre_video_merged",
                    "uploaded_file_video_merged",
                    "Pre Compressed Video Path",
                    "Compressed Video Path",
                ]
                if c in self.df_video_info.columns
            ]
            for idx, row in df_experiment_results.iterrows():
                sn = row["Participant SN"]
                m = self.df_video_info[self.df_video_info["Participant SN"] == sn]
                if not m.empty:
                    for col in upload_cols:
                        if col not in df_experiment_results.columns:
                            df_experiment_results[col] = None
                        df_experiment_results.iat[
                            idx, df_experiment_results.columns.get_loc(col)
                        ] = m.iloc[0][col]

            df_experiment_results = self.update_df_with_individual_prompts(
                df_input=df_experiment_results,
                df_features=self.df_features,
                tools=self.interpreters,
                template_main=self.template_main,
                template_cot=self.template_cot,
                template_knowledge_injection=self.template_knowledge_injection,
                template_tools=self.template_tools,
                include_stai_pre_in_prompt=self.include_stai_pre_in_prompt,
            )

        df_result = self.run_llm_on_files_multithread(
            df_experiment_results=df_experiment_results,
            enable_llm=True,
            video_only=self.video_only,
            use_pre_and_post=self.use_pre_and_post,
            max_workers=max_workers,
            over_ride_existing=over_ride_existing,
            default_prompt_text=None,
            verbose=verbose,
            attach_video=self.attach_video,
        )

        return df_result

    def run_trial_before_formal(
        self,
        n_trial: int = 1,
        over_ride_existing: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Trial Run (Pilot / Debugging Run) on the first `n_trial` participant(s) before formal execution.
        Verifies end-to-end:
          1. Raw / audio-merged video file existence
          2. FFmpeg video compression (with automatic codec selection: libx265/libx264/libopenh264/mpeg4)
          3. Gemini File API upload (verifying a valid active URI so 0-video requests never occur)
          4. Gemini MLLM inference and strict parsing of (stai_aa, stai_ap) in [10, 40]
        Caches the uploaded video and valid LLM result into the formal pickle files so the subsequent
        formal run (over_ride_existing=False) automatically reuses the completed trial participant(s).
        """
        full_sn_range = list(self.participant_sn_range)
        trial_sns = full_sn_range[: max(1, int(n_trial))]
        step_tag = self.experiment_step.upper()
        mode_tag = "Video Only (Silent)" if self.video_only else "Video + Audio (Merged)"
        print(
            f"=== [TRIAL RUN / DEBUG CHECK] Step={step_tag} | Mode={mode_tag} | "
            f"Testing {len(trial_sns)} participant(s): {trial_sns} ==="
        )

        df_full_video = self.get_df_video_path()
        df_trial_video = (
            df_full_video[
                pd.to_numeric(df_full_video["Participant SN"], errors="coerce").isin(trial_sns)
            ]
            .copy()
            .reset_index(drop=True)
        )
        if df_trial_video.empty:
            raise RuntimeError(f"[TRIAL FAILED] No video rows found for trial participant(s): {trial_sns}")

        # 1. Compress & upload only the trial subset
        df_trial_video = self.prepare_video_info(
            df_video_info=df_trial_video,
            enable_compression=True,
            video_only=self.video_only,
            use_pre_and_post=self.use_pre_and_post,
            scale_width=640,
            fps=30,
            crf=32,
            temp_dir=self.path_compressed_video,
            max_workers=min(2, len(trial_sns)),
            verbose=verbose,
        )

        # Merge uploaded file handles back into self.df_video_info
        col_up = "uploaded_file_video_only" if self.video_only else "uploaded_file_video_merged"
        for col in [col_up, "Compressed Video Path"]:
            if col not in self.df_video_info.columns:
                self.df_video_info[col] = None
        for _, r in df_trial_video.iterrows():
            sn = r["Participant SN"]
            idxs = self.df_video_info.index[self.df_video_info["Participant SN"] == sn].tolist()
            for idx_full in idxs:
                self.df_video_info.iat[idx_full, self.df_video_info.columns.get_loc(col_up)] = r[col_up]
                self.df_video_info.at[idx_full, "Compressed Video Path"] = r["Compressed Video Path"]

        # 2. Prepare prompts and run LLM on the trial subset
        df_experiment_results = None
        if self.path_pickle_df_experiment_results and os.path.exists(self.path_pickle_df_experiment_results):
            try:
                df_experiment_results = self.get_df_experiment_results()
            except Exception:
                df_experiment_results = None

        if df_experiment_results is None:
            df_experiment_results = self.update_df_with_individual_prompts(
                df_input=self.df_video_info,
                df_features=self.df_features,
                tools=self.interpreters,
                template_main=self.template_main,
                template_cot=self.template_cot,
                template_knowledge_injection=self.template_knowledge_injection,
                template_tools=self.template_tools,
                include_stai_pre_in_prompt=self.include_stai_pre_in_prompt,
            )
        else:
            for idx_full, row_full in df_experiment_results.iterrows():
                sn = row_full["Participant SN"]
                m = self.df_video_info[self.df_video_info["Participant SN"] == sn]
                if not m.empty:
                    if col_up not in df_experiment_results.columns:
                        df_experiment_results[col_up] = None
                    df_experiment_results.iat[
                        idx_full, df_experiment_results.columns.get_loc(col_up)
                    ] = m.iloc[0][col_up]
                    df_experiment_results.at[idx_full, "Compressed Video Path"] = m.iloc[0]["Compressed Video Path"]
            df_experiment_results = self.update_df_with_individual_prompts(
                df_input=df_experiment_results,
                df_features=self.df_features,
                tools=self.interpreters,
                template_main=self.template_main,
                template_cot=self.template_cot,
                template_knowledge_injection=self.template_knowledge_injection,
                template_tools=self.template_tools,
                include_stai_pre_in_prompt=self.include_stai_pre_in_prompt,
            )

        try:
            self.participant_sn_range = trial_sns
            df_trial_res = self.run_llm_on_files_multithread(
                df_experiment_results=df_experiment_results,
                enable_llm=True,
                video_only=self.video_only,
                use_pre_and_post=self.use_pre_and_post,
                max_workers=min(2, len(trial_sns)),
                over_ride_existing=over_ride_existing,
                default_prompt_text=None,
                verbose=verbose,
                attach_video=self.attach_video,
            )
        finally:
            self.participant_sn_range = full_sn_range

        # 3. Verify trial predictions
        df_trial_res = self.segment_llm_response(df_trial_res)
        for _, r in df_trial_res.iterrows():
            sn = r["Participant SN"]
            comp_p = str(r.get("Compressed Video Path", ""))
            up_obj = r.get(col_up, None)
            aa = r.get("stai_aa", np.nan)
            ap = r.get("stai_ap", np.nan)
            tot = r.get("stai_total", np.nan)
            uri_str = getattr(up_obj, "uri", "") if up_obj is not None else ""
            comp_kb = os.path.getsize(comp_p) / 1024.0 if comp_p and os.path.exists(comp_p) else 0.0
            print(
                f"  [Participant {sn} ({step_tag})] Compressed={comp_kb:.1f} KB | "
                f"Gemini URI={'OK' if uri_str else 'MISSING'} | "
                f"Predicted (stai_aa={aa}, stai_ap={ap}, stai_total={tot})"
            )
            if not self.check_llm_result(r.get("LLM_Result", "")):
                raise RuntimeError(
                    f"[TRIAL FAILED] Participant {sn} ({step_tag}) did not produce valid (stai_aa, stai_ap) scores in [10, 40]. "
                    f"Raw LLM_Result: {r.get('LLM_Result', '')}"
                )

        print(f"=== [TRIAL RUN PASSED] All {len(df_trial_res)} trial participant(s) verified successfully! ===\n")
        return df_trial_res

    # Alias for seamless compatibility with experiments_1_9.ipynb calls
    run_exp_moral_elevation = run_exp_pet_therapy

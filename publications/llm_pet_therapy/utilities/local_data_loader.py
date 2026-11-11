# Copyright (c) 2026 Ivan Liu.
# Licensed under CC BY 4.0.
"""Project data loader for Pet Therapy Study 2 (Multimodal LLM Analysis):
- Automatically extracts video/audio archives from the raw multimodal archive folder if not yet extracted.
- Loads participant metadata including both STAI psychometric subscales from Study 1:
    * STAI_AA (Anxiety-Absent / Absence of Calmness & Security Subscale, 10 items, range 10-40)
    * STAI_AP (Anxiety-Present / Presence of Acute Tension Subscale, 10 items, range 10-40)
    * STAI (Total State Anxiety = STAI_AA + STAI_AP, 20 items, range 20-80)
  for Pre-test, Post-test, and Improvement (Pre - Post), along with physiological/multimodal features.
- Merges pre and post video (.mp4) with corresponding audio (.wav) via FFmpeg.
"""

import os
import pickle
import zipfile
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np


class DataLoader:
    """Manage loading, caching, archive extraction, and audio-video merging for Pet Therapy Study 2."""

    def __init__(
        self,
        path_tabular_data: str = "",
        path_multimodal_zips: str = "",
        path_ps035_workspace: str = "",
    ):
        self.path_tabular_data = path_tabular_data
        self.path_multimodal_zips = path_multimodal_zips
        self.path_ps035_workspace = path_ps035_workspace

    def save_to_cache(self, data, cache_file):
        """Save data to a pickle cache file."""
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)

    def load_from_cache(self, cache_file):
        """Load data from a pickle cache file."""
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    def ensure_tabular_data(self):
        """Ensure tabular CSVs exist in path_tabular_data; if missing, extract from Archive/analysis2-20260928_v1.zip."""
        os.makedirs(self.path_tabular_data, exist_ok=True)
        part_csv = os.path.join(self.path_tabular_data, "participants.csv")
        if os.path.exists(part_csv):
            return

        archive_zip = os.path.join(
            self.path_ps035_workspace, "Archive", "analysis2-20260928_v1.zip"
        )
        if os.path.exists(archive_zip):
            print(f"[INFO] Extracting tabular data from {archive_zip}...")
            with zipfile.ZipFile(archive_zip, "r") as zf:
                for member in zf.namelist():
                    if member.startswith("04_复现输入与参考结果/data/") and not member.endswith("/"):
                        filename = os.path.basename(member)
                        target_path = os.path.join(self.path_tabular_data, filename)
                        with zf.open(member) as src, open(target_path, "wb") as dst:
                            dst.write(src.read())

    def load_participants_and_features(self, only_prediction_scope: bool = True):
        """
        Load participants.csv and feature tables (pre, post, delta), mapping conditions to Group IDs:
          1: COMB (combined: Petting=1, Feeding=1)
          2: PET  (petting:  Petting=1, Feeding=0)
          3: FEED (feeding:  Petting=0, Feeding=1)
          4: OBS  (observation: Petting=0, Feeding=0)
        """
        self.ensure_tabular_data()
        part_path = os.path.join(self.path_tabular_data, "participants.csv")
        df_part = pd.read_csv(part_path, dtype={"participant_id": str})
        df_part["participant_id"] = df_part["participant_id"].str.zfill(3)
        df_part["user_id"] = df_part["participant_id"].astype(int)
        df_part["Participant SN"] = df_part["participant_id"]

        condition_to_group = {
            "combined": 1,
            "petting": 2,
            "feeding": 3,
            "observation": 4,
        }
        group_names = {1: "COMB", 2: "PET", 3: "FEED", 4: "OBS"}

        df_part["Group"] = df_part["condition"].str.lower().map(condition_to_group)
        df_part["Group Name"] = df_part["Group"].map(group_names)

        if only_prediction_scope and "in_prediction_scope" in df_part.columns:
            mask = df_part["in_prediction_scope"].astype(str).str.lower() == "true"
            df_part = df_part[mask].reset_index(drop=True)

        # Load feature tables
        df_pre = pd.read_csv(
            os.path.join(self.path_tabular_data, "features_pre.csv"),
            dtype={"participant_id": str},
        )
        df_post = pd.read_csv(
            os.path.join(self.path_tabular_data, "features_post.csv"),
            dtype={"participant_id": str},
        )
        df_delta = pd.read_csv(
            os.path.join(self.path_tabular_data, "features_delta.csv"),
            dtype={"participant_id": str},
        )
        df_fdef = pd.read_csv(
            os.path.join(self.path_tabular_data, "feature_definitions.csv")
        )

        for d, pfx in [(df_pre, "pre"), (df_post, "post"), (df_delta, "delta")]:
            d["user_id"] = d["participant_id"].str.zfill(3).astype(int)

        df_pre_prefixed = df_pre.rename(
            columns={
                c: f"pre_{c}"
                for c in df_pre.columns
                if c not in ("participant_id", "user_id")
            }
        )
        df_post_prefixed = df_post.rename(
            columns={
                c: f"post_{c}"
                for c in df_post.columns
                if c not in ("participant_id", "user_id")
            }
        )
        df_delta_prefixed = df_delta.rename(
            columns={
                c: f"delta_{c}"
                for c in df_delta.columns
                if c not in ("participant_id", "user_id")
            }
        )

        df_features = (
            df_part.merge(
                df_pre_prefixed.drop(columns=["participant_id"], errors="ignore"),
                on="user_id",
                how="left",
            )
            .merge(
                df_post_prefixed.drop(columns=["participant_id"], errors="ignore"),
                on="user_id",
                how="left",
            )
            .merge(
                df_delta_prefixed.drop(columns=["participant_id"], errors="ignore"),
                on="user_id",
                how="left",
            )
        )

        return df_part, df_features, df_fdef

    def ensure_multimodal_files_extracted(self, path_data_experiment: str):
        """
        Check if raw videos and audios are extracted in path_data_experiment.
        If not, automatically extract the raw video/audio zip archives
        and organize them into standardized folders:
          - {path_data_experiment}/video/{sn}/{sn}_{pre|post}.mp4
          - {path_data_experiment}/audio/{sn}/{sn}_{pre|post}.wav
          - {path_data_experiment}/rr/{sn}/{sn}_{pre|post}.txt
        """
        video_root = os.path.join(path_data_experiment, "video")
        audio_root = os.path.join(path_data_experiment, "audio")
        rr_root = os.path.join(path_data_experiment, "rr")

        # Check if already extracted (e.g., 004_post.mp4 and 004_post.wav exist)
        sample_video = os.path.join(video_root, "004", "004_post.mp4")
        sample_audio = os.path.join(audio_root, "004", "004_post.wav")
        if os.path.exists(sample_video) and os.path.exists(sample_audio):
            return video_root, audio_root, rr_root

        os.makedirs(video_root, exist_ok=True)
        os.makedirs(audio_root, exist_ok=True)
        os.makedirs(rr_root, exist_ok=True)

        zip_video = os.path.join(self.path_multimodal_zips, "视频及RR.zip")
        zip_audio1 = os.path.join(self.path_multimodal_zips, "音频1.zip")
        zip_audio2 = os.path.join(self.path_multimodal_zips, "音频2.zip")

        if os.path.exists(zip_video) and not os.path.exists(sample_video):
            print(f"[INFO] Extracting videos and RR from {zip_video} ...")
            with zipfile.ZipFile(zip_video, "r") as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    parts = Path(info.filename).parts
                    fname = parts[-1]
                    if fname.lower().endswith(".mp4") and len(parts) >= 2:
                        sn = parts[-2].zfill(3)
                        target_dir = os.path.join(video_root, sn)
                        os.makedirs(target_dir, exist_ok=True)
                        target_file = os.path.join(target_dir, fname)
                        if not os.path.exists(target_file):
                            with zf.open(info) as src, open(target_file, "wb") as dst:
                                dst.write(src.read())
                    elif fname.lower().endswith(".txt") and len(parts) >= 2:
                        sn = parts[-2].zfill(3)
                        target_dir = os.path.join(rr_root, sn)
                        os.makedirs(target_dir, exist_ok=True)
                        target_file = os.path.join(target_dir, fname)
                        if not os.path.exists(target_file):
                            with zf.open(info) as src, open(target_file, "wb") as dst:
                                dst.write(src.read())

        for z_audio in [zip_audio1, zip_audio2]:
            if os.path.exists(z_audio):
                print(f"[INFO] Extracting audio from {z_audio} ...")
                with zipfile.ZipFile(z_audio, "r") as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        parts = Path(info.filename).parts
                        fname = parts[-1]
                        if fname.lower().endswith(".wav") and len(parts) >= 2:
                            sn = parts[-2].zfill(3)
                            target_dir = os.path.join(audio_root, sn)
                            os.makedirs(target_dir, exist_ok=True)
                            target_file = os.path.join(target_dir, fname)
                            if not os.path.exists(target_file):
                                with zf.open(info) as src, open(target_file, "wb") as dst:
                                    dst.write(src.read())

        return video_root, audio_root, rr_root

    def _merge_single_video_audio(
        self,
        video_path: str,
        audio_path: str,
        output_file: str,
        path_ffmpeg_bin: str = "ffmpeg",
    ):
        """Merge a single .mp4 video and .wav audio file into output_file using FFmpeg."""
        if os.path.exists(output_file) and os.path.getsize(output_file) > 1024:
            return output_file

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        if not os.path.exists(video_path):
            return ""

        if not os.path.exists(audio_path):
            # Copy video directly if no audio exists
            subprocess.run(
                [
                    path_ffmpeg_bin,
                    "-y",
                    "-i",
                    video_path,
                    "-c",
                    "copy",
                    output_file,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return output_file

        # Merge video and audio
        cmd = [
            path_ffmpeg_bin,
            "-y",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            output_file,
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0 or not os.path.exists(output_file) or os.path.getsize(output_file) < 1024:
            # Re-encode video if stream copy fails (compatible with both libx264 and Conda libopenh264/mpeg4)
            for v_args in (["-c:v", "libopenh264", "-b:v", "600k"], ["-c:v", "mpeg4", "-q:v", "6"]):
                cmd_reencode = (
                    [path_ffmpeg_bin, "-y", "-i", video_path, "-i", audio_path]
                    + v_args
                    + ["-c:a", "aac", "-shortest", output_file]
                )
                r2 = subprocess.run(cmd_reencode, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if r2.returncode == 0 and os.path.exists(output_file) and os.path.getsize(output_file) > 1024:
                    break

        return output_file

    def merge_and_get_video_with_audio(
        self,
        path_data_experiment,
        df_user_info,
        participant_sn_range,
        experiment_name="PS-035",
        experiment_step="pre",
        path_pickle=None,
        path_ffmpeg_bin="ffmpeg",
        path_ffprobe_bin="ffprobe",
    ):
        """
        Build the video metadata DataFrame for all participants in participant_sn_range.
        Supports:
          - experiment_step="pre": prepares Pre-test video only (Video Path = {sn}_pre.mp4, Merged File Path = {sn}_pre_merged.mp4)
          - experiment_step="post": prepares Post-test video only (Video Path = {sn}_post.mp4, Merged File Path = {sn}_post_merged.mp4)
          - experiment_step="pre_post": prepares both Pre and Post videos
        Caches result to path_pickle.
        """
        step_norm = str(experiment_step).strip().lower()

        if path_pickle is not None:
            path_pickle_obj = Path(path_pickle)
            if path_pickle_obj.exists():
                print(f"[INFO] Pickle file exists; reading it directly: {path_pickle_obj}")
                df_cached = pd.read_pickle(path_pickle_obj)
                if not df_cached.empty:
                    unique_sns = set(df_cached["Participant SN"].astype(int).unique())
                    req_sns = set(int(s) for s in participant_sn_range)
                    if req_sns.issubset(unique_sns):
                        return df_cached[
                            df_cached["Participant SN"].astype(int).isin(req_sns)
                        ].reset_index(drop=True)

        video_root, audio_root, _ = self.ensure_multimodal_files_extracted(
            path_data_experiment
        )
        merged_root = os.path.join(path_data_experiment, "video_with_audio")
        os.makedirs(merged_root, exist_ok=True)

        rows = []
        for sn_val in participant_sn_range:
            sn_str = f"{int(sn_val):03d}"
            v_pre = os.path.join(video_root, sn_str, f"{sn_str}_pre.mp4")
            v_post = os.path.join(video_root, sn_str, f"{sn_str}_post.mp4")
            a_pre = os.path.join(audio_root, sn_str, f"{sn_str}_pre.wav")
            a_post = os.path.join(audio_root, sn_str, f"{sn_str}_post.wav")

            if not os.path.exists(v_post) and not os.path.exists(v_pre):
                continue

            m_pre = os.path.join(merged_root, sn_str, f"{sn_str}_pre_merged.mp4")
            m_post = os.path.join(merged_root, sn_str, f"{sn_str}_post_merged.mp4")

            if step_norm in ("pre", "pre_post") and os.path.exists(v_pre):
                self._merge_single_video_audio(v_pre, a_pre, m_pre, path_ffmpeg_bin)
            if step_norm in ("post", "pre_post") and os.path.exists(v_post):
                self._merge_single_video_audio(v_post, a_post, m_post, path_ffmpeg_bin)

            if step_norm == "pre":
                active_video = v_pre if os.path.exists(v_pre) else ""
                active_merged = m_pre if os.path.exists(m_pre) else ""
            else:
                active_video = v_post if os.path.exists(v_post) else ""
                active_merged = m_post if os.path.exists(m_post) else ""

            fsize = os.path.getsize(active_video) if active_video and os.path.exists(active_video) else 0

            rows.append(
                {
                    "Experiment Name": experiment_name,
                    "Participant SN": sn_str,
                    "Experiment Step": step_norm,
                    "Pre Video Path": v_pre if os.path.exists(v_pre) else "",
                    "Post Video Path": v_post if os.path.exists(v_post) else "",
                    "Video Path": active_video,
                    "Pre Audio Path": a_pre if os.path.exists(a_pre) else "",
                    "Post Audio Path": a_post if os.path.exists(a_post) else "",
                    "Pre Merged File Path": m_pre if os.path.exists(m_pre) else "",
                    "Post Merged File Path": m_post if os.path.exists(m_post) else "",
                    "Merged File Path": active_merged,
                    "File Size": fsize,
                    "LLM_Result": "",
                    "stai_aa": np.nan,
                    "stai_ap": np.nan,
                    "stai_total": np.nan,
                    "explanation": "",
                }
            )

        df = pd.DataFrame(rows)
        if not df.empty and df_user_info is not None:
            df["user_id"] = df["Participant SN"].astype(int)
            cols_to_merge = [
                c
                for c in [
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
                if c in df_user_info.columns
            ]
            df = df.merge(
                df_user_info[cols_to_merge].drop_duplicates("user_id"),
                on="user_id",
                how="left",
            )

        if path_pickle is not None and not df.empty:
            os.makedirs(os.path.dirname(path_pickle), exist_ok=True)
            df.to_pickle(path_pickle)
            print(f"[INFO] Video metadata saved to pickle: {path_pickle}")

        return df

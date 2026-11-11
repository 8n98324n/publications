# Copyright (c) 2026 Ivan Liu.
# Licensed under CC BY 4.0.
"""Experiment driver: builds prompts, calls the LLM client per participant (serially or in
parallel), parses responses, and caches intermediate results.
"""

# Standard Library Imports

import os
import time
import re
from pathlib import Path
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import pickle
import random

# Additional imports for concurrent execution with timeout support
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from pathlib import Path

class ExperimentHelper:
    """
    ExperimentHelper is a utility class responsible for managing
    multimodal experiment pipelines.

    Its responsibilities include:
    - Managing experiment metadata
    - Loading feature data
    - Running LLM inference
    - Handling video preprocessing
    - Parsing LLM responses
    - Coordinating participant-level processing
    """

    # Constructor

    def __init__(
            self,
            experiment_name,
            data_loader,
            df_helper,
            video_helper,
            llm_client,
            path_data_experiment,
            path_pickle_list_df_features,
            path_pickle_df_names,
            path_ffmpeg_bin=str,
            path_ffprobe_bin=str
        ):
        """
        Initialize the experiment environment and required helper modules.

        Parameters
        ----------
        experiment_name : str
            Name of the experiment.

        data_loader : object
            Helper class responsible for loading datasets.

        df_helper : object
            Helper class for DataFrame processing.

        video_helper : object
            Utility class for video processing operations.

        llm_client : object
            Client interface used to communicate with the LLM.

        path_data_experiment : str
            Root directory containing all experimental data.

        path_pickle_list_df_features : str
            Path to the pickle file storing a list of feature DataFrames.

        path_pickle_df_names : str
            Path to the pickle file storing the corresponding DataFrame names.

        path_ffmpeg_bin : str
            Path to the ffmpeg binary used for video processing.

        path_ffprobe_bin : str
            Path to the ffprobe binary used for video metadata extraction.
        """

        # Store experiment metadata
        self.experiment_name = experiment_name

        # Store helper utilities
        self.data_loader = data_loader
        self.df_helper = df_helper
        self.video_helper = video_helper
        self.llm_client = llm_client

        # Store dataset path
        self.path_data_experiment = path_data_experiment

        # Store pickle file paths
        self.path_pickle_df_list = path_pickle_list_df_features
        self.path_pickle_df_names = path_pickle_df_names

        # Store external video processing tools
        self.path_ffmpeg_bin = path_ffmpeg_bin
        self.path_ffprobe_bin = path_ffprobe_bin

        # Flag indicating whether the experiment configuration has been initialized
        self.experiment_defined = False

    def get_llm_client(self):
        """Return the LLM client instance."""
        return self.llm_client

    def set_llm_client(self, llm_client):
        """Set the LLM client instance for this object."""
        self.llm_client = llm_client

    def copy(self):
        """Return a new ExperimentHelper instance with the same configuration and helper objects as this instance.
        """
        return ExperimentHelper(
            experiment_name=self.experiment_name,
            data_loader=self.data_loader,
            df_helper=self.df_helper,
            video_helper=self.video_helper,
            llm_client=self.llm_client,
            path_data_experiment=self.path_data_experiment,
            path_pickle_list_df_features=self.path_pickle_df_list,
            path_pickle_df_names=self.path_pickle_df_names,
            path_ffmpeg_bin=self.path_ffmpeg_bin,
            path_ffprobe_bin=self.path_ffprobe_bin
        )

    # Text Normalization Utility

    def _normalize_llm_text(self, text: str) -> str:
        """
        Normalize punctuation and whitespace in LLM-generated text.

        This function converts full-width punctuation to standard ASCII
        equivalents and collapses multiple spaces into a single space.

        Parameters
        ----------
        text : str
            Raw text returned by the LLM.

        Returns
        -------
        str
            Normalized text string.
        """

        # Handle None input safely
        if text is None:
            return ""

        s = str(text)

        # Replace full-width punctuation with ASCII equivalents
        s = s.replace("（", "(").replace("）", ")").replace("，", ",").replace("：", ":").replace("＝", "=")
        s = s.replace("“", '"').replace("”", '"').replace("—", "-").replace("–", "-")

        # Collapse repeated whitespace
        s = re.sub(r"\s+", " ", s).strip()

        return s

    # Parse LLM Output

    def segment_llm_response(self, df, column="LLM_Result"):
        """
        Extract sympathy and inspiration scores from LLM output.

        Expected output structure:

            (sympathy, inspiration) = (x, y)

            Explanation text ...

        Behavior rules:
        - Only overwrite df["sympathy"] and df["inspiration"] if valid values exist
        - Extract remaining text as df["explanation"]
        - If structure is not found, leave existing values unchanged

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing LLM outputs.

        column : str
            Name of the column containing the raw LLM output text.

        Returns
        -------
        pandas.DataFrame
            Updated DataFrame with parsed scores and explanations.
        """

        # Work on a copy to avoid modifying the original DataFrame
        df = df.copy()


        pattern = re.compile(
            r"sympathy\s*,\s*inspiration.*?\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)",
            re.IGNORECASE | re.DOTALL
        )

        # Iterate through each row
        for i, raw in enumerate(df[column].fillna("").values):

            raw_str = str(raw).strip()
            match = pattern.search(raw_str)

            # If expected structure not found, skip this row
            if not match:
                continue

            try:
                # Extract numeric scores
                new_sym = int(float(match.group(1)))
                new_insp = int(float(match.group(2)))

                # df.at[idx, "sympathy"] = new_sym
                # df.at[idx, "inspiration"] = new_insp
                df.iloc[i, df.columns.get_loc("sympathy")] = new_sym
                df.iloc[i, df.columns.get_loc("inspiration")] = new_insp

                # Extract explanation text

                # Everything after the score pattern
                explanation = raw_str[match.end():].strip()

                # Remove leading line breaks
                explanation = explanation.lstrip("\n").strip()
                if "explanation" not in df.columns:
                    df["explanation"] = None
                if explanation:
                    #df.at[idx, "explanation"] = explanation
                    df.iloc[i, df.columns.get_loc("explanation")] = explanation

            except Exception:
                # Parsing failure → leave original values unchanged
                continue

        return df

    # Load DataFrame from Pickle

    def read_from_pickle(self, path_pickle):
        """
        Load a serialized Python object (typically a DataFrame) from a pickle file.

        Parameters
        ----------
        path_pickle : str
            Path to the pickle file.

        Returns
        -------
        object
            Loaded Python object (commonly a pandas DataFrame).
        """

        with open(path_pickle, "rb") as f:
            df = pickle.load(f)

        return df

    # Participant-Level Processing

    def process_single_participant(
        self,
        participant_sn,
        sub_df,
        llm_client,
        video_helper,
        path_ffmpeg_bin,
        enable_compression,
        enable_llm,
        video_only,
        scale_width,
        fps,
        crf,
        temp_dir,
        temp_suffix,
        over_ride_existing=False,
        default_prompt_text=None,
        verbose=False,
        attach_video=True
    ):
        """
        Process one participant's experiment data.

        Core logic:
        - If this participant already has an LLM_Result, skip processing
        - Otherwise perform video preparation and LLM inference

        Special prompt logic:
        If column "individual_prompt" exists and contains a non-empty value,
        the first non-empty prompt will be used instead of the default prompt.

        Parameters
        ----------
        participant_sn : str or int
            Participant identifier.

        sub_df : pandas.DataFrame
            DataFrame containing all rows related to the participant.

        enable_llm : bool
            Whether LLM inference should be executed.

        attach_video : bool
            Whether video files should be included in the LLM request.

        Returns
        -------
        tuple
            (participant_sn, llm_result)
        """


        # Convert values to string for safe comparison
        sub_df = sub_df.loc[:, ~sub_df.columns.duplicated()]
        llm_values = sub_df["LLM_Result"].fillna("").astype(str)

        # If any row already contains a non-empty result → skip processing
        if any(len(v.strip()) > 10 for v in llm_values) and not over_ride_existing:

            if verbose:
                print(f"[Participant {participant_sn}] Existing LLM_Result detected. Skipping processing.\n")

            # Return the first existing result
            existing_result = llm_values.iloc[0]

            return participant_sn, existing_result


        # If all LLM_Result values are empty → run the pipeline

        sub_df = sub_df.sort_values("Experiment Step")

        for _, row in sub_df.iterrows():


            if enable_llm:

                # Default prompt
                prompt_to_use = default_prompt_text

                # If participant-specific prompt exists, override default
                if "individual_prompt" in sub_df.columns:
                    try:
                        prompts_series = sub_df["individual_prompt"].fillna("").astype(str).str.strip()
                        non_empty = prompts_series[prompts_series != ""]

                        if len(non_empty) > 0:
                            prompt_to_use = non_empty.iloc[0]

                            if verbose:
                                print(f"Using participant-specific prompt for participant {participant_sn}.\n")

                    except Exception:
                        # If any error occurs, fall back to default prompt
                        prompt_to_use = default_prompt_text

                if verbose:
                    print(f"[Thread] Participant {participant_sn} sending request to LLM. Prompt length: {len(prompt_to_use)}\n")

                # Prepare video attachments

                if attach_video:

                    if video_only:
                        uploaded_files = [
                            row["uploaded_file_video_only"]
                        ]
                    else:
                        uploaded_files = [
                            row["uploaded_file_video_merged"]
                        ]

                    # Append additional videos if defined
                    for add_video in self.additional_video:
                        uploaded_files.append(add_video)

                else:
                    uploaded_files = []

                # Call LLM API

                result = llm_client.chat(
                    prompt=prompt_to_use,
                    template_used=False,
                    stream=False,
                    verbose=verbose,
                    extra_info=uploaded_files
                )

            else:
                result = ""

        return participant_sn, result

    
    # Video Processing: Single Row Processing

    def process_single_video_info(
        self,
        idx,
        row,
        df,
        video_only,
        enable_compression,
        temp_suffix,
        temp_dir,
        video_helper,
        path_ffmpeg_bin,
        scale_width,
        fps,
        crf,
        llm_client,
        verbose
    ):
        """
        Process a single row of video information.

        Responsibilities:
        1. Determine whether the video requires compression.
        2. Compress the video if needed.
        3. Upload the processed video to the LLM storage (Gemini).
        4. Return metadata required to update the experiment DataFrame.

        Parameters
        ----------
        idx : int
            Row index of the DataFrame.

        row : pandas.Series
            Current row containing experiment metadata.

        df : pandas.DataFrame
            Full experiment results DataFrame.

        video_only : bool
            Indicates whether the workflow uses only the original video
            or a merged video file.

        enable_compression : bool
            Whether video compression should be applied.

        temp_suffix : str
            Suffix used when generating compressed video filenames.

        temp_dir : Path
            Temporary directory where compressed videos are stored.

        video_helper : object
            Utility responsible for video compression operations.

        path_ffmpeg_bin : str
            Path to the ffmpeg binary.

        scale_width : int
            Width used during video resizing.

        fps : int
            Frame rate used during compression.

        crf : int
            Compression quality parameter (Constant Rate Factor).

        llm_client : object
            Client used to upload files to the LLM platform.

        verbose : bool
            Enables verbose logging.

        Returns
        -------
        tuple
            (row_index, uploaded_file_reference, uploaded_video_path, error_message)
        """

        from datetime import datetime, timezone

        try:


            compress_needed = False
            path_video_for_compression = None
            path_video_to_upload = None
            uploaded_text = ""

            if video_only:

                # Use the original video file
                path_video_for_compression = row["Video Path"]
                path_video_to_upload = df.at[idx, "Video Path"]

                # Check whether the video has already been uploaded
                if (
                    df.at[idx, "uploaded_file_video_only"] is None
                    or df.at[idx, "uploaded_file_video_only"] == ""
                    or df.at[idx, "uploaded_file_video_only"] == "None"
                    or pd.isna(df.at[idx, "uploaded_file_video_only"])
                    or not Path(df.at[idx, "Compressed Video Path"] ).exists()
                ):
                    compress_needed = True
                else:
                    uploaded_text = df.at[idx, "uploaded_file_video_only"]

            else:

                # Use merged video file
                path_video_for_compression = row["Merged File Path"]
                path_video_to_upload = df.at[idx, "Merged File Path"]

                if (
                    df.at[idx, "uploaded_file_video_merged"] is None
                    or df.at[idx, "uploaded_file_video_merged"] == ""
                    or df.at[idx, "uploaded_file_video_merged"] == "None"
                    or pd.isna(df.at[idx, "uploaded_file_video_merged"])
                    or not Path(df.at[idx, "Compressed Video Path"] ).exists()
                ):
                    compress_needed = True
                else:
                    uploaded_text = df.at[idx, "uploaded_file_video_merged"]


            if (
                uploaded_text is not None
                and hasattr(uploaded_text, "expiration_time")
                and datetime.now(timezone.utc) < uploaded_text.expiration_time
            ):

                # The uploaded file reference is still valid
                if verbose:
                    print(f"{idx} video already uploaded")

                return idx, uploaded_text, path_video_to_upload, None, False

            else:

                # Store compressed video path

                if isinstance(path_video_for_compression, str):

                    path_video_for_compression = Path(path_video_for_compression)

                    new_name = path_video_for_compression.stem + temp_suffix
                    compressed_path = temp_dir / new_name

                    if "Compressed Video Path" not in df.columns:
                        df["Compressed Video Path"] = ""

                    df.at[idx, "Compressed Video Path"] = compressed_path


                if enable_compression and compress_needed:

                    if compressed_path and Path(compressed_path).exists():

                        if verbose:
                            print(
                                f"{path_video_for_compression} compressed file already exists, skipping compression: {compressed_path}"
                            )

                    else:
                        if Path(path_video_for_compression).exists():
                            video_helper.compress_video(
                                input_path=str(path_video_for_compression),
                                output_path=str(compressed_path),
                                path_ffmpeg_bin=path_ffmpeg_bin,
                                scale_width=scale_width,
                                fps=fps,
                                crf=crf,
                            )
                        else:
                            print("path_video_for_compression doesn't exist.")
                            return None, None, None, None, False

                    path_video_to_upload = compressed_path

                    # Print file size comparison

                    size_mb_original = 0

                    try:
                        size_mb_original = os.path.getsize(path_video_for_compression) / (1024 * 1024)
                    except Exception:
                        pass

                    size_mb = os.path.getsize(compressed_path) / (1024 * 1024)

                    if verbose:
                        sn = df.at[idx, "Participant SN"]

                        print(f"[Participant {sn}] Original size: {size_mb_original:.2f} MB → {path_video_for_compression}")
                        print(f"[Participant {sn}] Compressed size: {size_mb:.2f} MB → {compressed_path}")

                else:

                    path_video_to_upload = compressed_path


                uploaded_text = None
                
                try:
                    if  Path(path_video_to_upload).exists():
                        uploaded_text = llm_client.upload_file_to_gemini(
                            current_file=path_video_to_upload
                        )
                    else:
                        print("path_video_to_upload doesn't exist.")
                        return None, None, None, None, False
                except Exception:
                    uploaded_text = None

                return idx, uploaded_text, path_video_to_upload, None, True

        except Exception as e:

            err_msg = traceback.format_exc()
            print("[Error]" + err_msg)

            return idx, None, None, None


    # Parallel Video Processing

    def process_video_info_parallelly(
        self,
        df_video_info,
        video_only,
        enable_compression,
        temp_suffix,
        temp_dir,
        video_helper,
        path_ffmpeg_bin,
        scale_width,
        fps,
        crf,
        llm_client,
        verbose=False,
        max_workers=1,
    ):
        """
        Process video information for all rows in the experiment DataFrame
        using parallel threads.

        Each row is submitted as a separate task to the thread pool.

        After processing:
        - Upload results are written back to the DataFrame
        - The DataFrame is immediately saved to a pickle file (auto checkpoint)

        Parameters
        ----------
        df_experiment_results : pandas.DataFrame
            Experiment result DataFrame.

        max_workers : int
            Number of worker threads.

        Returns
        -------
        pandas.DataFrame
            Updated DataFrame with uploaded video references.
        """

        # Convert temporary directory to Path if necessary
        if isinstance(temp_dir, str):
            temp_dir = Path(temp_dir)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:

            # Submit parallel jobs

            futures = {
                executor.submit(
                    self.process_single_video_info,
                    idx,
                    row,
                    df_video_info,
                    video_only,
                    enable_compression,
                    temp_suffix,
                    temp_dir,
                    video_helper,
                    path_ffmpeg_bin,
                    scale_width,
                    fps,
                    crf,
                    llm_client,
                    verbose,
                ): idx
                for idx, row in df_video_info.iterrows()
            }

            # Collect results as tasks complete

            for future in as_completed(futures):

                idx = futures[future]

                try:

                    row_idx, uploaded_text, path_video_to_upload, err, uploaded = future.result()

                    # If the row-level task returned an error

                    if err is not None:

                        print(f"[Error] Row {idx} encountered an error:\n{err}")
                        continue

                    # Write results back to the DataFrame

                    if row_idx is not None:
                        if uploaded_text is not None and path_video_to_upload:
                            if video_only:
                                # df_video_info.at[row_idx, "uploaded_file_video_only"] = str(uploaded_text)
                                col_idx = df_video_info.columns.get_loc("uploaded_file_video_only")
                                df_video_info.iat[row_idx, col_idx] = uploaded_text
                            else:
                                #df_video_info.at[row_idx, "uploaded_file_video_merged"] = str(uploaded_text)         
                                col_idx = df_video_info.columns.get_loc("uploaded_file_video_merged")
                                df_video_info.iat[row_idx, col_idx] = uploaded_text

                            df_video_info.at[row_idx, "path_video_to_upload"] = path_video_to_upload
                        else:
                            print(f"[Error] Row {idx} upload failed or returned empty result.")
                        # Immediately save checkpoint (pickle cache)

                        df_to_cache = df_video_info.copy()

                        df_to_cache.to_pickle(self.path_pickle_df_video)

                        if verbose:
                            if uploaded:
                                print(f"[AutoSaved] Row {row_idx} uploaded and saved to pickle")
                    else:
                        pass

                except Exception:

                    print(f"[Error] Future encountered an unexpected exception:\n{traceback.format_exc()}")

        return df_video_info


    # LLM Result Validation

    def check_llm_result(self, result):
        """
        Check whether an LLM result is valid.

        A result is considered valid if:
        - It is not None
        - It is not an empty string
        - It is not the text "none" or "nan"

        Parameters
        ----------
        result : object

        Returns
        -------
        bool
            True if the result is valid, False otherwise.
        """

        if (
            result is not None
            and str(result).strip() != ""
            and str(result).strip().lower() != "none"
            and str(result).strip().lower() != "nan"
        ):
            return True

        return False
 
    # Compress and Upload Videos

    def prepare_video_info(
        self,
        df_video_info,
        llm_client,
        video_helper,
        path_ffmpeg_bin,
        enable_compression=True,
        enable_llm=True,
        video_only=False,
        scale_width=1280,
        fps=12,
        crf=28,
        temp_dir="",
        temp_suffix="_compressed.mp4",
        max_workers=4,
        task_timeout=120,
        over_ride_existing=False,
        verbose=False,
    ):
        """
        Parameters
        ----------
        llm_client : object
            Client responsible for communicating with the LLM API.

        video_helper : object
            Utility used for video processing operations.

        path_ffmpeg_bin : str
            Path to the ffmpeg executable.

        enable_compression : bool
            Whether video compression should be performed.

        enable_llm : bool
            Whether LLM inference should be executed.

        video_only : bool
            Whether only raw videos are used or merged videos.

        temp_dir : str
            Temporary directory used for storing compressed videos.

        temp_suffix : str
            Suffix appended to compressed video filenames.

        max_workers : int
            Maximum number of parallel threads.

        task_timeout : int
            Maximum time (seconds) allowed for each participant task.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing processed results.
        """

        from pathlib import Path
        from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
        import threading

        # Step A: Ensure compression directory exists

        temp_dir = Path(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Step B: Ensure required columns exist

        if "uploaded_file_video_only" not in df_video_info.columns:
            df_video_info["uploaded_file_video_only"] = None

        if "uploaded_file_video_merged" not in df_video_info.columns:
            df_video_info["uploaded_file_video_merged"] = None

        if "LLM_Result" not in df_video_info.columns:
            df_video_info["LLM_Result"] = None

        # Stage 1: Video compression and upload (row-level)

        df_video_info = self.process_video_info_parallelly(
            df_video_info=df_video_info,
            video_only=video_only,
            enable_compression=enable_compression,
            temp_suffix=temp_suffix,
            temp_dir=temp_dir,
            video_helper=video_helper,
            path_ffmpeg_bin=path_ffmpeg_bin,
            scale_width=scale_width,
            fps=fps,
            crf=crf,
            llm_client=llm_client,
            verbose=verbose,
            max_workers=max_workers
        )
        return df_video_info

    # Multithreaded LLM Inference Pipeline

    def run_llm_on_files_multithread(
        self,
        df_experiment_results,
        llm_client,
        video_helper,
        path_ffmpeg_bin,
        enable_compression=True,
        enable_llm=True,
        video_only=False,
        scale_width=1280,
        fps=12,
        crf=28,
        temp_dir="",
        temp_suffix="_compressed.mp4",
        max_workers=4,
        task_timeout=120,
        over_ride_existing=False,
        default_prompt_text=None,
        verbose=False,
        attach_video=True
    ):
        """
        Execute the full experiment pipeline using multithreading.

        Pipeline stages
        ----------------
        Stage 1: Row-level processing
            - Compress videos if required
            - Upload videos to the LLM platform

        Stage 2: Participant-level processing
            - Run LLM inference for each participant
            - Parse LLM outputs
            - Autosave intermediate results

        Parameters
        ----------
        df_experiment_results : pandas.DataFrame
            DataFrame containing experiment metadata and results.

        llm_client : object
            Client responsible for communicating with the LLM API.

        video_helper : object
            Utility used for video processing operations.

        path_ffmpeg_bin : str
            Path to the ffmpeg executable.

        enable_compression : bool
            Whether video compression should be performed.

        enable_llm : bool
            Whether LLM inference should be executed.

        video_only : bool
            Whether only raw videos are used or merged videos.

        temp_dir : str
            Temporary directory used for storing compressed videos.

        temp_suffix : str
            Suffix appended to compressed video filenames.

        max_workers : int
            Maximum number of parallel threads.

        task_timeout : int
            Maximum time (seconds) allowed for each participant task.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing processed results.
        """

        from pathlib import Path
        from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
        import threading


        # Validate DataFrame
        if df_experiment_results is None or df_experiment_results.empty:
            print("[Error 1]: DataFrame is empty. LLM inference cannot proceed.")
            return None

        # Select only participants within the experiment range

        df_selected = df_experiment_results[
            pd.to_numeric(df_experiment_results["Participant SN"], errors="coerce")
            .isin(self.participant_sn_range)
        ].copy()

        # Group rows by participant

        grouped = df_selected.groupby("Participant SN")

        # Thread-safe lock for pickle saving
        pickle_lock = threading.Lock()

        # Stage 2: Participant-level LLM inference

        with ThreadPoolExecutor(max_workers=max_workers) as executor:

            futures = []

            for participant_sn, sub_df in grouped:

                future = executor.submit(
                    self.process_single_participant,
                    participant_sn,
                    sub_df,
                    llm_client,
                    video_helper,
                    path_ffmpeg_bin,
                    enable_compression,
                    enable_llm,
                    video_only,
                    scale_width,
                    fps,
                    crf,
                    temp_dir,
                    temp_suffix,
                    over_ride_existing,
                    default_prompt_text,
                    verbose,
                    attach_video
                )

                future.participant_sn = participant_sn
                futures.append(future)

            # Process results as tasks complete

            for future in as_completed(futures):

                sn = future.participant_sn

                try:

                    participant_sn, result = future.result(timeout=task_timeout)

                    if self.check_llm_result(result):

                        # Update DataFrames
                        df_selected.loc[df_selected["Participant SN"] == participant_sn, "LLM_Result"] = result
                        df_selected = self.segment_llm_response(df_selected)

                        df_experiment_results.loc[
                            df_experiment_results["Participant SN"] == participant_sn,
                            "LLM_Result"
                        ] = result

                        df_experiment_results = self.segment_llm_response(df_experiment_results)

                        # Autosave updated results to pickle

                        with pickle_lock:
                            if self.path_pickle_df_experiment_results is not None and self.path_pickle_df_experiment_results != "":
                                df_experiment_results.to_pickle(self.path_pickle_df_experiment_results)
                            else:
                                print("[Error]: path_pickle_df_experiment_results is not defined.")
                        if verbose:
                            print(f"[AUTOSAVE-PARTICIPANT] {participant_sn} saved to {self.path_pickle_df_experiment_results}")

                    else:

                        if verbose:
                            print(f"[NO RESULT] Participant {sn} did not produce a valid LLM output.")

                except TimeoutError:

                    # df_selected.loc[df_selected["Participant SN"] == sn, "LLM_Result"] = None


                    if verbose:
                        print(f"[TIMEOUT] Participant {sn} exceeded {task_timeout}s. Autosaved.")

                except Exception as e:

                    if verbose:
                        print(f"[Error 2]: Participant {sn} encountered an exception: {e}")

                    continue

        # Final result

        return df_selected


    # Prompt Template Builder

    def get_prompt_text_template(
        self,
        template_experiment: str,
        template_cot: str = None,
        template_knowledge_injection: str = None,
        template_tools: str = None,
        tools_results: list = None,
    ) -> str:
        """
        Generate the final prompt text by replacing template placeholders.

        Supported placeholders
        ----------------------
        [[COT]]        -> Chain-of-Thought instructions
        [[KNOWLEDGE]]  -> Knowledge injection text
        [[TOOLS]]      -> Tool outputs formatted using template_tools
        """

        prompt_text = template_experiment

        # Replace [[TOOLS]] section

        if template_tools is not None and tools_results is not None and len(tools_results) > 0:
            tools_text = template_tools.format(*tools_results)
        else:
            tools_text = ""

        if "[[TOOLS]]" in prompt_text:
            prompt_text = prompt_text.replace("[[TOOLS]]", tools_text)

        # Replace [[COT]]

        cot_text = template_cot if template_cot is not None else ""

        if "[[COT]]" in prompt_text:
            prompt_text = prompt_text.replace("[[COT]]", cot_text)

        # Replace [[KNOWLEDGE]]

        knowledge_text = template_knowledge_injection if template_knowledge_injection is not None else ""

        if "[[KNOWLEDGE]]" in prompt_text:
            prompt_text = prompt_text.replace("[[KNOWLEDGE]]", knowledge_text)

        return prompt_text


    # Retrieve User Features

    def get_user_features_from_df(self, df, participant_sn: int) -> dict:
        """
        Retrieve user feature values from a DataFrame.

        The function first attempts to match the participant using
        the 'user_id' column. If not found, it attempts to fall back
        to positional indexing.
        """

        try:

            mask = df["user_id"] == participant_sn

            if not mask.any():
                mask = df["user_id"].astype(str) == str(participant_sn)

        except Exception:
            mask = None

        if mask is not None and mask.any():

            features = df.loc[mask].iloc[0].to_dict()

        else:

            try:
                idx = int(participant_sn)
                features = df.iloc[idx].to_dict()

            except Exception as e:
                raise ValueError(
                    f"Error 3: Cannot locate row for user_id={participant_sn} in the DataFrame: {e}"
                )

        return features


    # Experiment Configuration

    def define_experiment(
        self,
        participant_sn_range,
        path_pickle_df_experiment_results,
        video_only: bool,
        experiment_step: int,
        interpreters: list,
        template_main: str,
        template_cot: str,
        template_knowledge_injection: str,
        template_tools: str,
        path_pickle_df_video: str,
        path_compressed_video: str = "",
        attach_video: bool = True,
        additional_video: list = None,
        all_data=None,
        all_df_name=None,
        df_video_info=None,
        verbose: bool = True
    ):
        """
        Define experiment configuration and initialize required datasets.
        """

        if additional_video is None:
            additional_video = []
        self.participant_sn_range = participant_sn_range
        self.path_pickle_df_video = path_pickle_df_video
        self.path_pickle_df_experiment_results = path_pickle_df_experiment_results
        self.video_only = video_only
        self.experiment_step = experiment_step
        self.interpreters = interpreters

        self.template_main = template_main
        self.template_cot = template_cot
        self.template_knowledge_injection = template_knowledge_injection
        self.template_tools = template_tools

        self.attach_video = attach_video
        self.experiment_defined = True
        self.additional_video = additional_video
        self.verbose = verbose
        self.path_compressed_video = path_compressed_video

        # Load cached data if not provided

        if all_data is None or all_df_name is None:

            self.all_data = self.data_loader.load_from_cache(self.path_pickle_df_list)
            self.all_df_name = self.data_loader.load_from_cache(self.path_pickle_df_names)

        else:

            self.all_data = all_data
            self.all_df_name = all_df_name

        # Prepare user and feature DataFrames

        self.df_users = self.all_data[0]

        self.df_features = self.df_helper.concat_dataframes_by_keys(
            self.all_data,
            keys=("user_id", "Group"),
            how="outer"
        )

        # Load or generate video metadata

        if df_video_info is None:

            self.df_video_info = self.data_loader.merge_and_get_video_with_audio(
                self.path_data_experiment,
                self.df_users,
                experiment_step=[self.experiment_step],
                experiment_name=self.experiment_name,
                participant_sn_range=self.participant_sn_range,
                path_pickle=self.path_pickle_df_video,
                path_ffmpeg_bin=self.path_ffmpeg_bin,
                path_ffprobe_bin=self.path_ffprobe_bin
            )

        else:

            self.df_video_info = df_video_info

        # Initialize column
        if "llm_result" in self.df_video_info.columns:
            self.df_video_info.rename(columns={"llm_result": "LLM_Result"}, inplace=True)
        self.df_video_info["LLM_Result"] = ""


    # Random Fold Generator

    def generate_random_folds(self, list_users, num_folds):
        """
        Randomly shuffle users and split them into folds.
        """

        formatted_list_users = [str(u).zfill(3) for u in list_users]

        random.shuffle(formatted_list_users)

        fold_size = len(formatted_list_users) // num_folds

        folds = [
            formatted_list_users[i:i + fold_size]
            for i in range(0, len(formatted_list_users), fold_size)
        ]

        return folds

    # Few-Shot Learning Experiment Pipeline

    def run_few_shot_learning(
        self,
        df_video_info,
        folds,
        num_folds,
        genai,
        template_main,
        experiment_helper,
        video_only=True,
        similarity_based_sampling=True,
        similarity_based_sampling_by_group=True,
        number_of_few_shot_examples=2,
        over_ride_existing=False,
        max_workers=8,
        verbose=False
    ):
        """
        Main function for running Few-Shot Learning experiments.

        Two operation modes are supported:

        Mode 1: Similarity-based retrieval
            - Select few-shot examples based on explanation text similarity.

        Mode 2: Traditional fold-based few-shot learning
            - Use cross-validation style folds for selecting examples.

        Important notes:
        - Videos are NOT uploaded again.
        - Existing Gemini file objects stored in df_video_info are reused.
        """

        import pandas as pd
        import numpy as np
        import random
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        all_results = []

        # Mode 1: Similarity-based Few-Shot Retrieval

        if similarity_based_sampling:

            print("=== Starting similarity-based Few-Shot retrieval mode ===")

            # Reset index to prevent index misalignment issues
            df_video_info = df_video_info.reset_index(drop=True)


            explanations = df_video_info["explanation"].fillna("").astype(str).tolist()

            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(explanations)


            similarity_matrix = cosine_similarity(tfidf_matrix)


            for idx in range(len(df_video_info)):
                if verbose:
                    print(f"Processing video {idx+1}/{len(df_video_info)}")

                target_row = df_video_info.iloc[idx]
                target_sn = int(target_row["Participant SN"])


                if similarity_based_sampling_by_group:

                    similarities = similarity_matrix[idx].copy()

                    # Exclude the current sample
                    similarities[idx] = -1

                    top_k_idx = []

                    unique_groups = sorted(df_video_info["Group"].dropna().unique())

                    for group_value in unique_groups:

                        group_indices = df_video_info.index[
                            df_video_info["Group"] == group_value
                        ].tolist()

                        group_indices = [i for i in group_indices if i != idx]

                        if len(group_indices) == 0:
                            continue

                        group_similarities = similarities[group_indices]

                        best_idx_in_group = group_indices[np.argmax(group_similarities)]

                        top_k_idx.append(best_idx_in_group)

                else:

                    similarities = similarity_matrix[idx].copy()

                    similarities[idx] = -1

                    top_k_idx = np.argsort(similarities)[::-1][:number_of_few_shot_examples]

                few_shot_examples = []
                additional_video = []

                int_video_number = 2


                for sim_idx in top_k_idx:

                    example_row = df_video_info.iloc[sim_idx]

                    if video_only:
                        video_obj = example_row["uploaded_file_video_only"]
                    else:
                        video_obj = example_row["uploaded_file_video_merged"]

                    additional_video.append(video_obj)

                    group = example_row["Group"]

                    if group == 1:
                        sympathy, inspiration = 9, 9
                    elif group == 2:
                        sympathy, inspiration = 9, 1
                    elif group == 3:
                        sympathy, inspiration = 1, 9
                    else:
                        sympathy, inspiration = 1, 1

                    few_shot_examples.append(
                        f"The prediction of example video {int_video_number}: (sympathy, inspiration) = ({sympathy}, {inspiration})\n"
                    )

                    int_video_number += 1


                enhanced_template = (
                    "The following analysis uses a few-shot learning design. "
                    "Several labeled examples are provided as references. "
                    "You should analyze Video 1, while using the results from "
                    "Video 2 and the subsequent videos as examples to guide your judgment.\n\n"
                    + "\n".join(few_shot_examples)
                    + "\n\n"
                    + template_main
                )
                if verbose:
                    print(f"enhanced_template: {enhanced_template}")


                experiment_helper_current = ExperimentHelper(
                    experiment_name=self.experiment_name,
                    data_loader=self.data_loader,
                    df_helper=self.df_helper,
                    video_helper=self.video_helper,
                    llm_client=self.llm_client,
                    path_data_experiment=self.path_data_experiment,
                    path_pickle_list_df_features=self.path_pickle_df_list,
                    path_pickle_df_names=self.path_pickle_df_names,
                    path_ffmpeg_bin=self.path_ffmpeg_bin,
                    path_ffprobe_bin=self.path_ffprobe_bin
                )

                experiment_helper_current.define_experiment(
                    participant_sn_range=[target_sn],
                    path_pickle_df_video=self.path_pickle_df_video,
                    path_pickle_df_experiment_results= self.path_pickle_df_experiment_results,
                    experiment_step=self.experiment_step,
                    video_only=video_only,
                    interpreters=[],
                    template_main=enhanced_template,
                    template_cot="",
                    template_knowledge_injection="",
                    template_tools="",
                    attach_video=True,
                    additional_video=additional_video,
                    verbose=self.verbose,
                    path_compressed_video=self.path_compressed_video,
                    all_data=self.all_data,
                    all_df_name=self.all_df_name,
                    df_video_info=self.df_video_info,
                )

                df_result = experiment_helper_current.run_exp_moral_elevation(
                    over_ride_existing=over_ride_existing,
                    max_workers=max_workers,
                    verbose=self.verbose,
                )

                df_result = df_result.copy()

                df_result["retrieval_mode"] = "similarity"
                df_result["target_index"] = idx

                all_results.append(df_result)

            return pd.concat(all_results, ignore_index=True)

        # Mode 2: Traditional Fold-based Few-Shot Learning

        else:

            print("=== Starting traditional Fold-based Few-Shot mode ===")

            all_fold_dfs = []

            formatted_list_users = [str(u).zfill(3) for u in all_users]

            for i in range(len(folds)):

                print(f"--- Fold {i+1}/{num_folds} ---")

                test_users = folds[i]

                train_pool = [u for u in formatted_list_users if u not in test_users]

                few_shot_sns = random.sample(train_pool, number_of_few_shot_examples)

                few_shot_examples = []
                additional_video = []

                int_video_number = 2

                for sn in few_shot_sns:

                    row = df_video_info[
                        df_video_info["Participant SN"].astype(str) == str(sn)
                    ].iloc[0]

                    if video_only:
                        video_obj = row["uploaded_file_video_only"]
                    else:
                        video_obj = row["uploaded_file_video_merged"]

                    additional_video.append(video_obj)

                    group = row["Group"]

                    if group == 1:
                        sympathy, inspiration = 9, 9
                    elif group == 2:
                        sympathy, inspiration = 9, 1
                    elif group == 3:
                        sympathy, inspiration = 1, 9
                    else:
                        sympathy, inspiration = 1, 1

                    few_shot_examples.append(
                        f"The prediction of video number:{int_video_number}: (sympathy, inspiration) = ({sympathy}, {inspiration})\n"
                    )

                enhanced_template = (
                    "The following analysis uses a few-shot learning design. "
                    "Several labeled examples are provided as references. "
                    "You should analyze Video 1, using the results from "
                    "other similar videos as examples.\n\n"
                    + "\n".join(few_shot_examples)
                    + "\n\n"
                    + template_main
                )

                experiment_helper.define_experiment(
                    participant_sn_range=[int(sn) for sn in test_users],
                    path_pickle_df_video=path_pickle_df_video,
                    path_pickle_df_experiment=template_path_pickle_df_experiment.format(i=i),
                    experiment_step=23,
                    video_only=video_only,
                    interpreters=[],
                    template_main=enhanced_template,
                    template_cot="",
                    template_knowledge_injection="",
                    template_tools="",
                    attach_video=True,
                    additional_video=additional_video,
                    verbose=self.verbose,
                )

                df_fold_result = experiment_helper.run_exp_moral_elevation(
                    over_ride_existing=True,
                    max_workers=1,
                    verbose=self.verbose,
                )

                df_fold_result["fold"] = i

                all_fold_dfs.append(df_fold_result)

            return pd.concat(all_fold_dfs, ignore_index=True)


    # Generate Individual Prompts for Each Video

    def update_df_with_individual_prompts(
        self,
        df_input: pd.DataFrame,
        df_features: pd.DataFrame,
        tools: list,
        template_main: str,
        template_cot: str,
        template_knowledge_injection: str,
        template_tools: str
    ) -> pd.DataFrame:
        """
        Generate the 'individual_prompt' column for each row of the video DataFrame.
        """

        df_extended = df_input.copy()

        if "individual_prompt" not in df_extended.columns:
            df_extended["individual_prompt"] = ""

        for idx, row in df_extended.iterrows():

            try:
                participant_sn = int(row["Participant SN"])
            except Exception:
                participant_sn = row["Participant SN"]

            features = self.get_user_features_from_df(df_features, participant_sn)

            interpreter_results = []

            for tool in tools:
                interpreter_results.append(tool.upload_and_get_text_output(features))

            try:

                prompt = self.get_prompt_text_template(
                    template_experiment=template_main,
                    template_cot=template_cot,
                    template_knowledge_injection=template_knowledge_injection,
                    template_tools=template_tools,
                    tools_results=interpreter_results
                )

            except Exception as e:

                prompt = f"Error 4: Prompt generation failed: {e}"

            df_extended.at[idx, "individual_prompt"] = prompt

        return df_extended


    # Retrieve Video Metadata DataFrame

    def get_df_video_path(self) -> pd.DataFrame:

        """Return cached video information dataframe, or load and cache it from the data loader if not yet initialized.
        """
        if self.df_video_info is not None and len(self.df_video_info)>0:
            return self.df_video_info

        else:

            self.df_video_info = self.data_loader.merge_and_get_video_with_audio(
                self.path_data_experiment,
                self.df_users,
                experiment_name="PS-9",
                experiment_step=[self.experiment_step],
                participant_sn_range=self.participant_sn_range,
                path_pickle=self.path_pickle_df_video,
                path_ffmpeg_bin=self.path_ffmpeg_bin,
                path_ffprobe_bin=self.path_ffprobe_bin
            )

            return self.df_video_info


    # Retrieve Experiment Result DataFrame

    def get_df_experiment_results(self) -> pd.DataFrame:

        """Return the pickled DataFrame from the stored path, or None if the path is not set."""
        if self.path_pickle_df_experiment_results is None or self.path_pickle_df_experiment_results == "":
            return None

        pd_read = pd.read_pickle(self.path_pickle_df_experiment_results)
        return pd_read


    # Main Moral Elevation Experiment Runner

    def run_exp_moral_elevation(
        self,
        max_workers=8,
        over_ride_existing=False,
        add_explanation=False,
        verbose: bool=True
    ) -> pd.DataFrame:

        """Run a moral elevation experiment by preparing video data with compression settings and processing metadata through an LLM client, optionally overriding existing results and generating explanations, returning experiment results as a DataFrame.
        """
        print("===run_exp_moral_elevation start===")
        df_experiment_results = None

        # Video compression settings

        setting_fps = 30
        setting_crf = 32
        setting_width = 640
        setting_compression = True

        # Prepare Videoinfo
        

        self.df_video_info = self.get_df_video_path()

        self.df_video_info = self.prepare_video_info(
            llm_client=self.llm_client,
            df_video_info=self.df_video_info,
            video_helper=self.video_helper,
            path_ffmpeg_bin=self.path_ffmpeg_bin,
            enable_compression=setting_compression,
            scale_width=setting_width,
            fps=setting_fps,
            crf=setting_crf,
            video_only=self.video_only,
            over_ride_existing=over_ride_existing,
            verbose=verbose,
            max_workers=max_workers,
            temp_dir=self.path_compressed_video
        )

        # Attempt to load cached experiment results

        if self.path_pickle_df_experiment_results and not over_ride_existing:

            try:

                df_experiment_results = self.get_df_experiment_results()

            except Exception:

                print("[INFO] Existing experiment results could not be loaded. The experiment will be re-executed.")

        else:

            print("[INFO] No cached experiment results were found. The experiment will be executed.")

        # If results do not exist, generate them

        if df_experiment_results is None:

            # Create df_experiment_results based on df_video_info
            df_experiment_results = self.update_df_with_individual_prompts(
                df_input=self.df_video_info,
                df_features=self.df_features,
                tools=self.interpreters,
                template_main=self.template_main,
                template_cot=self.template_cot,
                template_knowledge_injection=self.template_knowledge_injection,
                template_tools=self.template_tools
            )
        else:
            df_experiment_results = self.update_df_with_individual_prompts(
                df_input=df_experiment_results,
                df_features=self.df_features,
                tools=self.interpreters,
                template_main=self.template_main,
                template_cot=self.template_cot,
                template_knowledge_injection=self.template_knowledge_injection,
                template_tools=self.template_tools
            )

        if add_explanation:

            df_experiment_results["individual_prompt"] = df_experiment_results.apply(
                lambda row: f"{row['individual_prompt']}\n\nHere is a complete description of this video:\n{row.get('explanation', '')}",
                axis=1
            )

        if df_experiment_results is None:

            print("Error 5: df_experiment_results is empty. Experiment cannot continue.")

            return None



        # Execute multithreaded LLM inference

        df_result = self.run_llm_on_files_multithread(
            df_experiment_results=df_experiment_results,
            llm_client=self.llm_client,
            video_helper=self.video_helper,
            path_ffmpeg_bin=self.path_ffmpeg_bin,
            enable_compression=setting_compression,
            scale_width=setting_width,
            fps=setting_fps,
            crf=setting_crf,
            enable_llm=True,
            video_only=self.video_only,
            over_ride_existing=over_ride_existing,
            default_prompt_text=None,
            verbose=verbose,
            max_workers=max_workers,
            attach_video=self.attach_video,
            temp_dir=self.path_compressed_video
        )

        if df_result is None:

            print("Error 6: df_result is empty. Cannot display experiment results.")

            return None

        if verbose:
            pd.set_option('display.max_colwidth', None)
            print(df_result["LLM_Result"])
            pd.set_option('display.max_colwidth', 10)

        return df_result



    #     Two modes:
    #     1. similarity_based_sampling=False  -> original fold logic
    #     2. similarity_based_sampling=True   -> Retrieve videos one by one based on explanation similarity


















    #                         df_video_info["Group"] == group_value






































    #                     df_video_info["Participant SN"].astype(str) == str(sn)





    #                     f"The prediction of video number:{int_video_number}: (sympathy, inspiration) = ({sympathy}, {inspiration})\n"












    #         df_extended.at[idx, "individual_prompt"] = prompt

















    #     df_result



# Copyright (c) 2026 Ivan Liu.
# Licensed under CC BY 4.0.
"""Fine-tuning support: builds tuning samples from dataframes, writes JSONL,
generates ground truth, and evaluates tuned-model predictions.
"""
import os













# Standard Library
import time
import re
import json
import pandas as pd
from tqdm import tqdm
import jsonlines
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.genai import types





class FineTuningHelper:

    """Generate a list of fine-tuning instances for Gemini video fine-tuning from a DataFrame containing video file names and target responses, skipping rows with missing required fields.
    """
    def __init__(
            self,
            gcs_storage_helper,
            experiment_helper
        ):
        self.gcs_storage_helper = gcs_storage_helper
        self.experiment_helper = experiment_helper






    def create_video_tuning_samples_from_df(self, df: pd.DataFrame):
            """
            Based on a Pandas DataFrame containing video file names and target responses, generate a list of training samples for Gemini video fine-tuning.
            
            Args:
                df (pd.DataFrame): the DataFrame containing the fine-tuning data. It must contain the following columns:
                    - 'video_file_name' (str): the relative path or file name of the video file in the GCS bucket (e.g.: 'clip1.mp4').
                    - 'instance' (str): the correct response the model should generate (i.e. the detailed description/summary of the video).

            Returns:
                list: a list of fine-tuning instance dictionaries. Each dictionary represents one complete user-model conversation turn.
            """
            tuning_instances = []
            
            # Iterate over each row in the DataFrame
            for index, row in df.iterrows():
                # Get the video file name and instance
                video_file_name = row.get('url_bucket_file')
                task_prompt = row.get('individual_prompt')
                suffix = row.get('ground_truth')
                
                # Ensure all required fields exist and are non-empty
                if not video_file_name or not suffix:
                    print(f"Warning: skipping the sample at index {index} because the 'video_file_name' or 'instance' key/value is missing.")
                    continue
                    
                instance = {
                    "contents": [
                        {
                            "role": "user",  # User input content
                            "parts": [  # Contains the video file reference and the text prompt
                                {
                                    "fileData": {
                                        # fileUri is the full GCS path of the video file
                                        "mimeType": "video/mp4",  # <--- video MIME type
                                        "fileUri":video_file_name,  # <--- construct the GCS video path
                                    }
                                },
                                {"text": task_prompt}, # The user's text instruction
                            ],
                        },
                        {
                            "role": "model",  # The model's target output (the correct answer)
                            "parts": [{"text": suffix}],  # The text response expected from the model
                        },
                    ]
                }
                tuning_instances.append(instance)
                
            return tuning_instances

    def update_ground_truth_column(self, df, llm_client, verbose=False):
        """
        Prepare fine-tuning data and append tuning-related columns
        to the original experiment DataFrame.

        Args:
            df: Experiment DataFrame
            llm_client: LLM client for fine-tuning
            verbose: Whether to print verbose logs
        """

        ground_truth_mapping = {
            "INSP": "(1,9)",
            "SYMP": "(9,1)",
            "ELVT": "(9,9)",
        }

        # 1. Generate ground truth
        df_updated = self.generate_ground_truth(df, ground_truth_mapping, llm_client, verbose=verbose)

        return df_updated

    def fine_tuning_data_preparation(self, df, prefix, path_jsonl, llm_client, verbose=False):
        """
        Prepare fine-tuning data and append tuning-related columns
        to the original experiment DataFrame.

        Args:
            df: Experiment DataFrame
            prefix: GCS prefix for uploaded JSONL
            path_jsonl: Local path to save JSONL
            llm_client: LLM client for fine-tuning
            verbose: Whether to print verbose logs
        """

        ground_truth_mapping = {
            "INSP": "(1,9)",
            "SYMP": "(9,1)",
            "ELVT": "(9,9)",
        }

        # 1. Generate ground truth
        df = self.generate_ground_truth(df, ground_truth_mapping, llm_client, verbose=verbose)

        # 2. Upload videos and update file URI column inside df
        df = self.gcs_storage_helper.upload_videos_from_df(
            df,
            "Compressed Video Path",
        )

        # 3. Generate tuning samples
        tuning_data = self.create_video_tuning_samples_from_df(df)

        # 4. Save JSONL locally
        self.save_jsonlines(path_jsonl, tuning_data)

        # 5. Upload JSONL to GCS
        path_uploaded_file = self.gcs_storage_helper.upload_jsonl(
            prefix=prefix,
            path_local_file=path_jsonl,
        )

        # 6. Append tuning fields to the original DataFrame
        df.loc[:, "tuning_file_uri"] = [
            inst["contents"][0]["parts"][0]["fileData"]["fileUri"]
            for inst in tuning_data
        ]

        df.loc[:, "tuning_prompt"] = [
            inst["contents"][0]["parts"][1]["text"]
            for inst in tuning_data
        ]

        df.loc[:, "tuning_ground_truth"] = [
            inst["contents"][1]["parts"][0]["text"]
            for inst in tuning_data
        ]

        return df, tuning_data, path_uploaded_file

    def save_jsonlines(self,file, instances):
        """
        Saves a list of json instances to a jsonlines file.
        """
        with jsonlines.open(file, mode="w") as writer:
            writer.write_all(instances)

        

    def generate_ground_truth(self, df_experiment, mapping, llm_client, verbose=False):
        """
        Generate new ground truth column by:
        1. Replacing tuple with mapped ground truth
        2. Rewriting explanation using LLM so it supports the mapped scores
        """

        df_experiment = df_experiment.copy()

        df_experiment["ground_truth"] = df_experiment.apply(
            self.replace_tuple_and_rewrite_explanation_row,
            axis=1,
            mapping=mapping,
            llm_client=llm_client,
            verbose=verbose
        )

        return df_experiment

    def replace_tuple_and_rewrite_explanation_row(
        self,
        row,
        mapping: dict,
        llm_client,
        verbose=False
    ):
        """
        For each row:
        - Extract original LLM_Result
        - Replace tuple with ground-truth tuple from mapping
        - Ask LLM to rewrite explanation so it supports the ground-truth scores (1-10 scale)
        """

        existing_gt = row.get("ground_truth")

        if isinstance(existing_gt, str):
            word_count = len(existing_gt.strip().split())
            if word_count > 10:
                return existing_gt


        raw = row.get("LLM_Result")

        if pd.isna(raw):
            return raw

        raw = str(raw)

        pattern = re.compile(
            r'\(\s*sympathy\s*,\s*inspiration\s*\)\s*=\s*'
            r'[\(\（]\s*(\d+)\s*[,，]\s*(\d+)\s*[\)\）]',
            re.IGNORECASE
        )

        match = pattern.search(raw)
        if not match:
            return raw

        original_explanation = raw[match.end():].strip()

        group = row.get("Group Name")
        key = str(group).strip().upper() if not pd.isna(group) else None
        mapped = mapping.get(key, "(1,1)")

        m = re.search(r'(\d+)\s*[,，]\s*(\d+)', str(mapped))
        if not m:
            sympathy_gt, inspiration_gt = 1, 1
        else:
            sympathy_gt = int(m.group(1))
            inspiration_gt = int(m.group(2))

        mapped_norm = f"(sympathy, inspiration) = ({sympathy_gt}, {inspiration_gt})"

        prompt_to_use = f"""
    You are an expert in multimodal emotion analysis.

    The ground-truth emotional ratings are:
    Sympathy = {sympathy_gt}
    Inspiration = {inspiration_gt}

    Scale definition:
    1 = extremely low
    10 = extremely high

    Your task:
    Rewrite the explanation below so that it logically justifies why
    sympathy is rated {sympathy_gt} and inspiration is rated {inspiration_gt}.

    Requirements:
    - The reasoning must align with observable evidence.
    - The explanation must clearly support the numerical intensity.
    - Do not mention that this is rewritten.
    - Do not mention the scale description.
    - Write in an analytical academic tone.
    - 80–150 words.

    Original explanation:
    {original_explanation}
    """

        try:
            result = llm_client.chat(
                prompt=prompt_to_use,
                template_used=False,
                stream=False,
                verbose=verbose,
                extra_info=None
            )

            if isinstance(result, dict):
                rewritten_explanation = result.get("content", "").strip()
            else:
                rewritten_explanation = str(result).strip()

        except Exception as e:
            print(f"LLM rewrite failed: {e}")
            rewritten_explanation = original_explanation

        final_text = f"{mapped_norm}\n\n{rewritten_explanation}"

        return final_text

    def run_eval(self, client, val_df, model, max_workers=3, retry_sleep=30):
        """
        Runs evaluation on the validation dataset in parallel.

        Args:
            client: Gemini client
            val_df: Validation DataFrame
            model: Model name or endpoint
            max_workers: Number of parallel threads
            retry_sleep: Seconds to wait before retry on failure

        Returns:
            List of predictions aligned with val_df index order
        """

        def _predict_one(idx, row):
            try:
                return idx, self.get_prediction(
                    client,
                    row["tuning_prompt"],
                    row["tuning_file_uri"],
                    model
                )
            except Exception:
                time.sleep(retry_sleep)
                return idx, self.get_prediction(
                    client,
                    row["tuning_prompt"],
                    row["tuning_file_uri"],
                    model
                )

        predictions = [None] * len(val_df)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_predict_one, idx, row): idx
                for idx, row in val_df.iterrows()
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Running parallel evaluation"
            ):
                idx, prediction = future.result()
                predictions[idx] = prediction

        return predictions


    def extend_df_with_prediction(self, client, df, model, **run_eval_kwargs):
        """
        Extends DataFrame with model predictions.

        Args:
            client: Gemini client
            df: Input DataFrame
            model: Model name or endpoint
            **run_eval_kwargs: Passed to run_eval (e.g., max_workers)

        Returns:
            DataFrame with Predictions column added
        """
        df_output = df.copy()
        predictions = self.run_eval(
            client,
            df_output,
            model=model,
            **run_eval_kwargs
        )
        df_output["sympathy"] = None
        df_output["inspiration"] = None
        df_output.loc[:, "Fine_Tuned_LLM_Text"] = predictions
        df_output = self.experiment_helper.segment_llm_response(df_output, "Fine_Tuned_LLM_Text")
        return df_output




    #     df.loc[:, "Predictions"] = predictions

            
    def get_prediction(self, client, task_prompt, query_video_uri, model):
        """Gets the prediction for a given instance.

        Args:
        query_image: The path to the query image.
        candidates: A list of paths to the candidate images.
        generation_model: The generation model to use for prediction.

        Returns:
        A string containing the prediction.
        """

        response = client.models.generate_content(
            model=model,
            contents=[
                # 1. Specify the URI and MIME type of the video file
                types.Part.from_uri(file_uri=str(query_video_uri), mime_type="video/mp4"),
                # 2. Change to a prompt for the video description
                task_prompt,
            ],
            # Optional config
            config={
                "temperature": 0.0,
            },
        )

        return response.text.strip()



















    def fine_tune_model(
        self,
        client,
        base_model,
        model_name,
        path_uploaded_jsonl_training,
        path_uploaded_jsonl_validation,
        metadata_dir="./tuning_metadata",
        poll_interval=60,
        retry_interval=30,
        max_retry=5,
    ):
        """
        Enhanced fine-tuning function (retains the original return structure)

        ✔ returns tuned_model_url on success
        ✔ raises RuntimeError on failure
        ✔ supports resuming
        ✔ detailed logging
        """
    
        import os
        import json
        import time
        import traceback
        from datetime import datetime
        
        os.makedirs(metadata_dir, exist_ok=True)

        metadata_path = os.path.join(metadata_dir, f"{model_name}_metadata.json")
        log_path = os.path.join(metadata_dir, f"{model_name}_log.txt")

        # Unified log output
        def log(msg):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{timestamp}] {msg}"
            print(line)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()

        log("====================================")
        log(f"Starting to fine-tune the model: {model_name}")
        log("====================================")

        tuning_job_name = None

        if os.path.exists(metadata_path):
            log("Existing metadata found; attempting to resume the task...")

            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                tuning_job_name = metadata.get("tuning_job_name")

                if tuning_job_name:
                    log(f"Successfully read the existing tuning job: {tuning_job_name}")

            except Exception as e:
                log("An error occurred while reading metadata; the task will be recreated.")
                log(str(e))

        if tuning_job_name is None:

            log("Creating a new fine-tuning task...")

            training_dataset = {
                "gcs_uri": path_uploaded_jsonl_training,
            }

            validation_dataset = types.TuningValidationDataset(
                gcs_uri=path_uploaded_jsonl_validation
            )

            sft_tuning_job = client.tunings.tune(
                base_model=base_model,
                training_dataset=training_dataset,
                config=types.CreateTuningJobConfig(
                    adapter_size="ADAPTER_SIZE_EIGHT",
                    epoch_count=1,
                    tuned_model_display_name=model_name,
                    validation_dataset=validation_dataset,
                ),
            )

            tuning_job_name = sft_tuning_job.name

            # Save the task name
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "tuning_job_name": tuning_job_name,
                        "status": "CREATED",
                        "created_time": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            log(f"Fine-tuning task created: {tuning_job_name}")

        else:
            log(f"Continuing to monitor the existing task: {tuning_job_name}")

        TERMINAL_STATES = {
            "JobState.JOB_STATE_SUCCEEDED",
            "JobState.JOB_STATE_FAILED",
            "JobState.JOB_STATE_CANCELLED",
        }

        retry_count = 0

        while True:
            try:
                sft_tuning_job = client.tunings.get(name=tuning_job_name)
                state = str(sft_tuning_job.state)

                log(f"Current task status: {state}")

                if state in TERMINAL_STATES:
                    break

                time.sleep(poll_interval)
                retry_count = 0

            except Exception as e:
                retry_count += 1
                log("A connection error occurred while polling; preparing to retry...")
                log(str(e))
                log(traceback.format_exc())

                if retry_count > max_retry:
                    log("Exceeded the maximum number of retries; stopping execution.")
                    raise

                log(f"Retrying in {retry_interval} seconds...")
                time.sleep(retry_interval)

        final_state = str(sft_tuning_job.state)

        if final_state == "JobState.JOB_STATE_SUCCEEDED":

            tuned_model_url = sft_tuning_job.tuned_model.checkpoints[0].endpoint
            experiment_name = sft_tuning_job.experiment

            log("Fine-tuning completed successfully!")
            log(f"Experiment name: {experiment_name}")
            log(f"Model endpoint: {tuned_model_url}")

            # Update metadata
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "tuning_job_name": tuning_job_name,
                        "status": "SUCCEEDED",
                        "endpoint": tuned_model_url,
                        "experiment": experiment_name,
                        "finished_time": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            log("Fine-tuning process finished (success)")
            log("====================================")

            return tuned_model_url  #  Keep the original return

        else:

            log(f"Fine-tuning failed or was cancelled, status: {final_state}")

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "tuning_job_name": tuning_job_name,
                        "status": final_state,
                        "finished_time": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            log("Fine-tuning process finished (failure)")
            log("====================================")

            raise RuntimeError(f"Tuning failed or cancelled: {final_state}")
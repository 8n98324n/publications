# Copyright (c) 2026 Ivan Liu.
# Licensed under CC BY 4.0.
"""Project data loader: reads per-subject folders, caches objects, and applies
summarisation, baseline subtraction, and IQR-based outlier handling.
"""
import pandas as pd
import local_utilities
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv
import os
import pickle
from pathlib import Path
from glob import glob
import subprocess

class DataLoader:

    """Manage loading, caching, and processing of data files with filtering and summarization capabilities. Initialize with environment configuration from dotenv, a DataHelper instance, and an empty DataFrame. Provide methods to serialize/deserialize data via pickle, process folders with file filtering and user-based merging, and augment DataFrames with summary columns.
    """
    def __init__(self):
        self.data_helper = local_utilities.DataHelper()
        self.df = pd.DataFrame()
        load_dotenv(override=True)
        self.huggingface_cache_location = os.getenv("huggingface_cache_location")
        self.datasets_cache_location = os.getenv("datasets_cache_location")
        self.resutls_location = os.getenv("results_location")


    def save_to_cache(self, data, cache_file):
        """Save data to a pickle cache file."""
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)

    def load_from_cache(self, cache_file):
        """Load data from a pickle cache file."""
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    def prosss_folder(self, base_directory, filters, user_ids, merging_pdf, modality_value= None):

        """Filter files in a directory by attributes and user IDs, merge with group information, select the largest file per user and part, and return the resulting dataframe.
        """
        file_filter = local_utilities.FileFilter(base_directory, attribute_filters=filters, modality_value=modality_value)

        df = file_filter.get_matching_files(user_ids)

        # Merge the two dataframes based on the 'user_id' column
        df_updated = pd.merge(df, merging_pdf[['user_id', 'Group']], on='user_id', how='left')

        # Find the row with the largest file_size for each user_id
        df_largest = df_updated.loc[df_updated.groupby(['user_id', 'part_2'])['file_size'].idxmax()]
        

        # Reset the index if needed
        df_largest.reset_index(drop=True, inplace=True)

        # Display the result
        print(df_largest)

        return df_largest


    def add_summary_columns(self, df: pd.DataFrame, strategies=None):
        """
        Add summary columns to the DataFrame and return both merged and extended summarized DataFrames.

        Parameters:
        - df (pd.DataFrame): Input DataFrame containing a 'user_id' column.
        - strategies (list): List of strategies for summarizing features.

        Returns:
        - tuple: A tuple containing:
        - df_merged (pd.DataFrame): Merged DataFrame with summarized features.
        - df_summarized_extended (pd.DataFrame): Summarized DataFrame with an additional 'Group' column.
        """
        # Add the "seq" column
        if strategies is None:
            strategies = []
        df['seq'] = df.groupby('user_id').cumcount()

        # Extract 'user_id' and 'Group' information
        user_group_mapping = df[['user_id', 'Group']].drop_duplicates().set_index('user_id')['Group']

        # Summarize features using the specified strategies
        processor = psychai.feature.feature_extraction.feature_processor.FeatureProcessor()
        df_summarized = processor.summarize_features_by_column(df.iloc[:, 1:], strategies, group_by_column="user_id")

        # Add 'Group' information to the summarized DataFrame
        df_summarized_extended = df_summarized.copy()
        df_summarized_extended['Group'] = df_summarized_extended['user_id'].map(user_group_mapping)

        # Reorder columns: Move 'Group' and 'user_id' to the front and drop the original first column
        last_two_columns = ['Group', 'user_id']
        other_columns = [col for col in df_summarized_extended.columns if col not in last_two_columns]
        df_summarized_extended = df_summarized_extended[last_two_columns + other_columns]

        # Merge the original DataFrame with the summarized DataFrame
        df_merged = pd.merge(df[:-1], df_summarized, on='user_id', how='left')

        return df_merged, df_summarized_extended


    def remove_outliers_and_calculate_median(self, df_baseline, columns_to_process = None, remove_outlier = False,  outlier_multiplier=1.5):
        """
        Remove outliers for each column in df_baseline according to IQR criteria for each user_id,
        then calculate the mean of each user_id for each column and return the resulting DataFrame.
        
        Args:
            df_baseline (pd.DataFrame): DataFrame containing a 'user_id' column and other numeric columns.

        Returns:
            pd.DataFrame: A new DataFrame with user_id as a column and the median for each column.
        """
        # Separate user_ids

        if columns_to_process is None:
            columns_to_process = []
        if remove_outlier:
            user_groups = df_baseline.groupby('user_id')

            # Function to remove outliers from a column based on IQR
            def remove_outliers(series):
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - outlier_multiplier * IQR
                upper_bound = Q3 + outlier_multiplier * IQR
                return series[(series >= lower_bound) & (series <= upper_bound)]

            # Process each group and calculate medians after outlier removal
            filtered_data = []
            for user_id, group in user_groups:
                filtered_group = group.copy()
                for column in group.columns:
                    if column != 'user_id':
                        filtered_group[column] = remove_outliers(group[column])
                filtered_data.append(filtered_group)

            filtered_df = pd.concat(filtered_data)
        else:
            filtered_df = df_baseline

        # Calculate median for each user_id
        df_baseline_median = filtered_df.groupby('user_id').median().reset_index()
        if columns_to_process != []:
            df_results = df_baseline_median["user_id" + columns_to_process]
        else:
            df_results = df_baseline_median

        return df_results

    def remove_baseline_from_target_df(self, df_target, df_baseline_median, column_to_process = None, outlier_multiplier = 3):
        """
        Subtract all values of each user_id in all columns in df_target that exist in df_baseline_median.
        
        Args:
            df_target (pd.DataFrame): DataFrame containing a 'user_id' column and other numeric columns.
            df_baseline_median (pd.DataFrame): DataFrame with user_id as a column and numeric columns.

        Returns:
            pd.DataFrame: A new DataFrame with the same structure as df_target with values adjusted.
        """



        if column_to_process is None:
            column_to_process = []
        if len(column_to_process) == 0:
            column_to_process = [col for col in df_baseline_median.columns if col in df_target.columns and col != 'user_id']


        # Merge df_target with only common columns from df_baseline_median
        df_merged = pd.merge(
            df_target,
            df_baseline_median[column_to_process.append(pd.Index(['user_id']))],
            on='user_id',
            suffixes=('', '_baseline'),
            how='left',  # Ensures all rows in df_target are kept
        )

        # Subtract baseline values from target values for matching columns
        for col in column_to_process:
            df_merged[col] = df_merged[col] - df_merged[f'{col}_baseline']

        # Retain only columns from df_target
        df_merged = df_merged[df_target.columns]

        return df_merged
    

    def remove_outliers_by_iqr(self, df_target, column_to_process=None, outlier_multiplier=3):
        """
        Remove outliers in the specified column(s) in df_target based on the interquartile range (IQR) for each user_id.

        Args:
            df_target (pd.DataFrame): DataFrame containing a 'user_id' column and other numeric columns.
            column_to_process (list): List of column names to process. If empty, all numeric columns except 'user_id' will be processed.
            outlier_multiplier (float): The multiplier for the IQR to determine outlier thresholds.

        Returns:
            pd.DataFrame: A DataFrame with outliers removed for the specified columns.
        """

        if column_to_process is None:
            column_to_process = []
        if len(column_to_process) == 0:
            column_to_process = [col for col in df_target.columns if col != 'user_id' and pd.api.types.is_numeric_dtype(df_target[col])]

        # Process each column for each user_id
        def remove_column_outliers(group):
            for col in column_to_process:
                q1 = group[col].quantile(0.25)
                q3 = group[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - outlier_multiplier * iqr
                upper_bound = q3 + outlier_multiplier * iqr
                group = group[(group[col] >= lower_bound) & (group[col] <= upper_bound)]
            return group

        # Apply the outlier removal function grouped by user_id
        df_filtered = df_target.groupby('user_id', group_keys=False).apply(remove_column_outliers)

        # Reset index to retain original structure
        df_filtered = df_filtered.reset_index(drop=True)

        return df_filtered


    def cap_outliers_by_iqr(self, df_target, column_to_process=None, outlier_multiplier=3):
        """
        Cap outliers in the specified column(s) in df_target based on the interquartile range (IQR) for each user_id.

        Args:
            df_target (pd.DataFrame): DataFrame containing a 'user_id' column and other numeric columns.
            column_to_process (list): List of column names to process. If empty, all numeric columns except 'user_id' will be processed.
            outlier_multiplier (float): The multiplier for the IQR to determine outlier thresholds.

        Returns:
            pd.DataFrame: A DataFrame with outliers capped for the specified columns.
        """

        if column_to_process is None:
            column_to_process = []
        if len(column_to_process) == 0:
            column_to_process = [col for col in df_target.columns if col != 'user_id' and pd.api.types.is_numeric_dtype(df_target[col])]

        # Process each column for each user_id
        def cap_column_outliers(group):
            for col in column_to_process:
                q1 = group[col].quantile(0.25)
                q3 = group[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - outlier_multiplier * iqr
                upper_bound = q3 + outlier_multiplier * iqr
                group[col] = group[col].clip(lower=lower_bound, upper=upper_bound)
            return group

        # Apply the outlier capping function grouped by user_id
        df_capped = df_target.groupby('user_id').apply(cap_column_outliers)

        # Reset index to retain original structure
        df_capped = df_capped.reset_index(drop=True)

        return df_capped


    def trim_data_frame(self, df:pd.DataFrame, feature_columns = None, prefix = "" ):
        """Return a copy of the data frame with selected columns, optionally prefixed, ensuring Group and user_id appear first.
        """
        if feature_columns is None:
            feature_columns = []
        if len(feature_columns) == 0:
            trimmed_columns = df.columns
        else:
            if type(feature_columns) == type([]):
                trimmed_columns = feature_columns
                trimmed_columns.insert(0, "user_id")
                trimmed_columns.insert(0, "Group")
            elif type(feature_columns) == type(df.columns):
                trimmed_columns = feature_columns
                trimmed_columns = trimmed_columns.insert(0, "user_id")
                trimmed_columns = trimmed_columns.insert(0, "Group")                
        df_trimmed = df.copy()[trimmed_columns]
        if prefix != "":
            df_trimmed.columns = [col if i < 2 else f"{prefix}-{col}" for i, col in enumerate(trimmed_columns)]
        else:
            df_trimmed.columns = [col if i < 2 else f"{col}" for i, col in enumerate(trimmed_columns)]
        return df_trimmed
    
    def get_concatenated_speech_sentiment_data_frame(self):
        """Load concatenated speech sentiment data from CSV, extract feature columns starting from index 3, and return a trimmed data frame with the "concatenated_sentiment" prefix applied.
        """
        file_path = r"./results/speech_text/csv/result_speech_sentiment_concatenated_20241112.csv"
        df_read = pd.read_csv(file_path)
        x_columns = df_read.columns[3:]
        prefix =  "concatenated_sentiment"
        return self.trim_data_frame(df_read, x_columns, prefix)

    def get_facial_df(self, file_target_path, file_baseline_path , outlier_multiplier = 3, to_trim = True, only_summary = True, prefix= "facial"):
        """Load facial data from target and baseline CSV files, filter columns by emotion, pose, and action unit features, remove rows with zero AU01 values and duplicates, add a sequence counter per user, and optionally return only summary statistics.
        """
        df_target = self.get_facial_df_from_csv(file_target_path, prefix)
        df_base_line = self.get_facial_df_from_csv(file_baseline_path, prefix)
        def trimm_df(df, to_trim):
            if to_trim:
                # List of columns containing the keyword 'AU'

                # Additional features to plot
                columns_emotions = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
                columns_pose = ["Pitch","Roll","Yaw"]
                # Filter columns containing any of the substrings

                columns_to_run = ['Group','user_id']
                columns_to_run.extend([col for col in df if any(key in col for key in columns_emotions)])
                columns_to_run.extend([col for col in df if any(key in col for key in columns_pose)])
                columns_to_run.extend([col for col in df if any(key in col for key in ['AU'])])
                columns_to_run.extend([col for col in df if any(key in col for key in ['Face'])])

            
            else:
                columns_to_exclude = ["frame",'Group','user_id']
                columns_to_run = ['Group','user_id']
                columns_to_run.extend([col for col in df if not any(key in col for key in columns_to_exclude)])               

            df_trimmed = df[columns_to_run]

            # Add the "seq" column
            df_trimmed['seq'] = df_trimmed.groupby('user_id').cumcount()

            df_trimmed = df_trimmed[df["AU01"]>0]


            # Assuming your DataFrame is named 'df'
            # Remove duplicate rows
            df_trimmed = df_trimmed.drop_duplicates()

            # Reset index
            df_trimmed = df_trimmed.reset_index(drop=True)

            return df_trimmed
        
        df_target_trimmed = trimm_df(df_target, to_trim)
        df_baseline_trimmed = trimm_df(df_base_line, to_trim)

           
        df_baseline_median = self.remove_outliers_and_calculate_median(df_baseline_trimmed, remove_outlier= False,  outlier_multiplier= outlier_multiplier)
        df_target_centered = self.remove_baseline_from_target_df(df_target_trimmed, df_baseline_median, column_to_process=df_target_trimmed.columns[2:])
        # Extract features using multiple strategies and plot them
        strategies = ['median', 'variance', 'max', 'end_to_begin']
        
        df_with_summary, df_summary = self.add_summary_columns(df_target_centered, strategies)

        threshold = 0.3
        outlier_multiplier = 3
        if only_summary:
            data_prepare_helper = psychai.data_preparation.data_preparation.DataPreparationHelper(df_summary.copy())
        else:
            data_prepare_helper = psychai.data_preparation.data_preparation.DataPreparationHelper(df_with_summary.copy())
        df_prepared = data_prepare_helper.get_data_frame()
        return df_prepared

    def get_facial_df_from_csv(self, file_path, prefix= "facial"):

        """Load a CSV file, filter to rows with complete data, select columns from index 16 onward, and trim the resulting dataframe with the specified feature columns.
        """
        df_read = pd.read_csv(file_path)

        # Convert the existing selection to a list and add the new column at the beginning
        selected_columns = df_read.columns[16:].tolist()

        # Reorder df_filtered_speech_acoustics with the new column order
        # Ensure the new column exists in the DataFrame; otherwise, this will raise an error

        df_read = df_read[~df_read.isnull().any(axis=1)]
        df_read= df_read.reset_index(drop=True)
        prefix =  ""
        return self.trim_data_frame(df_read, feature_columns = selected_columns,prefix=  prefix)
    
    def get_facial_expression_data_frame(self, file_path):
        
        """Read a CSV file and return a DataFrame containing only columns matching facial expression keywords (AU, emotion labels) plus Group and user_id columns positioned first.
        """
        df_read = pd.read_csv(file_path)
        # Define the list of strings to search for
        keywords = ["AU", "happiness", "sadness", "disgust", "anger", "fear", "surprise", "neutral"]

        # Create a regex pattern that matches any of the keywords
        pattern = "|".join(keywords)  # This will create a pattern like "AU|Face"

        # Filter columns based on the pattern
        x_columns = df_read.columns[df_read.columns.str.contains(pattern, case=False)]

        selected_columns = x_columns
        selected_columns = selected_columns.insert(0, "user_id")
        selected_columns = selected_columns.insert(0, "Group")

        # Reorder df_filtered_speech_acoustics with the new column order
        # Ensure the new column exists in the DataFrame; otherwise, this will raise an error
        df_read_explainable = df_read[selected_columns]
        return df_read_explainable
    

    
    def get_eye_movement_df(self, record_path, prefix = "eye",strategies = ['median','max']):
        """Load eye movement data from a CSV file, compute summary statistics using multiple strategies, remove outliers, and return the prepared dataframe.
        """
        df = self.get_eye_movement_df_from_csv(record_path, prefix)
        # Extract features using multiple strategies and plot them
        df_with_summary, df_summary = self.add_summary_columns(df, strategies)
        threshold = 0.3
        outlier_multiplier = 3
        data_prepare_helper = psychai.data_preparation.data_preparation.DataPreparationHelper(df_summary)
        df_prepared = data_prepare_helper.get_data_frame()
        return df_prepared
    
    
    def get_eye_movement_df_from_csv(self, record_path, prefix = "eye"):
        """Load eye movement data from a CSV file, remove rows with missing values, select columns from index 16 onward plus Group and user_id, filter out frame-related columns, and return the trimmed dataframe with the specified prefix applied to data columns.
        """
        df_filtered = pd.read_csv(record_path)
        df_filtered = df_filtered[~df_filtered.isnull().any(axis=1)]
        df_filtered= df_filtered.reset_index(drop=True)
        selected_columns = df_filtered.columns[16:]
        selected_columns = selected_columns.insert(0, "user_id")
        selected_columns = selected_columns.insert(0, "Group")
        df_trimmed = df_filtered[selected_columns]
        df_trimmed = df_trimmed.loc[:, ~df_trimmed.columns.str.contains('frame')]
        x_columns = df_trimmed.columns[2:]
        return self.trim_data_frame(df_trimmed, x_columns, prefix)
    
    def get_pose_features_without_landmarks_data_frame(self, record_path, prefix = "pose"):
        """Load a CSV file, filter to pose-related columns excluding landmarks, remove rows with missing values, and return the trimmed result.
        """
        df_read = pd.read_csv(record_path)
        selected_columns = df_read.columns[16:]

        keywords = ["landmark"]
        # Create a regex pattern that matches any of the keywords
        pattern = "|".join(keywords)  # This will create a pattern like "AU|Face"   
        selected_columns = selected_columns[~selected_columns.str.contains(pattern, case=False)]        
        selected_columns = selected_columns.insert(0, "user_id")
        selected_columns = selected_columns.insert(0, "Group")
        df_read = df_read[selected_columns]

        df_filtered = df_read[~df_read.isnull().any(axis=1)]
        df_filtered= df_filtered.reset_index(drop=True)

        x_columns = df_filtered.columns[2:]
        trimmed_df= self.trim_data_frame(df_filtered, x_columns, prefix)
        return trimmed_df
    
    def get_pose_features_landmarks_data_frame(self):
        """Load pose features from a cached CSV, filter to landmark columns, remove rows with missing values, and return the trimmed data frame with pose prefix applied to feature names.
        """
        record_path = r"./results/pose/pose_features_cache_20241101_2.csv"
        df_read = pd.read_csv(record_path)
        selected_columns = df_read.columns[16:]

        keywords = ["landmark"]
        # Create a regex pattern that matches any of the keywords
        pattern = "|".join(keywords)  # This will create a pattern like "AU|Face"   
        selected_columns = selected_columns[selected_columns.str.contains(pattern, case=False)]        
        selected_columns = selected_columns.insert(0, "user_id")
        selected_columns = selected_columns.insert(0, "Group")
        df_read = df_read[selected_columns]

        df_filtered = df_read[~df_read.isnull().any(axis=1)]
        df_filtered= df_filtered.reset_index(drop=True)

        x_columns = df_filtered.columns[2:]
        prefix =  "pose"
        return self.trim_data_frame(df_filtered, x_columns, prefix)
    
    def get_r_ppg_df(self, record_path, df_info, prefix = "r_ppg"):
        """Load photoplethysmography data from CSV, compute median-based summary statistics, identify outliers using a threshold and multiplier, and return the prepared dataframe.
        """
        df = self.get_r_ppg_df_from_csv(record_path, df_info, prefix)
        # Extract features using multiple strategies and plot them
        strategies = ['median']
        df_with_summary, df_summary = self.add_summary_columns(df, strategies)
        threshold = 0.3
        outlier_multiplier = 3
        data_prepare_helper = psychai.data_preparation.data_preparation.DataPreparationHelper(df_summary)
        df_prepared = data_prepare_helper.get_data_frame()
        return df_prepared
    
    def get_r_ppg_df_from_csv(self, record_path, df_info, prefix = "r_ppg"):
        """Load a CSV file, extract user IDs from filenames, merge with metadata, reorder columns to place Group and user_id first, then trim and return the resulting dataframe.
        """
        df_read = pd.read_csv(record_path)
        df_read['user_id'] = df_read['File_Name'].str.extract(r'PS-9_(\d{3})')[0].astype(int)
        df_merged = pd.merge(df_read, df_info, on='user_id', how='left')
        df_merged= df_merged.reset_index(drop=True)
        selected_columns = df_merged.columns.copy()
        selected_columns = selected_columns[1:-3]
        selected_columns = selected_columns.insert(0, "user_id")
        selected_columns = selected_columns.insert(0, "Group")
        df_trimmed = df_merged[selected_columns]
        x_columns = df_trimmed.columns[2:]
        return self.trim_data_frame(df_trimmed, x_columns, prefix)
    
    def get_mean_remote_ppg_data_frame(self):
        """Return a DataFrame containing mean PPG data aggregated by user ID, loaded from the remote PPG CSV source.
        """
        df= self.get_r_ppg_df_from_csv()
        df_mean = self.data_helper.calculate_mean_by_user_id(df)
        return df_mean
    
    def get_thermal_df(self, record_path, prefix =  "thermal"):

        """Load thermal data from a CSV file, prepare it using DataPreparationHelper, and return the processed DataFrame.
        """
        df = self.get_thermal_df_from_csv(record_path, prefix)
        data_prepare_helper = psychai.data_preparation.data_preparation.DataPreparationHelper(df)
        df_prepared = data_prepare_helper.get_data_frame()
        return df_prepared
        
    def get_thermal_df_from_csv(self, record_path, prefix =  "thermal"):

        """Load a CSV file, select columns from index 16 onward plus Group and user_id, and return the trimmed dataframe with the prefix applied.
        """
        df_read = pd.read_csv(record_path)
        selected_columns = df_read.columns[16:]
        selected_columns = selected_columns.insert(0, "user_id")
        selected_columns = selected_columns.insert(0, "Group")
        df_read = df_read[selected_columns]
        x_columns = df_read.columns[2:]
        return self.trim_data_frame(df_read, x_columns, prefix)
    
    def get_mm_llm_df(self, file_path, cap_outlier = True, outlier_multiplier= 3, prefix =  "mm_llm"):
        """Load and prepare multimodal LLM data from CSV by dropping rows with excessive missing values, imputing missing values with means, optionally capping outliers, and returning the cleaned dataframe.
        """
        df_prepared = self.get_mm_llm_df_from_csv(file_path, prefix = prefix)
        threshold = 0.1
        data_prepare_helper = psychai.data_preparation.data_preparation.DataPreparationHelper(df_prepared)
        data_prepare_helper.drop_missing_data(threshold=0.1)
        data_prepare_helper.handle_missing_values("mean", df_prepared.columns[0:].tolist())
        if cap_outlier:
            df_prepared = data_prepare_helper.process_outliers(strategy = 'cap', columns = df_prepared.columns[2:], threshold=threshold, outlier_multiplier= outlier_multiplier)
        df_prepared = data_prepare_helper.get_data_frame()
        return df_prepared
    def get_mm_llm_df_from_csv(self, file_path, prefix =  "mm_llm"):
        """Load a CSV file, filter columns by excluding those containing 'frame', 'Group', or 'sequence', reorder to place 'Group' and 'user_id' first, then trim the resulting dataframe using the remaining columns as features.
        """
        df_read = pd.read_csv(file_path)
        selected_columns = df_read.columns
        selected_columns = selected_columns[3:-1].tolist()
        keywords = ['frame', 'Group', 'sequence']
        selected_columns = [col for col in selected_columns if not any(keyword in col for keyword in keywords)]
        selected_columns.insert(0, "user_id")
        selected_columns.insert(0, "Group")
        df_read = df_read[selected_columns]
        x_columns = df_read.columns[2:]
        return self.trim_data_frame(df_read, x_columns, prefix)
        
    def get_speech_sentiment_data_frame(self, file_path):
    
        """Load a CSV file, filter columns starting from index 17 while excluding those containing 'frame', 'Group', or 'sequence', add a 'sentiment-' prefix, and return the trimmed data frame.
        """
        df_read = pd.read_csv(file_path)
        selected_columns = df_read.columns
        selected_columns = selected_columns[17:].tolist()
        keywords = ['frame', 'Group', 'sequence']
        selected_columns = [col for col in selected_columns if not any(keyword in col for keyword in keywords)]
        prefix =  "sentiment-"
        return self.trim_data_frame(df_read, selected_columns, prefix)
    
    def get_mean_speech_sentiment_data_frame(self):
        """Return a DataFrame containing the mean sentiment score for each user across all their speeches.
        """
        df= self.get_speech_sentiment_data_frame()
        df_mean = self.data_helper.calculate_mean_by_user_id(df)
        return df_mean
    
    def get_speech_text_df(self, file_path, parameters: dict =  None, prefix = "speech_text"): # do_log_transform = False, remove_outlier = False):
        """Load speech text data from CSV, extract the first two and last 28 columns, compute median/variance/max summary statistics, and return the prepared dataset.
        """
        if parameters is None:
            parameters = {}
        df_prepared = self.get_speech_text_df_from_csv(file_path, prefix = prefix)
        df_prepared = df_prepared[df_prepared.columns[:2].append(df_prepared.columns[-28:])]
        strategies = ['median','variance','max']
        df_with_summary, df_summary = self.add_summary_columns(df_prepared, strategies)
        data_prepare_helper = psychai.data_preparation.data_preparation.DataPreparationHelper(df_summary)
        df_prepared = data_prepare_helper.prepare_data()
        return df_prepared
    
    
    def get_speech_text_df_from_csv(self, file_path, prefix = "speech_text"):
        """Load a CSV file, filter columns by excluding those containing frame, Group, sequence, or attribute_value, and return the trimmed DataFrame with the specified prefix applied.
        """
        df_read = pd.read_csv(file_path)
        selected_columns = df_read.columns
        selected_columns = selected_columns[3:].tolist()
        keywords = ['frame', 'Group', 'sequence','attribute_value']
        selected_columns = [col for col in selected_columns if not any(keyword in col for keyword in keywords)]
        return self.trim_data_frame(df_read, selected_columns, prefix)
    
    def get_mean_speech_text_data_frame(self):
        """Return a DataFrame containing mean speech text values grouped by user ID."""
        df= self.get_speech_text_df_from_csv()
        df_mean = self.data_helper.calculate_mean_by_user_id(df)
        return df_mean
    
    def get_acoustics_df(self, path_target, path_baseline, prefix="acoustics"):
        """Load acoustic data from target and baseline CSV files, remove outliers and baseline median from target, compute median/variance/max summary statistics, and return prepared data with capping and missing-value imputation applied.
        """
        df_target = self.get_acoustics_df_from_csv(begin_column=14, record_path= path_target, prefix = prefix)
        df_baseline = self.get_acoustics_df_from_csv(begin_column=3, record_path= path_baseline)
        df_baseline_median = self.remove_outliers_and_calculate_median(df_baseline, outlier_multiplier= 3)
        df_target_centered = self.remove_baseline_from_target_df(df_target, df_baseline_median, column_to_process=df_target.columns[2:])
        strategies =  ['median','variance','max']
        df_with_summary, df_summary = self.add_summary_columns(df_target_centered, strategies)
        data_prepare_helper = psychai.data_preparation.data_preparation.DataPreparationHelper(df_summary)
        parameter = {
            "cap_data": True,
            "outlier_multiplier": 3,
            "replace_missing_data_with_mean":True
        }
        df_prepared = data_prepare_helper.prepare_data(parameter)
        return df_prepared

    def get_acoustics_df_from_csv(self, begin_column = None, end_column = None, record_path = "", prefix="acoustics"):
        """Load audio features from a CSV file, optionally filter columns by index range, remove columns matching specific substrings, trim the dataframe with a prefix, and return the cleaned result with dropped empty rows and reset index.
        """
        if record_path == "":
            record_path =  os.path.join(self.resutls_location, "examples", "paper5_mmer_moral", "audio","audio_features_merged_20241219.csv")
        df_read = pd.read_csv(record_path)
        if begin_column is not None and end_column is not None:
            x_columns = df_read.columns[begin_column:end_column]
        elif begin_column:
            x_columns = df_read.columns[begin_column:]
        elif begin_column:
            x_columns = df_read.columns[:end_column]
        else:
            x_columns = df_read.columns

        # Define the substrings to match for removal
        removal_substrings = ["seq","part","attribute","file","Group","user_id",]
        # Use a list comprehension to filter out columns containing any of the substrings
        x_columns_filtered = [col for col in x_columns if not any(substring in col for substring in removal_substrings)]
        matching_columns = [col for col in x_columns if any(sub in col for sub in removal_substrings)]
        print(f"Filtered out columns: {matching_columns}")

        df_read =  self.trim_data_frame(df_read, x_columns_filtered, prefix)


        df_cleaned = df_read.dropna(how='all')
        df_cleaned= df_cleaned.reset_index(drop=True)

        return df_cleaned

    def get_mean_acoustics_data_frame(self):
        """Load acoustics data from CSV, calculate the mean by user ID, and return the aggregated data frame.
        """
        df_read = self.get_acoustics_df_from_csv()
        df_mean_speech_acoustics = self.data_helper.calculate_mean_by_user_id(df_read)
        return df_mean_speech_acoustics


    def merge_user_info(self, df_users, df_files):
        # First convert Participant SN into a number
        """Merge user group information into the files dataframe by matching participant serial numbers, map group codes to names, and return the enriched dataframe.
        """
        df_files["Participant_SN_int"] = df_files["Participant SN"].astype(int)

        # Merge
        df_files = df_files.merge(
            df_users[["user_id", "Group"]],
            left_on="Participant_SN_int",
            right_on="user_id",
            how="left"
        )

        # user_id and the temporary column are not needed and can be deleted
        df_files = df_files.drop(columns=["user_id"], errors="ignore")

        # Generate the Group Name
        group_names = {
            1: "ELVT",
            2: "SYMP",
            3: "INSP",
            4: "CTRL"
        }

        df_files["Group Name"] = df_files["Group"].map(group_names)
        return df_files

    def merge_and_get_video_with_audio(
        self,
        path_data_experiment,
        df_user_info,
        participant_sn_range,
        experiment_name=None,
        experiment_step=None,
        month=None,
        day=None,
        hour=None,
        minute=None,
        second=None,
        path_pickle=None,              # ★ New parameter: pickle storage location
        path_ffmpeg_bin = "ffmpeg",
        path_ffprobe_bin= "ffprobe"
    ):
        """
        New feature:
        If the pickle file pointed to by path_pickle exists and use_pickle_if_exists=True,
        read and return that pickle directly, without performing any processing (including searching for videos and merging).

        Original features retained:
        Filter videos → find audio → automatically concatenate → trim/pad to length → merge audio.
        """

        # ★ Step 0: Handle the pickle first
        missing_sns = participant_sn_range
        to_extend = False
        if path_pickle is not None:
            path_pickle = Path(path_pickle)
            if path_pickle.exists():
                print(f"[INFO] Pickle file exists; reading it directly: {path_pickle}")
                pd_read_from_pickle =  pd.read_pickle(path_pickle)
                # Check whether self.participant_sn_range is in the loaded DataFrame
                if not pd_read_from_pickle.empty:
                    unique_sns = pd_read_from_pickle["Participant SN"].astype(int).unique()
                    to_extend = True
                    if all(sn in unique_sns for sn in participant_sn_range):
                        return pd_read_from_pickle
                    else:
                        print("[WARNING] The Participant SN in the pickle file is incomplete; the videos and audio will be reprocessed.")
                        # List the SNs that are not within the current range
                        missing_sns = [sn for sn in participant_sn_range if sn not in unique_sns]
                        print(f"Missing Participant SN: {missing_sns}")

        # ★ The following is the original complete workflow (kept consistent)

        def to_str(x):
            return None if x is None else str(x)

        def to_sn_str(x):
            return None if x is None else f"{int(x):03d}"

        experiment_name = to_str(experiment_name)

        # Supports step being a single value or a collection
        if experiment_step is None:
            experiment_step_set = None
        else:
            if isinstance(experiment_step, (list, tuple, set)):
                experiment_step_set = {to_str(s) for s in experiment_step}
            else:
                experiment_step_set = {to_str(experiment_step)}

        month, day = to_str(month), to_str(day)
        hour, minute, second = to_str(hour), to_str(minute), to_str(second)

        rows = []

        # --------- Search for videos ---------
        for sn in missing_sns:
            sn_folder = f"{sn:03d}"
            use_alternative = False
            folder_rec = os.path.join(path_data_experiment, sn_folder, "RecordedVideo")
            folder_rec_alternative = os.path.join(path_data_experiment, sn_folder, "Segment_WebCam")
            folder_audio = os.path.join(path_data_experiment, sn_folder, "Segment_Audio")
            folder_output = os.path.join(path_data_experiment, sn_folder, "video_with_audio")

            if not os.path.exists(folder_rec):
                if not os.path.exists(folder_rec_alternative):
                    continue
                else:
                    folder_rec = folder_rec_alternative
                    use_alternative = True

            os.makedirs(folder_output, exist_ok=True)

            for filename in os.listdir(folder_rec):
                 if not filename.lower().endswith((".mp4", ".avi", ".mov")):
                    folder_rec = folder_rec_alternative
                    use_alternative = True         

            for filename in os.listdir(folder_rec):
                if not filename.lower().endswith((".mp4", ".avi", ".mov")):
                    continue

                name_only = filename.split(".")[0]
                parts = name_only.split("_")
                if len(parts) < 8:
                    continue

                f_experiment_name, f_participant_sn, f_step = parts[0], parts[1], parts[2]
                f_month, f_day, f_hour, f_minute, f_second = parts[3:8]

                # Filter
                if experiment_name and f_experiment_name != experiment_name:
                    continue
                if experiment_step_set and f_step not in experiment_step_set:
                    continue
                if month and f_month != month.zfill(2): continue
                if day and f_day != day.zfill(2): continue
                if hour and f_hour != hour.zfill(2): continue
                if minute and f_minute != minute.zfill(2): continue
                if second and f_second != second.zfill(2): continue
                if int(f_participant_sn) not in participant_sn_range:
                    continue

                video_path = os.path.join(folder_rec, filename)
                video_size = os.path.getsize(video_path)

                rows.append({
                    "Experiment Name": f_experiment_name,
                    "Participant SN": f_participant_sn,
                    "Experiment Step": f_step,
                    "Month": f_month,
                    "Day": f_day,
                    "Hour": f_hour,
                    "Minute": f_minute,
                    "Second": f_second,
                    "Video Path": video_path,
                    "File Size": video_size,
                    "Audio Folder": folder_audio,
                    "Output Folder": folder_output,
                    "Base Name": name_only,
                    "LLM_Result": "",
                    "sympathy": "",
                    "inspiration": "",
                    "Use_Alternative": use_alternative
                })

        df = pd.DataFrame(rows)
        if not df.empty:

            # Keep the largest file per participant + step
            df = df.sort_values("File Size", ascending=False).groupby(
                ["Participant SN", "Experiment Step"], as_index=False
            ).first()

            # Start the audio merge
            final_output_list = []

            for idx, row in df.iterrows():

                video_path = row["Video Path"]
                folder_audio = row["Audio Folder"]
                folder_output = row["Output Folder"]
                exp, sn, step = row["Experiment Name"], row["Participant SN"], row["Experiment Step"]
                base_name = row["Base Name"]
                use_alternative = row["Use_Alternative"]

                audio_pattern = f"{exp}_{sn}_{step}_*.wav"
                audio_files = sorted(glob(os.path.join(folder_audio, audio_pattern)))

                output_file = os.path.join(folder_output, base_name + ".mp4")

                # If the output already exists → do not process
                if os.path.exists(output_file):
                    final_output_list.append(output_file)
                    continue

                if not use_alternative:

                    # No audio → copy directly
                    if len(audio_files) == 0:
                        subprocess.call([
                            path_ffmpeg_bin, "-y",
                            "-i", video_path,
                            "-c", "copy",
                            output_file
                        ])
                        final_output_list.append(output_file)
                        print(f"[INFO] No audio; the video has been copied: {output_file}")
                        continue

                    # Has audio → concat
                    concat_list = os.path.join(folder_output, "_tmp_audio_list.txt")
                    with open(concat_list, "w") as f:
                        for ap in audio_files:
                            f.write(f"file '{ap}'\n")

                    concat_audio = os.path.join(folder_output, "_tmp_concat.wav")

                    subprocess.call([
                        path_ffmpeg_bin, "-y",
                        "-f", "concat",
                        "-safe", "0",
                        "-i", concat_list,
                        "-c", "copy",
                        concat_audio
                    ])

                    # Video length
                    cmd = [
                        path_ffprobe_bin, "-v", "error", "-show_entries",
                        "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                        video_path
                    ]
                    video_duration = float(subprocess.check_output(cmd).decode().strip())

                    trimmed_audio = os.path.join(folder_output, "_tmp_trimmed.wav")
                    subprocess.call([
                        path_ffmpeg_bin, "-y",
                        "-i", concat_audio,
                        "-t", str(video_duration),
                        "-c", "copy",
                        trimmed_audio
                    ])

                    # Merge
                    subprocess.call([
                        path_ffmpeg_bin, "-y",
                        "-i", video_path,
                        "-i", trimmed_audio,
                        "-c:v", "copy",
                        "-c:a", "aac",
                        "-shortest",
                        output_file
                    ])
                    print(f"[INFO] Video merged: {output_file}")

                    # Clear tmp
                    for tmp in [concat_list, concat_audio, trimmed_audio]:
                        if os.path.exists(tmp):
                            os.remove(tmp)
                else:
                    # alternative does not need merging
                    output_file = video_path

                final_output_list.append(output_file)

            df["Merged File Path"] = final_output_list

            # merge user info
            df = self.merge_user_info(df_user_info, df)
        
        
        if to_extend:
            df_extended = pd_read_from_pickle
            if not df.empty:
                df_extended = pd.concat([df_extended, df], ignore_index=True)

            # ★ Step X: If the path_pickle parameter is provided → automatically save as a pickle
            if path_pickle is not None:
                df_extended.to_pickle(path_pickle)
                print(f"[INFO] The result has been saved as a pickle: {path_pickle}")
            return df_extended
        else:
            return df
        

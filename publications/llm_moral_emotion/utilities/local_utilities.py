# Copyright (c) 2026 Ivan Liu.
# Licensed under CC BY 4.0.
"""Project utilities: single-modality analysis, machine-learning runners, dataframe helpers,
file filtering, and object caching.
"""

import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

import psychai.visualization.chart_helper
import psychai.machine_learning.classification.classification_supervise_learning

import sys
current_dir = os.path.abspath('../')
sys.path.append(current_dir)

import local_data_loader
import local_utilities
import seaborn as sns
from scipy.stats import ttest_1samp, ttest_ind, pearsonr, f_oneway
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy.stats import boxcox
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import f_classif

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelBinarizer


class SingleModalityAnalysis:

    """Analyze files in a directory by filtering, merging with metadata, and returning the largest file per user and part.
    """
    def __init__(self):
        pass

    def prosss_folder(self, base_directory, filters, user_ids, merging_pdf, modality_value= None):

        """Process a folder by filtering files, merging with provided data, and returning the largest file for each user and part combination.
        """
        file_filter = FileFilter(base_directory, attribute_filters=filters, modality_value=modality_value)

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

class MachineLearning:
    """Manage machine learning workflows including data preparation and model training. Initialize with a DataFrame copy. Prepare data by optionally relabeling groups, applying preprocessing transformations (feature selection, PCA, outlier handling), and splitting into train/test sets. Run supervised learning classification with grid search over model parameters.
    """
    def __init__(self, df):
        self.df = df.copy()
        pass


    def prepare_data(self, group_labels, x_columns, y_column, column_to_split= "",
                            explained_variance_threshold=0.95,
                            remove_redundant_features=True, remove_low_variance=False, 
                            correlation_threshold=0.9, 
                            variance_threshold=0.01, use_pca=False,
                            outlier_strategies = ['Replace with Median'],
                            random_state = 1,
                            k=1.5):
        


        # Create a copy of the DataFrame and relabel the 'Group' column according to task specifications
        """Relabel the Group column if group_labels are provided, then prepare training and test datasets by preprocessing features and target variable according to specified parameters, returning the split and processed data.
        """
        if group_labels != {}:
            self.df["Group"] = self.df["Group"].map(group_labels)

        data_preprocessor = psychai.data_preparation.preprocessing.DataPreprocessor()
        # Prepare data
        X_train, y_train, X_test, y_test = data_preprocessor.prepare_training_data(self.df, x_columns, y_column,
                        column_to_split=column_to_split, explained_variance_threshold=explained_variance_threshold, 
                        correlation_threshold=correlation_threshold, 
                        remove_redundant_features=remove_redundant_features, remove_low_variance=remove_low_variance, 
                        variance_threshold=variance_threshold, use_pca=use_pca, outlier_strategies= outlier_strategies, k=k, 
                        random_state = random_state)
        return X_train, y_train, X_test, y_test
    
    def run_machine_learning_with_splitted_data(self, task_name, x_columns, y_column, 
                                                X_train, y_train, X_test, y_test, parameters= None):
        """Train a Random Forest classifier on the provided training data, perform grid search to optimize hyperparameters, evaluate the model on the test set, and display the best parameters, cross-validation score, and top feature importances.
        """
        if parameters is None:
            parameters = {}
        if parameters == {}:
            parameters = {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 3, 5, 7],
            }
        
        # Run grid search for best model parameters
        
        rf_classifier = psychai.machine_learning.classification.classification_supervise_learning.Classification(X_train.copy(), 
                        y_train.copy(), X_test.copy(), y_test.copy(), model_name=f"Random Forest - {task_name}", column_name=x_columns)
        grid_search_results = rf_classifier.grid_search_by_name("Random Forest", parameters=parameters, cv=10)
        
        # Display best hyperparameters and CV score
        print(f"Task: {task_name}")
        print("Best Hyperparameters:", grid_search_results.best_params_)
        print("Best CV Score:", grid_search_results.best_score_)
        
        # Evaluate the best model on the test set
        rf_classifier.evaluate_model()
        print("\n")

        rf_classifier.feature_importance(10)


    from sklearn.metrics import accuracy_score, f1_score











        







    def run_machine_learning(
        self,
        group_labels,               # Dictionary for relabeling classes (e.g., {"1": "positive", "2": "negative"})
        task_name,                  # Name of the classification task for display
        x_columns,                  # List of feature columns
        y_column,                   # Target column to predict
        column_to_split="",         # Optional column to split data for training/summarizetesting (if needed)
        cv=10,
        explained_variance_threshold=1,
        correlation_threshold=0.9,
        remove_redundant_features=True,
        remove_low_variance=False,
        variance_threshold=0.01,
        use_pca=False,
        model_type="Random Forest",  # "Random Forest" or "Decision Tree"
        grid_search_parameters=None,
        n_top_features=10,
        outlier_strategies=['Replace with Median'],
        k=1.5,
        random_state=1,
        model_parameters=None
    ):
        """
        A function to run a machine learning pipeline, including data preparation, feature selection,
        hyperparameter tuning, and evaluation metrics.
        Returns accuracy and F1 scores for each class.
        """

        if grid_search_parameters is None:
            grid_search_parameters = {}
        if model_parameters is None:
            model_parameters = {}
        X_train, y_train, X_test, y_test = self.prepare_data(
            group_labels,
            x_columns,
            y_column,
            column_to_split=column_to_split,
            correlation_threshold=correlation_threshold,
            explained_variance_threshold=explained_variance_threshold,
            remove_redundant_features=remove_redundant_features,
            remove_low_variance=remove_low_variance,
            variance_threshold=variance_threshold,
            use_pca=use_pca,
            outlier_strategies=outlier_strategies,
            k=k,
            random_state=random_state
        )

        classifier = psychai.machine_learning.classification.classification_supervise_learning.Classification(
            X_train,
            y_train,
            X_test,
            y_test,
            model_label=f"{model_type}-{task_name}",
            model_name=model_type,
            column_name=x_columns,
            random_state=random_state,
            model_parameters=model_parameters
        )

        if not grid_search_parameters:
            classifier.train_model()
        else:
            classifier.grid_search(
                grid_search_parameters=grid_search_parameters,
                cv=cv
            )

        unique_labels = list(dict.fromkeys(group_labels.values())) if group_labels else list(set(y_train))

        accuracy, f1_scores = classifier.evaluate_model(labels_order=unique_labels, font_size=16)

        feature_importance = None
        if model_type == "Random Forest":
            feature_importance = classifier.feature_importance(n_top_features)

        if model_type == "Decision Tree":
            from sklearn.tree import plot_tree
            import matplotlib.pyplot as plt

            # Get the trained Decision Tree model
            model = classifier.get_trained_model()

            # Plot the decision tree with proper feature names and labels
            plt.figure(figsize=(20, 10))
            plot_tree(
                model,
                feature_names=x_columns,      # Ensure feature names are displayed correctly
                class_names=unique_labels,   # Ensure result labels are displayed
                filled=True,
                fontsize=12
            )
            plt.show()

        # Return results for further use if needed
        return accuracy, f1_scores, feature_importance


class DataHelper:

    """Provide utilities for merging multiple DataFrames on user_id while preserving group assignments, extracting text between markers, and calculating numeric means grouped by user_id.
    """
    def __init__(self):
        pass
   
    def merge_dataframe(self, dataframes):
        # Verify each DataFrame contains 'user_id' and 'Group' columns
        """Merge multiple DataFrames on user_id using an outer join, validate that each contains user_id and Group columns, deduplicate Group values by keeping the first occurrence per user, and return a combined DataFrame with Group as the second column.
        """
        for i, df in enumerate(dataframes):
            if 'user_id' not in df.columns or 'Group' not in df.columns:
                raise KeyError(f"DataFrame at index {i} is missing 'user_id' or 'Group' column.")

        # Concatenate all DataFrames on 'user_id', performing an outer join and excluding 'Group' columns
        merged_data = pd.concat([df.set_index('user_id').drop(columns='Group') for df in dataframes], axis=1, join="outer").reset_index()

        # Combine 'Group' columns by taking the first available 'Group' for each 'user_id'
        group_data = pd.concat([df[['user_id', 'Group']].drop_duplicates(subset='user_id') for df in dataframes])
        group_data = group_data.drop_duplicates(subset='user_id', keep='first')

        # Merge the 'Group' column back into the final DataFrame as the second column
        df_merged = pd.merge(group_data, merged_data, on='user_id', how='outer')

        return df_merged

    # Define a function to clean up the text in each row
    def clean_llm_returned_text(self, text):
        # Check if both markers exist in the text
        """Extract and return text between <|notimestamps|> and <|endoftext|> markers if both are present, otherwise return the original text with leading and trailing whitespace removed.
        """
        start_marker = "<|notimestamps|>"
        end_marker = "<|endoftext|>"
        
        if start_marker in text and end_marker in text:
            # Find the start and end positions of the markers
            start_idx = text.find(start_marker) + len(start_marker)
            end_idx = text.find(end_marker)
            
            # Extract the text between the markers
            return text[start_idx:end_idx].strip()  # .strip() to remove any leading/trailing spaces
        else:
            # If markers are not found, return the original text
            return text

    # Group by user_id
    def calculate_mean_by_user_id(self, df):
        """Group data by user_id, check non-numerical column consistency, compute numeric means, and merge back non-numerical columns.
        """
        df_consistent = df.groupby('user_id').apply(self.check_non_numerical_consistency).reset_index(drop=True)

        df_mean = df.groupby('user_id').mean(numeric_only=True).reset_index()

        # Ensure 'user_id' is present in non_numerical_columns subset
        non_numerical_columns = df_consistent[['user_id'] + df_consistent.select_dtypes(include=['object']).columns.tolist()].drop_duplicates(subset=['user_id'])
        
        df_mean = pd.merge(df_mean, non_numerical_columns, on='user_id', how='left')

        return df_mean

    # Function to check consistency in non-numerical columns
    def check_non_numerical_consistency(self, group):
        """Verify that all object-type columns in the group contain only a single unique value, raising ValueError if inconsistency is found, then return the first row.
        """
        for col in group.select_dtypes(include=['object']).columns:
            if group[col].nunique() > 1:
                raise ValueError(f"Inconsistent values found in column '{col}' for user_id '{group.name}'")
        return group.iloc[0]  # Return the first row for consistent non-numerical columns


class FileFilter:
    """Initialize the FileFilter with a base directory, attribute filter tuples specifying (index, value) pairs for filename splitting, and a target modality folder name.
    """
    def __init__(self, base_directory, attribute_filters, modality_value):
        """
        Initializes the FileFilter object with base directory, list of attribute filters, and modality value.

        Args:
            base_directory (str): The base directory that contains the user folders (e.g., 001, 002, ..., 101).
            attribute_filters (list of tuples): List of (attribute_index, attribute_value) pairs to filter by.
            modality_value (str): The value you are looking for in the modality folder (e.g., 'PPG', 'Segment_Audio').
        """
        self.base_directory = base_directory  # Base directory path where the user folders are located
        self.attribute_filters = attribute_filters  # List of (attribute_index, attribute_value) tuples
        self.modality_value = modality_value  # Modality folder name
        self.matching_files = []  # List to store matching file information

    def filter_files_by_attributes(self, file_name):
        """
        Filters files based on multiple attributes in the file name.

        Args:
            file_name (str): The name of the file to check (e.g., SUBJ_001_1_10_21_14_30_45.txt).

        Returns:
            bool: True if the file matches all the desired attributes, otherwise False.
        """
        # Split the file name into parts by the underscore character
        parts = file_name.split('_')
        
        # Check if the file name format has enough parts
        for attribute_index, attribute_value in self.attribute_filters:
            if attribute_index < len(parts) and parts[attribute_index] == attribute_value:
                return True
        return False

    def process_user_folder(self, user_id):
        """
        Processes the files in a specific user folder and checks for matching files.

        Args:
            user_id (int): The ID of the user folder to process (e.g., 1 for '001').
        """
        # Format the user folder ID as a three-digit string (e.g., 001, 002, etc.)
        user_folder = f"{user_id:03d}"

        # Construct the full path to the modality folder inside the user's directory
        modality_folder = os.path.join(self.base_directory, user_folder, self.modality_value)

        # Check if the modality folder exists for this user
        if os.path.exists(modality_folder):
            # Iterate through all files in the modality folder
            for file_name in os.listdir(modality_folder):
                # Check if the file matches the desired attributes
                if self.filter_files_by_attributes(file_name):
                    # Split the file name into parts and append a dictionary with all necessary information
                    parts = file_name.split('_')
                    file_path = os.path.join(modality_folder, file_name)
                    file_info = {
                        'user_id': user_id,
                        'modality': self.modality_value,
                        'attribute_index': 2, #[index for index, _ in self.attribute_filters],
                        'attribute_value': parts[2], #[value for _, value in self.attribute_filters],
                        'files': file_path,
                        'file_size': os.path.getsize(file_path)
                    }
                    # Add parts as separate columns
                    for i, part in enumerate(parts):
                        file_info[f'part_{i}'] = part
                    
                    self.matching_files.append(file_info)
        else:
            # Print a message if the modality folder is not found
            print(f"{self.modality_value} folder not found for user {user_folder}")

        pass

    def find_matching_files(self, user_range):
        """
        Loops through the user folders to find files that match the specified attributes.

        Args:
            user_range (range): The range of user IDs to check (e.g., range(1, 14) for users 001 to 013).
        """
        # Iterate through each user folder in the specified range
        for user_id in user_range:
            # Process the files for each user folder
            self.process_user_folder(user_id)

        # Print out the total number of matching files found
        print(f"Total matching files found: {len(self.matching_files)}")

    def get_matching_files(self, user_range):
        """
        Finds the matching files and returns them as a DataFrame, including parts of the file name.

        Args:
            user_range (range): The range of user IDs to check (e.g., range(1, 14) for users 001 to 013).

        Returns:
            pd.DataFrame: A DataFrame containing 'user_id', 'attribute_index', 'attribute_value', 'files',
                          and individual parts of the file name.
        """
        # Clear previous matching files
        self.matching_files = []

        # Find matching files in the specified user range
        self.find_matching_files(user_range)

        # Convert matching files to a pandas DataFrame
        df = pd.DataFrame(self.matching_files)

        return df




class IO:
    """Cache an object using pickle, loading from disk if available and not overridden, otherwise creating it via the provided function and saving it.
    """
    def __init__(self):
        pass

    def cache_object(self, obj_name, folder, create_function, override=False):
        """
        Cache the object using pickle.
        
        Parameters:
        - obj_name (str): Name of the object to be cached, used as the filename.
        - create_function (function): A function that generates the object.
        - override (bool): If True, always recreate the object and overwrite the cache.
        
        Returns:
        - The object, either from cache or newly created.
        """
        # Create the filename based on the object's name
        filename = f"{folder}\\{obj_name}.pkl"
        
        # Check if the file exists and whether override is False
        if os.path.exists(filename) and not override:
            # Load and return the object from cache
            with open(filename, 'rb') as f:
                print(f"Loading {obj_name} from cache.")
                return pickle.load(f)
        else:
            # Create the object using the provided function
            print(f"Creating {obj_name} using the provided function.")
            obj = create_function()
            
            # Save the object to the cache file
            with open(filename, 'wb') as f:
                pickle.dump(obj, f)
                print(f"{obj_name} saved to cache.")
            
            return obj
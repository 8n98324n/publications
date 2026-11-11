# Copyright (c) 2026 Ivan Liu.
# Licensed under CC BY 4.0.
"""Emotion-classification task engine: runs classification over feature tables or datasets
and collects per-run analyses.
"""

import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
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
import psychai.visualization.chart_helper
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelBinarizer


class EmotionClassificationTask:

    """Classify emotions by evaluating multiple feature engineering and model selection strategies across parameter combinations. Run analyses on input data using specified classifiers, feature selection methods (ANOVA, RFE), and optional hyperparameter grid search, accumulating evaluation results and predictions.
    """
    def __init__(self, chart_helper, feature_enginnering_helper, classifier):
        self.chart_helper = chart_helper
        self.feature_enginnering_helper = feature_enginnering_helper
        self.classifier = classifier
        pass

    def run_analysis(self, 
        df, 
        df_name, 
        param_combinations, 
        random_state = 1,
        remove_high_correlation = False, 
        verbose = False, 
        model_grid_search_parameters = None, 
        experiment_name = "",
        parameters= None):

        
        # Initialize a DataFrame to store evaluation results
        # This will hold the outcomes of each experiment for comparison
        """Execute feature selection and model evaluation across multiple parameter combinations, iterating through each configuration to set up RFE models, log task details, and accumulate evaluation results in DataFrames.
        """
        if model_grid_search_parameters is None:
            model_grid_search_parameters = {}
        if parameters is None:
            parameters = {}
        df_all_evaluations = pd.DataFrame()
        df_all_predictions = pd.DataFrame()
        # Iterate over each combination of parameters
        for n_kept, do_anova, do_rfe, do_model_grid_search, model_name, rfe_model_name, group_column in param_combinations:

            # Set up the RFE model based on the name
            # RFE (Recursive Feature Elimination) requires a model to rank features
            if rfe_model_name == "Random Forest":
                rfe_model = RandomForestClassifier(random_state=random_state)  # Random Forest as the RFE estimator
            elif rfe_model_name == "SVM":
                rfe_model = SVC(kernel="linear", random_state=random_state)  # SVM with linear kernel for RFE
            else:
                # Raise an error if the RFE model is not supported
                raise ValueError(f"Unsupported RFE model: {rfe_model_name}")

            # Log the current configuration for tracking
            print(f"=== Task: {df_name}, Model: {model_name}, RFE Model: {rfe_model_name}, "
                f"n_kept: {n_kept}, ANOVA: {do_anova}, RFE: {do_rfe}, Grid Search: {do_model_grid_search} ===")

            if model_grid_search_parameters == {}:
                model_grid_search_parameters= {
                    # Parameters to tune during Grid Search for the Random Forest model
                    'n_estimators': [50,100],  # Number of trees in the forest
                    'max_depth': [2, 3, 5, 10, None],  # Maximum depth of the tree
                    'min_samples_split': [2,3],  # Minimum samples required to split an internal node
                    'min_samples_leaf': [1,2],  # Minimum samples required to be a leaf node
                    'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
                }            
            # Define the parameters for the emotion classification task
            # This dictionary contains all the configuration needed for running the task
            if parameters == {}:
                parameters = {
                    "model_name": model_name,  # Name of the classification model
                    "group_column": group_column, # Name of the column to group rows by
                    "run_1": False,  # First experimental run
                    "run_2": False,  # Second experimental run
                    "run_3": True,  # Third experimental run
                    "verbose": verbose,  # Whether to print detailed logs during training
                    "visualization": True,  # Whether to create visual outputs (e.g., confusion matrices)
                    "do_anova": do_anova,  # Whether to apply ANOVA for feature selection
                    "n_cross_validation": 5,  # Number of folds for cross-validation
                    "percentage_kept_by_anova": 0.7,  # Proportion of features to retain after ANOVA
                    "do_rfe": do_rfe,  # Whether to apply Recursive Feature Elimination
                    "rfe_model": rfe_model,  # Model to be used for RFE
                    "n_kept": n_kept,  # Number of features to keep after RFE
                    "do_model_grid_search": do_model_grid_search,  # Whether to perform Grid Search
                    "random_state": random_state,  # Random seed for reproducibility
                    "model_grid_search_parameters": model_grid_search_parameters,
                    "experiment_name": experiment_name,
                    "remove_high_correlation": remove_high_correlation
                }
            else:
                parameters_input = parameters.copy()
                parameters_to_update = {
                    "model_name": model_name,  # Name of the classification model
                    "group_column": group_column, # Name of the column to group rows by
                    "run_1": False,  # First experimental run
                    "run_2": False,  # Second experimental run
                    "run_3": True,  # Third experimental run
                    "verbose": verbose,  # Whether to print detailed logs during training
                    "visualization": True,  # Whether to create visual outputs (e.g., confusion matrices)
                    "do_anova": do_anova,  # Whether to apply ANOVA for feature selection
                    "n_cross_validation": 5,  # Number of folds for cross-validation
                    "percentage_kept_by_anova": 0.7,  # Proportion of features to retain after ANOVA
                    "do_rfe": do_rfe,  # Whether to apply Recursive Feature Elimination
                    "rfe_model": rfe_model,  # Model to be used for RFE
                    "n_kept": n_kept,  # Number of features to keep after RFE
                    "do_model_grid_search": do_model_grid_search,  # Whether to perform Grid Search
                    "random_state": random_state,  # Random seed for reproducibility
                    "model_grid_search_parameters": model_grid_search_parameters,
                    "experiment_name": experiment_name,
                    "remove_high_correlation": remove_high_correlation
                }

                if parameters_input.get("run_1") is not None:
                    parameters_to_update["run_1"] = parameters_input["run_1"]

                if parameters_input.get("run_2") is not None:
                    parameters_to_update["run_2"] = parameters_input["run_2"]

                if parameters_input.get("run_3") is not None:
                    parameters_to_update["run_3"] = parameters_input["run_3"]

                if parameters_input.get("verbose") is not None:
                    parameters_to_update["verbose"] = parameters_input["verbose"]

                if parameters_input.get("n_cross_validation") is not None:
                    parameters_to_update["n_cross_validation"] = parameters_input["n_cross_validation"]

                parameters = parameters_to_update.copy()

            # Add special parameters for MLP (Multi-Layer Perceptron) if applicable
            if model_name == "MLP":
                parameters["model_initial_parameters"] = {
                    "hidden_layer_sizes": (128, 64)  # MLP hidden layer sizes, customizable
                }

            # Run the emotion classification task with the defined parameters
            # This function should handle the training, validation, and evaluation of the model
            (accuracy_1, accuracy_2, accuracy_3, all_y_pred, 
            all_y_true, all_group) = self.run_emotion_classification_tasks(df, parameters=parameters)

            # Prepare evaluation results for this iteration
            # Store the configuration and resulting accuracy in a DataFrame
            result_evaluations = pd.DataFrame({
                'df_name': [df_name],  # Name of the dataset
                'model_name': [model_name],  # Classification model used
                'do_rfe': [do_rfe],  # Model used for RFE
                'rfe_model_name': [rfe_model_name],  # Model used for RFE
                'n_kept': [n_kept],  # Number of features retained
                'do_model_grid_search': [do_model_grid_search],  # Whether Grid Search was applied
                'do_anova': [do_anova],  # Whether ANOVA was applied
                'Accuracy1': [accuracy_1],  # Save the third accuracy value as an example
                'Accuracy2': [accuracy_2],  # Save the third accuracy value as an example
                'Accuracy3': [accuracy_3]  # Save the third accuracy value as an example
            })

            result_prediction = pd.DataFrame({
                'df_name': [df_name]*len(all_y_pred),  # Name of the dataset
                'all_y_pred': all_y_pred,  # Classification model used
                'all_y_true': all_y_true,  # Model used for RFE
                'all_group': all_group,  # Number of features retained
            })

            # Append results to the cumulative DataFrame
            # Concatenate results from each experiment for comparison
            df_all_evaluations = pd.concat([df_all_evaluations, result_evaluations], ignore_index=True)
            df_all_predictions = pd.concat([df_all_predictions, result_prediction], ignore_index=True)     

        # Final evaluation DataFrame
        # This DataFrame contains the results of all experiments and can be exported or analyzed further
        return df_all_evaluations, df_all_predictions

    def classify(
        self,
        df_input,
        group_labels:dict,               # Dictionary for relabeling classes (e.g., {"1": "positive", "2": "negative"})
        model_name = "Random Forest",
        task_name = "",                  # Name of the classification task for display
        feature_columns = None,
        target_column = "Group",
        group_column = "user_id",
        ticklabel = None,
        n_cross_validation = 10,
        remove_high_correlation = False,
        corr_threshold = 0.95,
        random_state = 1,
        do_anova = False,
        percentage_kept_by_anova = 0.5,
        do_rfe = False,
        rfe_model = None,
        n_kept = 10,
        do_pca = False,
        explained_variance_threshold = 0.85,
        do_model_grid_search = False,
        model_grid_search_parameters = None,
        verbose = False,
        visualization = True,
        show_percentage = False,
        n_top_features = 10,  # Adjust `n` to your desired number of top features
        model_initial_parameters = None
    ):
        """
        A function to run a machine learning pipeline, including data preparation, feature selection,
        hyperparameter tuning, and evaluation metrics.
        Returns accuracy and F1 scores for each class.
        """
        if feature_columns is None:
            feature_columns = []
        if ticklabel is None:
            ticklabel = []
        if model_grid_search_parameters is None:
            model_grid_search_parameters = {}
        if model_initial_parameters is None:
            model_initial_parameters = {}
        fe_helper = psychai.feature.feature_engineering.feature_engineering_helper.FeatureEngineeringHelper()
        print(f"random_state:{random_state}")

        df = df_input.copy()
        df[target_column] = df[target_column].map(group_labels)


        # Prepare feature columns
        if len(feature_columns)==0:
            feature_columns = df.columns.tolist()

        columns_to_filter_out = [target_column,group_column,"Unnamed: 0","seq"]
        feature_columns = [col for col in feature_columns if col not in columns_to_filter_out]


        X = df[feature_columns].to_numpy()
        y = df[target_column].to_numpy()

        # Initialize cross-validator
        if group_column:
            cv = GroupKFold(n_splits=n_cross_validation)
            groups = df[group_column].to_numpy()
        else:
            cv = KFold(n_splits=n_cross_validation, shuffle=True, random_state=random_state)
            groups = None

        # Initialize Random Forest model

        # Store cross-validation scores
        cv_scores = []
        all_y_true = []
        all_y_pred = []
        all_y_test_binary = []
        all_y_prob = []
        all_groups = []

        # Initialize an empty list to store SHAP values and feature names across splits
        # Initialize an empty DataFrame to store all iterations
        df_all_feature_importance = pd.DataFrame()

        iteration_indx = 1
        for train_idx, val_idx in cv.split(X, y, groups):

            kept_columns = feature_columns
            # Split data into train and validation sets
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            if groups is not None:
                groups_train = groups[train_idx]
                groups_val = groups[val_idx]
            else:
                groups_train = None
                groups_val = None
            # ===================== Scale features ========================================================
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

            # ===================== Remove High Correlated Columns ========================================

            if remove_high_correlation:
                X_train, X_val, kept_columns = fe_helper.remove_high_correlation(X_train, X_val , kept_columns, corr_threshold = corr_threshold, verbose= verbose)

            # ===================== Keep only high Anova F-values =========================================
            if do_anova:
                n_features_kept_by_anova = int(len(kept_columns)*percentage_kept_by_anova)
                if verbose:
                    print(f"{n_features_kept_by_anova} columns out of {len(kept_columns)} columns selected by anova")
                X_train, X_val, kept_columns = fe_helper.keep_high_anova(X_train, X_val, y_train, kept_columns, n_features_kept_by_anova, verbose= verbose)

            # ===================== Perform Recursive Feature Elimination (RFE) ===========================
            if do_rfe:
                X_train, X_val, kept_columns = fe_helper.RFE(X_train, X_val,  y_train, random_state, kept_columns, n_kept, verbose= verbose, rfe_model = rfe_model)

            #====================== PCA ===================================================================
            if do_pca:
                X_train, X_val = fe_helper.pca(X_train, X_val ,explained_variance_threshold, verbose= verbose)
                
            # Train the Random Forest model
            # Shuffle training data
            if groups is not None:
                X_train, y_train, groups_train = shuffle(X_train, y_train, groups_train, random_state=random_state)
            else:
                X_train, y_train = shuffle(X_train, y_train, random_state=random_state)

            #X_train, y_train, groups_train = shuffle(X_train, y_train, groups_train, random_state=random_state)

            self.classifier.init(
                X_train = X_train , y_train = y_train, X_test = X_val, y_test = y_val, model_name = model_name,
                do_model_grid_search = do_model_grid_search, grid_search_parameters = model_grid_search_parameters,
                group_info = groups_train, model_initial_parameters = model_initial_parameters
            )
            self.classifier.fit()

            # Evaluate the model on the validation set
            y_pred = self.classifier.predict()
            
            score = accuracy_score(y_val, y_pred)
            cv_scores.append(score)

            if model_name == "Random Forest" and not do_pca:
                trained_model = self.classifier.get_trained_model() 
                feature_importances = trained_model.feature_importances_  # Get feature importances
                df_feature_importance = pd.DataFrame([feature_importances], columns=kept_columns)
                df_all_feature_importance = pd.concat([df_all_feature_importance, df_feature_importance], axis=0, ignore_index=True)

            

            if verbose:
                print(f"The {iteration_indx}-th run fold score: {score}")
                iteration_indx = iteration_indx+1

            # Store results for evaluation after all folds
            all_y_true.extend(y_val)
            all_y_pred.extend(y_pred)
            if groups is not None:
                all_groups.extend(groups_val)

            if len(np.unique(y_val)) == 2:
                lb = LabelBinarizer()
                y_test_binary = lb.fit_transform(y_val).ravel()  # Binarize the labels (0 and 1)
                # Calculate the predicted probabilities and AUC score
                y_prob = self.classifier.predict_proba()[:, 1]
                all_y_test_binary.extend(y_test_binary)
                all_y_prob.extend(y_prob)



        if verbose:
            print(f"Cross-Validation Scores: {cv_scores}")
            print(f"Mean Accuracy: {np.mean(cv_scores)}")


        # Overall metrics
        accuracy = accuracy_score(all_y_true, all_y_pred)
        recall = recall_score(all_y_true, all_y_pred, average='weighted')
        f1 = f1_score(all_y_true, all_y_pred, average='weighted')




        # Print metrics
        print(f"Total Accuracy: {accuracy}")
        print(f"Total Recall: {recall}")
        print(f"Total F1 Score: {f1}")

        # Call the function with an option for percentage
        if visualization:
            title = f"{task_name} (Acc:{accuracy:.2f}, F1:{f1:.2f})"
            if self.chart_helper:
                if len(ticklabel)>0:
                    cm = confusion_matrix(all_y_true, all_y_pred,  labels=ticklabel)
                    self.chart_helper.plot_confusion_matrix(cm, ticklabels=ticklabel, show_percentage=show_percentage, task_name=task_name, title = title)
                else:
                    cm = confusion_matrix(all_y_true, all_y_pred)
                    self.chart_helper.plot_confusion_matrix(cm, ticklabels=np.unique(all_y_true), show_percentage=show_percentage, task_name=task_name, title = title)

            
        if visualization:
            if len(np.unique(y_val)) == 2:
                auc = roc_auc_score(all_y_test_binary, all_y_prob)
                print(f"ROC-AUC for {task_name}: {auc:.4f}")

                # Plot ROC Curve
                fpr, tpr, thresholds = roc_curve(all_y_test_binary, all_y_prob)
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve for {task_name}')
                plt.legend(loc="lower right")
                plt.show()


        if model_name == "Random Forest" and not do_pca:
            # Fill NaN values with 0 (if missing features are assumed unimportant)
            df_all_feature_importance = df_all_feature_importance.fillna(0)

            # Calculate the mean value of each feature across iterations
            mean_feature_importances = df_all_feature_importance.mean()

            # Select the top n features
            
            top_features = mean_feature_importances.nlargest(n_top_features)

            top_features = top_features.sort_values()

            # Print the column names as a list
            print(f"Top {n_top_features} Features by Mean Importance:{top_features.index.tolist()}")

            if visualization:
                # Plot the top n features as a horizontal bar chart
                plt.figure(figsize=(10, 6))
                top_features.plot(kind='barh', color='skyblue')  # Sort for better visualization
                plt.xlabel("Mean Feature Importance")
                plt.title(f"Top {n_top_features} Features by Mean Importance")
                plt.tight_layout()
                plt.show()







        feature_importance = []
        return accuracy, f1, feature_importance, all_y_pred, all_y_true, all_groups

    def run_emotion_classification_tasks_Test(self, df_merged, 
                                         correlation_threshold = 0.9, 
                                         remove_redundant_features = True,
                                         column_to_split = "",
                                         cv = 3,
                                         random_state = 1,
                                         model_type = "Random Forest", # "Random Forest"
                                         grid_search_parameters = None,
                                         model_parameters = None,
                                         n_top_features= 5,
                                         outlier_strategies=None):

        # We assume that 'Group' is the label column and all other columns from the third onward are feature columns.
        """Execute emotion classification tasks on a merged dataframe using the specified model type and hyperparameters, with optional feature redundancy removal, outlier handling, and cross-validation-based model selection.
        """
        if grid_search_parameters is None:
            grid_search_parameters = {}
        if model_parameters is None:
            model_parameters = {}
        if outlier_strategies is None:
            outlier_strategies = []
        x_columns = df_merged.columns[2:]  # Columns from the 3rd onward are features
        y_column = "Group"  # Target label column

        # The parameter grid is used to test different configurations for optimal performance.
        if grid_search_parameters == {}:
            if model_type =="XGBoost":
                grid_search_parameters = {
                    # Number of trees (boosting rounds)
                    'n_estimators': [50, 100, 200],  # Common choices for boosting rounds

                    # Maximum depth of each tree, controlling complexity
                    'max_depth': [3, 6, 10],  # Typical values are 3, 6, 10

                    # Learning rate to control the contribution of each tree
                    'learning_rate': [0.01, 0.1, 0.2],  # Common values to try, slower rates can improve accuracy

                    # Fraction of columns (features) to be used by each tree
                    'colsample_bytree': [0.6, 0.8, 1.0],  # Reduces overfitting by using a subset of features

                    # Fraction of rows to be used by each tree
                    'subsample': [0.8, 1.0],  # Usually set to values between 0.5 and 1.0 to prevent overfitting

                    # Minimum loss reduction required to make a further partition
                    'gamma': [0, 0.1, 0.2],  # Controls tree splitting; use 0 or small values initially

                    # Minimum child weight, a regularization to control tree depth
                    'min_child_weight': [1, 5, 10],  # Higher values prevent deeper splits, reducing overfitting

                    # L2 regularization term on weights (ridge regression)
                    'lambda': [1, 10],  # Helps reduce overfitting

                    # Seed for reproducibility
                    'random_state': [1],  # Standard value for reproducibility
                }

            else:
                grid_search_parameters = {
                    'bootstrap': [True],                # Whether to use bootstrap samples
                    'ccp_alpha': [0.0],                 # Complexity parameter used for minimal cost-complexity pruning
                    'class_weight': ['balanced'],       # Adjust weights to handle class imbalance
                    'criterion': ['gini'],              # Splitting criterion
                    'max_depth': [None],        # Limits tree depth to avoid overfitting
                    'max_features': ['sqrt'],           # Number of features considered at each split
                    'max_leaf_nodes': [None],           # Maximum leaf nodes
                    'max_samples': [None],              # If bootstrap is true, it limits samples
                    'min_impurity_decrease': [0.0],     # Threshold for node impurity
                    'min_samples_leaf': [1],         # Minimum samples at each leaf node
                    'min_samples_split': [2],        # Minimum samples required to split a node
                    'min_weight_fraction_leaf': [0.0],  # Minimum fraction of samples at each leaf node
                    'n_estimators': [100],    # Number of trees in the forest
                    'n_jobs': [None],                   # Number of jobs to run in parallel
                    'oob_score': [False],               # Use out-of-bag samples for validation
                    'verbose': [0],                     # Controls verbosity of output
                    'warm_start': [False],              # Reuse previous solution for additional trees
                    'random_state': [random_state]      # Seed for reproducibility
                }
                # Task 3: 2x2 Analysis with Original Group Labels
        machine_learning = MachineLearning(df_merged.copy())
        group_labels_task3 = {
            1: 0,
            2: 1,
            3: 2,
            4: 3
        }
        accuracy, f1_scores, feature_importance = machine_learning.run_machine_learning(
            group_labels_task3,
            "Multi-Class Classification",
            x_columns=x_columns,
            y_column=y_column,
            cv = cv,
            remove_redundant_features=remove_redundant_features,
            grid_search_parameters=grid_search_parameters,
            correlation_threshold = correlation_threshold,
            column_to_split = column_to_split,
            model_type = model_type,
            use_pca= False,
            n_top_features = n_top_features,
            outlier_strategies=outlier_strategies
        )
        return accuracy, f1_scores, feature_importance 


    def extend_list(self, input_list, requested_length):
        # If the input list is shorter than the requested length, add empty strings
        """Extend the input list with empty strings until it reaches the requested length, then return it.
        """
        if len(input_list) < requested_length:
            input_list.extend([""] * (requested_length - len(input_list)))
        return input_list
    

    def run_emotion_classification_datasets(self, df_list, 
                                            correlation_threshold = 0.7, 
                                            n_top_features = 5,
                                            k =1.5,
                                            random_state = 1,
                                            cv = 10,
                                            outlier_strategies = ['Replace with Median'],
                                            column_to_split = "",
                                            use_pca = False):
        """Execute emotion classification tasks across multiple datasets and aggregate evaluation metrics for each, adjusting the correlation threshold for datasets with fewer than 20 columns, then return DataFrames containing evaluation scores and feature importance rankings from three classification models.
        """
        df_evaluations = pd.DataFrame()
        df_features = pd.DataFrame()
        for df in df_list:
            if (len(df.columns)<20):
                correlation_threshold = 1

            accuracy_1, f1_scores_dict_1, feature_importance_1, accuracy_2, f1_scores_dict_2, feature_importance_2, accuracy_3, f1_scores_dict_3, feature_importance_3 = self.run_emotion_classification_tasks(
                df, 
                n_top_features= n_top_features,
                correlation_threshold = correlation_threshold,
                outlier_strategies = outlier_strategies,
                random_state = random_state,
                cv = cv,
                k = k,
                column_to_split= column_to_split,
                use_pca = use_pca)
            
            result_evaluation = [
                f"{accuracy_1:.2f}",
                *[f"{score:.2f}" for score in f1_scores_dict_1.values()],
                f"{accuracy_2:.2f}",
                *[f"{score:.2f}" for score in f1_scores_dict_2.values()],
                f"{accuracy_3:.2f}",
                *[f"{score:.2f}" for score in f1_scores_dict_3.values()],
            ]
            df_evaluation_row = pd.DataFrame([result_evaluation])
            df_evaluations = pd.concat([df_evaluations, df_evaluation_row], ignore_index=True)
            feature_importance_1.reverse()
            feature_importance_2.reverse()
            feature_importance_3.reverse()
            result_feature =  self.extend_list(feature_importance_1,n_top_features) + self.extend_list(feature_importance_2,n_top_features) + self.extend_list(feature_importance_3,n_top_features)
            df_feature_row = pd.DataFrame([result_feature])  # Make it a row (2D array)
            df_features = pd.concat([df_features, df_feature_row], ignore_index=True)
        return df_evaluations, df_features


    def run_emotion_classification_tasks(self, 
                                         df_prepared, 
                                         parameters = None  
                                        ):
        
        """Run emotion classification tasks on prepared data with configurable model selection, feature engineering, and cross-validation parameters, returning predictions and evaluation metrics.
        """
        if parameters is None:
            parameters = {}
        model_name = "Random Forest"
        random_state = 42
        n_cross_validation = 5
        remove_high_correlation = False
        corr_threshold = 0.95
        do_anova = True
        percentage_kept_by_anova = 0.7
        do_rfe = True
        rfe_model = None
        n_kept = 6
        do_pca = False
        explained_variance_threshold = 0.7
        verbose = True
        visualization = True
        do_model_grid_search = False
        model_grid_search_parameters = {}
        run_1 = False
        run_2 = False
        run_3 = False
        model_initial_parameters = {}

        all_y_true_list = []
        all_y_pred_list = []
        all_group_list = []

        if parameters.get("model_name") is not None:
            model_name = parameters.get("model_name")
        if parameters.get("random_state") is not None:
            random_state = parameters.get("random_state")
        if parameters.get("group_column") is not None:
            group_column = parameters.get("group_column")
        if parameters.get("n_cross_validation") is not None:
            n_cross_validation = parameters.get("n_cross_validation")
        if parameters.get("remove_high_correlation") is not None:
            remove_high_correlation = parameters.get("remove_high_correlation")
        if parameters.get("corr_threshold") is not None:
            corr_threshold = parameters.get("corr_threshold")
        if parameters.get("do_anova") is not None:
            do_anova = parameters.get("do_anova")
        if parameters.get("percentage_kept_by_anova") is not None:
            percentage_kept_by_anova = parameters.get("percentage_kept_by_anova")
        if parameters.get("do_rfe") is not None:
            do_rfe = parameters.get("do_rfe")
        if parameters.get("rfe_model") is not None:
            rfe_model = parameters.get("rfe_model")
        if parameters.get("n_kept") is not None:
            n_kept = parameters.get("n_kept")
        if parameters.get("do_pca") is not None:
            do_pca = parameters.get("do_pca")
        if parameters.get("explained_variance_threshold") is not None:
            explained_variance_threshold = parameters.get("explained_variance_threshold")
        if parameters.get("verbose") is not None:
            verbose = parameters.get("verbose")
        if parameters.get("visualization") is not None:
            visualization = parameters.get("visualization")
        if parameters.get("do_model_grid_search") is not None:
            do_model_grid_search = parameters.get("do_model_grid_search")
        if parameters.get("model_grid_search_parameters") is not None:
            model_grid_search_parameters = parameters.get("model_grid_search_parameters")
        if parameters.get("run_1") is not None:
            run_1 = parameters.get("run_1")
        if parameters.get("run_2") is not None:
            run_2 = parameters.get("run_2")
        if parameters.get("run_3") is not None:
            run_3 = parameters.get("run_3")
        if parameters.get("model_initial_parameters") is not None:
            model_initial_parameters = parameters.get("model_initial_parameters")
        if parameters.get("experiment_name") is not None:
            experiment_name = parameters.get("experiment_name")

        chart_helper = psychai.visualization.chart_helper.ChartHelper()
        accuracy_1 = 0
        accuracy_2 = 0
        accuracy_3 = 0
        if run_1:
            group_labels_task1 = {1: "CMPS", 2: "CMPS", 3: "N-CMPS", 4: "N-CMPS"}
            accuracy_1, f1_scores_dict_1, feature_importance_1, all_y_pred, all_y_true, all_groups =  self.classify(
                df_input= df_prepared,
                target_column = "Group",
                group_column = group_column,
                group_labels = group_labels_task1,
                n_cross_validation = n_cross_validation,
                remove_high_correlation = remove_high_correlation,
                corr_threshold = corr_threshold,
                random_state = random_state,
                do_anova = do_anova,
                percentage_kept_by_anova =percentage_kept_by_anova,
                do_rfe = do_rfe,
                n_kept = n_kept,
                do_pca = do_pca,
                explained_variance_threshold = explained_variance_threshold,
                verbose = verbose,
                do_model_grid_search = do_model_grid_search,
                model_grid_search_parameters = model_grid_search_parameters,
                model_initial_parameters = model_initial_parameters,
                model_name= model_name,
                rfe_model = rfe_model,
                visualization = visualization,
                task_name="Task1:Compassion"
            )
            all_y_true_list.extend(all_y_true)
            all_y_pred_list.extend(all_y_pred)
            all_group_list.extend(all_groups)
        if run_2:
            group_labels_task2 = {1: "INSP", 3: "INSP", 2: "U-INSP", 4: "U-INSP"}
            accuracy_2, f1_scores_dict_1, feature_importance_1, all_y_pred, all_y_true, all_groups = self.classify(
                df_input= df_prepared,
                target_column = "Group",
                group_column = group_column,
                group_labels = group_labels_task2,
                n_cross_validation = n_cross_validation,
                remove_high_correlation = remove_high_correlation,
                corr_threshold = corr_threshold,
                random_state = random_state,
                do_anova = do_anova,
                percentage_kept_by_anova =percentage_kept_by_anova,
                do_rfe = do_rfe,
                n_kept = n_kept,
                do_pca = do_pca,
                explained_variance_threshold = explained_variance_threshold,
                verbose = verbose,
                do_model_grid_search = do_model_grid_search,
                model_grid_search_parameters = model_grid_search_parameters,
                model_initial_parameters = model_initial_parameters,
                model_name= model_name,
                rfe_model = rfe_model,
                visualization = visualization,
                task_name="Task2:Inspiration"
            )
            all_y_true_list.extend(all_y_true)
            all_y_pred_list.extend(all_y_pred)
            all_group_list.extend(all_groups)
        if run_3:
            
            group_labels_task3 = {1: "ELVT",2: "SYMP",3: "INSP",4: "CTRL"}
            ticklabel= ["ELVT","SYMP","INSP","CTRL"]

            if model_name in ["XGBoost"]:
                group_labels_task3 = {1:0,2:1,3:2,4:3}
                ticklabel = [0,1,2,3]
            accuracy_3, f1_scores_dict_1, feature_importance_1, all_y_pred, all_y_true, all_groups = self.classify(
                df_input= df_prepared,
                target_column = "Group",
                group_column = group_column,
                group_labels = group_labels_task3,
                n_cross_validation = n_cross_validation,
                remove_high_correlation = remove_high_correlation,
                corr_threshold = corr_threshold,
                random_state = random_state,
                do_anova = do_anova,
                percentage_kept_by_anova =percentage_kept_by_anova,
                do_rfe = do_rfe,
                n_kept = n_kept,
                do_pca = do_pca,
                explained_variance_threshold = explained_variance_threshold,
                verbose = verbose,
                do_model_grid_search = do_model_grid_search,
                model_grid_search_parameters = model_grid_search_parameters,
                ticklabel= ticklabel,
                model_initial_parameters = model_initial_parameters,
                model_name= model_name,
                rfe_model = rfe_model,
                visualization = visualization,
                task_name=experiment_name #"Task3:Moral Elevation"
            )
            all_y_true_list.extend(all_y_true)
            all_y_pred_list.extend(all_y_pred)
            all_group_list.extend(all_groups)
            return accuracy_1, accuracy_2, accuracy_3, all_y_pred_list,  all_y_true_list, all_group_list
    

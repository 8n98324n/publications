# Copyright (c) 2026 Ivan Liu.
# Licensed under CC BY 4.0.
"""Analysis helper that merges prediction tables and reports accuracy."""
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class DataAnalysisHelper:

    """Class providing utilities for merging tables and computing aggregate statistics."""
    def __init__(self):
        pass























    # 1. Merge two tables and calculate average values



    #         df_merged[["sympathy_23","sympathy_25","inspiration_23","inspiration_25"]].isna().any(axis=1)










































    # 0. Group name






































    #         df["sympathy_avg"] += weights[i] * df[f"sympathy_{i}"]
    #         df["inspiration_avg"] += weights[i] * df[f"inspiration_{i}"]


























    def merge_and_show_accuracy(
        self,
        df_list,
        chart_helper,
        title="",
        threshold_type="median",
        weights=None,
        font_size = 12
    ):

        """Merge multiple dataframes with weighted averaging, compute classification metrics (accuracy, F1, confusion matrix) across groups, and display results via a chart helper with configurable threshold and formatting.
        """
        import pandas as pd
        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

        # 0. basic checks
        if not isinstance(df_list, list) or len(df_list) == 0:
            raise ValueError("df_list must be a non-empty list of dataframes")

        n = len(df_list)

        if weights is None:
            weights = [1 / n] * n

        if len(weights) != n:
            raise ValueError("weights length must match number of dataframes")

        weights = np.array(weights, dtype=float)
        weights = weights / weights.sum()

        # 1. group names
        group_names = {1: "ELVT", 2: "SYMP", 3: "INSP", 4: "CTRL"}
        group_labels = [group_names[i] for i in [1, 2, 3, 4]]

        # 2. rename columns to avoid collision
        renamed_dfs = []

        for i, df in enumerate(df_list):

            df_tmp = df.copy()

            df_tmp = df_tmp.rename(
                columns={
                    "sympathy": f"sympathy_{i}",
                    "inspiration": f"inspiration_{i}",
                    "Group": f"Group_{i}"
                }
            )

            renamed_dfs.append(df_tmp)

        # 3. merge all dfs
        df = renamed_dfs[0]

        for df_next in renamed_dfs[1:]:
            df = pd.merge(df, df_next, on="Participant SN", how="inner")

        # 4. convert numeric
        sym_cols = [f"sympathy_{i}" for i in range(n)]
        insp_cols = [f"inspiration_{i}" for i in range(n)]

        df[sym_cols] = df[sym_cols].apply(pd.to_numeric, errors="coerce")
        df[insp_cols] = df[insp_cols].apply(pd.to_numeric, errors="coerce")

        # 5. NaN-safe weighted average
        sym_values = df[sym_cols].values
        insp_values = df[insp_cols].values
        weights_arr = weights

        mask_sym = ~np.isnan(sym_values)
        mask_insp = ~np.isnan(insp_values)

        sym_weight_sum = np.sum(mask_sym * weights_arr, axis=1)
        insp_weight_sum = np.sum(mask_insp * weights_arr, axis=1)

        sym_weight_sum[sym_weight_sum == 0] = np.nan
        insp_weight_sum[insp_weight_sum == 0] = np.nan

        df["sympathy_avg"] = np.nansum(sym_values * weights_arr, axis=1) / sym_weight_sum
        df["inspiration_avg"] = np.nansum(insp_values * weights_arr, axis=1) / insp_weight_sum

        # 6. ground truth
        df["GT_Group"] = pd.to_numeric(df["Group_0"], errors="coerce")

        # 7. compute thresholds from ALL input dfs
        sympathy_all = []
        inspiration_all = []

        for src_df in df_list:

            sympathy_all.append(
                pd.to_numeric(src_df["sympathy"], errors="coerce")
            )

            inspiration_all.append(
                pd.to_numeric(src_df["inspiration"], errors="coerce")
            )

        sympathy_all = pd.concat(sympathy_all).dropna()
        inspiration_all = pd.concat(inspiration_all).dropna()

        def truncated_mean(series, lower_q=0.25, upper_q=0.75):
            """
            Compute truncated mean using only the central quantile range.
            Default keeps the central 50% (25%–75%).
            """
            lower = series.quantile(lower_q)
            upper = series.quantile(upper_q)
            trimmed = series[(series >= lower) & (series <= upper)]
            return trimmed.mean()

        if threshold_type == "median":

            sympathy_threshold = sympathy_all.median()
            inspiration_threshold = inspiration_all.median()

        elif threshold_type == "mean":

            sympathy_threshold = sympathy_all.mean()
            inspiration_threshold = inspiration_all.mean()

        elif threshold_type == "truncated_mean_80":

            sympathy_threshold = truncated_mean(sympathy_all, lower_q=0.1, upper_q=0.9)
            inspiration_threshold = truncated_mean(inspiration_all, lower_q=0.1, upper_q=0.9)

        elif threshold_type == "truncated_mean_50":

            sympathy_threshold = truncated_mean(sympathy_all, lower_q=0.25, upper_q=0.75)
            inspiration_threshold = truncated_mean(inspiration_all, lower_q=0.25, upper_q=0.75)

        elif threshold_type == "q1":

            sympathy_threshold = sympathy_all.quantile(0.25)
            inspiration_threshold = inspiration_all.quantile(0.25)

        elif threshold_type == "q3":

            sympathy_threshold = sympathy_all.quantile(0.75)
            inspiration_threshold = inspiration_all.quantile(0.75)

        else:

            sympathy_threshold = sympathy_all.mean()
            inspiration_threshold = inspiration_all.mean()

        print("Sympathy threshold:", sympathy_threshold)
        print("Inspiration threshold:", inspiration_threshold)

        # 8. classification
        def classify(sym, insp):

            if pd.isna(sym) or pd.isna(insp):
                return np.nan

            if sym > sympathy_threshold and insp > inspiration_threshold:
                return 1
            elif sym > sympathy_threshold and insp <= inspiration_threshold:
                return 2
            elif sym <= sympathy_threshold and insp > inspiration_threshold:
                return 3
            else:
                return 4

        df["Pred_Group"] = df.apply(
            lambda r: classify(r["sympathy_avg"], r["inspiration_avg"]),
            axis=1
        )

        # 9. invalid rows
        invalid_rows = df[df[["GT_Group", "Pred_Group"]].isna().any(axis=1)]

        if not invalid_rows.empty:

            print("\nRows skipped due to NaN values:")

            display(
                invalid_rows[
                    [
                        "Participant SN",
                        "sympathy_avg",
                        "inspiration_avg",
                        "GT_Group",
                        "Pred_Group"
                    ]
                ]
            )

        # 10. evaluation
        df_eval = df.dropna(subset=["GT_Group", "Pred_Group"])

        y_true = df_eval["GT_Group"].astype(int)
        y_pred = df_eval["Pred_Group"].astype(int)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")

        print("Accuracy:", acc)
        print("Macro F1:", f1)

        # 11. confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4])

        # Normalize each column (avoid division by zero)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_normalized = np.divide(cm, row_sums, where=row_sums != 0)

        # Get diagonal (accuracy for each class)
        per_class_accuracy = np.diag(cm_normalized)


        # Corresponding label
        values = []
        for i, acc_i in enumerate(per_class_accuracy):
            print(f"Class {group_labels[i]} accuracy: {acc_i:.2%}")
            values.append(f"{acc_i:.4f}")  # Or use .2% depending on your needs

        # Excel one column (tab-separated)
        excel_row = "\t".join(values)
        print("\nExcel (row):")
        print(excel_row)

        chart_helper.plot_confusion_matrix(
            cm,
            ticklabels=group_labels,
            show_percentage=True,
            task_name="",
            title=f"{title} (Acc:{acc:.2f}, F1:{f1:.2f})",
            xticklabels=group_labels,
            yticklabels=group_labels,
            font_size = font_size
        )

        return df
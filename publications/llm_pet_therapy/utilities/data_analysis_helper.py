# Copyright (c) 2026 Ivan Liu.
# Licensed under CC BY 4.0.
"""Analysis helper for Pet Therapy Multimodal LLM Analysis (Study 2):
- Evaluates the two psychometric subscales of the Spielberger State-Trait Anxiety Inventory (STAI-S):
    * STAI-AA (Anxiety-Absent / Absence of Calmness & Security Subscale, 10 items, range 10-40)
    * STAI-AP (Anxiety-Present / Presence of Acute Tension Subscale, 10 items, range 10-40)
    * STAI-Total (STAI-AA + STAI-AP, 20 items, range 20-80)
- Computes regression metrics (MAE, RMSE, R^2, Pearson r, Spearman rho) separately for:
    * Pre-Test predictions (experiment_step="pre")
    * Post-Test predictions (experiment_step="post")
    * Subtracted Pre-minus-Post Improvement predictions (Improvement = Pre - Post)
- Evaluates Reliable Change Index (RCI) binary classification (RCI_STAI >= 8, RCI_AA >= 5) from Study 1,
  as well as two-component threshold-based 4-Condition (COMB, PET, FEED, OBS) and binary factor classification.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from scipy.stats import pearsonr, spearmanr


class DataAnalysisHelper:
    """Utilities for merging tables and computing STAI two-component regression and classification statistics."""

    def __init__(self):
        pass

    def merge_and_show_accuracy(
        self,
        df_list,
        chart_helper,
        title="",
        threshold_type="mean",
        weights=None,
        font_size=16,
        show_stai_regression=True,
    ):
        """
        Merge one or more experiment result DataFrames (e.g., subtracted Pre-Post df_diff,
        or single-step df_pre / df_post) with weighted averaging, and evaluate:
          1. Psychometric STAI Two-Component & Total Regression (Pre, Post, and Pre - Post Improvement)
          2. Study 1 Reliable Change Index (RCI_STAI >= 8, RCI_AA >= 5) Classification
          3. Two-Component (STAI-AA, STAI-AP) Threshold-Based 4-Condition & Binary Factor Classification
        """
        if not isinstance(df_list, list) or len(df_list) == 0:
            raise ValueError("df_list must be a non-empty list of dataframes")

        n = len(df_list)
        if weights is None:
            weights = [1.0 / n] * n
        weights = np.array(weights, dtype=float)
        weights = weights / weights.sum()

        group_names = {1: "COMB", 2: "PET", 3: "FEED", 4: "OBS"}
        group_labels = [group_names[i] for i in [1, 2, 3, 4]]

        pred_base_cols = [
            "stai_aa",
            "stai_ap",
            "stai_total",
            "stai_aa_pre_pred",
            "stai_ap_pre_pred",
            "stai_total_pre_pred",
            "stai_aa_post_pred",
            "stai_ap_post_pred",
            "stai_total_post_pred",
            "stai_aa_imp_pred",
            "stai_ap_imp_pred",
            "stai_total_imp_pred",
        ]

        renamed_dfs = []
        for i, d in enumerate(df_list):
            df_tmp = d.copy()
            rename_map = {"Group": f"Group_{i}"}
            for col in pred_base_cols:
                if col in df_tmp.columns:
                    rename_map[col] = f"{col}_{i}"
            df_tmp = df_tmp.rename(columns=rename_map)
            renamed_dfs.append(df_tmp)

        df = renamed_dfs[0]
        for df_next in renamed_dfs[1:]:
            cols_next = ["Participant SN"] + [
                c
                for c in df_next.columns
                if c.endswith(tuple(f"_{i}" for i in range(1, n)))
            ]
            df = pd.merge(df, df_next[cols_next], on="Participant SN", how="inner")

        def _weighted_avg(base_col):
            cols = [f"{base_col}_{i}" for i in range(n) if f"{base_col}_{i}" in df.columns]
            if not cols:
                return np.full(len(df), np.nan)
            df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
            vals = df[cols].values
            mask = ~np.isnan(vals)
            w_sum = np.sum(mask * weights[: len(cols)], axis=1)
            w_sum[w_sum == 0] = np.nan
            return np.nansum(vals * weights[: len(cols)], axis=1) / w_sum

        for base_col in pred_base_cols:
            if any(f"{base_col}_{i}" in df.columns for i in range(n)):
                df[f"{base_col}_avg"] = _weighted_avg(base_col)

        df["GT_Group"] = pd.to_numeric(df["Group_0"], errors="coerce")

        # Compute thresholds across all input dfs on the two STAI components (stai_aa, stai_ap)
        aa_all = pd.concat(
            [pd.to_numeric(src["stai_aa"], errors="coerce") for src in df_list if "stai_aa" in src.columns]
        ).dropna()
        ap_all = pd.concat(
            [pd.to_numeric(src["stai_ap"], errors="coerce") for src in df_list if "stai_ap" in src.columns]
        ).dropna()

        def truncated_mean(series, lower_q=0.25, upper_q=0.75):
            lower = series.quantile(lower_q)
            upper = series.quantile(upper_q)
            trimmed = series[(series >= lower) & (series <= upper)]
            return trimmed.mean()

        if len(aa_all) == 0 or len(ap_all) == 0:
            print(f"=== {title} Evaluation ===")
            print("[WARNING] No valid (stai_aa, stai_ap) predictions found yet.")
            return df

        if threshold_type == "median":
            aa_th = aa_all.median()
            ap_th = ap_all.median()
        elif threshold_type == "mean":
            aa_th = aa_all.mean()
            ap_th = ap_all.mean()
        elif threshold_type == "truncated_mean_80":
            aa_th = truncated_mean(aa_all, 0.1, 0.9)
            ap_th = truncated_mean(ap_all, 0.1, 0.9)
        elif threshold_type == "truncated_mean_50":
            aa_th = truncated_mean(aa_all, 0.25, 0.75)
            ap_th = truncated_mean(ap_all, 0.25, 0.75)
        else:
            aa_th = aa_all.mean()
            ap_th = ap_all.mean()

        print(f"=== {title} Evaluation ===")
        print(f"STAI-AA component threshold ({threshold_type}): {aa_th:.3f}")
        print(f"STAI-AP component threshold ({threshold_type}): {ap_th:.3f}")

        # 1. Evaluate STAI Two-Component & Total Regression (Pre, Post, and Pre - Post Improvement)
        if show_stai_regression:
            self.evaluate_stai_predictions(df, title=title)

        # 2. Two-Component (STAI-AA, STAI-AP) Threshold Classification
        def classify(aa_val, ap_val):
            if pd.isna(aa_val) or pd.isna(ap_val):
                return np.nan
            if aa_val > aa_th and ap_val > ap_th:
                return 1  # COMB (high improvement on both STAI-AA and STAI-AP)
            elif aa_val > aa_th and ap_val <= ap_th:
                return 2  # PET  (selective improvement on STAI-AA calmness/security)
            elif aa_val <= aa_th and ap_val > ap_th:
                return 3  # FEED (predominant change on STAI-AP tension)
            else:
                return 4  # OBS  (low improvement on both STAI-AA and STAI-AP)

        df["Pred_Group"] = df.apply(
            lambda r: classify(r["stai_aa_avg"], r["stai_ap_avg"]), axis=1
        )
        df["Pred_Petting"] = np.where(
            df["stai_aa_avg"].isna(), np.nan, (df["stai_aa_avg"] > aa_th).astype(int)
        )
        df["Pred_Feeding"] = np.where(
            df["stai_ap_avg"].isna(), np.nan, (df["stai_ap_avg"] > ap_th).astype(int)
        )

        invalid_rows = df[df[["GT_Group", "Pred_Group"]].isna().any(axis=1)]
        if not invalid_rows.empty:
            print(f"\nRows skipped due to NaN values: {len(invalid_rows)}")

        df_eval = df.dropna(subset=["GT_Group", "Pred_Group"]).copy()
        if df_eval.empty:
            print("[WARNING] No valid predictions available yet to evaluate.")
            return df

        y_true = df_eval["GT_Group"].astype(int)
        y_pred = df_eval["Pred_Group"].astype(int)

        acc = accuracy_score(y_true, y_pred)
        bacc = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")

        print(f"\n[Two-Component Threshold Classification (n={len(df_eval)})]:")
        print(f"  4-Condition Accuracy          : {acc:.4f} ({acc:.2%})")
        print(f"  4-Condition Balanced Accuracy : {bacc:.4f} ({bacc:.2%})")
        print(f"  4-Condition Macro F1          : {f1:.4f}")

        if "Petting" in df_eval.columns:
            yt_pet = df_eval["Petting"].astype(int)
            yp_pet = df_eval["Pred_Petting"].astype(int)
            acc_pet = accuracy_score(yt_pet, yp_pet)
            bacc_pet = balanced_accuracy_score(yt_pet, yp_pet)
            f1_pet = f1_score(yt_pet, yp_pet, average="macro")
            try:
                auc_pet = roc_auc_score(yt_pet, df_eval["stai_aa_avg"])
            except Exception:
                auc_pet = np.nan
            print(
                f"  Petting Factor (via STAI-AA)  : Acc={acc_pet:.2%}, BalAcc={bacc_pet:.2%}, F1={f1_pet:.3f}, AUC={auc_pet:.3f}"
            )

        if "Feeding" in df_eval.columns:
            yt_feed = df_eval["Feeding"].astype(int)
            yp_feed = df_eval["Pred_Feeding"].astype(int)
            acc_feed = accuracy_score(yt_feed, yp_feed)
            bacc_feed = balanced_accuracy_score(yt_feed, yp_feed)
            f1_feed = f1_score(yt_feed, yp_feed, average="macro")
            try:
                auc_feed = roc_auc_score(yt_feed, df_eval["stai_ap_avg"])
            except Exception:
                auc_feed = np.nan
            print(
                f"  Feeding Factor (via STAI-AP)  : Acc={acc_feed:.2%}, BalAcc={bacc_feed:.2%}, F1={f1_feed:.3f}, AUC={auc_feed:.3f}"
            )

        if "Petting" in df_eval.columns and "Feeding" in df_eval.columns:
            yt_dir = ((df_eval["Petting"].astype(int) == 1) | (df_eval["Feeding"].astype(int) == 1)).astype(int)
            yp_dir = ((df_eval["Pred_Petting"].astype(int) == 1) | (df_eval["Pred_Feeding"].astype(int) == 1)).astype(int)
            score_dir = df_eval["stai_total_avg"].values if "stai_total_avg" in df_eval.columns else (df_eval["stai_aa_avg"].values + df_eval["stai_ap_avg"].values)
            acc_dir = accuracy_score(yt_dir, yp_dir)
            bacc_dir = balanced_accuracy_score(yt_dir, yp_dir)
            f1_dir = f1_score(yt_dir, yp_dir, average="macro")
            try:
                auc_dir = roc_auc_score(yt_dir, score_dir)
            except Exception:
                auc_dir = np.nan
            print(
                f"  Direct Interaction vs Obs     : Acc={acc_dir:.2%}, BalAcc={bacc_dir:.2%}, F1={f1_dir:.3f}, AUC={auc_dir:.3f}"
            )

        cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4])
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_normalized = np.divide(cm, row_sums, where=row_sums != 0)
        per_class_accuracy = np.diag(cm_normalized)

        values = []
        print("\nPer-Class Accuracy:")
        for i, acc_i in enumerate(per_class_accuracy):
            print(f"  Class {group_labels[i]} accuracy: {acc_i:.2%}")
            values.append(f"{acc_i:.4f}")

        print("\nExcel (row):")
        print("\t".join(values))

        if chart_helper is not None:
            chart_helper.plot_confusion_matrix(
                cm,
                ticklabels=group_labels,
                show_percentage=True,
                task_name="",
                title=f"{title} (Acc:{acc:.2f}, F1:{f1:.2f})",
                xticklabels=group_labels,
                yticklabels=group_labels,
                font_size=font_size,
            )

        return df

    def _compute_regression_row(self, df: pd.DataFrame, gt_col: str, pred_col: str) -> dict:
        if gt_col not in df.columns or pred_col not in df.columns:
            return None
        sub = df[[gt_col, pred_col]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) < 3:
            return None
        yt = sub[gt_col].values
        yp = sub[pred_col].values
        mae = mean_absolute_error(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))
        r2 = r2_score(yt, yp)
        r_val, p_r = pearsonr(yt, yp)
        rho_val, p_rho = spearmanr(yt, yp)
        return {
            "n": len(sub),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "r": r_val,
            "p_r": p_r,
            "rho": rho_val,
            "p_rho": p_rho,
        }

    def evaluate_stai_predictions(self, df: pd.DataFrame, title: str = ""):
        """
        Evaluate MLLM predictions across both STAI psychometric subscales (STAI-AA and STAI-AP)
        and Total State Anxiety (STAI-Total = STAI-AA + STAI-AP) for:
          - Pre-Test (STAI_AA_pre, STAI_AP_pre, STAI_pre)
          - Post-Test (STAI_AA_post, STAI_AP_post, STAI_post)
          - Subtracted Improvement: Pre - Post (STAI_AA_improvement, STAI_AP_improvement, STAI_improvement)
        Plus Reliable Change Index (RCI_STAI >= 8 and RCI_AA >= 5) binary classification from Study 1.
        """
        def _resolve_col(candidates):
            for c in candidates:
                if c in df.columns and df[c].notna().any():
                    return c
            return None

        has_pre_post_diff = _resolve_col(["stai_aa_imp_pred_avg", "stai_aa_imp_pred"]) is not None
        step_mode = (
            str(df["Experiment Step"].iloc[0]).strip().lower()
            if "Experiment Step" in df.columns and len(df) > 0
            else ""
        )

        eval_pairs = []
        if has_pre_post_diff:
            eval_pairs.extend(
                [
                    (
                        "Pre-Test  | STAI-AA (Calmness/Security Absence, 10-40)",
                        "STAI_AA_pre",
                        _resolve_col(["stai_aa_pre_pred_avg", "stai_aa_pre_pred"]),
                    ),
                    (
                        "Pre-Test  | STAI-AP (Acute Tension Presence, 10-40)   ",
                        "STAI_AP_pre",
                        _resolve_col(["stai_ap_pre_pred_avg", "stai_ap_pre_pred"]),
                    ),
                    (
                        "Pre-Test  | STAI-Total (State Anxiety Total, 20-80)   ",
                        "STAI_pre",
                        _resolve_col(["stai_total_pre_pred_avg", "stai_total_pre_pred"]),
                    ),
                    (
                        "Post-Test | STAI-AA (Calmness/Security Absence, 10-40)",
                        "STAI_AA_post",
                        _resolve_col(["stai_aa_post_pred_avg", "stai_aa_post_pred"]),
                    ),
                    (
                        "Post-Test | STAI-AP (Acute Tension Presence, 10-40)   ",
                        "STAI_AP_post",
                        _resolve_col(["stai_ap_post_pred_avg", "stai_ap_post_pred"]),
                    ),
                    (
                        "Post-Test | STAI-Total (State Anxiety Total, 20-80)   ",
                        "STAI_post",
                        _resolve_col(["stai_total_post_pred_avg", "stai_total_post_pred"]),
                    ),
                    (
                        "Subtracted (Pre - Post) | Delta STAI-AA Improvement   ",
                        "STAI_AA_improvement",
                        _resolve_col(["stai_aa_imp_pred_avg", "stai_aa_imp_pred", "stai_aa_avg", "stai_aa"]),
                    ),
                    (
                        "Subtracted (Pre - Post) | Delta STAI-AP Improvement   ",
                        "STAI_AP_improvement",
                        _resolve_col(["stai_ap_imp_pred_avg", "stai_ap_imp_pred", "stai_ap_avg", "stai_ap"]),
                    ),
                    (
                        "Subtracted (Pre - Post) | Delta STAI-Total Improvement",
                        "STAI_improvement",
                        _resolve_col(["stai_total_imp_pred_avg", "stai_total_imp_pred", "stai_total_avg", "stai_total"]),
                    ),
                ]
            )
        elif step_mode == "pre":
            eval_pairs.extend(
                [
                    (
                        "Pre-Test  | STAI-AA (Calmness/Security Absence, 10-40)",
                        "STAI_AA_pre",
                        _resolve_col(["stai_aa_avg", "stai_aa"]),
                    ),
                    (
                        "Pre-Test  | STAI-AP (Acute Tension Presence, 10-40)   ",
                        "STAI_AP_pre",
                        _resolve_col(["stai_ap_avg", "stai_ap"]),
                    ),
                    (
                        "Pre-Test  | STAI-Total (State Anxiety Total, 20-80)   ",
                        "STAI_pre",
                        _resolve_col(["stai_total_avg", "stai_total"]),
                    ),
                ]
            )
        elif step_mode == "post":
            eval_pairs.extend(
                [
                    (
                        "Post-Test | STAI-AA (Calmness/Security Absence, 10-40)",
                        "STAI_AA_post",
                        _resolve_col(["stai_aa_avg", "stai_aa"]),
                    ),
                    (
                        "Post-Test | STAI-AP (Acute Tension Presence, 10-40)   ",
                        "STAI_AP_post",
                        _resolve_col(["stai_ap_avg", "stai_ap"]),
                    ),
                    (
                        "Post-Test | STAI-Total (State Anxiety Total, 20-80)   ",
                        "STAI_post",
                        _resolve_col(["stai_total_avg", "stai_total"]),
                    ),
                ]
            )

        print(f"\n[STAI Two-Component & Total Psychometric Regression ({title})]:")
        for label, gt_col, pred_col in eval_pairs:
            if not pred_col:
                continue
            res = self._compute_regression_row(df, gt_col, pred_col)
            if res is None:
                continue
            print(
                f"  {label} (n={res['n']}): "
                f"MAE={res['MAE']:.3f}, RMSE={res['RMSE']:.3f}, R²={res['R2']:.3f}, "
                f"Pearson r={res['r']:.3f} (p={res['p_r']:.4f}), Spearman ρ={res['rho']:.3f} (p={res['p_rho']:.4f})"
            )

        # Study 1 Reliable Change Index (RCI) Evaluation when Pre - Post differences are present
        if has_pre_post_diff:
            print(f"\n[Study 1 Reliable Change Index (RCI) Classification ({title})]:")
            col_tot_imp = _resolve_col(["stai_total_imp_pred_avg", "stai_total_imp_pred", "stai_total_avg", "stai_total"])
            col_aa_imp = _resolve_col(["stai_aa_imp_pred_avg", "stai_aa_imp_pred", "stai_aa_avg", "stai_aa"])

            if "STAI_improvement" in df.columns and col_tot_imp:
                sub_tot = df[["STAI_improvement", col_tot_imp]].apply(pd.to_numeric, errors="coerce").dropna()
                if len(sub_tot) >= 3:
                    yt_rci = (sub_tot["STAI_improvement"] >= 8).astype(int)
                    yp_rci = (sub_tot[col_tot_imp] >= 8).astype(int)
                    acc_rci = accuracy_score(yt_rci, yp_rci)
                    bacc_rci = balanced_accuracy_score(yt_rci, yp_rci)
                    f1_rci = f1_score(yt_rci, yp_rci, average="macro", zero_division=0)
                    try:
                        auc_rci = roc_auc_score(yt_rci, sub_tot[col_tot_imp])
                    except Exception:
                        auc_rci = np.nan
                    print(
                        f"  RCI_STAI (Delta STAI-Total >= 8) : Acc={acc_rci:.2%}, BalAcc={bacc_rci:.2%}, Macro-F1={f1_rci:.3f}, AUC={auc_rci:.3f}"
                    )

            if "STAI_AA_improvement" in df.columns and col_aa_imp:
                sub_aa = df[["STAI_AA_improvement", col_aa_imp]].apply(pd.to_numeric, errors="coerce").dropna()
                if len(sub_aa) >= 3:
                    yt_rci_aa = (sub_aa["STAI_AA_improvement"] >= 5).astype(int)
                    yp_rci_aa = (sub_aa[col_aa_imp] >= 5).astype(int)
                    acc_aa = accuracy_score(yt_rci_aa, yp_rci_aa)
                    bacc_aa = balanced_accuracy_score(yt_rci_aa, yp_rci_aa)
                    f1_aa = f1_score(yt_rci_aa, yp_rci_aa, average="macro", zero_division=0)
                    try:
                        auc_aa = roc_auc_score(yt_rci_aa, sub_aa[col_aa_imp])
                    except Exception:
                        auc_aa = np.nan
                    print(
                        f"  RCI_AA   (Delta STAI-AA >= 5)    : Acc={acc_aa:.2%}, BalAcc={bacc_aa:.2%}, Macro-F1={f1_aa:.3f}, AUC={auc_aa:.3f}"
                    )

    def compare_modality_experiments(self, experiment_dfs: dict) -> pd.DataFrame:
        """
        Build and display a comprehensive comparison table across the three modality ablation experiments:
          - Experiment 1: Video Only
          - Experiment 2: Video + Audio (Non-Semantic Acoustics Only)
          - Experiment 3: Video + Audio + Speech Context (Full Audiovisual + Speech Semantics)
        """
        rows = []
        target_specs = [
            ("Pre-Test", "STAI-AA (10-40)", "STAI_AA_pre", "stai_aa_pre_pred"),
            ("Pre-Test", "STAI-AP (10-40)", "STAI_AP_pre", "stai_ap_pre_pred"),
            ("Pre-Test", "STAI-Total (20-80)", "STAI_pre", "stai_total_pre_pred"),
            ("Post-Test", "STAI-AA (10-40)", "STAI_AA_post", "stai_aa_post_pred"),
            ("Post-Test", "STAI-AP (10-40)", "STAI_AP_post", "stai_ap_post_pred"),
            ("Post-Test", "STAI-Total (20-80)", "STAI_post", "stai_total_post_pred"),
            ("Improvement (Pre - Post)", "Delta STAI-AA", "STAI_AA_improvement", "stai_aa_imp_pred"),
            ("Improvement (Pre - Post)", "Delta STAI-AP", "STAI_AP_improvement", "stai_ap_imp_pred"),
            ("Improvement (Pre - Post)", "Delta STAI-Total", "STAI_improvement", "stai_total_imp_pred"),
        ]

        for exp_name, df_exp in experiment_dfs.items():
            if df_exp is None or df_exp.empty:
                continue
            for stage, construct, gt_col, pred_col in target_specs:
                res = self._compute_regression_row(df_exp, gt_col, pred_col)
                if res is not None:
                    rows.append(
                        {
                            "Experiment": exp_name,
                            "Stage": stage,
                            "Construct": construct,
                            "n": res["n"],
                            "MAE": round(res["MAE"], 3),
                            "RMSE": round(res["RMSE"], 3),
                            "R2": round(res["R2"], 3),
                            "Pearson_r": round(res["r"], 3),
                            "p_r": round(res["p_r"], 4),
                            "Spearman_rho": round(res["rho"], 3),
                            "p_rho": round(res["p_rho"], 4),
                        }
                    )

        df_summary = pd.DataFrame(rows)
        if not df_summary.empty:
            print("=== Cross-Modality Comparison Summary (Pre, Post, and Pre - Post Improvement) ===")
            print(df_summary.to_string(index=False))
        else:
            print("[INFO] No completed experiment predictions found yet for cross-modality comparison.")
        return df_summary

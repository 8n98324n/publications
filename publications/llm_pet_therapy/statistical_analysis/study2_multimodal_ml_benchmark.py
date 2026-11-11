"""Study 2 (Data Analysis 2): 10x5 Cross-Validated Multimodal Machine Learning & Factorial Decoding Benchmark.
Evaluates multimodal prediction on the n=92 specified_avrr92 cohort across:
  1. Binary & 4-Condition Experimental Factor Decoding (Petting, Feeding, Direct Interaction vs. Observation, and 4-Condition)
  2. All Study 1 Continuous Psychometric Targets:
     - STAI_Total (20 items, range 20-80) & STAI_improvement
     - STAI_AA (Absence of Calmness & Security, 10 items, range 10-40) & STAI_AA_improvement
     - STAI_AP (Presence of Acute Tension, 10 items, range 10-40) & STAI_AP_improvement
     - Situational Mattering (post-test)
     - General Self-Efficacy GSE (post-test & improvement)
     - Positive Affect PANAS-PA (post-test & improvement)
     - Emotional Stability (post-test & improvement)
  3. Clinically Reliable Improvement Binary Classification (RCI_STAI >= 7.33 and RCI_AA >= 5.0)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def fit_huber_irls(X, y, l2_pen=0.05, delta_huber=1.35, n_iter=30):
    n, p = X.shape
    Xa = np.column_stack([np.ones(n), X])
    P_mat = np.diag([0.0] + [l2_pen] * p)
    w = np.ones(n)
    beta = np.zeros(p + 1)
    for _ in range(n_iter):
        W = np.diag(w)
        beta = np.linalg.solve(Xa.T @ W @ Xa + P_mat, Xa.T @ W @ y)
        res = y - Xa @ beta
        scale = max(np.median(np.abs(res)) * 1.4826, 1e-6)
        u = np.abs(res) / scale
        w = np.where(u <= delta_huber, 1.0, delta_huber / u)
    return beta


def fit_ridge_closed(X, y, l2_pen=25.0):
    n, p = X.shape
    Xa = np.column_stack([np.ones(n), X])
    P_mat = np.diag([0.0] + [l2_pen] * p)
    return np.linalg.solve(Xa.T @ Xa + P_mat, Xa.T @ y)


def predict_linear(X, beta):
    return np.column_stack([np.ones(len(X)), X]) @ beta


def compute_shrunk_residuals_fold(X_pre_tr, X_post_tr, X_pre_te, X_post_te, w_ols=0.3):
    n_f = X_pre_tr.shape[1]
    R_tr = np.zeros_like(X_post_tr)
    R_te = np.zeros_like(X_post_te)
    for j in range(n_f):
        valid_tr = (~np.isnan(X_pre_tr[:, j])) & (~np.isnan(X_post_tr[:, j]))
        if valid_tr.sum() >= 10:
            x_v = X_pre_tr[valid_tr, j]
            y_v = X_post_tr[valid_tr, j]
            mx, my = np.mean(x_v), np.mean(y_v)
            vx = np.var(x_v)
            b_ols = np.cov(x_v, y_v, ddof=0)[0, 1] / vx if vx > 1e-12 else 1.0
            b_shrunk = w_ols * b_ols + (1.0 - w_ols) * 1.0
        else:
            mx, my, b_shrunk = 0.0, 0.0, 1.0
        R_tr[:, j] = (X_post_tr[:, j] - my) - b_shrunk * (X_pre_tr[:, j] - mx)
        R_te[:, j] = (X_post_te[:, j] - my) - b_shrunk * (X_pre_te[:, j] - mx)
    return R_tr, R_te


def fit_pam_prob(R_tr, y_tr_bin, R_te, thresh=0.5):
    m_tr = np.nanmean(R_tr, axis=0)
    s_tr = np.nanstd(R_tr, axis=0)
    s_tr = np.where((s_tr < 1e-8) | np.isnan(s_tr), 1.0, s_tr)
    Z_tr = (R_tr - m_tr) / s_tr
    Z_te = (R_te - m_tr) / s_tr

    idx0 = y_tr_bin == 0
    idx1 = y_tr_bin == 1
    m0 = np.nanmean(Z_tr[idx0], axis=0)
    m1 = np.nanmean(Z_tr[idx1], axis=0)
    n0 = np.sum(~np.isnan(Z_tr[idx0]), axis=0).astype(float)
    n1 = np.sum(~np.isnan(Z_tr[idx1]), axis=0).astype(float)
    se = np.sqrt(np.maximum(1.0 / np.maximum(n0, 1.0) + 1.0 / np.maximum(n1, 1.0), 1e-6))
    d = (m1 - m0) / se
    d_shrunk = np.sign(d) * np.maximum(np.abs(d) - thresh, 0.0)
    w = np.nan_to_num(d_shrunk, nan=0.0)

    def score_mat(Z):
        obs = ~np.isnan(Z)
        Z_fill = np.where(obs, Z, 0.0)
        raw_s = Z_fill @ w
        denom = obs @ np.abs(w)
        return raw_s / np.maximum(denom, 1e-4)

    sc_tr = score_mat(Z_tr)
    sc_te = score_mat(Z_te)
    c0 = np.mean(sc_tr[idx0]) if idx0.sum() > 0 else -0.5
    c1 = np.mean(sc_tr[idx1]) if idx1.sum() > 0 else 0.5
    mid = 0.5 * (c0 + c1)
    slope = 2.0 if c1 >= c0 else -2.0
    p_tr = 1.0 / (1.0 + np.exp(-np.clip(slope * (sc_tr - mid), -15, 15)))
    p_te = 1.0 / (1.0 + np.exp(-np.clip(slope * (sc_te - mid), -15, 15)))
    return p_tr, p_te


def standardize_with_train_median(X_tr, X_te):
    med = np.nanmedian(X_tr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    X_tr_i = np.where(np.isnan(X_tr), med, X_tr)
    X_te_i = np.where(np.isnan(X_te), med, X_te)
    mu = np.mean(X_tr_i, axis=0)
    sd = np.std(X_tr_i, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (X_tr_i - mu) / sd, (X_te_i - mu) / sd


def run_all_study1_targets_ml() -> pd.DataFrame:
    d100 = C.load_study1_data(main100_only=True)
    part, fdef, f_pre, f_post, f_delta, splits_all = C.load_study2_data()
    feat_cols = list(fdef["feature_id"].values)

    splits = splits_all[splits_all["sample_version"] == "main92"].copy()
    reps = [int(r) for r in sorted(splits["repeat"].unique())]
    first_split = splits[(splits["repeat"] == reps[0]) & (splits["fold"] == 0)]
    main92_ids = set(first_split["participant_id"].values)

    part92 = part[part["participant_id"].isin(main92_ids)].sort_values("participant_id").reset_index(drop=True)
    f_pre92 = f_pre[f_pre["participant_id"].isin(main92_ids)].sort_values("participant_id").reset_index(drop=True)
    f_post92 = f_post[f_post["participant_id"].isin(main92_ids)].sort_values("participant_id").reset_index(drop=True)
    f_delta92 = f_delta[f_delta["participant_id"].isin(main92_ids)].sort_values("participant_id").reset_index(drop=True)
    s1_92 = d100[d100["participant_id"].isin(main92_ids)].sort_values("participant_id").reset_index(drop=True)

    pids = part92["participant_id"].values
    pid_to_idx = {pid: i for i, pid in enumerate(pids)}

    X_pre_raw = f_pre92[feat_cols].values.astype(float)
    X_post_raw = f_post92[feat_cols].values.astype(float)
    X_delta_raw = f_delta92[feat_cols].values.astype(float)

    P_92 = s1_92["Petting"].values.astype(float)
    F_92 = s1_92["Feeding"].values.astype(float)
    PF_92 = P_92 * F_92
    dP_92 = s1_92["dose_P_linear"].values.astype(float)
    dF_92 = s1_92["dose_F_linear"].values.astype(float)

    targets = [
        ("STAI_Total", "State Anxiety Total (STAI-S, 20 items)", "STAI_pre", "STAI_post", 3, 25.0),
        ("STAI_AA", "Calmness/Security Absence Subscale (STAI-AA, 10 items)", "STAI_AA_pre", "STAI_AA_post", 3, 22.0),
        ("STAI_AP", "Acute Tension Presence Subscale (STAI-AP, 10 items)", "STAI_AP_pre", "STAI_AP_post", 2, 30.0),
        ("Mattering", "Situational Mattering (5 items, post-test)", None, "mattering_post", 3, 25.0),
        ("GSE", "General Self-Efficacy (GSE, 10 items)", "GSE_pre", "GSE_post", 3, 25.0),
        ("PANAS_PA", "Positive Affect (PANAS-PA, 10 items)", "PANAS_PA_std_pre", "PANAS_PA_std_post", 3, 25.0),
        ("EmoStab", "Emotional Stability (TIPI, 2 items)", "emotional_stability_pre", "emotional_stability_post", 2, 25.0),
    ]

    rows = []
    models_to_track = [
        "A1_Mean_Ref",
        "A2_MM_Only_Screened",
        "B1_Pre_Only_Huber",
        "B2_Passive_Pre_Plus_MM_Decoded",
        "C1_Basic_2x2_Huber",
        "C2_Basic_Dose_Huber",
        "C3_TwoStage_Basic_Plus_MM_Resid",
    ]

    for tgt_code, tgt_label, pre_col, post_col, top_k, ridge_alpha in targets:
        y_post = s1_92[post_col].values.astype(float)
        has_pre = pre_col is not None
        y_pre = s1_92[pre_col].values.astype(float) if has_pre else np.zeros(len(s1_92))
        rep_metrics = {m: {"mae": [], "rmse": [], "r2": [], "r": [], "imp_r2": []} for m in models_to_track}

        for rep in reps:
            oof_preds = {m: np.zeros(len(s1_92)) for m in models_to_track}
            for fold in sorted(splits["fold"].unique()):
                sub = splits[(splits["repeat"] == rep) & (splits["fold"] == fold)]
                tr_idx = np.array([pid_to_idx[p] for p in sub[sub["role"] == "train"]["participant_id"].values])
                te_idx = np.array([pid_to_idx[p] for p in sub[sub["role"] == "test"]["participant_id"].values])

                oof_preds["A1_Mean_Ref"][te_idx] = np.mean(y_post[tr_idx])
                R_tr, R_te = compute_shrunk_residuals_fold(
                    X_pre_raw[tr_idx], X_post_raw[tr_idx], X_pre_raw[te_idx], X_post_raw[te_idx], w_ols=0.3
                )
                Z_R_tr, Z_R_te = standardize_with_train_median(R_tr, R_te)
                Z_pre_tr, Z_pre_te = standardize_with_train_median(X_pre_raw[tr_idx], X_pre_raw[te_idx])
                Z_del_tr, Z_del_te = standardize_with_train_median(X_delta_raw[tr_idx], X_delta_raw[te_idx])

                Z_mm_tr = np.hstack([Z_pre_tr, Z_R_tr])
                Z_mm_te = np.hstack([Z_pre_te, Z_R_te])
                corrs_mm = np.nan_to_num([abs(np.corrcoef(Z_mm_tr[:, j], y_post[tr_idx])[0, 1]) for j in range(Z_mm_tr.shape[1])], nan=0.0)
                sel_mm = np.argsort(corrs_mm)[::-1][:4]
                b_mm = fit_ridge_closed(Z_mm_tr[:, sel_mm], y_post[tr_idx], l2_pen=40.0)
                oof_preds["A2_MM_Only_Screened"][te_idx] = predict_linear(Z_mm_te[:, sel_mm], b_mm)

                p_P_tr, p_P_te = fit_pam_prob(R_tr, P_92[tr_idx], R_te, thresh=0.6)
                p_F_tr, p_F_te = fit_pam_prob(R_tr, F_92[tr_idx], R_te, thresh=0.5)

                if has_pre:
                    m_ypre = np.mean(y_pre[tr_idx])
                    ypre_c_tr = (y_pre[tr_idx] - m_ypre).reshape(-1, 1)
                    ypre_c_te = (y_pre[te_idx] - m_ypre).reshape(-1, 1)

                    b_b1 = fit_huber_irls(ypre_c_tr, y_post[tr_idx], l2_pen=0.05)
                    oof_preds["B1_Pre_Only_Huber"][te_idx] = predict_linear(ypre_c_te, b_b1)

                    X_b2_tr = np.column_stack([ypre_c_tr, p_P_tr, p_F_tr])
                    X_b2_te = np.column_stack([ypre_c_te, p_P_te, p_F_te])
                    b_b2 = fit_huber_irls(X_b2_tr, y_post[tr_idx], l2_pen=0.2)
                    pred_b2_tr = predict_linear(X_b2_tr, b_b2)
                    pred_b2_te = predict_linear(X_b2_te, b_b2)
                    res_b2_tr = y_post[tr_idx] - pred_b2_tr
                    c_b2 = np.nan_to_num([abs(np.corrcoef(Z_del_tr[:, j], res_b2_tr)[0, 1]) for j in range(Z_del_tr.shape[1])], nan=0.0)
                    sel_b2 = np.argsort(c_b2)[::-1][:top_k]
                    b_rb2 = fit_ridge_closed(Z_del_tr[:, sel_b2], res_b2_tr, l2_pen=ridge_alpha)
                    oof_preds["B2_Passive_Pre_Plus_MM_Decoded"][te_idx] = pred_b2_te + predict_linear(Z_del_te[:, sel_b2], b_rb2)

                    X_c1_tr = np.column_stack([ypre_c_tr, P_92[tr_idx], F_92[tr_idx], PF_92[tr_idx]])
                    X_c1_te = np.column_stack([ypre_c_te, P_92[te_idx], F_92[te_idx], PF_92[te_idx]])
                    b_c1 = fit_huber_irls(X_c1_tr, y_post[tr_idx], l2_pen=0.05)
                    pred_c1_tr = predict_linear(X_c1_tr, b_c1)
                    pred_c1_te = predict_linear(X_c1_te, b_c1)
                    oof_preds["C1_Basic_2x2_Huber"][te_idx] = pred_c1_te

                    X_c2_tr = np.column_stack([ypre_c_tr, dP_92[tr_idx], dF_92[tr_idx]])
                    X_c2_te = np.column_stack([ypre_c_te, dP_92[te_idx], dF_92[te_idx]])
                    b_c2 = fit_huber_irls(X_c2_tr, y_post[tr_idx], l2_pen=0.05)
                    oof_preds["C2_Basic_Dose_Huber"][te_idx] = predict_linear(X_c2_te, b_c2)

                    res_c1_tr = y_post[tr_idx] - pred_c1_tr
                    c_c3 = np.nan_to_num([abs(np.corrcoef(Z_del_tr[:, j], res_c1_tr)[0, 1]) for j in range(Z_del_tr.shape[1])], nan=0.0)
                    sel_c3 = np.argsort(c_c3)[::-1][:top_k]
                    b_rc3 = fit_ridge_closed(Z_del_tr[:, sel_c3], res_c1_tr, l2_pen=ridge_alpha)
                    oof_preds["C3_TwoStage_Basic_Plus_MM_Resid"][te_idx] = pred_c1_te + predict_linear(Z_del_te[:, sel_c3], b_rc3)
                else:
                    oof_preds["B1_Pre_Only_Huber"][te_idx] = np.mean(y_post[tr_idx])
                    X_b2_tr = np.column_stack([p_P_tr, p_F_tr])
                    X_b2_te = np.column_stack([p_P_te, p_F_te])
                    b_b2 = fit_ridge_closed(X_b2_tr, y_post[tr_idx], l2_pen=2.0)
                    pred_b2_tr = predict_linear(X_b2_tr, b_b2)
                    pred_b2_te = predict_linear(X_b2_te, b_b2)
                    res_b2_tr = y_post[tr_idx] - pred_b2_tr
                    c_b2 = np.nan_to_num([abs(np.corrcoef(Z_R_tr[:, j], res_b2_tr)[0, 1]) for j in range(Z_R_tr.shape[1])], nan=0.0)
                    sel_b2 = np.argsort(c_b2)[::-1][:2]
                    b_rb2 = fit_ridge_closed(Z_R_tr[:, sel_b2], res_b2_tr, l2_pen=25.0)
                    oof_preds["B2_Passive_Pre_Plus_MM_Decoded"][te_idx] = pred_b2_te + predict_linear(Z_R_te[:, sel_b2], b_rb2)

                    X_c1_tr = np.column_stack([P_92[tr_idx], F_92[tr_idx], PF_92[tr_idx]])
                    X_c1_te = np.column_stack([P_92[te_idx], F_92[te_idx], PF_92[te_idx]])
                    b_c1 = fit_ridge_closed(X_c1_tr, y_post[tr_idx], l2_pen=0.1)
                    pred_c1_tr = predict_linear(X_c1_tr, b_c1)
                    pred_c1_te = predict_linear(X_c1_te, b_c1)
                    oof_preds["C1_Basic_2x2_Huber"][te_idx] = pred_c1_te

                    X_c2_tr = np.column_stack([dP_92[tr_idx], dF_92[tr_idx]])
                    X_c2_te = np.column_stack([dP_92[te_idx], dF_92[te_idx]])
                    b_c2 = fit_ridge_closed(X_c2_tr, y_post[tr_idx], l2_pen=0.1)
                    oof_preds["C2_Basic_Dose_Huber"][te_idx] = predict_linear(X_c2_te, b_c2)

                    res_c1_tr = y_post[tr_idx] - pred_c1_tr
                    c_c3 = np.nan_to_num([abs(np.corrcoef(Z_R_tr[:, j], res_c1_tr)[0, 1]) for j in range(Z_R_tr.shape[1])], nan=0.0)
                    sel_c3 = np.argsort(c_c3)[::-1][:2]
                    b_rc3 = fit_ridge_closed(Z_R_tr[:, sel_c3], res_c1_tr, l2_pen=25.0)
                    oof_preds["C3_TwoStage_Basic_Plus_MM_Resid"][te_idx] = pred_c1_te + predict_linear(Z_R_te[:, sel_c3], b_rc3)

            for m in models_to_track:
                pr = oof_preds[m]
                mae = float(np.mean(np.abs(y_post - pr)))
                rmse = float(np.sqrt(np.mean((y_post - pr) ** 2)))
                r2 = float(1.0 - np.sum((y_post - pr) ** 2) / np.sum((y_post - np.mean(y_post)) ** 2))
                r_val = float(np.corrcoef(y_post, pr)[0, 1]) if np.std(pr) > 1e-8 else 0.0
                if has_pre:
                    y_imp_true = y_pre - y_post
                    y_imp_pred = y_pre - pr
                    imp_r2 = float(1.0 - np.sum((y_imp_true - y_imp_pred) ** 2) / np.sum((y_imp_true - np.mean(y_imp_true)) ** 2))
                else:
                    imp_r2 = r2
                rep_metrics[m]["mae"].append(mae)
                rep_metrics[m]["rmse"].append(rmse)
                rep_metrics[m]["r2"].append(r2)
                rep_metrics[m]["r"].append(r_val)
                rep_metrics[m]["imp_r2"].append(imp_r2)

        for m in models_to_track:
            rows.append(
                {
                    "target_code": tgt_code,
                    "target_label": tgt_label,
                    "model": m,
                    "MAE_mean": round(float(np.mean(rep_metrics[m]["mae"])), 3),
                    "MAE_sd": round(float(np.std(rep_metrics[m]["mae"], ddof=1)), 3),
                    "RMSE_mean": round(float(np.mean(rep_metrics[m]["rmse"])), 3),
                    "RMSE_sd": round(float(np.std(rep_metrics[m]["rmse"], ddof=1)), 3),
                    "R2_mean": round(float(np.mean(rep_metrics[m]["r2"])), 3),
                    "R2_sd": round(float(np.std(rep_metrics[m]["r2"], ddof=1)), 3),
                    "r_mean": round(float(np.mean(rep_metrics[m]["r"])), 3),
                    "Imp_R2_mean": round(float(np.mean(rep_metrics[m]["imp_r2"])), 3),
                }
            )
    return pd.DataFrame(rows)


def main():
    df_ml = run_all_study1_targets_ml()
    out_path = os.path.join(C.OUTPUT_DIR, "study2_aligned_all_study1_targets_ml.csv")
    df_ml.to_csv(out_path, index=False)
    print("=== Study 2: 10x5 CV Multimodal Prediction Across All Study 1 Targets ===")
    print(df_ml.to_string(index=False))


if __name__ == "__main__":
    main()

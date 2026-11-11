"""Study 2 (Data Analysis 2): 19-Feature Multimodal Statistical Analysis & Cross-Channel Alignment with Study 1.
Reproduces:
  1. 2x2 Factorial ANCOVA (OLS & HC3 robust SEs), Exposure Time-Dose ANCOVA (10 min = 1.0, 5 min = 0.5),
     Direct Interaction vs. Observation contrasts, and Net Single-Component contrasts across all 19 multimodal features.
  2. Cross-channel associations (Table 6) between multimodal feature changes (raw delta & baseline-adjusted ANCOVA residuals)
     and Study 1's core psychological targets (STAI-Total, STAI-AA, STAI-AP, Situational Mattering, GSE, PANAS-PA).
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def pearson_r_pval(x: np.ndarray, y: np.ndarray):
    m = (~np.isnan(x)) & (~np.isnan(y))
    xv, yv = x[m], y[m]
    n = len(xv)
    if n < 4 or np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
        return 0.0, 1.0
    r = float(np.corrcoef(xv, yv)[0, 1])
    if abs(r) >= 1.0:
        return r, 0.0
    t_val = r * np.sqrt((n - 2.0) / max(1.0 - r * r, 1e-12))
    return r, C.t_pval_2sided(t_val, n - 2)


def spearman_r_pval(x: np.ndarray, y: np.ndarray):
    m = (~np.isnan(x)) & (~np.isnan(y))
    xv, yv = x[m], y[m]
    if len(xv) < 4:
        return 0.0, 1.0
    rx = pd.Series(xv).rank(method="average").values
    ry = pd.Series(yv).rank(method="average").values
    return pearson_r_pval(rx, ry)


def compute_multimodal_feature_effects():
    d100 = C.load_study1_data(main100_only=True)
    part, fdef, f_pre, f_post, f_delta, _ = C.load_study2_data()
    feat_cols = list(fdef["feature_id"].values)
    feat_labels = dict(zip(fdef["feature_id"], fdef["feature_name"]))

    rows = []
    for f in feat_cols:
        x_pre_all = f_pre[f].values.astype(float)
        y_post_all = f_post[f].values.astype(float)
        valid = (~np.isnan(x_pre_all)) & (~np.isnan(y_post_all))
        n_sub = int(valid.sum())

        scale_mult = 100.0 if f in [
            "au05_prevalence",
            "au07_prevalence",
            "au12_prevalence",
            "auto_silent_interval_proportion",
            "auto_periodic_vocalization_proportion",
        ] else 1.0
        x_pre = (x_pre_all[valid] - np.mean(x_pre_all[valid])) * scale_mult
        y = y_post_all[valid] * scale_mult
        P = d100["Petting"].values[valid].astype(float)
        F = d100["Feeding"].values[valid].astype(float)
        PF = P * F

        X_2x2 = np.column_stack([np.ones(n_sub), x_pre, P, F, PF])
        fit_2x2 = C.ols_fit(y, X_2x2)
        c_P = C.linear_contrast(fit_2x2, [0, 0, 1, 0, 0.5])
        c_F = C.linear_contrast(fit_2x2, [0, 0, 0, 1, 0.5])
        c_Int = C.linear_contrast(fit_2x2, [0, 0, 0, 0, 1.0])
        c_Dir = C.linear_contrast(fit_2x2, [0, 0, 2/3, 2/3, 1/3])
        c_NetP = C.linear_contrast(fit_2x2, [0, 0, 1, 0, 0])
        c_NetF = C.linear_contrast(fit_2x2, [0, 0, 0, 1, 0])

        dP = np.where((P == 1) & (F == 0), 1.0, np.where((P == 1) & (F == 1), 0.5, 0.0))
        dF = np.where((F == 1) & (P == 0), 1.0, np.where((P == 1) & (F == 1), 0.5, 0.0))
        X_dose = np.column_stack([np.ones(n_sub), x_pre, dP, dF])
        fit_dose = C.ols_fit(y, X_dose)
        c_dP = C.linear_contrast(fit_dose, [0, 0, 1, 0])
        c_dF = C.linear_contrast(fit_dose, [0, 0, 0, 1])

        rows.append(
            {
                "feature": f,
                "label": feat_labels.get(f, f),
                "n": n_sub,
                "P_2x2_b": round(c_P["est"], 4),
                "P_2x2_p": round(c_P["p_ols"], 4),
                "P_2x2_hc3_p": round(c_P["p_hc3"], 4),
                "F_2x2_b": round(c_F["est"], 4),
                "F_2x2_p": round(c_F["p_ols"], 4),
                "F_2x2_hc3_p": round(c_F["p_hc3"], 4),
                "Int_2x2_b": round(c_Int["est"], 4),
                "Int_2x2_p": round(c_Int["p_ols"], 4),
                "P_dose_b": round(c_dP["est"], 4),
                "P_dose_se": round(c_dP["se_ols"], 4),
                "P_dose_p": round(c_dP["p_ols"], 4),
                "P_dose_hc3_p": round(c_dP["p_hc3"], 4),
                "F_dose_b": round(c_dF["est"], 4),
                "F_dose_se": round(c_dF["se_ols"], 4),
                "F_dose_p": round(c_dF["p_ols"], 4),
                "F_dose_hc3_p": round(c_dF["p_hc3"], 4),
                "Direct_vs_Obs_b": round(c_Dir["est"], 4),
                "Direct_vs_Obs_p": round(c_Dir["p_ols"], 4),
                "Direct_vs_Obs_hc3_p": round(c_Dir["p_hc3"], 4),
                "NetP_10m_b": round(c_NetP["est"], 4),
                "NetP_10m_p": round(c_NetP["p_ols"], 4),
                "NetF_10m_b": round(c_NetF["est"], 4),
                "NetF_10m_p": round(c_NetF["p_ols"], 4),
            }
        )
    return pd.DataFrame(rows)


def compute_cross_channel_associations():
    d100 = C.load_study1_data(main100_only=True)
    part, fdef, f_pre, f_post, f_delta, _ = C.load_study2_data()
    feat_cols = list(fdef["feature_id"].values)
    feat_labels = dict(zip(fdef["feature_id"], fdef["feature_name"]))

    rows = []
    for f in feat_cols:
        x_pre_all = f_pre[f].values.astype(float)
        y_post_all = f_post[f].values.astype(float)
        valid = (~np.isnan(x_pre_all)) & (~np.isnan(y_post_all))
        n_sub = int(valid.sum())

        xv_pre = x_pre_all[valid]
        yv_post = y_post_all[valid]
        dx = yv_post - xv_pre
        fx_res = C.ols_fit(yv_post, np.column_stack([np.ones(n_sub), xv_pre]))["resid"]

        stai_imp = d100["STAI_improvement"].values[valid].astype(float)
        aa_imp = d100["STAI_AA_improvement"].values[valid].astype(float)
        ap_imp = d100["STAI_AP_improvement"].values[valid].astype(float)
        stai_res = -C.ols_fit(d100["STAI_post"].values[valid].astype(float), np.column_stack([np.ones(n_sub), d100["STAI_pre"].values[valid].astype(float)]))["resid"]
        aa_res = -C.ols_fit(d100["STAI_AA_post"].values[valid].astype(float), np.column_stack([np.ones(n_sub), d100["STAI_AA_pre"].values[valid].astype(float)]))["resid"]
        ap_res = -C.ols_fit(d100["STAI_AP_post"].values[valid].astype(float), np.column_stack([np.ones(n_sub), d100["STAI_AP_pre"].values[valid].astype(float)]))["resid"]
        mat_val = d100["mattering_post"].values[valid].astype(float)
        gse_res = C.ols_fit(d100["GSE_post"].values[valid].astype(float), np.column_stack([np.ones(n_sub), d100["GSE_pre"].values[valid].astype(float)]))["resid"]
        pa_res = C.ols_fit(d100["PANAS_PA_std_post"].values[valid].astype(float), np.column_stack([np.ones(n_sub), d100["PANAS_PA_std_pre"].values[valid].astype(float)]))["resid"]

        P = d100["Petting"].values[valid].astype(float)
        F = d100["Feeding"].values[valid].astype(float)
        X_cond = np.column_stack([np.ones(n_sub), P, F, P * F])
        fx_res_c = C.ols_fit(fx_res, X_cond)["resid"]
        stai_res_c = C.ols_fit(stai_res, X_cond)["resid"]
        aa_res_c = C.ols_fit(aa_res, X_cond)["resid"]
        ap_res_c = C.ols_fit(ap_res, X_cond)["resid"]

        rho_stai, p_rho_stai = spearman_r_pval(dx, stai_imp)
        rho_aa, p_rho_aa = spearman_r_pval(dx, aa_imp)
        rho_ap, p_rho_ap = spearman_r_pval(dx, ap_imp)
        rho_mat, p_rho_mat = spearman_r_pval(dx, mat_val)

        r_stai_res, p_stai_res = pearson_r_pval(fx_res, stai_res)
        r_aa_res, p_aa_res = pearson_r_pval(fx_res, aa_res)
        r_ap_res, p_ap_res = pearson_r_pval(fx_res, ap_res)
        r_mat_res, p_mat_res = pearson_r_pval(fx_res, mat_val)
        r_gse_res, p_gse_res = pearson_r_pval(fx_res, gse_res)
        r_pa_res, p_pa_res = pearson_r_pval(fx_res, pa_res)

        r_stai_part, p_stai_part = pearson_r_pval(fx_res_c, stai_res_c)
        r_aa_part, p_aa_part = pearson_r_pval(fx_res_c, aa_res_c)
        r_ap_part, p_ap_part = pearson_r_pval(fx_res_c, ap_res_c)

        rows.append(
            {
                "feature": f,
                "label": feat_labels.get(f, f),
                "n": n_sub,
                "raw_rho_STAI": round(rho_stai, 4),
                "raw_p_STAI": round(p_rho_stai, 4),
                "raw_rho_AA": round(rho_aa, 4),
                "raw_p_AA": round(p_rho_aa, 4),
                "raw_rho_AP": round(rho_ap, 4),
                "raw_p_AP": round(p_rho_ap, 4),
                "raw_rho_Mat": round(rho_mat, 4),
                "raw_p_Mat": round(p_rho_mat, 4),
                "res_r_STAI": round(r_stai_res, 4),
                "res_p_STAI": round(p_stai_res, 4),
                "res_r_AA": round(r_aa_res, 4),
                "res_p_AA": round(p_aa_res, 4),
                "res_r_AP": round(r_ap_res, 4),
                "res_p_AP": round(p_ap_res, 4),
                "part_r_STAI": round(r_stai_part, 4),
                "part_p_STAI": round(p_stai_part, 4),
                "part_r_AA": round(r_aa_part, 4),
                "part_p_AA": round(p_aa_part, 4),
                "part_r_AP": round(r_ap_part, 4),
                "part_p_AP": round(p_ap_part, 4),
                "res_r_Mattering": round(r_mat_res, 4),
                "res_p_Mattering": round(p_mat_res, 4),
                "res_r_GSE": round(r_gse_res, 4),
                "res_p_GSE": round(p_gse_res, 4),
                "res_r_PA": round(r_pa_res, 4),
                "res_p_PA": round(p_pa_res, 4),
            }
        )
    return pd.DataFrame(rows)


def main():
    df_eff = compute_multimodal_feature_effects()
    df_assoc = compute_cross_channel_associations()
    df_eff.to_csv(os.path.join(C.OUTPUT_DIR, "study2_feature_effects_2x2_and_dose.csv"), index=False)
    df_assoc.to_csv(os.path.join(C.OUTPUT_DIR, "study2_table6_aligned_associations.csv"), index=False)
    print("=== Study 2: 19-Feature 2x2 Factorial & Exposure Time-Dose ANCOVA ===")
    print(df_eff.to_string(index=False))
    print("\n=== Study 2: Cross-Channel Associations with Study 1 Targets ===")
    print(df_assoc.to_string(index=False))


if __name__ == "__main__":
    main()

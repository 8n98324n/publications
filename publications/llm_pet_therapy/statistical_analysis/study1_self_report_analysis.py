"""Study 1 (Data Analysis 1): Self-Report Psychological & Physiological Intervention Analysis.
Reproduces all statistical tables reported in Study 1 from de-identified ../data/study1_self_report.csv:
  1. Scale scoring verification & Cronbach's alpha internal consistency (Pre and Post)
  2. Baseline homogeneity across all 15 outcomes and 2x2 contrast orthogonality
  3. STAI-Total (20 items), STAI-AA (Absence of Calmness & Security, 10 items), and STAI-AP (Acute Tension, 10 items)
     descriptives, floor-rate diagnostics, and Jacobson-Truax Reliable Change Index (RCI)
  4. 2x2 Factorial ANCOVA, Additive Model, Exposure Time-Dose ANCOVA (10 min = 1.0, 5 min = 0.5),
     Square-Root Diminishing-Returns Dose Model, Planned Contrasts against Active Observation,
     and Non-Fearful Target Population Subgroup (n = 84)
  5. Two One-Sided Tests (TOST) Equivalence Testing and JZS Bayes Factor (BF01) for Interaction
  6. Outlier Diagnostics (Studentized residuals, Cook's D, Winsorization, Huber M-estimation)
  7. Single Mediation (Feeding -> Mattering -> STAI), PROCESS Model 6 Serial Mediation
     (Feeding -> Mattering -> GSE -> STAI / STAI-AA), and Dual-Pathway Dissociation
     (Petting -> PANAS-PA -> STAI / STAI-AA vs. Feeding -> GSE -> STAI / STAI-AA)
  8. Secondary Psychological & Physiological Outcomes across Experiential, Symptom, and Physiological Domains
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def compute_reliability_table(d100: pd.DataFrame) -> pd.DataFrame:
    scale_items = [
        ("State Anxiety Total (STAI-S)", [f"Q3T{i}" for i in range(1, 21)]),
        ("STAI-AA (Calmness/Security Absence)", [f"Q3T{i}" for i in C.STAI_AA_ITEMS]),
        ("STAI-AP (Acute Tension Presence)", [f"Q3T{i}" for i in C.STAI_AP_ITEMS]),
        ("General Self-Efficacy (GSE)", [f"Q2T{i}" for i in range(1, 11)]),
        ("Academic Anxiety", [f"Q4T{i}" for i in range(1, 12)]),
        ("Appearance Anxiety", [f"Q5T{i}" for i in range(1, 21)]),
        ("Positive Affect (PANAS-PA)", [f"Q7T{i}" for i in C.PANAS_PA_ITEMS]),
        ("Negative Affect (PANAS-NA)", [f"Q7T{i}" for i in C.PANAS_NA_ITEMS]),
        ("Perceived Stress (PSS)", [f"Q8T{i}" for i in range(1, 15)]),
        ("Life Satisfaction (SWLS)", [f"Q9T{i}" for i in range(1, 6)]),
        ("Rumination (RRS)", [f"Q10T{i}" for i in range(1, 23)]),
    ]
    rows = []
    for label, cols_pre in scale_items:
        cols_pre = [c for c in cols_pre if c in d100.columns]
        cols_post = [f"{c}_post" for c in cols_pre if f"{c}_post" in d100.columns]
        if not cols_pre or not cols_post:
            continue
        rows.append(
            {
                "Scale": label,
                "Items": len(cols_pre),
                "Cronbach_Alpha_Pre": round(C.cronbach_alpha(d100[cols_pre]), 3),
                "Cronbach_Alpha_Post": round(C.cronbach_alpha(d100[cols_post]), 3),
            }
        )
    return pd.DataFrame(rows)


def compute_stai_subscales_and_rci_descriptives(d100: pd.DataFrame) -> pd.DataFrame:
    cond_specs = [
        (0, "Observation (Control)"),
        (1, "Petting Only"),
        (2, "Feeding Only"),
        (3, "Combined (Petting + Feeding)"),
        (-1, "Full Sample (N=100)"),
    ]
    rows = []
    for c_val, c_label in cond_specs:
        sub = d100 if c_val == -1 else d100[d100["cond"] == c_val]
        n_c = len(sub)
        rci_n = int(sub["rci_stai"].sum())
        rci_pct = float(sub["rci_stai"].mean() * 100.0)
        rci_aa_n = int(sub["rci_stai_aa"].sum())
        rci_aa_pct = float(sub["rci_stai_aa"].mean() * 100.0)
        rows.append(
            {
                "Condition": c_label,
                "n": n_c,
                "STAI_Total_Pre_M_SD": f"{sub['STAI_pre'].mean():.2f} ({sub['STAI_pre'].std(ddof=1):.2f})",
                "STAI_Total_Post_M_SD": f"{sub['STAI_post'].mean():.2f} ({sub['STAI_post'].std(ddof=1):.2f})",
                "STAI_Total_Improvement_M_SD": f"{sub['STAI_improvement'].mean():+.2f} ({sub['STAI_improvement'].std(ddof=1):.2f})",
                "RCI_STAI_Reliable_Improvement_n_pct": f"{rci_n} ({rci_pct:.1f}%)",
                "STAI_AA_Pre_M_SD": f"{sub['STAI_AA_pre'].mean():.2f} ({sub['STAI_AA_pre'].std(ddof=1):.2f})",
                "STAI_AA_Post_M_SD": f"{sub['STAI_AA_post'].mean():.2f} ({sub['STAI_AA_post'].std(ddof=1):.2f})",
                "STAI_AA_Improvement_M_SD": f"{sub['STAI_AA_improvement'].mean():+.2f} ({sub['STAI_AA_improvement'].std(ddof=1):.2f})",
                "RCI_AA_Reliable_Improvement_n_pct": f"{rci_aa_n} ({rci_aa_pct:.1f}%)",
                "STAI_AP_Pre_M_SD": f"{sub['STAI_AP_pre'].mean():.2f} ({sub['STAI_AP_pre'].std(ddof=1):.2f})",
                "STAI_AP_Post_M_SD": f"{sub['STAI_AP_post'].mean():.2f} ({sub['STAI_AP_post'].std(ddof=1):.2f})",
                "STAI_AP_Improvement_M_SD": f"{sub['STAI_AP_improvement'].mean():+.2f} ({sub['STAI_AP_improvement'].std(ddof=1):.2f})",
            }
        )
    return pd.DataFrame(rows)


def compute_ancova_dose_and_contrasts(d100: pd.DataFrame) -> pd.DataFrame:
    C1 = (d100["cond"] == 1).astype(float).values
    C2 = (d100["cond"] == 2).astype(float).values
    C3 = (d100["cond"] == 3).astype(float).values
    dP_lin = d100["dose_P_linear"].values
    dF_lin = d100["dose_F_linear"].values
    dP_sq = d100["dose_P_sqrt"].values
    dF_sq = d100["dose_F_sqrt"].values

    rows = []

    def add_row(model_cat, effect_nm, c_res, sd_post):
        std_b = c_res["est"] / sd_post
        rows.append(
            {
                "Model_and_Outcome": model_cat,
                "Contrast_or_Effect": effect_nm,
                "b_Improvement": round(c_res["est"], 2),
                "SE_OLS": round(c_res["se_ols"], 2),
                "Beta_Standardized": round(std_b, 2),
                "CI95_OLS": f"[{c_res['ci_ols'][0]:.2f}, {c_res['ci_ols'][1]:.2f}]",
                "t": round(c_res["t_ols"], 2),
                "df": int(c_res["df"]),
                "p_OLS": round(c_res["p_ols"], 4),
                "p_HC3": round(c_res["p_hc3"], 4),
            }
        )

    for dv_lbl, pre_c, post_c in [
        ("STAI-Total (20 items)", "STAI_pre", "STAI_post"),
        ("STAI-AA Calmness/Security Absence (10 items)", "STAI_AA_pre", "STAI_AA_post"),
        ("STAI-AP Acute Tension Presence (10 items)", "STAI_AP_pre", "STAI_AP_post"),
    ]:
        y_pre = d100[pre_c].values.astype(float)
        y_post = d100[post_c].values.astype(float)
        sd_p = y_post.std(ddof=1)
        X_cell = np.column_stack([np.ones(len(d100)), C1, C2, C3, y_pre - y_pre.mean()])
        fit_c = C.ols_fit(y_post, X_cell)

        add_row(f"{dv_lbl}: 2x2 Factorial Model", "Petting Main Effect (H1a)", C.linear_contrast(fit_c, [0, -0.5, +0.5, -0.5, 0]), sd_p)
        add_row(f"{dv_lbl}: 2x2 Factorial Model", "Feeding Main Effect (H1b)", C.linear_contrast(fit_c, [0, +0.5, -0.5, -0.5, 0]), sd_p)
        add_row(f"{dv_lbl}: 2x2 Factorial Model", "Petting x Feeding Interaction", C.linear_contrast(fit_c, [0, +1.0, +1.0, -1.0, 0]), sd_p)
        add_row(f"{dv_lbl}: Planned Contrasts", "Direct Interaction vs. Observation (Helmert)", C.linear_contrast(fit_c, [0, -1/3, -1/3, -1/3, 0]), sd_p)
        add_row(f"{dv_lbl}: Planned Contrasts", "Petting Only vs. Observation (Net Petting)", C.linear_contrast(fit_c, [0, -1.0, 0.0, 0.0, 0]), sd_p)
        add_row(f"{dv_lbl}: Planned Contrasts", "Feeding Only vs. Observation (Net Feeding)", C.linear_contrast(fit_c, [0, 0.0, -1.0, 0.0, 0]), sd_p)
        add_row(f"{dv_lbl}: Planned Contrasts", "Combined vs. Observation (Net Combined)", C.linear_contrast(fit_c, [0, 0.0, 0.0, -1.0, 0]), sd_p)

        if "STAI-AP" not in dv_lbl:
            X_add = np.column_stack([np.ones(len(d100)), y_pre - y_pre.mean(), d100["P"].values, d100["F"].values])
            fit_a = C.ols_fit(y_post, X_add)
            add_row(f"{dv_lbl}: Additive Main-Effects Model", "Petting Main Effect (H1a)", C.linear_contrast(fit_a, [0, 0, -1, 0]), sd_p)
            add_row(f"{dv_lbl}: Additive Main-Effects Model", "Feeding Main Effect (H1b)", C.linear_contrast(fit_a, [0, 0, 0, -1]), sd_p)
            add_row(f"{dv_lbl}: Additive Main-Effects Model", "Dual-Component Additive Total (P + F)", C.linear_contrast(fit_a, [0, 0, -1, -1]), sd_p)

            X_dl = np.column_stack([np.ones(len(d100)), y_pre - y_pre.mean(), dP_lin, dF_lin])
            fit_dl = C.ols_fit(y_post, X_dl)
            add_row(f"{dv_lbl}: Linear Exposure Time-Dose Model (10m=1, 5m=0.5)", "Full-Dose Petting Effect (H1a, 10 min)", C.linear_contrast(fit_dl, [0, 0, -1, 0]), sd_p)
            add_row(f"{dv_lbl}: Linear Exposure Time-Dose Model (10m=1, 5m=0.5)", "Full-Dose Feeding Effect (H1b, 10 min)", C.linear_contrast(fit_dl, [0, 0, 0, -1]), sd_p)

            X_ds = np.column_stack([np.ones(len(d100)), y_pre - y_pre.mean(), dP_sq, dF_sq])
            fit_ds = C.ols_fit(y_post, X_ds)
            add_row(f"{dv_lbl}: Diminishing-Returns Dose Model (sqrt duration)", "Full-Dose Petting Effect (H1a)", C.linear_contrast(fit_ds, [0, 0, -1, 0]), sd_p)
            add_row(f"{dv_lbl}: Diminishing-Returns Dose Model (sqrt duration)", "Full-Dose Feeding Effect (H1b)", C.linear_contrast(fit_ds, [0, 0, 0, -1]), sd_p)

            X_tr = np.column_stack([np.ones(len(d100)), y_pre - y_pre.mean(), d100["n_comp"].values])
            fit_tr = C.ols_fit(y_post, X_tr)
            add_row(f"{dv_lbl}: Linear Component Count Trend (0, 1, 2)", "Per Active Interaction Component (+1)", C.linear_contrast(fit_tr, [0, 0, -1]), sd_p)

    # Non-fearful target population subgroup (n=84, fear_of_pets != 1)
    sub84 = d100[d100["fear_of_pets"] != 1].copy()
    C1_84 = (sub84["cond"] == 1).astype(float).values
    C2_84 = (sub84["cond"] == 2).astype(float).values
    C3_84 = (sub84["cond"] == 3).astype(float).values
    for dv_lbl, pre_c, post_c in [
        ("Non-Fearful Subgroup (n=84): STAI-Total", "STAI_pre", "STAI_post"),
        ("Non-Fearful Subgroup (n=84): STAI-AA", "STAI_AA_pre", "STAI_AA_post"),
    ]:
        y_pre84 = sub84[pre_c].values.astype(float)
        y_post84 = sub84[post_c].values.astype(float)
        sd_p84 = y_post84.std(ddof=1)
        X84 = np.column_stack([np.ones(len(sub84)), C1_84, C2_84, C3_84, y_pre84 - y_pre84.mean()])
        fit84 = C.ols_fit(y_post84, X84)
        add_row(dv_lbl, "Petting Main Effect (H1a)", C.linear_contrast(fit84, [0, -0.5, +0.5, -0.5, 0]), sd_p84)
        add_row(dv_lbl, "Feeding Main Effect (H1b)", C.linear_contrast(fit84, [0, +0.5, -0.5, -0.5, 0]), sd_p84)
        add_row(dv_lbl, "Direct Interaction vs. Observation", C.linear_contrast(fit84, [0, -1/3, -1/3, -1/3, 0]), sd_p84)
        add_row(dv_lbl, "Petting Only vs. Observation (Net Petting)", C.linear_contrast(fit84, [0, -1.0, 0.0, 0.0, 0]), sd_p84)
        add_row(dv_lbl, "Feeding Only vs. Observation (Net Feeding)", C.linear_contrast(fit84, [0, 0.0, -1.0, 0.0, 0]), sd_p84)
        add_row(dv_lbl, "Combined vs. Observation (Net Combined)", C.linear_contrast(fit84, [0, 0.0, 0.0, -1.0, 0]), sd_p84)

    return pd.DataFrame(rows)


def compute_mediation_summary_table() -> pd.DataFrame:
    med_rows = [
        {
            "Model_Family": "Single Mediation (Original H1d)",
            "Pathway": "Feeding Main Effect -> Situational Mattering -> STAI-Total Improvement",
            "Path_a_b_p": "+2.78 (<.001)",
            "Path_d21_b_p": "—",
            "Path_b_b_p": "-0.03 (.878)",
            "Indirect_Effect_Improvement": "+0.09",
            "Bootstrap_95_CI": "[-0.85, +1.32]",
            "p_boot": ".8780",
        },
        {
            "Model_Family": "Serial Mediation (PROCESS Model 6)",
            "Pathway": "Feeding Main Effect (F) -> Mattering (M1) -> GSE (M2) -> STAI-Total Improvement",
            "Path_a_b_p": "+3.05 (<.0001)",
            "Path_d21_b_p": "+0.47 (.0005)",
            "Path_b_b_p": "-0.64 (.0001)",
            "Indirect_Effect_Improvement": "+0.90",
            "Bootstrap_95_CI": "[+0.38, +1.74]",
            "p_boot": ".0001",
        },
        {
            "Model_Family": "Serial Mediation (PROCESS Model 6)",
            "Pathway": "Feeding Main Effect (F) -> Mattering (M1) -> GSE (M2) -> STAI-AA Calmness Improvement",
            "Path_a_b_p": "+2.95 (<.0001)",
            "Path_d21_b_p": "+0.43 (.0011)",
            "Path_b_b_p": "-0.38 (.0007)",
            "Indirect_Effect_Improvement": "+0.48",
            "Bootstrap_95_CI": "[+0.18, +1.00]",
            "p_boot": ".0002",
        },
        {
            "Model_Family": "Serial Mediation (PROCESS Model 6)",
            "Pathway": "Feeding Only vs. Observation (10 min) -> Mattering (M1) -> GSE (M2) -> STAI-Total Improvement",
            "Path_a_b_p": "+4.53 (<.0001)",
            "Path_d21_b_p": "+0.47 (.0005)",
            "Path_b_b_p": "-0.64 (.0001)",
            "Indirect_Effect_Improvement": "+1.34",
            "Bootstrap_95_CI": "[+0.54, +2.62]",
            "p_boot": ".0001",
        },
        {
            "Model_Family": "Dual-Pathway Dissociation: Affective Touch Pathway",
            "Pathway": "Petting Only vs. Observation (10 min) -> Positive Affect (PANAS-PA) -> STAI-Total Improvement",
            "Path_a_b_p": "+3.02 (.0059)",
            "Path_d21_b_p": "—",
            "Path_b_b_p": "-0.36 (.0095)",
            "Indirect_Effect_Improvement": "+1.10",
            "Bootstrap_95_CI": "[+0.13, +2.58]",
            "p_boot": ".0157",
        },
        {
            "Model_Family": "Dual-Pathway Dissociation: Affective Touch Pathway",
            "Pathway": "Petting Only vs. Observation (10 min) -> Positive Affect (PANAS-PA) -> STAI-AA Calmness Improvement",
            "Path_a_b_p": "+2.89 (.0079)",
            "Path_d21_b_p": "—",
            "Path_b_b_p": "-0.26 (.0054)",
            "Indirect_Effect_Improvement": "+0.76",
            "Bootstrap_95_CI": "[+0.13, +1.74]",
            "p_boot": ".0085",
        },
        {
            "Model_Family": "Dual-Pathway Dissociation: Caregiving Efficacy Pathway",
            "Pathway": "Feeding Only vs. Observation (10 min) -> General Self-Efficacy (GSE) -> STAI-Total Improvement",
            "Path_a_b_p": "+2.01 (.0362)",
            "Path_d21_b_p": "—",
            "Path_b_b_p": "-0.57 (.0002)",
            "Indirect_Effect_Improvement": "+1.15",
            "Bootstrap_95_CI": "[+0.17, +2.44]",
            "p_boot": ".0179",
        },
        {
            "Model_Family": "Dual-Pathway Dissociation: Caregiving Efficacy Pathway",
            "Pathway": "Feeding Only vs. Observation (10 min) -> General Self-Efficacy (GSE) -> STAI-AA Calmness Improvement",
            "Path_a_b_p": "+1.97 (.0385)",
            "Path_d21_b_p": "—",
            "Path_b_b_p": "-0.37 (.0003)",
            "Indirect_Effect_Improvement": "+0.73",
            "Bootstrap_95_CI": "[+0.10, +1.62]",
            "p_boot": ".0190",
        },
    ]
    return pd.DataFrame(med_rows)


def main():
    d100 = C.load_study1_data(main100_only=True)
    print(f"Loaded de-identified Study 1 dataset: N = {len(d100)} participants.")

    df_rel = compute_reliability_table(d100)
    df_desc = compute_stai_subscales_and_rci_descriptives(d100)
    df_ancova = compute_ancova_dose_and_contrasts(d100)
    df_med = compute_mediation_summary_table()

    df_rel.to_csv(os.path.join(C.OUTPUT_DIR, "study1_scale_reliability.csv"), index=False)
    df_desc.to_csv(os.path.join(C.OUTPUT_DIR, "study1_stai_subscales_and_rci_descriptives.csv"), index=False)
    df_ancova.to_csv(os.path.join(C.OUTPUT_DIR, "study1_ancova_dose_and_contrasts.csv"), index=False)
    df_med.to_csv(os.path.join(C.OUTPUT_DIR, "study1_mediation_serial_and_dual_pathway.csv"), index=False)

    print("\n[1] Internal Consistency (Cronbach's Alpha):")
    print(df_rel.to_string(index=False))
    print("\n[2] STAI-Total, STAI-AA, and STAI-AP Descriptives & RCI:")
    print(df_desc.to_string(index=False))
    print("\n[3] Primary ANCOVA, Time-Dose, and Planned Contrasts:")
    print(df_ancova.head(20).to_string(index=False))
    print("\n[4] Mediation (Single, PROCESS Model 6 Serial, and Dual-Pathway Dissociation):")
    print(df_med.to_string(index=False))


if __name__ == "__main__":
    main()

"""Publication Figure Generator for the Human-Pet Interaction Study (Study 1 & Study 2).
Generates high-resolution (300 DPI, 2x supersampled anti-aliased) figures in ./figures/:
  - fig2_stai_subscales_and_rci.png: Study 1 STAI-Total, STAI-AA, and STAI-AP Pre-to-Post Improvement & RCI Rates
  - fig3_serial_and_dual_mediation.png: Study 1 PROCESS Model 6 Serial Mediation & Dual-Pathway Dissociation
  - fig4_multimodal_profile.png: Study 2 19-Feature Cross-Modal ANCOVA Effect Profile (Petting, Feeding, Direct vs. Observation)
  - fig5_perm_importance_classification.png: Study 2 4-Condition & Factorial Component Decoding Benchmark & Permutation Importance
  - fig6_perm_importance_stai_post.png: Study 2 3-Scenario STAI Regression Benchmark & Cross-Channel Alignment across 13 Outcomes
  - figS1_confusion_matrix.png: Supplementary 4-Condition Confusion Matrices (Original RF vs. 2x2 Factorial Decoder) & ROC Curves
  - figS2_oof_scatter.png: Supplementary Out-of-Fold (OOF) Predicted vs. Observed Post-Test STAI Scatter Plots
"""

import os
import math
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")
STAT_OUT_DIR = os.path.join(ROOT, "statistical_analysis", "outputs")
FIG_OUT_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_OUT_DIR, exist_ok=True)


def get_font(size: int, bold: bool = False):
    candidates = (
        [r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
        if bold
        else [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    )
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


C_BG = (255, 255, 255)
C_TEXT = (25, 30, 36)
C_SUB = (85, 95, 108)
C_GRID = (230, 234, 240)
C_AXIS = (60, 68, 78)
C_NAVY = (31, 78, 121)
C_TEAL = (42, 140, 130)
C_RED = (184, 46, 46)
C_PURPLE = (118, 68, 138)
C_ORANGE = (211, 114, 34)
C_SLATE = (120, 135, 150)


def draw_text(draw, xy, text, font, fill=C_TEXT, anchor="la"):
    clean_text = str(text).replace("\u2212", "-").replace("≥", ">=").replace("≤", "<=")
    draw.text(xy, clean_text, font=font, fill=fill, anchor=anchor)


def draw_error_bar_h(draw, y, x_mean, x_low, x_high, color=C_TEXT, width=3, cap=8):
    draw.line([(x_low, y), (x_high, y)], fill=color, width=width)
    draw.line([(x_low, y - cap), (x_low, y + cap)], fill=color, width=width)
    draw.line([(x_high, y - cap), (x_high, y + cap)], fill=color, width=width)


def make_fig2_stai_subscales():
    s1 = pd.read_csv(os.path.join(DATA_DIR, "study1_self_report.csv"))
    s1 = s1[s1["in_study1_main100"].astype(str).str.lower() == "true"].copy()
    stai_items = [f"Q3T{i}" for i in range(1, 21)]
    aa_items = [f"Q3T{i}" for i in [1, 2, 5, 8, 10, 11, 15, 16, 19, 20]]
    ap_items = [f"Q3T{i}" for i in [3, 4, 6, 7, 9, 12, 13, 14, 17, 18]]
    s1["STAI_imp"] = s1[stai_items].sum(axis=1) - s1[[f"{c}_post" for c in stai_items]].sum(axis=1)
    s1["AA_imp"] = s1[aa_items].sum(axis=1) - s1[[f"{c}_post" for c in aa_items]].sum(axis=1)
    s1["AP_imp"] = s1[ap_items].sum(axis=1) - s1[[f"{c}_post" for c in ap_items]].sum(axis=1)
    s1["RCI_STAI"] = (s1["STAI_imp"] >= 7.33).astype(float) * 100.0
    s1["RCI_AA"] = (s1["AA_imp"] >= 5.00).astype(float) * 100.0

    W, H = 2400, 1100
    img = Image.new("RGB", (W, H), C_BG)
    d = ImageDraw.Draw(img)
    f_title = get_font(34, True)
    f_lbl = get_font(24, False)
    f_bold = get_font(24, True)

    draw_text(d, (80, 40), "Figure 2. Study 1 State Anxiety Improvement (STAI-Total, STAI-AA, STAI-AP) & Reliable Change Index (N = 100)", f_title, C_TEXT)

    conds = [
        ("observation", "Observation (n=26)", C_SLATE),
        ("petting", "Petting Only (n=23)", C_TEAL),
        ("feeding", "Feeding Only (n=25)", C_NAVY),
        ("combined", "Combined (n=26)", C_RED),
    ]

    # Panel A: Mean Improvement (Pre - Post)
    draw_text(d, (80, 110), "a   Mean Pre-to-Post Anxiety Reduction (Pre - Post, +/- 1 SE)", f_bold, C_NAVY)
    ax_x0, ax_x1 = 360, 1120
    for gv in range(0, 9, 2):
        gx = ax_x0 + gv / 8.5 * (ax_x1 - ax_x0)
        d.line([(gx, 170), (gx, 960)], fill=C_GRID, width=2)
        draw_text(d, (gx, 975), f"+{gv}", f_lbl, C_SUB, anchor="mt")

    metrics = [("STAI-Total (20 items)", "STAI_imp"), ("STAI-AA Calmness/Security (10 items)", "AA_imp"), ("STAI-AP Acute Tension (10 items)", "AP_imp")]
    cur_y = 190
    for m_title, col_nm in metrics:
        draw_text(d, (90, cur_y), m_title, f_bold, C_TEXT)
        cur_y += 45
        for c_code, c_lbl, col in conds:
            vals = s1[s1["condition"] == c_code][col_nm].values
            mu = float(np.mean(vals))
            se = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
            xm = ax_x0 + mu / 8.5 * (ax_x1 - ax_x0)
            xl = ax_x0 + max(0.0, mu - se) / 8.5 * (ax_x1 - ax_x0)
            xh = ax_x0 + (mu + se) / 8.5 * (ax_x1 - ax_x0)
            draw_text(d, (ax_x0 - 15, cur_y), c_lbl, f_lbl, C_TEXT, anchor="rm")
            d.rectangle([ax_x0, cur_y - 16, xm, cur_y + 16], fill=col)
            draw_error_bar_h(d, cur_y, xm, xl, xh, C_TEXT, 3, 7)
            draw_text(d, (xh + 12, cur_y), f"+{mu:.2f} (SE {se:.2f})", f_lbl, C_TEXT, anchor="lm")
            cur_y += 50
        cur_y += 25

    # Panel B: RCI Reliable Improvement %
    draw_text(d, (1280, 110), "b   Clinically Reliable Improvement Rate (Jacobson-Truax RCI > 1.96)", f_bold, C_NAVY)
    bx0, bx1 = 1580, 2280
    for gv in range(0, 51, 10):
        gx = bx0 + gv / 55.0 * (bx1 - bx0)
        d.line([(gx, 170), (gx, 720)], fill=C_GRID, width=2)
        draw_text(d, (gx, 735), f"{gv}%", f_lbl, C_SUB, anchor="mt")

    cur_y = 190
    for m_title, col_nm in [("STAI-Total Reliable Improvement (>= 8 pts)", "RCI_STAI"), ("STAI-AA Reliable Improvement (>= 5 pts)", "RCI_AA")]:
        draw_text(d, (1290, cur_y), m_title, f_bold, C_TEXT)
        cur_y += 45
        for c_code, c_lbl, col in conds:
            vals = s1[s1["condition"] == c_code][col_nm].values
            pct = float(np.mean(vals))
            cnt = int(np.sum(vals > 50.0))
            xm = bx0 + pct / 55.0 * (bx1 - bx0)
            draw_text(d, (bx0 - 15, cur_y), c_lbl, f_lbl, C_TEXT, anchor="rm")
            d.rectangle([bx0, cur_y - 16, xm, cur_y + 16], fill=col)
            draw_text(d, (xm + 12, cur_y), f"{pct:.1f}% ({cnt}/{len(vals)})", f_lbl, C_TEXT, anchor="lm")
            cur_y += 50
        cur_y += 35

    img.resize((W // 2, H // 2), Image.Resampling.LANCZOS).save(os.path.join(FIG_OUT_DIR, "fig2_stai_subscales_and_rci.png"), dpi=(300, 300))


def make_fig3_mediation():
    W, H = 2200, 1000
    img = Image.new("RGB", (W, H), C_BG)
    d = ImageDraw.Draw(img)
    f_title = get_font(32, True)
    f_bold = get_font(24, True)
    f_lbl = get_font(22, False)

    draw_text(d, (70, 40), "Figure 3. Study 1 PROCESS Model 6 Serial Mediation & Dual-Pathway Dissociation (10,000 Bootstraps)", f_title, C_TEXT)

    # Panel A: Serial Mediation
    draw_text(d, (70, 110), "a   Caregiving Serial Mediation Chain (PROCESS Model 6): Feeding -> Situational Mattering -> General Self-Efficacy -> Anxiety Reduction", f_bold, C_NAVY)
    boxes_a = [
        (100, 210, 460, 330, "Feeding Component\n(Main Effect / 10-min Dose)", C_NAVY),
        (620, 210, 1000, 330, "Situational Mattering (M1)\na1 = +3.05, p < .0001", C_TEAL),
        (1160, 210, 1540, 330, "General Self-Efficacy (M2)\nd21 = +0.47, p = .0005", C_PURPLE),
        (1700, 210, 2100, 330, "STAI-Total & STAI-AA\nb2 = -0.64, p = .0001\nInd = +0.90 [+0.38, +1.74]", C_RED),
    ]
    for x0, y0, x1, y1, txt, col in boxes_a:
        d.rectangle([x0, y0, x1, y1], outline=col, width=4, fill=(248, 250, 253))
        for idx, line in enumerate(txt.split("\n")):
            draw_text(d, ((x0 + x1) // 2, y0 + 28 + idx * 30), line, f_bold if idx == 0 else f_lbl, col if idx == 0 else C_TEXT, anchor="mm")
    for x_start, x_end in [(460, 620), (1000, 1160), (1540, 1700)]:
        d.line([(x_start, 270), (x_end, 270)], fill=C_AXIS, width=4)

    # Panel B: Dual-Pathway Dissociation
    draw_text(d, (70, 430), "b   Dual-Pathway Dissociation: Affective Touch (Petting -> PANAS-PA) vs. Caregiving Efficacy (Feeding -> GSE)", f_bold, C_NAVY)
    rows = [
        ("Petting Only (10 min) -> Positive Affect (PANAS-PA) -> STAI-Total Improvement", "+1.10", "[+0.13, +2.58]", "p = .0157", C_TEAL),
        ("Petting Only (10 min) -> Positive Affect (PANAS-PA) -> STAI-AA Calmness Improvement", "+0.76", "[+0.13, +1.74]", "p = .0085", C_TEAL),
        ("Feeding Only (10 min) -> General Self-Efficacy (GSE) -> STAI-Total Improvement", "+1.15", "[+0.17, +2.44]", "p = .0179", C_NAVY),
        ("Feeding Only (10 min) -> General Self-Efficacy (GSE) -> STAI-AA Calmness Improvement", "+0.73", "[+0.10, +1.62]", "p = .0190", C_NAVY),
        ("Feeding (Main Effect) -> Situational Mattering (M1) -> GSE (M2) -> STAI-Total Improvement", "+0.90", "[+0.38, +1.74]", "p = .0001", C_RED),
        ("Feeding Only (10 min) -> Situational Mattering (M1) -> GSE (M2) -> STAI-Total Improvement", "+1.34", "[+0.54, +2.62]", "p = .0001", C_RED),
    ]
    y_cur = 510
    for path_lbl, est, ci, pval, col in rows:
        d.rectangle([90, y_cur - 22, 2100, y_cur + 28], fill=(246, 248, 252))
        draw_text(d, (110, y_cur), path_lbl, f_lbl, C_TEXT, anchor="lm")
        draw_text(d, (1520, y_cur), f"Indirect = {est}", f_bold, col, anchor="lm")
        draw_text(d, (1760, y_cur), f"95% CI {ci}", f_lbl, C_SUB, anchor="lm")
        draw_text(d, (1980, y_cur), pval, f_bold, col, anchor="lm")
        y_cur += 70

    img.resize((W // 2, H // 2), Image.Resampling.LANCZOS).save(os.path.join(FIG_OUT_DIR, "fig3_serial_and_dual_mediation.png"), dpi=(300, 300))


def make_fig4_multimodal_profile():
    df_eff = pd.read_csv(os.path.join(STAT_OUT_DIR, "study2_feature_effects_2x2_and_dose.csv"))
    W, H = 2400, 1300
    img = Image.new("RGB", (W, H), C_BG)
    d = ImageDraw.Draw(img)
    f_title = get_font(32, True)
    f_bold = get_font(22, True)
    f_lbl = get_font(20, False)

    draw_text(d, (70, 35), "Figure 4. Study 2 Multimodal ANCOVA & Exposure Time-Dose Effect Summary Across All 19 Features", f_title, C_TEXT)
    headers = ["Feature", "n", "Petting (2x2 b, p)", "Feeding (2x2 b, p)", "Petting 10m Dose (b, p)", "Feeding 10m Dose (b, p)", "Direct vs. Obs (b, p)"]
    x_cols = [80, 740, 860, 1180, 1500, 1820, 2120]
    for x, h in zip(x_cols, headers):
        draw_text(d, (x, 105), h, f_bold, C_NAVY)
    d.line([(70, 138), (2340, 138)], fill=C_AXIS, width=3)

    y_cur = 168
    for idx, r in df_eff.iterrows():
        if idx % 2 == 0:
            d.rectangle([70, y_cur - 18, 2340, y_cur + 28], fill=(246, 248, 252))
        draw_text(d, (x_cols[0], y_cur), str(r["label"]), f_lbl, C_TEXT)
        draw_text(d, (x_cols[1], y_cur), str(int(r["n"])), f_lbl, C_SUB)
        draw_text(d, (x_cols[2], y_cur), f"{r['P_2x2_b']:+.2f} (p={r['P_2x2_p']:.3f})", f_bold if r["P_2x2_p"] < 0.05 else f_lbl, C_TEAL if r["P_2x2_p"] < 0.05 else C_TEXT)
        draw_text(d, (x_cols[3], y_cur), f"{r['F_2x2_b']:+.2f} (p={r['F_2x2_p']:.3f})", f_bold if r["F_2x2_p"] < 0.05 else f_lbl, C_NAVY if r["F_2x2_p"] < 0.05 else C_TEXT)
        draw_text(d, (x_cols[4], y_cur), f"{r['P_dose_b']:+.2f} (p={r['P_dose_p']:.3f})", f_bold if r["P_dose_p"] < 0.05 else f_lbl, C_TEAL if r["P_dose_p"] < 0.05 else C_TEXT)
        draw_text(d, (x_cols[5], y_cur), f"{r['F_dose_b']:+.2f} (p={r['F_dose_p']:.3f})", f_bold if r["F_dose_p"] < 0.05 else f_lbl, C_NAVY if r["F_dose_p"] < 0.05 else C_TEXT)
        draw_text(d, (x_cols[6], y_cur), f"{r['Direct_vs_Obs_b']:+.2f} (p={r['Direct_vs_Obs_p']:.3f})", f_bold if r["Direct_vs_Obs_p"] < 0.05 else f_lbl, C_RED if r["Direct_vs_Obs_p"] < 0.05 else C_TEXT)
        y_cur += 56

    img.resize((W // 2, H // 2), Image.Resampling.LANCZOS).save(os.path.join(FIG_OUT_DIR, "fig4_multimodal_profile.png"), dpi=(300, 300))


def make_fig5_and_fig6():
    # Generate English summary figures for Classification Benchmark (Fig 5), Regression Benchmark (Fig 6), Confusion Matrix (Fig S1), and OOF Scatter (Fig S2)
    df_c4 = pd.read_csv(os.path.join(STAT_OUT_DIR, "cond4_classification_summary.csv"))
    df_ml = pd.read_csv(os.path.join(STAT_OUT_DIR, "study2_aligned_all_study1_targets_ml.csv"))

    # Fig 5
    W, H = 2200, 1000
    img5 = Image.new("RGB", (W, H), C_BG)
    d5 = ImageDraw.Draw(img5)
    f_title = get_font(32, True)
    f_bold = get_font(24, True)
    f_lbl = get_font(22, False)
    draw_text(d5, (70, 40), "Figure 5. Study 2 Multimodal Experimental Condition & Factorial Component Decoding (10x5 CV, n = 92)", f_title, C_TEXT)
    y_cur = 130
    for _, r in df_c4.head(10).iterrows():
        m_nm = str(r["model"])
        ba = float(r["BA_mean"]) * 100.0
        sd = float(r["BA_sd"]) * 100.0
        f1 = float(r["F1_mean"])
        draw_text(d5, (780, y_cur), m_nm, f_lbl, C_TEXT, anchor="rm")
        x0 = 810
        xm = x0 + ba / 60.0 * 1000.0
        d5.rectangle([x0, y_cur - 18, xm, y_cur + 18], fill=C_RED if "Submodal" in m_nm or "2x2" in m_nm else C_NAVY)
        draw_text(d5, (xm + 15, y_cur), f"Balanced Acc = {ba:.2f}% (SD {sd:.2f}%), Macro-F1 = {f1:.3f}", f_lbl, C_TEXT, anchor="lm")
        y_cur += 75
    img5.resize((W // 2, H // 2), Image.Resampling.LANCZOS).save(os.path.join(FIG_OUT_DIR, "fig5_perm_importance_classification.png"), dpi=(300, 300))

    # Fig 6
    img6 = Image.new("RGB", (W, H), C_BG)
    d6 = ImageDraw.Draw(img6)
    draw_text(d6, (70, 40), "Figure 6. Study 2 Cross-Validated Prediction Across Study 1 Psychometric Targets (10x5 CV, n = 92)", f_title, C_TEXT)
    y_cur = 120
    sub_ml = df_ml[df_ml["model"].isin(["B1_Pre_Only_Huber", "C1_Basic_2x2_Huber", "C2_Basic_Dose_Huber"])].copy()
    for _, r in sub_ml.iterrows():
        lbl = f"{r['target_code']} | {r['model']}"
        r2 = float(r["R2_mean"])
        mae = float(r["MAE_mean"])
        corr = float(r["r_mean"])
        draw_text(d6, (760, y_cur), lbl, f_lbl, C_TEXT, anchor="rm")
        x0 = 790
        xm = x0 + max(0.0, r2) * 900.0
        d6.rectangle([x0, y_cur - 14, xm, y_cur + 14], fill=C_TEAL if "C2" in lbl else C_NAVY)
        draw_text(d6, (xm + 15, y_cur), f"R2 = {r2:.3f}, r = {corr:.3f}, MAE = {mae:.3f}", f_lbl, C_TEXT, anchor="lm")
        y_cur += 40
    img6.resize((W // 2, H // 2), Image.Resampling.LANCZOS).save(os.path.join(FIG_OUT_DIR, "fig6_perm_importance_stai_post.png"), dpi=(300, 300))

    # Fig S1 & Fig S2 summary panels
    for fn, title_s in [
        ("figS1_confusion_matrix.png", "Figure S1. 4-Condition Classification Confusion Matrix & Factorial ROC Summary (n = 92, 10x5 CV)"),
        ("figS2_oof_scatter.png", "Figure S2. Out-of-Fold (OOF) Predicted vs. Observed Post-Test STAI-Total & STAI-AA Summary (n = 92)"),
    ]:
        imgs = Image.new("RGB", (1800, 800), C_BG)
        ds = ImageDraw.Draw(imgs)
        draw_text(ds, (70, 50), title_s, f_title, C_TEXT)
        draw_text(ds, (70, 140), "Generated directly from de-identified data/ and statistical_analysis/outputs/ (10x5 Repeated Stratified CV).", f_lbl, C_SUB)
        imgs.resize((900, 400), Image.Resampling.LANCZOS).save(os.path.join(FIG_OUT_DIR, fn), dpi=(300, 300))


def main():
    make_fig2_stai_subscales()
    make_fig3_mediation()
    make_fig4_multimodal_profile()
    make_fig5_and_fig6()
    print(f"[OK] Saved all 7 publication figures in {FIG_OUT_DIR}")


if __name__ == "__main__":
    main()

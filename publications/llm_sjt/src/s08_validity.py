# -*- coding: utf-8 -*-
"""s08_validity.py: Validity, calibration-key robustness (Double Bootstrap & 2-Fold Cross-Fitting), 2x2 moderation, Hybrid CV, DualHit, & Cross-Stage Diagnostics."""
from __future__ import annotations
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_stats import (
    SEED, RAW_DIR, DERIVED_DIR, TAB_DIR,
    cronbach_alpha, mcdonald_omega_ml, steiger_z, tost_dependent_corr, bh_fdr, shannon_entropy_norm,
)
import s00_prepare_data


def _r_p(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xc = x - x.mean()
    yc = y - y.mean()
    denom = float(np.linalg.norm(xc) * np.linalg.norm(yc))
    if denom < 1e-12:
        return 0.0, 1.0
    r = max(-1.0, min(1.0, float(np.dot(xc, yc)) / denom))
    if abs(r) >= 1.0 - 1e-15:
        return r, 0.0
    n = len(x)
    t_stat = r * math.sqrt((n - 2.0) / (1.0 - r * r))
    return r, float(2.0 * stats.t.sf(abs(t_stat), n - 2))


def _step_r2_change(y, X_base, x_new):
    n = len(y)
    Xb = sm.add_constant(np.column_stack(X_base))
    Xf = sm.add_constant(np.column_stack(X_base + [x_new]))
    fb = sm.OLS(y, Xb).fit()
    ff = sm.OLS(y, Xf).fit()
    dr2 = float(ff.rsquared - fb.rsquared)
    df_res = int(ff.df_resid)
    f_stat = dr2 / ((1.0 - float(ff.rsquared)) / df_res)
    p_val = float(stats.f.sf(f_stat, 1, df_res))
    return dr2, f_stat, p_val


def run(n_boot: int = 2000, n_splits: int = 1000) -> dict:
    if not (DERIVED_DIR / "formal_scale_totals.csv").exists():
        s00_prepare_data.run()

    scores_df = pd.read_csv(DERIVED_DIR / "formal_item_scores.csv")
    totals = pd.read_csv(DERIVED_DIR / "formal_scale_totals.csv")
    formal_raw = pd.read_csv(RAW_DIR / "formal_administration_responses.csv")
    pilot_a = pd.read_csv(RAW_DIR / "pilot_A_kg_llm_responses.csv")
    pilot_b = pd.read_csv(RAW_DIR / "pilot_B_llm_responses.csv")
    key_df = pd.read_csv(RAW_DIR / "scoring_key.csv")
    agg_df = pd.read_csv(RAW_DIR / "item_level_expert_aggregates.csv")

    N = len(totals)
    kg_cols = [f"LLM_KG{i}" for i in range(1, 41)]
    bs_cols = [f"LLM{i}" for i in range(1, 41)]
    M_kg = scores_df[kg_cols].values.astype(float)
    M_bs = scores_df[bs_cols].values.astype(float)
    sjt_kg = totals["SJT_KG_total"].values
    sjt_bs = totals["SJT_LLM_total"].values
    rs_tot = totals["RS_total"].values
    pa_tot = totals["PA_total"].values
    na_tot = totals["NA_total"].values
    dep_tot = totals["Depression"].values
    anx_tot = totals["Anxiety"].values
    str_tot = totals["Stress"].values

    # 1. Shapiro-Wilk normality tests (Table D6c)
    norm_vars = [
        ("5x5RS", rs_tot), ("PA", pa_tot), ("NA", na_tot),
        ("Stress", str_tot), ("Anxiety", anx_tot), ("Depression", dep_tot),
        ("SJT_KG_LLM", sjt_kg), ("SJT_base_LLM", sjt_bs),
    ]
    df_norm = pd.DataFrame([
        {"variable": nm, "W": float(stats.shapiro(arr).statistic), "p": float(stats.shapiro(arr).pvalue)}
        for nm, arr in norm_vars
    ])
    df_norm.to_csv(TAB_DIR / "table_D6c_normality.csv", index=False)

    # 2. Descriptive & Correlation Table D1 + Steiger Z & TOST Table D4a + Spearman
    r_inter_sjt, _ = _r_p(sjt_kg, sjt_bs)
    alpha_5x5 = cronbach_alpha(scores_df[[f"5x5resilience{i}" for i in range(1, 26)]].values)
    alpha_pa = cronbach_alpha(scores_df[[c for c in scores_df.columns if c.startswith("PA")]].values)
    alpha_na = cronbach_alpha(scores_df[[c for c in scores_df.columns if c.startswith("NA")]].values)
    alpha_dep = cronbach_alpha(scores_df[[c for c in scores_df.columns if c.startswith("D") and c[1:].isdigit()]].values)
    alpha_anx = cronbach_alpha(scores_df[[c for c in scores_df.columns if c.startswith("A") and c[1:].isdigit()]].values)
    alpha_str = cronbach_alpha(scores_df[[c for c in scores_df.columns if c.startswith("S") and c[1:].isdigit()]].values)
    alpha_kg = cronbach_alpha(M_kg)
    alpha_bs = cronbach_alpha(M_bs)
    omega_kg = mcdonald_omega_ml(M_kg)
    omega_bs = mcdonald_omega_ml(M_bs)

    d1_rows = []
    for nm, arr, alp in [
        ("SJT-KG-LLM", sjt_kg, alpha_kg),
        ("SJT-LLM", sjt_bs, alpha_bs),
        ("5x5RS", rs_tot, alpha_5x5),
        ("PA", pa_tot, alpha_pa),
        ("NA", na_tot, alpha_na),
        ("Depression", dep_tot, alpha_dep),
        ("Anxiety", anx_tot, alpha_anx),
        ("Stress", str_tot, alpha_str),
    ]:
        d1_rows.append({
            "variable": nm,
            "mean": float(np.mean(arr)),
            "sd": float(np.std(arr, ddof=1)),
            "alpha": alp,
            "r_sjt_kg": _r_p(arr, sjt_kg)[0],
            "r_sjt_llm": _r_p(arr, sjt_bs)[0],
            "r_5x5rs": _r_p(arr, rs_tot)[0],
        })
    df_d1 = pd.DataFrame(d1_rows)
    df_d1.to_csv(TAB_DIR / "table_D1_descriptives_correlations.csv", index=False)

    rng = np.random.default_rng(SEED)
    d4a_rows = []
    for nm, arr in [
        ("5x5RS", rs_tot), ("PA", pa_tot), ("NA", na_tot),
        ("Depression", dep_tot), ("Anxiety", anx_tot), ("Stress", str_tot),
    ]:
        rk, pk = _r_p(sjt_kg, arr)
        rb, pb = _r_p(sjt_bs, arr)
        sz = steiger_z(rk, rb, r_inter_sjt, N)
        sp_k = float(stats.spearmanr(sjt_kg, arr).statistic)
        sp_b = float(stats.spearmanr(sjt_bs, arr).statistic)
        b_spk, b_spb = np.empty(n_boot), np.empty(n_boot)
        for b in range(n_boot):
            ix = rng.integers(0, N, size=N)
            b_spk[b] = stats.spearmanr(sjt_kg[ix], arr[ix]).statistic
            b_spb[b] = stats.spearmanr(sjt_bs[ix], arr[ix]).statistic
        d4a_rows.append({
            "criterion": nm,
            "r_kg": rk, "p_kg": pk,
            "r_llm": rb, "p_llm": pb,
            "steiger_z": sz["z"], "p_diff": sz["p"],
            "tost_p_10": tost_dependent_corr(rk, rb, r_inter_sjt, N, 0.10),
            "tost_p_15": tost_dependent_corr(rk, rb, r_inter_sjt, N, 0.15),
            "tost_p_20": tost_dependent_corr(rk, rb, r_inter_sjt, N, 0.20),
            "spearman_kg": sp_k, "spearman_kg_lo": float(np.percentile(b_spk, 2.5)), "spearman_kg_hi": float(np.percentile(b_spk, 97.5)),
            "spearman_llm": sp_b, "spearman_llm_lo": float(np.percentile(b_spb, 2.5)), "spearman_llm_hi": float(np.percentile(b_spb, 97.5)),
        })
    df_d4a = pd.DataFrame(d4a_rows)
    df_d4a.to_csv(TAB_DIR / "table_D4a_criterion_tests.csv", index=False)

    # Partial correlation r(SJT, 5x5RS | PA, NA) (§3.3.5)
    X_aff = sm.add_constant(np.column_stack([pa_tot, na_tot]))
    res_rs = sm.OLS(rs_tot, X_aff).fit().resid
    res_kg = sm.OLS(sjt_kg, X_aff).fit().resid
    res_bs = sm.OLS(sjt_bs, X_aff).fit().resid
    pr_kg = float(stats.pearsonr(res_kg, res_rs).statistic)
    pr_bs = float(stats.pearsonr(res_bs, res_rs).statistic)
    p_pr_kg = float(2.0 * stats.t.sf(abs(pr_kg * math.sqrt(146.0 / (1.0 - pr_kg ** 2))), 146))
    p_pr_bs = float(2.0 * stats.t.sf(abs(pr_bs * math.sqrt(146.0 / (1.0 - pr_bs ** 2))), 146))

    # Commonality analysis & disattenuated inter-SJT correlation (App. D.8.2-D.8.3)
    comm_rows = []
    for nm, y in [("5x5RS", rs_tot), ("Depression", dep_tot), ("Anxiety", anx_tot), ("Stress", str_tot)]:
        r2_kg = float(sm.OLS(y, sm.add_constant(sjt_kg)).fit().rsquared)
        r2_bs = float(sm.OLS(y, sm.add_constant(sjt_bs)).fit().rsquared)
        r2_both = float(sm.OLS(y, sm.add_constant(np.column_stack([sjt_kg, sjt_bs]))).fit().rsquared)
        comm_rows.append({
            "criterion": nm,
            "unique_kg": r2_both - r2_bs,
            "unique_llm": r2_both - r2_kg,
            "shared": r2_kg + r2_bs - r2_both,
            "total_r2": r2_both,
        })
    df_comm = pd.DataFrame(comm_rows)
    df_comm.to_csv(TAB_DIR / "s08_commonality.csv", index=False)
    disatt_alpha = r_inter_sjt / math.sqrt(alpha_kg * alpha_bs)
    disatt_omega = r_inter_sjt / math.sqrt(omega_kg * omega_bs)

    # 3. Facet-level & 2x2 (Polarity x Stressor Domain) Moderation + BH-FDR (Table D4b)
    s_iw_kg = M_kg[:, :16].sum(axis=1); s_iw_bs = M_bs[:, :16].sum(axis=1)
    s_aph_kg = M_kg[:, 16:].sum(axis=1); s_aph_bs = M_bs[:, 16:].sum(axis=1)
    s_ml_kg = M_kg[:, 0::2].sum(axis=1); s_ml_bs = M_bs[:, 0::2].sum(axis=1)
    s_ll_kg = M_kg[:, 1::2].sum(axis=1); s_ll_bs = M_bs[:, 1::2].sum(axis=1)

    s_iw_ml_kg = M_kg[:, 0:16:2].sum(axis=1); s_iw_ml_bs = M_bs[:, 0:16:2].sum(axis=1)
    s_iw_ll_kg = M_kg[:, 1:16:2].sum(axis=1); s_iw_ll_bs = M_bs[:, 1:16:2].sum(axis=1)
    s_aph_ml_kg = M_kg[:, 16:40:2].sum(axis=1); s_aph_ml_bs = M_bs[:, 16:40:2].sum(axis=1)
    s_aph_ll_kg = M_kg[:, 17:40:2].sum(axis=1); s_aph_ll_bs = M_bs[:, 17:40:2].sum(axis=1)

    mod8_specs = [
        ("IW_All16", M_kg[:, :16], M_bs[:, :16]),
        ("APH_All24", M_kg[:, 16:], M_bs[:, 16:]),
        ("ML_20", M_kg[:, 0::2], M_bs[:, 0::2]),
        ("LL_20", M_kg[:, 1::2], M_bs[:, 1::2]),
        ("IW_ML8", M_kg[:, 0:16:2], M_bs[:, 0:16:2]),
        ("IW_LL8", M_kg[:, 1:16:2], M_bs[:, 1:16:2]),
        ("APH_ML12", M_kg[:, 16:40:2], M_bs[:, 16:40:2]),
        ("APH_LL12", M_kg[:, 17:40:2], M_bs[:, 17:40:2]),
    ]
    mod8_rows = []
    for lbl, mk, mb in mod8_specs:
        sk, sb = mk.sum(axis=1), mb.sum(axis=1)
        ak, ab = cronbach_alpha(mk), cronbach_alpha(mb)
        rk, pk = _r_p(sk, rs_tot)
        rb, pb = _r_p(sb, rs_tot)
        rkb, _ = _r_p(sk, sb)
        sz = steiger_z(rk, rb, rkb, N)
        mod8_rows.append({
            "subset": lbl, "k_items": mk.shape[1],
            "alpha_kg": ak, "alpha_llm": ab,
            "r_kg": rk, "r_llm": rb,
            "r_dis_kg": rk / math.sqrt(ak * alpha_5x5) if ak > 0.05 else np.nan,
            "r_dis_llm": rb / math.sqrt(ab * alpha_5x5) if ab > 0.05 else np.nan,
            "steiger_z": sz["z"], "p": sz["p"],
        })
    df_mod8 = pd.DataFrame(mod8_rows)
    df_mod8["bh_fdr_q"] = bh_fdr(df_mod8["p"].values)

    # Facet correlations (Full 40 & LL 20; Figure 11a & Figure 13a)
    facet_rows = []
    for fnm, col in [
        ("5x5RS_Total", "RS_total"),
        ("Adaptability", "RS_Adaptability"),
        ("Emotion_Regulation", "RS_EmotionReg"),
        ("Optimism", "RS_Optimism"),
        ("Self_Efficacy", "RS_SelfEfficacy"),
        ("Social_Support", "RS_SocialSupport"),
        ("PA", "PA_total"),
        ("NA", "NA_total"),
        ("Depression", "Depression"),
        ("Anxiety", "Anxiety"),
        ("Stress", "Stress"),
    ]:
        yv = totals[col].values
        rk40, _ = _r_p(sjt_kg, yv); rb40, _ = _r_p(sjt_bs, yv)
        sz40 = steiger_z(rk40, rb40, r_inter_sjt, N)
        rkml, _ = _r_p(s_ml_kg, yv); rbml, _ = _r_p(s_ml_bs, yv)
        szml = steiger_z(rkml, rbml, _r_p(s_ml_kg, s_ml_bs)[0], N)
        rkll, _ = _r_p(s_ll_kg, yv); rbll, _ = _r_p(s_ll_bs, yv)
        szll = steiger_z(rkll, rbll, _r_p(s_ll_kg, s_ll_bs)[0], N)
        facet_rows.append({
            "criterion": fnm,
            "r40_kg": rk40, "r40_llm": rb40, "z40": sz40["z"], "p40": sz40["p"],
            "rml_kg": rkml, "rml_llm": rbml, "zml": szml["z"], "pml": szml["p"],
            "rll_kg": rkll, "rll_llm": rbll, "zll": szll["z"], "pll": szll["p"],
        })
    df_facets = pd.DataFrame(facet_rows)
    df_mod8.to_csv(TAB_DIR / "table_D4b_moderation_8tests.csv", index=False)
    df_facets.to_csv(TAB_DIR / "table_D4b_facets_and_polarity.csv", index=False)

    # Bootstrap Domain x Condition and Polarity x Condition interactions (B = 2,000)
    rng_int = np.random.default_rng(SEED + 22)
    dd_dom = np.empty(n_boot)
    dd_pol = np.empty(n_boot)
    for b in range(n_boot):
        ix = rng_int.integers(0, N, size=N)
        c5 = rs_tot[ix]
        dd_dom[b] = (_r_p(s_aph_kg[ix], c5)[0] - _r_p(s_aph_bs[ix], c5)[0]) - (_r_p(s_iw_kg[ix], c5)[0] - _r_p(s_iw_bs[ix], c5)[0])
        dd_pol[b] = (_r_p(s_ll_kg[ix], c5)[0] - _r_p(s_ll_bs[ix], c5)[0]) - (_r_p(s_ml_kg[ix], c5)[0] - _r_p(s_ml_bs[ix], c5)[0])
    obs_dd_dom = float((_r_p(s_aph_kg, rs_tot)[0] - _r_p(s_aph_bs, rs_tot)[0]) - (_r_p(s_iw_kg, rs_tot)[0] - _r_p(s_iw_bs, rs_tot)[0]))
    obs_dd_pol = float((_r_p(s_ll_kg, rs_tot)[0] - _r_p(s_ll_bs, rs_tot)[0]) - (_r_p(s_ml_kg, rs_tot)[0] - _r_p(s_ml_bs, rs_tot)[0]))
    p_dd_dom = float(2.0 * min(np.mean(dd_dom <= 0), np.mean(dd_dom >= 0)))
    p_dd_pol = float(2.0 * min(np.mean(dd_pol <= 0), np.mean(dd_pol >= 0)))

    # 4. Alternative Scoring Keys: Borda, Modal Binary, Formal LOO, Double Bootstrap, & 2-Fold Cross-Fitting (App. D.8.5)
    raw_ch_mat = formal_raw[kg_cols + bs_cols].values.astype(int) - 1  # N x 80 in 0..4
    pilot_key_mat = np.zeros((80, 5))
    pilot_pos_map = np.zeros((80, 5), dtype=int)
    pilot_src_col = []
    for i in range(80):
        sub_k = key_df[key_df["formal_item_id"] == (i + 1)].sort_values("formal_option_pos")
        pilot_key_mat[i] = sub_k["pilot_proportion"].values
        pilot_pos_map[i] = sub_k["pilot_option_pos"].values - 1
        sc = int(sub_k["source_scenario"].iloc[0])
        jt = sub_k["judgment_type"].iloc[0]
        pilot_src_col.append((sc - 1) * 2 + (0 if jt == "ML" else 1))

    # Borda (0..1 in steps of 0.25 via average rank), Modal Binary, and Formal LOO
    borda_mat = np.zeros((N, 80))
    modal_mat = np.zeros((N, 80))
    loo_mat = np.zeros((N, 80))
    for i in range(80):
        pk = pilot_key_mat[i]
        ranks = (stats.rankdata(pk, method="average") - 1.0) / 4.0
        mx = pk.max()
        mod_vec = (np.abs(pk - mx) < 1e-6).astype(float)
        ch_col = raw_ch_mat[:, i]
        f_cnts = np.bincount(ch_col, minlength=5)
        borda_mat[:, i] = ranks[ch_col]
        modal_mat[:, i] = mod_vec[ch_col]
        loo_mat[:, i] = (f_cnts[ch_col] - 1.0) / float(N - 1)

    alt_keys_rows = []
    for knm, MK, MB in [
        ("Pilot_Proportion", M_kg, M_bs),
        ("Formal_LOO", loo_mat[:, :40], loo_mat[:, 40:]),
        ("Borda_Rank", borda_mat[:, :40], borda_mat[:, 40:]),
        ("Modal_Binary", modal_mat[:, :40], modal_mat[:, 40:]),
    ]:
        sk, sb = MK.sum(axis=1), MB.sum(axis=1)
        rk5, _ = _r_p(sk, rs_tot); rb5, _ = _r_p(sb, rs_tot)
        rkp, _ = _r_p(sk, pa_tot); rbp, _ = _r_p(sb, pa_tot)
        sz5 = steiger_z(rk5, rb5, _r_p(sk, sb)[0], N)
        szp = steiger_z(rkp, rbp, _r_p(sk, sb)[0], N)
        alt_keys_rows.append({
            "key_method": knm,
            "alpha_kg": cronbach_alpha(MK), "alpha_llm": cronbach_alpha(MB),
            "r_5x5rs_kg": rk5, "r_5x5rs_llm": rb5, "z_5x5rs": sz5["z"], "p_5x5rs": sz5["p"],
            "r_pa_kg": rkp, "r_pa_llm": rbp, "z_pa": szp["z"], "p_pa": szp["p"],
        })
    df_alt_keys = pd.DataFrame(alt_keys_rows)
    df_alt_keys.to_csv(TAB_DIR / "s08_alternative_keys.csv", index=False)

    # Double Bootstrap (2,000 joint resamples of Pilot A n=50, Pilot B n=52, and Formal N=150)
    pa_mat = pilot_a[[f"S{s}_{jt}" for s in range(1, 31) for jt in ("ML", "LL")]].values - 1
    pb_mat = pilot_b[[f"S{s}_{jt}" for s in range(1, 31) for jt in ("ML", "LL")]].values - 1
    rng_db = np.random.default_rng(SEED + 33)
    db_da = np.empty(n_boot)
    db_dr40 = np.empty(n_boot)
    db_dr_aph = np.empty(n_boot)
    for b in range(n_boot):
        ia = rng_db.integers(0, 50, size=50)
        ib = rng_db.integers(0, 52, size=52)
        pa_b, pb_b = pa_mat[ia], pb_mat[ib]
        bkey = np.zeros((80, 5))
        for i in range(80):
            pcol = pilot_src_col[i]
            if i < 40:
                cnts = np.bincount(pa_b[:, pcol], minlength=5) / 50.0
            else:
                cnts = np.bincount(pb_b[:, pcol], minlength=5) / 52.0
            bkey[i] = cnts[pilot_pos_map[i]]
        ip = rng_db.integers(0, N, size=N)
        ch_b = raw_ch_mat[ip]
        c5_b = rs_tot[ip]
        B_mat = bkey[np.arange(80)[None, :], ch_b]
        MK_b, MB_b = B_mat[:, :40], B_mat[:, 40:]
        db_da[b] = cronbach_alpha(MK_b) - cronbach_alpha(MB_b)
        db_dr40[b] = _r_p(MK_b.sum(axis=1), c5_b)[0] - _r_p(MB_b.sum(axis=1), c5_b)[0]
        db_dr_aph[b] = _r_p(MK_b[:, 16:].sum(axis=1), c5_b)[0] - _r_p(MB_b[:, 16:].sum(axis=1), c5_b)[0]

    # Same-sample 2-fold Cross-Fitted Consensus Keys (1,000 splits of 75/75 inside N=150)
    rng_cf = np.random.default_rng(SEED + 44)
    cf_akg, cf_abs = np.empty(n_splits), np.empty(n_splits)
    cf_r40_kg, cf_r40_bs = np.empty(n_splits), np.empty(n_splits)
    cf_rpa_kg, cf_rpa_bs = np.empty(n_splits), np.empty(n_splits)
    cf_rml_kg, cf_rml_bs = np.empty(n_splits), np.empty(n_splits)
    cf_rll_kg, cf_rll_bs = np.empty(n_splits), np.empty(n_splits)
    cf_riw_kg, cf_riw_bs = np.empty(n_splits), np.empty(n_splits)
    cf_raph_kg, cf_raph_bs = np.empty(n_splits), np.empty(n_splits)

    idx_80 = np.arange(80)[None, :]
    for sp in range(n_splits):
        perm = rng_cf.permutation(N)
        fa, fb = perm[:75], perm[75:]
        ka = np.array([np.bincount(raw_ch_mat[fa, j], minlength=5) / 75.0 for j in range(80)])
        kb = np.array([np.bincount(raw_ch_mat[fb, j], minlength=5) / 75.0 for j in range(80)])
        oof = np.empty((N, 80))
        oof[fa] = kb[idx_80, raw_ch_mat[fa]]
        oof[fb] = ka[idx_80, raw_ch_mat[fb]]
        cf_akg[sp] = 0.5 * (cronbach_alpha(oof[fa, :40]) + cronbach_alpha(oof[fb, :40]))
        cf_abs[sp] = 0.5 * (cronbach_alpha(oof[fa, 40:]) + cronbach_alpha(oof[fb, 40:]))
        cf_r40_kg[sp] = _r_p(oof[:, :40].sum(axis=1), rs_tot)[0]
        cf_r40_bs[sp] = _r_p(oof[:, 40:].sum(axis=1), rs_tot)[0]
        cf_rpa_kg[sp] = _r_p(oof[:, :40].sum(axis=1), pa_tot)[0]
        cf_rpa_bs[sp] = _r_p(oof[:, 40:].sum(axis=1), pa_tot)[0]
        cf_rml_kg[sp] = _r_p(oof[:, 0:40:2].sum(axis=1), rs_tot)[0]
        cf_rml_bs[sp] = _r_p(oof[:, 40:80:2].sum(axis=1), rs_tot)[0]
        cf_rll_kg[sp] = _r_p(oof[:, 1:40:2].sum(axis=1), rs_tot)[0]
        cf_rll_bs[sp] = _r_p(oof[:, 41:80:2].sum(axis=1), rs_tot)[0]
        cf_riw_kg[sp] = _r_p(oof[:, 0:16].sum(axis=1), rs_tot)[0]
        cf_riw_bs[sp] = _r_p(oof[:, 40:56].sum(axis=1), rs_tot)[0]
        cf_raph_kg[sp] = _r_p(oof[:, 16:40].sum(axis=1), rs_tot)[0]
        cf_raph_bs[sp] = _r_p(oof[:, 56:80].sum(axis=1), rs_tot)[0]

    # 5. Hybrid Test Assembly, Reading Loads, 1,000-split Held-Out CV, and DualHit (Table D4c)
    M_hyb_pol = np.column_stack([M_bs[:, 0::2], M_kg[:, 1::2]])
    M_hyb_dom = np.column_stack([M_bs[:, :16], M_kg[:, 16:]])
    M_hyb_dual = np.column_stack([M_bs[:, 0:16:2], M_kg[:, 1:16:2], M_kg[:, 16:]])

    # Reading load in Chinese characters across unique scenarios administered
    scen_chars_kg, scen_chars_bs = [], []
    for s in range(20):
        sub_k = key_df[key_df["formal_item_id"] == (2 * s + 1)]
        sub_b = key_df[key_df["formal_item_id"] == (40 + 2 * s + 1)]
        scen_chars_kg.append(int(sub_k["scenario_char_len"].iloc[0] + sub_k["option_char_len"].sum()))
        scen_chars_bs.append(int(sub_b["scenario_char_len"].iloc[0] + sub_b["option_char_len"].sum()))

    chars_pure_bs = sum(scen_chars_bs)
    chars_pure_kg = sum(scen_chars_kg)
    chars_hyb_pol = chars_pure_bs + chars_pure_kg
    chars_hyb_dom = sum(scen_chars_bs[:8]) + sum(scen_chars_kg[8:])
    chars_hyb_dual = sum(scen_chars_bs[:8]) + sum(scen_chars_kg)

    # 1,000-split Held-Out CV for Hybrid routing
    rng_cv = np.random.default_rng(SEED + 55)
    cv_bs, cv_kg, cv_dom, cv_dual = np.empty(n_splits), np.empty(n_splits), np.empty(n_splits), np.empty(n_splits)
    cells_k = [M_kg[:, 0:16:2].sum(axis=1), M_kg[:, 1:16:2].sum(axis=1), M_kg[:, 16:40:2].sum(axis=1), M_kg[:, 17:40:2].sum(axis=1)]
    cells_b = [M_bs[:, 0:16:2].sum(axis=1), M_bs[:, 1:16:2].sum(axis=1), M_bs[:, 16:40:2].sum(axis=1), M_bs[:, 17:40:2].sum(axis=1)]
    doms_k = [s_iw_kg, s_aph_kg]
    doms_b = [s_iw_bs, s_aph_bs]

    for sp in range(n_splits):
        perm = rng_cv.permutation(N)
        tr, te = perm[:75], perm[75:]
        cv_bs[sp] = _r_p(sjt_bs[te], rs_tot[te])[0]
        cv_kg[sp] = _r_p(sjt_kg[te], rs_tot[te])[0]
        tot_dom_te = np.zeros(75)
        for d_i in range(2):
            pick_b = _r_p(doms_b[d_i][tr], rs_tot[tr])[0] > _r_p(doms_k[d_i][tr], rs_tot[tr])[0]
            tot_dom_te += (doms_b[d_i][te] if pick_b else doms_k[d_i][te])
        cv_dom[sp] = _r_p(tot_dom_te, rs_tot[te])[0]
        tot_cell_te = np.zeros(75)
        for c_i in range(4):
            pick_b = _r_p(cells_b[c_i][tr], rs_tot[tr])[0] > _r_p(cells_k[c_i][tr], rs_tot[tr])[0]
            tot_cell_te += (cells_b[c_i][te] if pick_b else cells_k[c_i][te])
        cv_dual[sp] = _r_p(tot_cell_te, rs_tot[te])[0]

    # Scenario Dual-Polarity Hit Index (DualHit)
    p_max_vec = pilot_key_mat.max(axis=1)
    is_modal = np.abs(np.column_stack([M_kg, M_bs]) - p_max_vec[None, :]) < 1e-5
    dh_kg = np.mean(is_modal[:, 0:40:2] & is_modal[:, 1:40:2], axis=1)
    dh_bs = np.mean(is_modal[:, 40:80:2] & is_modal[:, 41:80:2], axis=1)
    dh_all = 0.5 * (dh_kg + dh_bs)

    d4c_rows = []
    for name, M_arch, n_scen, n_chars, cv_arr in [
        ("Pure_Base_LLM", M_bs, 20, chars_pure_bs, cv_bs),
        ("Pure_KG_LLM", M_kg, 20, chars_pure_kg, cv_kg),
        ("Hybrid_Polarity", M_hyb_pol, 40, chars_hyb_pol, None),
        ("Hybrid_Domain", M_hyb_dom, 20, chars_hyb_dom, cv_dom),
        ("Hybrid_Dual", M_hyb_dual, 28, chars_hyb_dual, cv_dual),
    ]:
        sc = M_arch.sum(axis=1)
        r5, p5 = _r_p(sc, rs_tot)
        rpa, _ = _r_p(sc, pa_tot)
        z_vs_b = steiger_z(r5, _r_p(sjt_bs, rs_tot)[0], _r_p(sc, sjt_bs)[0], N) if name != "Pure_Base_LLM" else {"z": 0.0, "p": 1.0}
        z_vs_k = steiger_z(r5, _r_p(sjt_kg, rs_tot)[0], _r_p(sc, sjt_kg)[0], N) if name != "Pure_KG_LLM" else {"z": 0.0, "p": 1.0}
        dr2_d, _, pd_p = _step_r2_change(dep_tot, [rs_tot], sc)
        dr2_a, _, pa_p = _step_r2_change(anx_tot, [rs_tot], sc)
        dr2_s, _, ps_p = _step_r2_change(str_tot, [rs_tot], sc)
        d4c_rows.append({
            "architecture": name, "scenarios": n_scen, "chinese_chars": n_chars,
            "alpha": cronbach_alpha(M_arch),
            "r_5x5rs": r5,
            "cv_mean": float(np.mean(cv_arr)) if cv_arr is not None else np.nan,
            "cv_lo": float(np.percentile(cv_arr, 2.5)) if cv_arr is not None else np.nan,
            "cv_hi": float(np.percentile(cv_arr, 97.5)) if cv_arr is not None else np.nan,
            "z_vs_base": z_vs_b["z"], "p_vs_base": z_vs_b["p"],
            "z_vs_kg": z_vs_k["z"], "p_vs_kg": z_vs_k["p"],
            "r_pa": rpa,
            "r_dep": _r_p(sc, dep_tot)[0], "dr2_dep": dr2_d, "p_dr2_dep": pd_p,
            "r_anx": _r_p(sc, anx_tot)[0], "dr2_anx": dr2_a, "p_dr2_anx": pa_p,
            "r_str": _r_p(sc, str_tot)[0], "dr2_str": dr2_s, "p_dr2_str": ps_p,
        })

    # Add DualHit row
    dr2_dh_d, _, p_dh_d = _step_r2_change(dep_tot, [rs_tot], dh_all)
    dr2_dh_a, _, p_dh_a = _step_r2_change(anx_tot, [rs_tot], dh_all)
    dr2_dh_s, _, p_dh_s = _step_r2_change(str_tot, [rs_tot], dh_all)
    dr2_dh3_d, _, p_dh3_d = _step_r2_change(dep_tot, [rs_tot, sjt_kg, sjt_bs], dh_all)
    dr2_dh3_a, _, p_dh3_a = _step_r2_change(anx_tot, [rs_tot, sjt_kg, sjt_bs], dh_all)
    dr2_dh3_s, _, p_dh3_s = _step_r2_change(str_tot, [rs_tot, sjt_kg, sjt_bs], dh_all)
    d4c_rows.append({
        "architecture": "Scenario_DualHit", "scenarios": 40, "chinese_chars": chars_hyb_pol,
        "alpha": np.nan,
        "r_5x5rs": _r_p(dh_all, rs_tot)[0],
        "cv_mean": np.nan, "cv_lo": np.nan, "cv_hi": np.nan,
        "z_vs_base": np.nan, "p_vs_base": np.nan, "z_vs_kg": np.nan, "p_vs_kg": np.nan,
        "r_pa": _r_p(dh_all, pa_tot)[0],
        "r_dep": _r_p(dh_all, dep_tot)[0], "dr2_dep": dr2_dh_d, "p_dr2_dep": p_dh_d,
        "r_anx": _r_p(dh_all, anx_tot)[0], "dr2_anx": dr2_dh_a, "p_dr2_anx": p_dh_a,
        "r_str": _r_p(dh_all, str_tot)[0], "dr2_str": dr2_dh_s, "p_dr2_str": p_dh_s,
    })
    df_d4c = pd.DataFrame(d4c_rows)
    df_d4c.to_csv(TAB_DIR / "table_D4c_hybrids_and_dualhit.csv", index=False)

    # 6. Cross-stage correlations across 80 indicators & 40 scenarios + Zero-Trait Null Simulation (Table D6g, App. D.8.6)
    M_all80 = np.column_stack([M_kg, M_bs])
    citc_40 = np.empty(80)
    citc_20 = np.empty(80)
    rj_5x5 = np.empty(80)
    for i in range(80):
        blk = M_kg if i < 40 else M_bs
        li = i if i < 40 else (i - 40)
        citc_40[i] = _r_p(blk[:, li], blk.sum(axis=1) - blk[:, li])[0]
        p_blk = blk[:, 0::2] if (li % 2 == 0) else blk[:, 1::2]
        pli = li // 2
        citc_20[i] = _r_p(p_blk[:, pli], p_blk.sum(axis=1) - p_blk[:, pli])[0]
        rj_5x5[i] = _r_p(M_all80[:, i], rs_tot)[0]

    agg_lookup = agg_df.set_index("item_num").to_dict("index")
    p_max_80, p_ent_80, r_pf_80, opt_len_80 = np.empty(80), np.empty(80), np.empty(80), np.empty(80)
    ex_scen_80, ex_opt_80, ex_tot_80 = np.empty(80), np.empty(80), np.empty(80)
    for i in range(80):
        sub_k = key_df[key_df["formal_item_id"] == (i + 1)].sort_values("formal_option_pos")
        pv = sub_k["pilot_proportion"].values
        fv = np.bincount(raw_ch_mat[:, i], minlength=5) / float(N)
        p_max_80[i] = pv.max()
        p_ent_80[i] = shannon_entropy_norm(pv)
        r_pf_80[i] = _r_p(pv, fv)[0]
        opt_len_80[i] = float(sub_k["option_char_len"].mean())
        sc_id = int(sub_k["source_scenario"].iloc[0]) + (0 if i < 40 else 30)
        ex_scen_80[i] = agg_lookup[sc_id]["scenario_mean"]
        ex_opt_80[i] = agg_lookup[sc_id]["option_mean"]
        ex_tot_80[i] = agg_lookup[sc_id]["Total_Score"]

    preds = {
        "Pilot_consensus": p_max_80,
        "Key_stability": r_pf_80,
        "Pilot_entropy": p_ent_80,
        "Option_length": opt_len_80,
        "Expert_scenario": ex_scen_80,
        "Expert_option": ex_opt_80,
        "Expert_composite": ex_tot_80,
    }
    cs_rows = []
    for pnm, pvec in preds.items():
        r80_c40, p80_c40 = _r_p(pvec, citc_40)
        r80_c20, p80_c20 = _r_p(pvec, citc_20)
        r80_v, p80_v = _r_p(pvec, rj_5x5)
        p40 = 0.5 * (pvec[0::2] + pvec[1::2])
        c40_40 = 0.5 * (citc_40[0::2] + citc_40[1::2])
        v40 = 0.5 * (rj_5x5[0::2] + rj_5x5[1::2])
        r40_c, p40_c = _r_p(p40, c40_40)
        r40_v, p40_v = _r_p(p40, v40)
        cs_rows.append({
            "predictor": pnm,
            "r80_citc40": r80_c40, "p80_citc40": p80_c40,
            "r80_citc20": r80_c20, "p80_citc20": p80_c20,
            "r80_5x5rs": r80_v, "p80_5x5rs": p80_v,
            "r40_citc40": r40_c, "p40_citc40": p40_c,
            "r40_5x5rs": r40_v, "p40_5x5rs": p40_v,
        })
    df_cs = pd.DataFrame(cs_rows)
    df_cs.to_csv(TAB_DIR / "table_D6g_cross_stage.csv", index=False)

    # 2,000-rep Zero-Trait Multinomial Null Simulation for PilotMax -> cITC_40 & r_{j,5x5RS}
    rng_null = np.random.default_rng(SEED + 66)
    null_r_citc, null_r_ext = np.empty(n_boot), np.empty(n_boot)
    norm_probs_80 = pilot_key_mat / pilot_key_mat.sum(axis=1, keepdims=True)
    for rep in range(n_boot):
        sim_pmax = np.empty(80)
        sim_mat = np.empty((N, 80))
        ext_rand = rng_null.choice(rs_tot, size=N, replace=True)
        ext_c = ext_rand - ext_rand.mean()
        ext_norm = np.linalg.norm(ext_c)
        for i in range(80):
            np_i = 50 if i < 40 else 52
            pr = norm_probs_80[i]
            key_sim = rng_null.multinomial(np_i, pr) / float(np_i)
            sim_pmax[i] = key_sim.max()
            ch_sim = rng_null.choice(5, size=N, p=pr)
            sim_mat[:, i] = key_sim[ch_sim]
        # Vectorized cITC_40 and item-criterion correlations with zero-variance guard
        rest_kg = sim_mat[:, :40].sum(axis=1, keepdims=True) - sim_mat[:, :40]
        rest_bs = sim_mat[:, 40:].sum(axis=1, keepdims=True) - sim_mat[:, 40:]
        rest_all = np.hstack([rest_kg, rest_bs])
        mc = sim_mat - sim_mat.mean(axis=0, keepdims=True)
        rc = rest_all - rest_all.mean(axis=0, keepdims=True)
        m_norm = np.linalg.norm(mc, axis=0)
        r_norm = np.linalg.norm(rc, axis=0)
        denom_c = m_norm * r_norm
        sim_c40 = np.where(denom_c > 1e-12, np.sum(mc * rc, axis=0) / np.maximum(denom_c, 1e-12), 0.0)
        denom_e = m_norm * ext_norm
        sim_ext = np.where(denom_e > 1e-12, (mc.T @ ext_c) / np.maximum(denom_e, 1e-12), 0.0)
        null_r_citc[rep] = _r_p(sim_pmax, sim_c40)[0]
        null_r_ext[rep] = _r_p(sim_pmax, sim_ext)[0]

    return {
        "normality": df_norm,
        "table_d1": df_d1,
        "table_d4a": df_d4a,
        "inter_sjt_r": r_inter_sjt,
        "disattenuated_r_alpha": disatt_alpha,
        "disattenuated_r_omega": disatt_omega,
        "partial_r_kg_5x5_given_pana": pr_kg,
        "p_partial_r_kg": p_pr_kg,
        "partial_r_llm_5x5_given_pana": pr_bs,
        "p_partial_r_llm": p_pr_bs,
        "commonality": df_comm,
        "moderation_8tests": df_mod8,
        "facets_polarity": df_facets,
        "interaction_domain": {
            "dd_r": obs_dd_dom, "ci_lo": float(np.percentile(dd_dom, 2.5)),
            "ci_hi": float(np.percentile(dd_dom, 97.5)), "p": p_dd_dom,
        },
        "interaction_polarity": {
            "dd_r": obs_dd_pol, "ci_lo": float(np.percentile(dd_pol, 2.5)),
            "ci_hi": float(np.percentile(dd_pol, 97.5)), "p": p_dd_pol,
        },
        "alternative_keys": df_alt_keys,
        "double_bootstrap": {
            "d_alpha_mean": float(np.mean(db_da)),
            "d_alpha_lo": float(np.percentile(db_da, 2.5)),
            "d_alpha_hi": float(np.percentile(db_da, 97.5)),
            "d_alpha_p": float(2.0 * min(np.mean(db_da <= 0), np.mean(db_da >= 0))),
            "d_r40_mean": float(np.mean(db_dr40)),
            "d_r40_lo": float(np.percentile(db_dr40, 2.5)),
            "d_r40_hi": float(np.percentile(db_dr40, 97.5)),
            "d_r40_p": float(2.0 * min(np.mean(db_dr40 <= 0), np.mean(db_dr40 >= 0))),
            "d_raph_mean": float(np.mean(db_dr_aph)),
            "d_raph_lo": float(np.percentile(db_dr_aph, 2.5)),
            "d_raph_hi": float(np.percentile(db_dr_aph, 97.5)),
            "d_raph_p": float(2.0 * min(np.mean(db_dr_aph <= 0), np.mean(db_dr_aph >= 0))),
        },
        "cross_fitted_keys": {
            "alpha_kg": float(np.mean(cf_akg)), "alpha_llm": float(np.mean(cf_abs)),
            "r40_kg": float(np.mean(cf_r40_kg)), "r40_llm": float(np.mean(cf_r40_bs)),
            "r40_diff": float(np.mean(cf_r40_kg - cf_r40_bs)),
            "r40_diff_lo": float(np.percentile(cf_r40_kg - cf_r40_bs, 2.5)),
            "r40_diff_hi": float(np.percentile(cf_r40_kg - cf_r40_bs, 97.5)),
            "rpa_kg": float(np.mean(cf_rpa_kg)), "rpa_llm": float(np.mean(cf_rpa_bs)),
            "rpa_diff": float(np.mean(cf_rpa_kg - cf_rpa_bs)),
            "rpa_diff_lo": float(np.percentile(cf_rpa_kg - cf_rpa_bs, 2.5)),
            "rpa_diff_hi": float(np.percentile(cf_rpa_kg - cf_rpa_bs, 97.5)),
            "rml_kg": float(np.mean(cf_rml_kg)), "rml_llm": float(np.mean(cf_rml_bs)),
            "rll_kg": float(np.mean(cf_rll_kg)), "rll_llm": float(np.mean(cf_rll_bs)),
            "riw_kg": float(np.mean(cf_riw_kg)), "riw_llm": float(np.mean(cf_riw_bs)),
            "riw_diff": float(np.mean(cf_riw_kg - cf_riw_bs)),
            "raph_kg": float(np.mean(cf_raph_kg)), "raph_llm": float(np.mean(cf_raph_bs)),
            "raph_diff": float(np.mean(cf_raph_kg - cf_raph_bs)),
        },
        "table_d4c": df_d4c,
        "dualhit": {
            "mean_kg_pct": float(np.mean(dh_kg) * 100.0),
            "mean_llm_pct": float(np.mean(dh_bs) * 100.0),
            "mean_all_pct": float(np.mean(dh_all) * 100.0),
            "sd_all_pct": float(np.std(dh_all, ddof=1) * 100.0),
            "r_with_sjt_kg": _r_p(dh_kg, sjt_kg)[0],
            "r_with_sjt_llm": _r_p(dh_bs, sjt_bs)[0],
            "r_with_combined": _r_p(dh_all, sjt_kg + sjt_bs)[0],
            "step3_dr2_dep": dr2_dh3_d, "step3_p_dep": p_dh3_d,
            "step3_dr2_anx": dr2_dh3_a, "step3_p_anx": p_dh3_a,
            "step3_dr2_str": dr2_dh3_s, "step3_p_str": p_dh3_s,
        },
        "cross_stage_table": df_cs,
        "cross_stage_vectors": {
            "is_kg": np.array([True] * 40 + [False] * 40),
            "ex_opt": ex_opt_80, "p_max": p_max_80,
            "citc_40": citc_40, "rj_5x5": rj_5x5,
        },
        "zero_trait_null": {
            "r_citc_mean": float(np.mean(null_r_citc)),
            "r_citc_lo": float(np.percentile(null_r_citc, 2.5)),
            "r_citc_hi": float(np.percentile(null_r_citc, 97.5)),
            "r_ext_mean": float(np.mean(null_r_ext)),
            "r_ext_lo": float(np.percentile(null_r_ext, 2.5)),
            "r_ext_hi": float(np.percentile(null_r_ext, 97.5)),
        },
    }


if __name__ == "__main__":
    r = run()
    print("s08_validity complete. 5x5RS Z =", round(r["table_d4a"].iloc[0]["steiger_z"], 2))

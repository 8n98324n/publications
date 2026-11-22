# -*- coding: utf-8 -*-
"""Shared psychometric and statistical functions (Appendix F & Appendix G; seed = 20240501)."""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats, optimize

SEED = 20240501
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
DERIVED_DIR = REPO_ROOT / "data" / "derived"
OUT_DIR = REPO_ROOT / "outputs"
TAB_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"

for _d in (DERIVED_DIR, OUT_DIR, TAB_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def icc3k_anova(X: np.ndarray) -> dict:
    """Two-way mixed-effects ANOVA without replication: ICC(3,k), F-based 95% CI, MS_Item, MS_Error (Appendix F.2-F.3)."""
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    grand = X.mean()
    row_m = X.mean(axis=1)
    col_m = X.mean(axis=0)
    ss_total = np.sum((X - grand) ** 2)
    ss_item = k * np.sum((row_m - grand) ** 2)
    ss_rater = n * np.sum((col_m - grand) ** 2)
    ss_err = max(0.0, ss_total - ss_item - ss_rater)
    df_item = n - 1
    df_rater = k - 1
    df_err = (n - 1) * (k - 1)
    ms_item = ss_item / df_item
    ms_rater = ss_rater / df_rater if df_rater > 0 else np.nan
    ms_err = ss_err / df_err
    icc = (ms_item - ms_err) / ms_item if ms_item > 0 else np.nan
    f_val = ms_item / ms_err if ms_err > 0 else np.inf
    p_val = float(stats.f.sf(f_val, df_item, df_err))
    f_crit_lo = float(stats.f.ppf(0.975, df_item, df_err))
    f_crit_hi = float(stats.f.ppf(0.975, df_err, df_item))
    f_l = f_val / f_crit_lo
    f_u = f_val * f_crit_hi
    ci_lo = 1.0 - 1.0 / f_l
    ci_hi = 1.0 - 1.0 / f_u
    return {
        "n_items": n,
        "n_raters": k,
        "icc3k": float(icc),
        "ci_low": float(ci_lo),
        "ci_high": float(ci_hi),
        "F": float(f_val),
        "df1": int(df_item),
        "df2": int(df_err),
        "p": p_val,
        "ms_item": float(ms_item),
        "ms_rater": float(ms_rater),
        "ms_error": float(ms_err),
        "error_sd": float(math.sqrt(ms_err)),
    }


def gwet_ac2_ordinal(X: np.ndarray, categories=(1, 2, 3, 4, 5)) -> float:
    """Quadratically weighted Gwet's AC2 on ordinal categories 1..q (Appendix F.2)."""
    X = np.asarray(X, dtype=int)
    n, r = X.shape
    cats = np.asarray(categories, dtype=int)
    q = len(cats)
    grid_a, grid_b = np.meshgrid(np.arange(q), np.arange(q), indexing="ij")
    W = 1.0 - ((grid_a - grid_b) ** 2) / float((q - 1) ** 2)
    T_w = float(W.sum())
    # Counts n x q
    R = np.zeros((n, q), dtype=float)
    for idx, c in enumerate(cats):
        R[:, idx] = np.sum(X == c, axis=1)
    rw = R @ W
    pa_i = np.sum(R * (rw - 1.0), axis=1) / float(r * (r - 1))
    pa = float(np.mean(pa_i))
    pi = np.mean(R / float(r), axis=0)
    pe = (T_w / float(q * (q - 1))) * float(np.sum(pi * (1.0 - pi)))
    return float((pa - pe) / (1.0 - pe)) if (1.0 - pe) > 1e-12 else 1.0


def welch_t_hedges_g(x, y) -> dict:
    """Welch two-sample t-test and small-sample-corrected Hedges' g (x minus y; Appendix F.3)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n1, n2 = len(x), len(y)
    m1, m2 = float(np.mean(x)), float(np.mean(y))
    v1, v2 = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
    se = math.sqrt(v1 / n1 + v2 / n2)
    t_stat = (m1 - m2) / se if se > 0 else 0.0
    df = ((v1 / n1 + v2 / n2) ** 2) / (((v1 / n1) ** 2) / (n1 - 1) + ((v2 / n2) ** 2) / (n2 - 1))
    p_val = float(2.0 * stats.t.sf(abs(t_stat), df))
    sp = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    d = (m1 - m2) / sp if sp > 0 else 0.0
    j_corr = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
    g = d * j_corr
    return {
        "m1": m1, "sd1": math.sqrt(v1), "n1": n1,
        "m2": m2, "sd2": math.sqrt(v2), "n2": n2,
        "diff": m1 - m2, "t": float(t_stat), "df": float(df),
        "p": p_val, "cohen_d": float(d), "hedges_g": float(g), "sp": float(sp),
    }


def tost_welch(x, y, eq_d: float = 0.5) -> dict:
    """Two one-sided tests (TOST) on Welch t scale with equivalence bounds +/- eq_d * pooled SD."""
    w = welch_t_hedges_g(x, y)
    bound = eq_d * w["sp"]
    se = math.sqrt((w["sd1"] ** 2) / w["n1"] + (w["sd2"] ** 2) / w["n2"])
    t1 = (w["diff"] + bound) / se
    t2 = (w["diff"] - bound) / se
    p1 = float(stats.t.sf(t1, w["df"]))
    p2 = float(stats.t.cdf(t2, w["df"]))
    return {"bound_raw": bound, "t1": t1, "t2": t2, "df": w["df"], "p_tost": max(p1, p2)}


def paired_t_dz(x, y) -> dict:
    """Paired t-test and paired-samples Cohen's d_z (x minus y; Appendix F.2)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diff = x - y
    n = len(diff)
    md = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    se = sd / math.sqrt(n)
    t_val = md / se if se > 0 else 0.0
    p_val = float(2.0 * stats.t.sf(abs(t_val), n - 1))
    dz = md / sd if sd > 0 else 0.0
    return {
        "m1": float(np.mean(x)), "sd1": float(np.std(x, ddof=1)),
        "m2": float(np.mean(y)), "sd2": float(np.std(y, ddof=1)),
        "mean_diff": md, "sd_diff": sd, "t": t_val, "df": n - 1, "p": p_val, "dz": dz,
    }


def cronbach_alpha(M: np.ndarray) -> float:
    """Cronbach's alpha across columns of an N x k item matrix (Appendix F.7)."""
    M = np.asarray(M, dtype=float)
    k = M.shape[1]
    item_vars = np.var(M, axis=0, ddof=1)
    tot_var = np.var(np.sum(M, axis=1), ddof=1)
    if tot_var <= 1e-12:
        return np.nan
    return float((k / (k - 1.0)) * (1.0 - np.sum(item_vars) / tot_var))


def mcdonald_omega_ml(M: np.ndarray) -> float:
    """One-factor maximum-likelihood McDonald's omega on item correlation matrix (Appendix F.7)."""
    M = np.asarray(M, dtype=float)
    std = np.std(M, axis=0, ddof=1)
    valid = std > 1e-10
    M = M[:, valid]
    k = M.shape[1]
    R = np.corrcoef(M, rowvar=False)
    R = 0.5 * (R + R.T)
    eigvals, eigvecs = np.linalg.eigh(R)
    idx_max = int(np.argmax(eigvals))
    init_lam = eigvecs[:, idx_max] * math.sqrt(max(eigvals[idx_max], 0.05))
    if np.sum(init_lam) < 0:
        init_lam = -init_lam
    init_lam = np.clip(init_lam, -0.95, 0.95)

    def nll(lam):
        psi = 1.0 - lam * lam
        inv_psi = 1.0 / psi
        u = lam * inv_psi
        denom = 1.0 + float(np.dot(lam, u))
        log_det = float(np.sum(np.log(psi))) + math.log(denom)
        ru = R @ u
        tr_term = float(np.sum(inv_psi)) - float(np.dot(u, ru)) / denom
        return log_det + tr_term - k

    bounds = [(-0.995, 0.995)] * k
    res = optimize.minimize(nll, init_lam, method="L-BFGS-B", bounds=bounds)
    lam = res.x
    if np.sum(lam) < 0:
        lam = -lam
    psi = 1.0 - lam * lam
    sum_lam_sq = float(np.sum(lam)) ** 2
    return float(sum_lam_sq / (sum_lam_sq + float(np.sum(psi))))


def steiger_z(r_xc: float, r_yc: float, r_xy: float, n: int) -> dict:
    """Steiger's (1980) Z test for two dependent correlations sharing a criterion (Appendix F.8)."""
    zx = math.atanh(max(-0.999999, min(0.999999, r_xc)))
    zy = math.atanh(max(-0.999999, min(0.999999, r_yc)))
    r_bar = 0.5 * (r_xc + r_yc)
    denom_c = (1.0 - r_bar * r_bar) ** 2
    c_bar = (r_xy * (1.0 - 2.0 * r_bar * r_bar) - 0.5 * (r_bar * r_bar) * (1.0 - 2.0 * r_bar * r_bar - r_xy * r_xy)) / denom_c
    c_bar = max(-0.999999, min(0.999999, c_bar))
    se = math.sqrt((2.0 - 2.0 * c_bar) / (n - 3.0))
    z = (zx - zy) / se
    p = float(2.0 * stats.norm.sf(abs(z)))
    return {"z": float(z), "p": p, "se_z": float(se), "r_bar": float(r_bar)}


def tost_dependent_corr(r_xc: float, r_yc: float, r_xy: float, n: int, delta_r: float) -> float:
    """Dependent-correlation TOST on Fisher z scale (Appendix F.8)."""
    st = steiger_z(r_xc, r_yc, r_xy, n)
    zx = math.atanh(r_xc)
    zy = math.atanh(r_yc)
    r_bar = st["r_bar"]
    delta_z = math.atanh(r_bar + 0.5 * delta_r) - math.atanh(r_bar - 0.5 * delta_r)
    z1 = ((zx - zy) + delta_z) / st["se_z"]
    z2 = ((zx - zy) - delta_z) / st["se_z"]
    p1 = float(stats.norm.sf(z1))
    p2 = float(stats.norm.cdf(z2))
    return max(p1, p2)


def bh_fdr(p_values) -> np.ndarray:
    """Benjamini-Hochberg FDR q-values."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q_sorted = np.empty(m, dtype=float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        val = min(prev, ranked[i] * m / float(i + 1))
        prev = val
        q_sorted[i] = min(1.0, val)
    q = np.empty(m, dtype=float)
    q[order] = q_sorted
    return q


def shannon_entropy_norm(probs) -> float:
    """Normalized Shannon entropy -sum(p ln p) / ln(K)."""
    p = np.asarray(probs, dtype=float)
    k = len(p)
    p = p[p > 1e-12]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)) / math.log(k))

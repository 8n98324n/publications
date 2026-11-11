"""Shared data loaders and statistical/psychometric methods for Study 1 and Study 2:
- Reads only de-identified CSV files in ../data/ (zero PII, zero external dependencies).
- Provides OLS + HC3 robust covariance estimation, Huber M-estimation, Tukey biweight regression,
  2x2 factorial and exposure time-dose contrasts, Two One-Sided Tests (TOST) equivalence and JZS Bayes Factor,
  Jacobson-Truax Reliable Change Index (RCI), and Bootstrap mediation (single & PROCESS Model 6 serial mediation).
"""

import os
import math
import json
import itertools
import numpy as np
import pandas as pd

try:
    from scipy import stats as _scipy_stats
    from scipy.integrate import quad as _scipy_quad
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONDITION_LABELS = {
    0: "Observation",
    1: "Petting",
    2: "Feeding",
    3: "Combined",
}

STAI_AA_ITEMS = [1, 2, 5, 8, 10, 11, 15, 16, 19, 20]
STAI_AP_ITEMS = [3, 4, 6, 7, 9, 12, 13, 14, 17, 18]
PANAS_PA_ITEMS = [1, 3, 5, 9, 10, 12, 14, 16, 17, 19]
PANAS_NA_ITEMS = [2, 4, 6, 7, 8, 11, 13, 15, 18, 20]


def _betacf(a: float, b: float, x: float) -> float:
    max_it = 200
    eps = 3.0e-14
    fpmin = 1.0e-30
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_it + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delt = d * c
        h *= delt
        if abs(delt - 1.0) < eps:
            break
    return h


def betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_pval_2sided(t_val: float, df: float) -> float:
    if _HAS_SCIPY:
        return float(2.0 * _scipy_stats.t.sf(abs(t_val), df))
    x = df / (df + t_val * t_val)
    return float(betai(0.5 * df, 0.5, x))


def t_cdf(t_val: float, df: float) -> float:
    if _HAS_SCIPY:
        return float(_scipy_stats.t.cdf(t_val, df))
    p2 = t_pval_2sided(t_val, df)
    return 1.0 - 0.5 * p2 if t_val >= 0 else 0.5 * p2


def t_sf(t_val: float, df: float) -> float:
    return 1.0 - t_cdf(t_val, df)


def t_ppf_975(df: float) -> float:
    if _HAS_SCIPY:
        return float(_scipy_stats.t.ppf(0.975, df))
    lo, hi = 1.95, 2.50
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if t_pval_2sided(mid, df) > 0.05:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def load_study1_data(main100_only: bool = True) -> pd.DataFrame:
    """Load de-identified Study 1 self-report and physiological dataset (N=100 main or N=101 sensitivity)."""
    path = os.path.join(DATA_DIR, "study1_self_report.csv")
    df = pd.read_csv(path, dtype={"participant_id": str})
    if main100_only:
        mask = df["in_study1_main100"].astype(str).str.lower() == "true"
        df = df[mask].reset_index(drop=True)
    df["cond"] = df["cond"].astype(int)
    df["P"] = df["Petting"].astype(float)
    df["F"] = df["Feeding"].astype(float)
    df["n_comp"] = df["P"] + df["F"]
    df["dose_P_linear"] = np.where(df["cond"] == 1, 1.0, np.where(df["cond"] == 3, 0.5, 0.0))
    df["dose_F_linear"] = np.where(df["cond"] == 2, 1.0, np.where(df["cond"] == 3, 0.5, 0.0))
    df["dose_P_sqrt"] = np.sqrt(df["dose_P_linear"])
    df["dose_F_sqrt"] = np.sqrt(df["dose_F_linear"])
    df["rci_stai"] = (df["STAI_improvement"] >= 7.33).astype(int)
    df["rci_stai_aa"] = (df["STAI_AA_improvement"] >= 5.0).astype(int)
    return df


def load_study2_data():
    """Load de-identified Study 2 multimodal datasets and CV splits."""
    part = pd.read_csv(os.path.join(DATA_DIR, "participants.csv"), dtype={"participant_id": str})
    fdef = pd.read_csv(os.path.join(DATA_DIR, "feature_definitions.csv"))
    f_pre = pd.read_csv(os.path.join(DATA_DIR, "features_pre.csv"), dtype={"participant_id": str})
    f_post = pd.read_csv(os.path.join(DATA_DIR, "features_post.csv"), dtype={"participant_id": str})
    f_delta = pd.read_csv(os.path.join(DATA_DIR, "features_delta.csv"), dtype={"participant_id": str})
    cv_path = os.path.join(DATA_DIR, "cv", "split_membership.csv")
    splits = pd.read_csv(cv_path, dtype={"participant_id": str}) if os.path.exists(cv_path) else None
    return part, fdef, f_pre, f_post, f_delta, splits


def cronbach_alpha(item_df: pd.DataFrame) -> float:
    """Compute Cronbach's alpha internal consistency coefficient."""
    dd = item_df.dropna()
    k = dd.shape[1]
    if k < 2:
        return np.nan
    return float(k / (k - 1.0) * (1.0 - dd.var(ddof=1).sum() / dd.sum(axis=1).var(ddof=1)))


def ols_fit(y: np.ndarray, X: np.ndarray) -> dict:
    """Fit ordinary least squares with both classical OLS and HC3 robust covariance."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    df_resid = n - k
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    sigma2 = float(np.sum(resid ** 2) / df_resid)
    cov_ols = sigma2 * XtX_inv

    H_diag = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    denom = np.maximum(1.0 - H_diag, 1e-6) ** 2
    meat = X.T @ ((resid ** 2 / denom)[:, None] * X)
    cov_hc3 = XtX_inv @ meat @ XtX_inv

    return {
        "beta": beta,
        "resid": resid,
        "cov_ols": cov_ols,
        "cov_hc3": cov_hc3,
        "df": df_resid,
        "n": n,
        "sigma2": sigma2,
    }


def linear_contrast(fit: dict, L: list | np.ndarray) -> dict:
    """Evaluate a linear contrast L' * beta under both OLS and HC3 standard errors."""
    L = np.asarray(L, dtype=float)
    est = float(L @ fit["beta"])
    se_ols = float(np.sqrt(max(L @ fit["cov_ols"] @ L, 1e-16)))
    se_hc3 = float(np.sqrt(max(L @ fit["cov_hc3"] @ L, 1e-16)))
    df = fit["df"]
    t_crit = t_ppf_975(df)
    t_ols = est / se_ols
    t_hc3 = est / se_hc3
    p_ols = t_pval_2sided(t_ols, df)
    p_hc3 = t_pval_2sided(t_hc3, df)
    return {
        "est": est,
        "se_ols": se_ols,
        "se_hc3": se_hc3,
        "t_ols": t_ols,
        "t_hc3": t_hc3,
        "p_ols": p_ols,
        "p_hc3": p_hc3,
        "ci_ols": (est - t_crit * se_ols, est + t_crit * se_ols),
        "ci_hc3": (est - t_crit * se_hc3, est + t_crit * se_hc3),
        "df": df,
    }


def tost_and_jzs_bf(est: float, se: float, df_resid: int, sesoi: float, n_eff: float, r_scale: float = 0.707) -> dict:
    """Compute Two One-Sided Tests (TOST) equivalence p-value and JZS Bayes Factor (BF10, BF01)."""
    p_tost = float(max(t_sf((est + sesoi) / se, df_resid), t_cdf((est - sesoi) / se, df_resid)))
    t_val = est / se

    def _integrand(g):
        return (
            ((1.0 + n_eff * g) ** -0.5)
            * ((1.0 + (t_val ** 2) / (df_resid * (1.0 + n_eff * g))) ** (-(df_resid + 1.0) / 2.0))
            * ((r_scale ** 2 / (2.0 * np.pi)) ** 0.5)
            * (g ** -1.5)
            * np.exp(-(r_scale ** 2) / (2.0 * g))
        )

    if _HAS_SCIPY:
        num = _scipy_quad(_integrand, 1e-8, np.inf, limit=200)[0]
    else:
        grid = np.logspace(-6, 4, 4000)
        vals = np.array([_integrand(g) for g in grid])
        num = float(np.trapz(vals, grid))
    den = (1.0 + (t_val ** 2) / df_resid) ** (-(df_resid + 1.0) / 2.0)
    bf10 = float(num / den)
    return {
        "estimate": est,
        "SESOI": float(sesoi),
        "TOST_p": p_tost,
        "BF10": bf10,
        "BF01": float(1.0 / bf10) if bf10 > 0 else np.inf,
    }


def holm_adjust(pvals) -> np.ndarray:
    """Holm step-down familywise error rate adjustment."""
    ps = np.asarray(pvals, dtype=float)
    m = len(ps)
    order = np.argsort(ps)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (m - k) * ps[i])
        adj[i] = min(running, 1.0)
    return adj


def bh_fdr_adjust(pvals) -> np.ndarray:
    """Benjamini-Hochberg false discovery rate q-values."""
    ps = np.asarray(pvals, dtype=float)
    m = len(ps)
    order = np.argsort(ps)
    q = np.empty(m, dtype=float)
    running = 1.0
    for k in range(m - 1, -1, -1):
        i = order[k]
        running = min(running, ps[i] * m / (k + 1.0))
        q[i] = running
    return q

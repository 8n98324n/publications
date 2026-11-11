# Copyright (c) 2026 Ivan Liu.
# Licensed under CC BY 4.0.
"""Shared loaders, constants and statistical helpers for the statistical_analysis notebooks.

Every notebook in this folder imports from this module so that the sample definition,
the discretization rule and the random seeds are identical across analyses. The
functions follow the definitions in Appendix E of the manuscript (Detailed Computational
Procedures of the Analysis Methods, sections E.1-E.6); the docstring of each function names
the appendix section it implements. Notebook NN_EK_* implements section E.K.

Experiment numbers are those of the manuscript. The committed CSV files keep the numbering in
force when the analyses were run: manuscript Exp 16 (vision + audio + physiology, no speech
semantics) is final/experiment_14.csv; manuscript Exp 14-15 are the human observer passes.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score

# --------------------------------------------------------------------------- paths
HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.normpath(os.path.join(HERE, "..", "final"))
REPRO = os.path.join(FINAL, "reproducibility")          # frozen outputs used for verification
OUT = os.path.join(HERE, "outputs")                     # outputs written by the notebooks
os.makedirs(OUT, exist_ok=True)

# --------------------------------------------------------------------------- constants
SEED = 42
NB = 10_000           # bootstrap resamples
NP = 10_000           # label permutations
NB_KAPPA = 2_000      # bootstrap resamples for the kappa interval
NB_DIM = 2_000        # permutations for the two-dimensional accuracy

CLS = ["ELVT", "SYMP", "INSP", "CTRL"]
GROUP_NAMES = {1: "ELVT", 2: "SYMP", 3: "INSP", 4: "CTRL"}        # numeric Group -> class code
SYM_HI = {"ELVT": 1, "SYMP": 1, "INSP": 0, "CTRL": 0}             # sympathy axis
INS_HI = {"ELVT": 1, "SYMP": 0, "INSP": 1, "CTRL": 0}             # inspiration axis

# manuscript experiment number -> configuration name
NAMES = {1: "Baseline(V)", 2: "V+A", 3: "V+A+T", 4: "V+A+T+P", 5: "CoT", 6: "KnowInj", 7: "FewShot",
         8: "FineTune", 9: "Agentic", 10: "ML V", 11: "ML V+A", 12: "ML V+A+T", 13: "ML V+A+T+P",
         14: "Observers, silent", 15: "Observers, audiovisual", 16: "V+A+P ctrl", 17: "Still frames, 6 MLLMs"}
MLLM_EXPS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 16]      # Exp 16 = supplementary control (Appendix D.1)
ML_EXPS = [10, 11, 12, 13]
OBS_EXPS = [14, 15]
GEE_EXPS = list(range(1, 14))                     # the 13 primary model configurations entering the GEE (E.5.1)

# manuscript experiment number -> committed CSV file (the files keep the numbering in force when
# the analyses were run; manuscript Exp 16 is experiment_14.csv)
MLLM_FILE = {e: f"experiment_{e}.csv" for e in range(1, 10)}
MLLM_FILE[16] = "experiment_14.csv"

# modalities used by the four machine-learning configurations (Exp 10-13)
ML_MODALITIES = {10: ["Facial", "pose", "Eye_Movement"],
                 11: ["Facial", "pose", "Eye_Movement", "Accoustics"],
                 12: ["Facial", "pose", "Eye_Movement", "Accoustics", "Speech Text"],
                 13: ["Facial", "pose", "Eye_Movement", "Accoustics", "Speech Text", "r-PPG", "thermal"]}


# --------------------------------------------------------------------------- discretization (E.1.1.1)
def mean_threshold_labels(symp, insp):
    """Appendix E.1.1.1: a score above the sample mean of its dimension is 'high'; crossing the two
    binary decisions gives the four classes. The thresholds are the means of the arrays passed in."""
    symp = np.asarray(symp, float); insp = np.asarray(insp, float)
    hs = symp > symp.mean(); hi = insp > insp.mean()
    return np.where(hs & hi, "ELVT", np.where(hs & ~hi, "SYMP", np.where(~hs & hi, "INSP", "CTRL")))


def labels_with_thresholds(symp, insp, ts, ti):
    """Same crossing rule with explicit thresholds (used by the supplementary sensitivity analysis, notebook 06)."""
    symp = np.asarray(symp, float); insp = np.asarray(insp, float)
    hs = symp > ts; hi = insp > ti
    return np.where(hs & hi, "ELVT", np.where(hs & ~hi, "SYMP", np.where(~hs & hi, "INSP", "CTRL")))


# --------------------------------------------------------------------------- loaders
def load_mllm_experiment(e: int) -> pd.DataFrame:
    """Return the frozen per-participant table for an MLLM configuration.

    Columns: pid, true (design label from the numeric Group column), symp, insp, pred.
    Rows without a model score or without a design label are dropped (the valid-N rule of
    Table A1 note); if a participant appears twice the first row is kept.
    """
    df = pd.read_csv(os.path.join(FINAL, MLLM_FILE[e]), low_memory=False)
    d = pd.DataFrame({"pid": pd.to_numeric(df["Participant_SN_int"], errors="coerce"),
                      "group": pd.to_numeric(df["Group"], errors="coerce"),
                      "symp": pd.to_numeric(df["sympathy"], errors="coerce"),
                      "insp": pd.to_numeric(df["inspiration"], errors="coerce")})
    d = d.dropna(subset=["pid", "group", "symp", "insp"]).drop_duplicates("pid", keep="first")
    d["pid"] = d.pid.astype(int); d["true"] = d.group.astype(int).map(GROUP_NAMES)
    d["pred"] = mean_threshold_labels(d.symp, d.insp)
    return d[["pid", "true", "pred", "symp", "insp"]].sort_values("pid").reset_index(drop=True)


def load_ml_predictions() -> pd.DataFrame:
    """Per-modality Random Forest predictions (final/machine_learning.csv).

    The file holds two acoustic runs (same five folds, differing only in the correlation-pruning
    flag), appended one after the other. The first block of acoustic rows is the run without
    pruning and is discarded; the second block (pruning on, the setting shared by every other
    modality) is kept so that acoustics contributes one vote per participant (Appendix C.2.4).
    Label codes 0-3 map to Group 1-4.
    """
    ml = pd.read_csv(os.path.join(FINAL, "machine_learning.csv"))
    ac = ml[ml.modality == "Accoustics"]
    kept_block = ac.iloc[len(ac) // 2:]
    ml = pd.concat([ml[ml.modality != "Accoustics"], kept_block], ignore_index=True)
    ml["true"] = (ml.actual.astype(int) + 1).map(GROUP_NAMES)
    ml["pred"] = (ml.predicted.astype(int) + 1).map(GROUP_NAMES)
    return ml.rename(columns={"user_id": "pid"})


def fuse_ml(ml: pd.DataFrame, e: int) -> pd.DataFrame:
    """Late fusion for Exp 10-13: the mode of the modality votes available for each participant;
    when votes tie, the smallest class code wins (pandas `Series.mode()` returns the tied values in
    ascending order and the first is taken), as in experiment_10_13.ipynb (Appendix C.2.4)."""
    sub = ml[ml.modality.isin(ML_MODALITIES[e])]
    rows = []
    for pid, g in sub.groupby("pid"):
        votes = g.predicted.astype(int)
        fused = int(votes.mode().iloc[0])
        rows.append(dict(pid=int(pid), true=GROUP_NAMES[int(g.actual.iloc[0]) + 1],
                         pred=GROUP_NAMES[fused + 1], n_votes=len(votes)))
    return pd.DataFrame(rows).sort_values("pid").reset_index(drop=True)


def design_labels(labels: dict) -> pd.DataFrame:
    """Design label for every participant seen in any configuration (pid -> true)."""
    return (pd.concat([v[["pid", "true"]] for v in labels.values()])
            .drop_duplicates("pid").sort_values("pid").reset_index(drop=True))


def load_observer_composites() -> pd.DataFrame:
    """Cleaned observer composites (final/reproducibility/observer_composite_scores_clean.csv):
    one row per participant x version x construct with the three raters and the complete-case
    consensus (mean of the three raters)."""
    return pd.read_csv(os.path.join(REPRO, "observer_composite_scores_clean.csv"))


def observer_panel(version: str, truth: pd.DataFrame) -> pd.DataFrame:
    """Appendix E.4.6: participants with a complete-case consensus on both constructs for the given
    pass, classified with the sample-mean rule applied to the consensus scores."""
    oc = load_observer_composites()
    c = (oc[oc.version == version].pivot(index="pid", columns="construct", values="consensus_complete_case")
         .dropna().reset_index().merge(truth, on="pid"))
    return pd.DataFrame(dict(pid=c.pid.astype(int).values, true=c.true.values,
                             pred=mean_threshold_labels(c.SYM, c.INS), symp=c.SYM.values, insp=c.INS.values))


def load_all_labels() -> dict:
    """Frozen per-participant label tables for every configuration with participant-level
    predictions (Exp 1-13, 16) plus the two observer passes (Exp 14-15)."""
    labels = {e: load_mllm_experiment(e) for e in MLLM_EXPS}
    ml = load_ml_predictions()
    for e in ML_EXPS:
        labels[e] = fuse_ml(ml, e)[["pid", "true", "pred"]]
    truth = design_labels(labels)
    labels[14] = observer_panel("silent", truth)
    labels[15] = observer_panel("audiovisual", truth)
    return labels


def load_frozen(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(REPRO, name))


# --------------------------------------------------------------------------- E.1 statistics
def axis_accuracy(true, pred):
    """Two-dimensional scoring (E.1.1.2): accuracy on the sympathy axis, on the inspiration axis,
    and their mean."""
    t = np.asarray(true); p = np.asarray(pred)
    sa = np.mean([SYM_HI[a] == SYM_HI[b] for a, b in zip(t, p)])
    ia = np.mean([INS_HI[a] == INS_HI[b] for a, b in zip(t, p)])
    return sa, ia, (sa + ia) / 2


def significance_battery(d: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Appendix E.1.1-E.1.2: accuracy, macro-F1, kappa with participant-bootstrap intervals, the one-sided
    permutation test (labels shuffled against fixed predictions, add-one correction) and the
    two-dimensional accuracy with its own permutation test."""
    t = d.true.values; p = d.pred.values; n = len(t)
    acc = (t == p).mean()
    f1 = f1_score(t, p, labels=CLS, average="macro")
    kap = cohen_kappa_score(t, p, labels=CLS)
    idx = rng.integers(0, n, (NB, n))
    accs = (t[idx] == p[idx]).mean(1)
    kaps = np.array([cohen_kappa_score(t[i], p[i], labels=CLS) for i in idx[:NB_KAPPA]])
    perm = np.array([(rng.permutation(t) == p).mean() for _ in range(NP)])
    p_perm = (np.sum(perm >= acc) + 1) / (NP + 1)
    sa, ia, da = axis_accuracy(t, p)
    dperm = np.array([axis_accuracy(rng.permutation(t), p)[2] for _ in range(NB_DIM)])
    return dict(n=n, acc=acc, acc_lo=np.percentile(accs, 2.5), acc_hi=np.percentile(accs, 97.5),
                macro_f1=f1, kappa=kap, kappa_lo=np.percentile(kaps, 2.5), kappa_hi=np.percentile(kaps, 97.5),
                p_perm=p_perm, dim_acc=da, symp_axis=sa, insp_axis=ia,
                p_perm_dim=(np.sum(dperm >= da) + 1) / (NB_DIM + 1))


def confusion(d: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(pd.Categorical(d.true, CLS), pd.Categorical(d.pred, CLS), dropna=False)


def per_class_recall(d: pd.DataFrame) -> dict:
    cm = confusion(d).values.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.diag(cm) / cm.sum(1)
    return dict(zip(CLS, r))


def expand_confusion(cm) -> tuple:
    """Turn a 4x4 confusion matrix (rows = true, columns = predicted) back into label vectors."""
    cm = np.asarray(cm, int); t = []; p = []
    for i in range(4):
        for j in range(4):
            t += [CLS[i]] * cm[i, j]; p += [CLS[j]] * cm[i, j]
    return np.array(t), np.array(p)


# --------------------------------------------------------------------------- E.2 statistics
def macro_auc(d: pd.DataFrame) -> float:
    """Supplementary sensitivity analysis (notebook 06; not reported in the manuscript): one-vs-rest macro-AUC from the two continuous scores. Each centred
    score pair is projected onto the four class directions (ELVT +S+I, SYMP +S-I, INSP -S+I,
    CTRL -S-I); the projection is the class propensity."""
    s = d.symp.values - d.symp.mean(); i = d.insp.values - d.insp.mean()
    dirs = {"ELVT": (1, 1), "SYMP": (1, -1), "INSP": (-1, 1), "CTRL": (-1, -1)}
    aucs = []
    for c, (a, b) in dirs.items():
        y = (d.true.values == c).astype(int)
        if y.sum() in (0, len(y)):
            continue
        aucs.append(roc_auc_score(y, a * s + b * i))
    return float(np.mean(aucs))


def anova_2x2(d: pd.DataFrame, e: int) -> pd.DataFrame:
    """Appendix E.2 (Table A2) and E.4.3: Type II two-way ANOVA of each continuous score on the two induction factors,
    partial eta squared with a 90% noncentral-F interval."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from scipy.optimize import brentq
    d = d.copy(); d["S"] = d.true.map(SYM_HI); d["I"] = d.true.map(INS_HI); out = []
    for score in ("symp", "insp"):
        m = smf.ols(f"{score} ~ C(S)*C(I)", data=d).fit(); t = sm.stats.anova_lm(m, typ=2)
        err = t.loc["Residual", "sum_sq"]; df_e = int(t.loc["Residual", "df"])
        for eff, name in [("C(S)", "Sympathy manip"), ("C(I)", "Inspiration manip"), ("C(S):C(I)", "Interaction")]:
            ss = t.loc[eff, "sum_sq"]; F = t.loc[eff, "F"]; p = t.loc[eff, "PR(>F)"]; df1 = int(t.loc[eff, "df"])
            eta = ss / (ss + err)

            def ncp(q):
                f = lambda nc: stats.ncf.cdf(F, df1, df_e, nc) - q
                if f(0) < 0:
                    return 0.0
                hi = 50
                while f(hi) > 0:
                    hi *= 2
                return brentq(f, 0, hi)
            lo, hi = ncp(.95), ncp(.05)
            out.append(dict(exp=e, N=len(d), score=score, effect=name, SS_type="II", df1=df1, df2=df_e, F=F, p=p,
                            eta_p2=eta, eta_lo90=lo / (lo + df1 + df_e + 1), eta_hi90=hi / (hi + df1 + df_e + 1)))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- correlations (E.3, E.4)
def spearman_boot(x, y, rng, B=NB):
    """Vectorised bootstrap of Spearman rho (ranks recomputed within each resample)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x); idx = rng.integers(0, n, size=(B, n))
    rx = stats.rankdata(x[idx], axis=1); ry = stats.rankdata(y[idx], axis=1)
    rx = rx - rx.mean(1, keepdims=True); ry = ry - ry.mean(1, keepdims=True)
    den = np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (rx * ry).sum(1) / den
    return r[np.isfinite(r)]


def spearman_with_ci(x, y, seed=SEED):
    rho, p = stats.spearmanr(x, y)
    r = spearman_boot(x, y, np.random.default_rng(seed))
    return rho, p, np.percentile(r, 2.5), np.percentile(r, 97.5)


def partial_spearman(x, y, Z):
    """Spearman partial correlation: rank x and y, residualise on [1, Z], Pearson of residuals."""
    rx = stats.rankdata(x); ry = stats.rankdata(y)
    X = np.column_stack([np.ones(len(rx)), Z])
    bx, *_ = np.linalg.lstsq(X, rx, rcond=None); by, *_ = np.linalg.lstsq(X, ry, rcond=None)
    ex = rx - X @ bx; ey = ry - X @ by
    r = np.corrcoef(ex, ey)[0, 1]
    n, k = len(rx), Z.shape[1]; df = n - 2 - k
    t = r * np.sqrt(df / max(1e-12, 1 - r ** 2))
    return r, 2 * stats.t.sf(abs(t), df)


def partial_spearman_boot(x, y, Z, seed=SEED, B=NB):
    x = np.asarray(x, float); y = np.asarray(y, float); Z = np.asarray(Z, float)
    r, p = partial_spearman(x, y, Z); rng = np.random.default_rng(seed); n = len(x); out = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        try:
            out[b] = partial_spearman(x[i], y[i], Z[i])[0]
        except Exception:
            out[b] = np.nan
    out = out[np.isfinite(out)]
    return r, p, np.percentile(out, 2.5), np.percentile(out, 97.5), n


def icc_mw(X):
    """McGraw & Wong (1996) two-way random-effects ICCs for an n x k complete matrix:
    A1 = ICC(A,1), Ak = ICC(A,k), C1 = ICC(C,1), Ck = ICC(C,k)."""
    X = np.asarray(X, float); n, k = X.shape
    gm = X.mean(); rm = X.mean(1); cm = X.mean(0)
    SSR = k * ((rm - gm) ** 2).sum(); SSC = n * ((cm - gm) ** 2).sum()
    SSE = ((X - rm[:, None] - cm[None, :] + gm) ** 2).sum()
    MSR, MSC, MSE = SSR / (n - 1), SSC / (k - 1), SSE / ((n - 1) * (k - 1))
    return dict(A1=(MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n),
                Ak=(MSR - MSE) / (MSR + (MSC - MSE) / n),
                C1=(MSR - MSE) / (MSR + (k - 1) * MSE),
                Ck=(MSR - MSE) / MSR)


def icc_boot(X, rng, B=NB):
    X = np.asarray(X, float); n = X.shape[0]; idx = rng.integers(0, n, size=(B, n))
    out = {k: np.empty(B) for k in ("A1", "Ak", "C1")}
    for b in range(B):
        r = icc_mw(X[idx[b]])
        for k in out:
            out[k][b] = r[k]
    return out


def cohen_d(a, b):
    """Cohen's d with pooled SD and its small-sample corrected form (Hedges' g)."""
    a, b = np.asarray(a, float), np.asarray(b, float); na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    d = (a.mean() - b.mean()) / sp
    return d, d * (1 - 3 / (4 * (na + nb) - 9))


# --------------------------------------------------------------------------- E.5 statistics
def long_correctness(labels: dict, exps) -> pd.DataFrame:
    """Long table (pid, exp, correct, symp_ok, insp_ok) for the GEE."""
    L = []
    for e in exps:
        d = labels[e].copy(); d["exp"] = e; d["correct"] = (d.true == d.pred).astype(int)
        d["symp_ok"] = [int(SYM_HI[a] == SYM_HI[b]) for a, b in zip(d.true, d.pred)]
        d["insp_ok"] = [int(INS_HI[a] == INS_HI[b]) for a, b in zip(d.true, d.pred)]
        L.append(d[["pid", "exp", "correct", "symp_ok", "insp_ok"]])
    out = pd.concat(L, ignore_index=True); out["pid"] = out.pid.astype(int)
    return out


def fit_gee(long: pd.DataFrame, outcome: str, exps):
    """Appendix E.5.1: population-averaged binomial GEE, participant clusters, exchangeable working
    correlation, robust covariance; Exp 1 (vision only) is the reference level."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    d = long[long.exp.isin(exps)].copy(); d["exp"] = pd.Categorical(d.exp, categories=exps)
    return smf.gee(f"{outcome} ~ C(exp)", groups="pid", data=d, family=sm.families.Binomial(),
                   cov_struct=sm.cov_struct.Exchangeable()).fit()


def gee_table(m, outcome: str, exps) -> pd.DataFrame:
    ci = m.conf_int(); rows = []
    for e in exps[1:]:
        k = f"C(exp)[T.{e}]"
        rows.append(dict(outcome=outcome, exp=e, name=NAMES[e], OR=np.exp(m.params[k]),
                         ci_lo=np.exp(ci.loc[k, 0]), ci_hi=np.exp(ci.loc[k, 1]), p=m.pvalues[k]))
    return pd.DataFrame(rows)


def gee_contrast(m, a: int, b: int) -> dict:
    """OR of configuration b relative to configuration a from one fitted GEE (difference of the two
    coefficients; Wald test with the robust covariance)."""
    names = list(m.params.index); c = np.zeros(len(names))

    def idx(e):
        return None if e == 1 else names.index(f"C(exp)[T.{e}]")
    if idx(b) is not None:
        c[idx(b)] += 1
    if idx(a) is not None:
        c[idx(a)] -= 1
    est = float(c @ m.params.values); se = float(np.sqrt(c @ m.cov_params().values @ c)); z = est / se
    return dict(OR=np.exp(est), or_lo=np.exp(est - 1.96 * se), or_hi=np.exp(est + 1.96 * se), z=z,
                p_gee=2 * stats.norm.sf(abs(z)))


def paired_descriptives(labels: dict, a: int, b: int, rng) -> dict:
    """Descriptive effect sizes for a contrast (E.5.4): raw accuracies on the shared participants,
    bootstrap interval of the accuracy difference, and the numbers improved / worsened."""
    m = labels[a][["pid", "true", "pred"]].merge(labels[b][["pid", "pred"]], on="pid", suffixes=("_a", "_b"))
    ca = (m.true == m.pred_a).values.astype(int); cb = (m.true == m.pred_b).values.astype(int)
    idx = rng.integers(0, len(m), (NB, len(m))); diffs = cb[idx].mean(1) - ca[idx].mean(1)
    return dict(n_paired=len(m), acc_a=ca.mean(), acc_b=cb.mean(), diff=cb.mean() - ca.mean(),
                diff_lo=np.percentile(diffs, 2.5), diff_hi=np.percentile(diffs, 97.5),
                improved=int(((ca == 0) & (cb == 1)).sum()), worsened=int(((ca == 1) & (cb == 0)).sum()))


# --------------------------------------------------------------------------- E.6 explanation coding
EXP4_TEXT_CUES = ["text_sad", "text_anger", "text_touched", "text_pride", "text_admire"]
EXP4_NONVERBAL_CUES = ["face_tears", "face_frown", "face_smile", "face_neutral", "vocal_subdued", "vocal_excited",
                       "body_still", "body_animated"]
EXP4_PHYSIO_CUES = ["cheek_up", "cheek_down", "forehead_up", "forehead_down", "nose_up", "nose_down",
                    "face_temp_up_other", "face_temp_down_other", "hr_up", "hr_down", "hrv_up", "hrv_down",
                    "phys_nodirection"]


def exp4_channel(cue: str) -> str:
    """Channel of a cue in the 26-cue schema (Appendix E.6.1): text, nonverbal or physio."""
    if cue in EXP4_TEXT_CUES:
        return "text"
    if cue in EXP4_NONVERBAL_CUES:
        return "nonverbal"
    return "physio"


def load_exp4_cue_table() -> pd.DataFrame:
    """Appendix E.6.1: the 26-cue coding of the 100 Exp 4 explanations.

    The five textual and eight nonverbal cues come from the first coding pass
    (final/exp4_feature_coding.csv, which also holds pid, the design label `true` and the
    mean-threshold predicted class `pred`); its seven undirected physiological codes (phys_*) are
    dropped. The thirteen directional physiological cues come from the second pass
    (final/exp4_physio_direction_coding.csv). The two files are joined on pid.
    """
    first = pd.read_csv(os.path.join(FINAL, "exp4_feature_coding.csv"))
    phys = pd.read_csv(os.path.join(FINAL, "exp4_physio_direction_coding.csv"))
    keep = ["pid", "true", "pred"] + EXP4_TEXT_CUES + EXP4_NONVERBAL_CUES
    out = first[keep].merge(phys[["pid"] + EXP4_PHYSIO_CUES], on="pid", how="inner", validate="one_to_one")
    assert len(out) == len(first) == len(phys), "the two coding passes must cover the same participants"
    return out


# --------------------------------------------------------------------------- reporting helpers
def fmt_p(p):
    return "< .001" if p < .001 else (f"= {p:.3f}" if p < .1 else f"= {p:.2f}").replace("0.", ".")


def compare(computed: pd.DataFrame, frozen: pd.DataFrame, keys, cols, atol=0.005) -> pd.DataFrame:
    """Align a computed table with its frozen counterpart on `keys` and report the maximum absolute
    difference per column. Used at the end of every notebook as the reproduction check."""
    m = computed.merge(frozen, on=keys, suffixes=("", "_frozen"))
    rows = []
    for c in cols:
        if c + "_frozen" not in m.columns:
            continue
        diff = (m[c].astype(float) - m[c + "_frozen"].astype(float)).abs()
        rows.append(dict(column=c, n_compared=int(diff.notna().sum()), max_abs_diff=float(diff.max()),
                         within_tolerance=bool(diff.max() <= atol)))
    return pd.DataFrame(rows)

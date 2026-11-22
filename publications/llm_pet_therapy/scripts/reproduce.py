from pathlib import Path
import hashlib
import argparse
import csv
import pandas as pd
import numpy as np
import scipy.stats as st
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "generated"

def verify_assets():
    with (ROOT / "manuscript_snapshot/figures.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        path = ROOT / row["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], path
    return rows

def load_participants():
    df = pd.read_csv(DATA / "participants.csv", dtype={"participant_id": str})
    assert len(df) == df.participant_id.nunique() == 100
    assert df.in_prediction_scope.sum() == 92
    for stage in ("pre", "post"):
        np.testing.assert_array_equal(df[f"STAI_{stage}"].to_numpy(),
                                      (df[f"STAI_AA_{stage}"] + df[f"STAI_AP_{stage}"]).to_numpy())
    return df

def design(df):
    df = df.copy()
    df["pet_dose"] = df.condition.map({"observation": 0, "petting": 1, "feeding": 0, "combined": .5})
    df["feed_dose"] = df.condition.map({"observation": 0, "petting": 0, "feeding": 1, "combined": .5})
    df["combined"] = df.condition.eq("combined").astype(int)
    assert df[["pet_dose", "feed_dose"]].notna().all().all()
    return df

def dose(df, baseline, post, relief, include_departure=True):
    frame = design(df).dropna(subset=[baseline, post])
    terms = ["pet_dose", "feed_dose"]
    if include_departure:
        terms.append("combined")
    fit = smf.ols(f"{post} ~ {baseline} + {' + '.join(terms)}", frame).fit()
    adjusted = fit.get_robustcov_results(cov_type="HC3")
    sign = -1 if relief else 1
    for contrast in terms:
        j = fit.model.exog_names.index(contrast)
        lo, hi = sorted(sign * fit.conf_int().iloc[j])
        yield dict(n=len(frame), contrast=contrast, b=sign*fit.params[contrast],
                   lower=lo, upper=hi, p=fit.pvalues[contrast], p_hc3=adjusted.pvalues[j])

def analysis1(df):
    effects, desc = [], []
    for scale in ("STAI", "STAI_AA", "STAI_AP"):
        pre, post = f"{scale}_pre", f"{scale}_post"
        for label, departure in [("additive_dose", False), ("with_departure", True)]:
            effects.extend(dict(scale=scale, model=label, **r)
                           for r in dose(df, pre, post, relief=True, include_departure=departure))
        for condition, group in df.groupby("condition"):
            delta = group[pre] - group[post]
            desc.append(dict(scale=scale, condition=condition, n=len(group), pre=group[pre].mean(),
                             pre_sd=group[pre].std(), post=group[post].mean(), post_sd=group[post].std(),
                             relief=delta.mean(), relief_sd=delta.std(),
                             reliable_relief_n=int(delta.ge(7.33).sum()) if scale=="STAI" else None))
    effects = pd.DataFrame(effects)
    aa_family = effects.scale.eq("STAI_AA") & effects.model.eq("additive_dose")
    effects.loc[aa_family, "p_holm_stai_aa_two_tests"] = multipletests(
        effects.loc[aa_family, "p"], method="holm")[1]
    return pd.DataFrame(desc), effects

def analysis2(df):
    features = pd.read_csv(DATA / "features_paired.csv", dtype={"participant_id": str})
    sample = features.loc[features.eligible_factorial].merge(df[["participant_id","condition"]], on="participant_id", validate="many_to_one")
    rows = []
    definitions = pd.read_csv(DATA / "feature_definitions.csv")
    proportion_features = set(definitions.loc[definitions.display_unit.eq("proportion (0-1)"), "feature_id"])
    for feature, part in sample.groupby("feature_id"):
        part = part.copy()
        if feature in proportion_features:
            part[["pre_value","post_value"]] *= 100
        rows.extend(dict(feature=feature, **r) for r in dose(
            part, "pre_value", "post_value", relief=False, include_departure=False))
        rows.extend(dict(feature=feature, **r) for r in dose(
            part, "pre_value", "post_value", relief=False) if r["contrast"]=="combined")
    result = pd.DataFrame(rows)
    result["p_bh_all_dose_and_departure"] = multipletests(result.p, method="fdr_bh")[1]
    return result

def mllm_replay():
    files = sorted((DATA / "model_predictions").glob("experiment_*.csv"))
    if len(files) != 4:
        raise FileNotFoundError("Expected four local score snapshots in data/model_predictions.")
    rows=[]
    for path in files:
        df=pd.read_csv(path, dtype={"participant_id": str})
        for key, scale in [("total","STAI"), ("aa","STAI_AA"), ("ap","STAI_AP")]:
            y=np.concatenate([df[f"{scale}_{stage}"].to_numpy() for stage in ("pre","post")])
            pred=np.concatenate([df[f"stai_{key}_{stage}_pred"].to_numpy() for stage in ("pre","post")])
            valid=np.isfinite(y)&np.isfinite(pred)
            y,pred=y[valid],pred[valid]
            rows.append(dict(experiment=path.stem, scale=scale, n=len(y), mae=np.abs(y-pred).mean(),
                             r2=1-((y-pred)**2).sum()/((y-y.mean())**2).sum(), pearson_r=st.pearsonr(y,pred).statistic))
    return pd.DataFrame(rows)

def residual_correlations():
    outcomes = pd.read_csv(DATA / "study1_self_report.csv", dtype={"participant_id": str})
    outcomes = outcomes.loc[outcomes.in_study1_main100]
    features = pd.read_csv(DATA / "features_paired.csv", dtype={"participant_id": str})
    paired = features.loc[features.eligible_correlation].merge(
        outcomes, on="participant_id", validate="many_to_one")
    rows = []
    for feature, part in paired.groupby("feature_id"):
        for scale in ("STAI", "STAI_AA", "STAI_AP", "GSE", "PANAS_PA", "mattering"):
            baseline, post = f"{scale}_pre", f"{scale}_post"
            cols = ["pre_value", "post_value", post]
            if scale != "mattering":
                cols.append(baseline)
            frame = part.dropna(subset=cols)
            fr = smf.ols("post_value ~ pre_value", frame).fit().resid
            yr = frame[post] if scale == "mattering" else smf.ols(f"{post} ~ {baseline}", frame).fit().resid
            if scale.startswith("STAI"):
                yr = -yr
            test = st.pearsonr(fr, yr)
            rows.append(dict(feature=feature, outcome=scale, n=len(frame), r=test.statistic, p=test.pvalue))
    result = pd.DataFrame(rows)
    assert len(result) == 114
    result["q"] = multipletests(result.p, method="fdr_bh")[1]
    return result

def make_figures(df, effects):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target=OUT / "figures"
    target.mkdir(parents=True, exist_ok=True)
    order=["observation","petting","feeding","combined"]
    fig, axs=plt.subplots(1,3,figsize=(12,3.6),layout="constrained")
    for ax, scale in zip(axs, ("STAI","STAI_AA","STAI_AP")):
        for name in order:
            part=df.loc[df.condition.eq(name),[f"{scale}_pre",f"{scale}_post"]]
            ax.errorbar([0,1],part.mean(),yerr=part.sem(),marker="o",label=name)
        ax.set(xticks=[0,1],xticklabels=["Pretest","Posttest"],title=scale)
    axs[-1].legend()
    fig.savefig(target / "figure_03_pre_post_replot.png",dpi=180)
    plt.close(fig)
    plot=effects.pivot(index="feature",columns="contrast",values="b")
    fig,axs=plt.subplots(1,len(plot.columns),figsize=(12,6),sharey=True,layout="constrained")
    for ax,column in zip(axs,plot.columns):
        ax.barh(plot.index,plot[column]); ax.axvline(0,color="black",lw=.65); ax.set_title(column)
    fig.savefig(target / "figure_07_dose_coefficients_replot.png",dpi=180)
    plt.close(fig)

def run():
    verify_assets()
    OUT.mkdir(exist_ok=True)
    df=load_participants()
    desc,effects=analysis1(df)
    feature_effects=analysis2(df)
    scores=mllm_replay()
    primary=effects.query("scale == 'STAI' and model == 'additive_dose'")
    np.testing.assert_array_equal(primary.b.round(2).to_numpy(), [3.17, 2.92])
    np.testing.assert_array_equal(scores.query("scale == 'STAI'").mae.round(2).to_numpy(),
                                  [7.66, 7.76, 8.18, 8.64])
    for name, table in [("stai_descriptives",desc),("stai_dose",effects),
                        ("feature_dose",feature_effects),("mllm_scores",scores),
                        ("feature_residual_correlations",residual_correlations())]:
        table.to_csv(OUT / f"{name}.csv", index=False)
    make_figures(df, feature_effects)
    return effects, scores

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Run the v1.29 tabular checks and partial figure redraws.")
    parser.parse_args()
    effects,scores=run()
    print(effects.to_string(index=False))
    print(scores.to_string(index=False))

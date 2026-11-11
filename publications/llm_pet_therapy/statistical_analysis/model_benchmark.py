from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC, SVC, SVR


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
REF_CV = DATA / "cv" / "split_membership.csv"
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20261102
CONDITION_ORDER = ["observation", "petting", "feeding", "combined"]


@dataclass
class Spec:
    name: str
    feature_set: str
    estimator: Any
    grid: dict[str, list[Any]]
    task: str
    target_mode: str = "post"
    selectable: bool = True


def standard_pipeline(model: Any, *, selector: bool = False) -> Pipeline:
    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
    if selector:
        steps.append(("select", SelectKBest(score_func=f_classif)))
    steps.append(("model", model))
    return Pipeline(steps)


def tree_pipeline(model: Any) -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def modality_transformer(feature_ids: list[str], prefix_offset: int = 0, include_basic: bool = False) -> ColumnTransformer:
    definitions = pd.read_csv(DATA / "feature_definitions.csv")
    category_map = dict(zip(definitions.feature_id, definitions.category))
    transformers: list[tuple[str, Any, list[int]]] = []
    if include_basic:
        transformers.append(("basic", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), list(range(4))))
    base = 4 if include_basic else 0
    for category in ["Facial", "Acoustic", "Thermal", "Cardiovascular"]:
        indices = [base + i + prefix_offset for i, fid in enumerate(feature_ids) if category_map[fid] == category]
        if not indices:
            continue
        transformers.append(
            (
                f"m{len(transformers)}",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                        ("pca", __import__("sklearn.decomposition").decomposition.PCA(n_components=1)),
                    ]
                ),
                indices,
            )
        )
    return ColumnTransformer(transformers, remainder="drop")


def load_data() -> tuple[pd.DataFrame, list[str], dict[str, np.ndarray]]:
    participants = pd.read_csv(DATA / "participants.csv", dtype={"participant_id": str})
    participants = participants.loc[participants.in_prediction_scope.astype(bool)].copy()
    participants["participant_id"] = participants.participant_id.str.zfill(3)
    participants["interaction"] = participants.Petting * participants.Feeding
    participants["condition"] = pd.Categorical(participants.condition, categories=CONDITION_ORDER, ordered=True)
    participants = participants.sort_values("participant_id").reset_index(drop=True)

    definitions = pd.read_csv(DATA / "feature_definitions.csv")
    feature_ids = definitions.feature_id.tolist()
    merged = participants.copy()
    for stage, filename in [("pre", "features_pre.csv"), ("post", "features_post.csv"), ("delta", "features_delta.csv")]:
        frame = pd.read_csv(DATA / filename, dtype={"participant_id": str})
        frame["participant_id"] = frame.participant_id.str.zfill(3)
        keep = ["participant_id"] + feature_ids
        frame = frame[keep].rename(columns={fid: f"{stage}__{fid}" for fid in feature_ids})
        merged = merged.merge(frame, on="participant_id", how="left", validate="one_to_one")

    arrays: dict[str, np.ndarray] = {}
    arrays["delta19"] = merged[[f"delta__{fid}" for fid in feature_ids]].to_numpy(float)
    arrays["post19"] = merged[[f"post__{fid}" for fid in feature_ids]].to_numpy(float)
    arrays["prepost38"] = merged[[f"pre__{fid}" for fid in feature_ids] + [f"post__{fid}" for fid in feature_ids]].to_numpy(float)
    basic_cols = ["STAI_pre", "Petting", "Feeding", "interaction"]
    arrays["basic4"] = merged[basic_cols].to_numpy(float)
    arrays["ext_post23"] = np.column_stack([arrays["basic4"], arrays["post19"]])
    arrays["ext_delta23"] = np.column_stack([arrays["basic4"], arrays["delta19"]])
    arrays["paired42"] = np.column_stack([arrays["basic4"], arrays["prepost38"]])
    return merged, feature_ids, arrays


def make_specs(feature_ids: list[str]) -> tuple[list[Spec], list[Spec]]:
    logit_grid = {"model__C": [0.01, 0.1, 1.0, 10.0]}
    class_specs = [
        Spec("dummy", "delta19", DummyClassifier(strategy="most_frequent"), {}, "classification", selectable=False),
        Spec("rf_fixed", "delta19", tree_pipeline(RandomForestClassifier(n_estimators=500, max_depth=3, min_samples_leaf=5, max_features="sqrt", random_state=SEED, n_jobs=1)), {}, "classification", selectable=False),
        Spec("logit_l2_delta", "delta19", standard_pipeline(LogisticRegression(penalty="l2", solver="lbfgs", max_iter=10000, class_weight="balanced", random_state=SEED)), logit_grid, "classification"),
        Spec("lda_shrink_delta", "delta19", standard_pipeline(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")), {}, "classification"),
        Spec("linear_svm_delta", "delta19", standard_pipeline(LinearSVC(dual="auto", class_weight="balanced", max_iter=20000, random_state=SEED)), {"model__C": [0.01, 0.1, 1.0, 10.0]}, "classification"),
        Spec("rbf_svm_delta", "delta19", standard_pipeline(SVC(kernel="rbf", class_weight="balanced", random_state=SEED)), {"model__C": [0.1, 1.0, 10.0], "model__gamma": ["scale", 0.03, 0.1, 0.3]}, "classification"),
        Spec("select_logit_delta", "delta19", standard_pipeline(LogisticRegression(penalty="l2", solver="lbfgs", max_iter=10000, class_weight="balanced", random_state=SEED), selector=True), {"select__k": [5, 10, 15, "all"], "model__C": [0.1, 1.0, 10.0]}, "classification"),
        Spec("logit_l2_paired", "prepost38", standard_pipeline(LogisticRegression(penalty="l2", solver="lbfgs", max_iter=10000, class_weight="balanced", random_state=SEED)), logit_grid, "classification"),
        Spec(
            "modality_pca_logit_delta",
            "delta19",
            Pipeline([("modalities", modality_transformer(feature_ids)), ("model", LogisticRegression(penalty="l2", solver="lbfgs", max_iter=10000, class_weight="balanced", random_state=SEED))]),
            {"model__C": [0.01, 0.1, 1.0, 10.0]},
            "classification",
        ),
    ]

    common_rf = dict(n_estimators=500, max_depth=3, min_samples_leaf=5, min_samples_split=2, bootstrap=True, max_features=1.0, random_state=SEED, n_jobs=1)
    reg_specs = [
        Spec("mean", "basic4", DummyRegressor(strategy="mean"), {}, "regression", selectable=False),
        Spec("rf_basic_fixed", "basic4", tree_pipeline(RandomForestRegressor(**common_rf)), {}, "regression", selectable=False),
        Spec("rf_post_fixed", "post19", tree_pipeline(RandomForestRegressor(**common_rf)), {}, "regression", selectable=False),
        Spec("rf_extended_fixed", "ext_post23", tree_pipeline(RandomForestRegressor(**common_rf)), {}, "regression", selectable=False),
        Spec("linear_basic", "basic4", standard_pipeline(LinearRegression()), {}, "regression"),
        Spec("ridge_basic", "basic4", standard_pipeline(Ridge()), {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}, "regression"),
        Spec("huber_basic", "basic4", standard_pipeline(HuberRegressor(max_iter=5000)), {"model__epsilon": [1.1, 1.35, 1.8], "model__alpha": [0.0001, 0.01, 1.0]}, "regression"),
        Spec("ridge_post", "post19", standard_pipeline(Ridge()), {"model__alpha": [0.1, 1.0, 10.0, 100.0]}, "regression"),
        Spec("elastic_post", "post19", standard_pipeline(ElasticNet(max_iter=20000, random_state=SEED)), {"model__alpha": [0.01, 0.1, 1.0], "model__l1_ratio": [0.1, 0.5, 0.9]}, "regression"),
        Spec("linear_svr_post", "post19", standard_pipeline(SVR(kernel="linear")), {"model__C": [0.01, 0.1, 1.0, 10.0], "model__epsilon": [0.1, 1.0, 2.0]}, "regression"),
        Spec("ridge_extended_post", "ext_post23", standard_pipeline(Ridge()), {"model__alpha": [0.1, 1.0, 10.0, 100.0]}, "regression"),
        Spec("elastic_extended_post", "ext_post23", standard_pipeline(ElasticNet(max_iter=20000, random_state=SEED)), {"model__alpha": [0.01, 0.1, 1.0], "model__l1_ratio": [0.1, 0.5, 0.9]}, "regression"),
        Spec("ridge_extended_delta", "ext_delta23", standard_pipeline(Ridge()), {"model__alpha": [0.1, 1.0, 10.0, 100.0]}, "regression"),
        Spec("ridge_paired", "paired42", standard_pipeline(Ridge()), {"model__alpha": [0.1, 1.0, 10.0, 100.0]}, "regression"),
        Spec("pls_extended_post", "ext_post23", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", PLSRegression(scale=False, max_iter=2000))]), {"model__n_components": [1, 2, 3, 5, 8]}, "regression"),
        Spec(
            "modality_pca_ridge_post",
            "ext_post23",
            Pipeline([("modalities", modality_transformer(feature_ids, include_basic=True)), ("model", Ridge())]),
            {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
            "regression",
        ),
        Spec("ridge_change_delta", "ext_delta23", standard_pipeline(Ridge()), {"model__alpha": [0.1, 1.0, 10.0, 100.0]}, "regression", target_mode="improvement"),
        Spec("elastic_change_delta", "ext_delta23", standard_pipeline(ElasticNet(max_iter=20000, random_state=SEED)), {"model__alpha": [0.01, 0.1, 1.0], "model__l1_ratio": [0.1, 0.5, 0.9]}, "regression", target_mode="improvement"),
    ]
    return class_specs, reg_specs


def split_index(frame: pd.DataFrame) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    splits = pd.read_csv(REF_CV, dtype={"participant_id": str})
    splits = splits.loc[splits.sample_version.eq("main92")].copy()
    id_to_idx = {pid: i for i, pid in enumerate(frame.participant_id)}
    result = []
    for repeat in sorted(splits.repeat.unique()):
        for fold in sorted(splits.loc[splits.repeat.eq(repeat), "fold"].unique()):
            block = splits.loc[(splits.repeat.eq(repeat)) & (splits.fold.eq(fold))]
            train = np.array([id_to_idx[x] for x in block.loc[block.role.eq("train"), "participant_id"]], dtype=int)
            test = np.array([id_to_idx[x] for x in block.loc[block.role.eq("test"), "participant_id"]], dtype=int)
            result.append((int(repeat), int(fold), train, test))
    return result


def json_params(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)


def class_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
    }


def reg_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    if np.std(pred) == 0 or np.std(y) == 0:
        r = np.nan
    else:
        r = float(pearsonr(y, pred).statistic)
    return {
        "mae": mean_absolute_error(y, pred),
        "rmse": mean_squared_error(y, pred) ** 0.5,
        "r2": r2_score(y, pred),
        "pearson_r": r,
    }


def fit_candidate(spec: Spec, X: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray, inner_labels: np.ndarray, seed: int) -> tuple[np.ndarray, float, dict[str, Any]]:
    estimator = clone(spec.estimator)
    inner = list(StratifiedKFold(n_splits=4, shuffle=True, random_state=seed).split(X[train], inner_labels))
    if spec.grid:
        scoring = "balanced_accuracy" if spec.task == "classification" else "neg_mean_absolute_error"
        fit = GridSearchCV(estimator, spec.grid, scoring=scoring, cv=inner, n_jobs=4, refit=True, error_score="raise")
        fit.fit(X[train], y[train])
        return np.asarray(fit.predict(X[test])).reshape(-1), float(fit.best_score_), fit.best_params_
    estimator.fit(X[train], y[train])
    pred = np.asarray(estimator.predict(X[test])).reshape(-1)
    # Untuned candidates remain eligible; estimate their inner score without leaking outer labels.
    inner_scores = []
    for inner_train, inner_test in inner:
        inner_est = clone(estimator)
        inner_est.fit(X[train][inner_train], y[train][inner_train])
        inner_pred = np.asarray(inner_est.predict(X[train][inner_test])).reshape(-1)
        if spec.task == "classification":
            inner_scores.append(balanced_accuracy_score(y[train][inner_test], inner_pred))
        else:
            inner_scores.append(-mean_absolute_error(y[train][inner_test], inner_pred))
    return pred, float(np.mean(inner_scores)), {}


def benchmark_classification(frame: pd.DataFrame, arrays: dict[str, np.ndarray], specs: list[Spec], splits: list[tuple[int, int, np.ndarray, np.ndarray]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    tasks = {
        "condition4": frame.condition.astype(str).to_numpy(),
        "petting": frame.Petting.to_numpy(int),
        "feeding": frame.Feeding.to_numpy(int),
    }
    records: list[dict[str, Any]] = []
    choices: list[dict[str, Any]] = []
    for task_name, y in tasks.items():
        fold_predictions: dict[tuple[str, int], np.ndarray] = {}
        for spec in specs:
            for repeat in range(10):
                fold_predictions[(spec.name, repeat)] = np.empty(len(frame), dtype=object if task_name == "condition4" else float)
        for repeat, fold, train, test in splits:
            best_name = None
            best_score = -np.inf
            best_pred = None
            for spec in specs:
                pred, inner_score, params = fit_candidate(
                    spec,
                    arrays[spec.feature_set],
                    y,
                    train,
                    test,
                    y[train],
                    SEED + repeat * 100 + fold,
                )
                fold_predictions[(spec.name, repeat)][test] = pred
                choices.append(
                    dict(task=task_name, repeat=repeat, fold=fold, model=spec.name, inner_score=inner_score, selected=False, params=json_params(params))
                )
                if spec.selectable and inner_score > best_score:
                    best_name, best_score, best_pred = spec.name, inner_score, pred
            fold_predictions.setdefault(("nested_selector", repeat), np.empty(len(frame), dtype=object if task_name == "condition4" else float))[test] = best_pred
            choices.append(dict(task=task_name, repeat=repeat, fold=fold, model=best_name, inner_score=best_score, selected=True, params=""))
            print(f"classification {task_name} repeat={repeat} fold={fold} selected={best_name} inner={best_score:.3f}", flush=True)

        model_names = [spec.name for spec in specs] + ["nested_selector"]
        for model in model_names:
            for repeat in range(10):
                pred = np.asarray(fold_predictions[(model, repeat)])
                if task_name != "condition4":
                    pred = pred.astype(int)
                scores = class_metrics(y, pred)
                for metric, value in scores.items():
                    records.append(dict(task=task_name, model=model, repeat=repeat, metric=metric, value=value, n=len(y)))
    return pd.DataFrame(records), pd.DataFrame(choices)


def benchmark_regression(frame: pd.DataFrame, arrays: dict[str, np.ndarray], specs: list[Spec], splits: list[tuple[int, int, np.ndarray, np.ndarray]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_post = frame.STAI_post.to_numpy(float)
    y_improvement = frame.STAI_improvement.to_numpy(float)
    conditions = frame.condition.astype(str).to_numpy()
    records: list[dict[str, Any]] = []
    choices: list[dict[str, Any]] = []
    fold_predictions: dict[tuple[str, int], np.ndarray] = {}
    for spec in specs:
        for repeat in range(10):
            fold_predictions[(spec.name, repeat)] = np.empty(len(frame), dtype=float)
    for repeat, fold, train, test in splits:
        best_name = None
        best_score = -np.inf
        best_pred_post = None
        for spec in specs:
            target = y_post if spec.target_mode == "post" else y_improvement
            pred, inner_score, params = fit_candidate(
                spec,
                arrays[spec.feature_set],
                target,
                train,
                test,
                conditions[train],
                SEED + 10000 + repeat * 100 + fold,
            )
            if spec.target_mode == "improvement":
                pred_post = frame.STAI_pre.to_numpy(float)[test] - pred
            else:
                pred_post = pred
            fold_predictions[(spec.name, repeat)][test] = pred_post
            choices.append(dict(task="stai_post", repeat=repeat, fold=fold, model=spec.name, inner_score=inner_score, selected=False, params=json_params(params)))
            if spec.selectable and inner_score > best_score:
                best_name, best_score, best_pred_post = spec.name, inner_score, pred_post
        fold_predictions.setdefault(("nested_selector", repeat), np.empty(len(frame), dtype=float))[test] = best_pred_post
        choices.append(dict(task="stai_post", repeat=repeat, fold=fold, model=best_name, inner_score=best_score, selected=True, params=""))
        print(f"regression repeat={repeat} fold={fold} selected={best_name} inner={best_score:.3f}", flush=True)

    for model in [spec.name for spec in specs] + ["nested_selector"]:
        for repeat in range(10):
            pred = fold_predictions[(model, repeat)]
            scores = reg_metrics(y_post, pred)
            for metric, value in scores.items():
                records.append(dict(task="stai_post", model=model, repeat=repeat, metric=metric, value=value, n=len(y_post)))
    return pd.DataFrame(records), pd.DataFrame(choices)


def summarize(perf: pd.DataFrame) -> pd.DataFrame:
    return (
        perf.groupby(["task", "model", "metric", "n"], as_index=False)
        .value.agg(mean="mean", sd="std", median="median", minimum="min", maximum="max")
        .sort_values(["task", "metric", "mean"], ascending=[True, True, False])
    )


def main() -> None:
    frame, feature_ids, arrays = load_data()
    splits = split_index(frame)
    if len(frame) != 92 or len(splits) != 50:
        raise RuntimeError(f"Unexpected analysis dimensions: n={len(frame)}, splits={len(splits)}")
    class_specs, reg_specs = make_specs(feature_ids)

    class_perf, class_choices = benchmark_classification(frame, arrays, class_specs, splits)
    class_perf.to_csv(OUT / "classification_performance_by_repeat.csv", index=False)
    class_choices.to_csv(OUT / "classification_inner_selection.csv", index=False)
    summarize(class_perf).to_csv(OUT / "classification_performance_summary.csv", index=False)

    reg_perf, reg_choices = benchmark_regression(frame, arrays, reg_specs, splits)
    reg_perf.to_csv(OUT / "regression_performance_by_repeat.csv", index=False)
    reg_choices.to_csv(OUT / "regression_inner_selection.csv", index=False)
    summarize(reg_perf).to_csv(OUT / "regression_performance_summary.csv", index=False)

    print("\nTOP CLASSIFICATION", flush=True)
    class_summary = summarize(class_perf)
    print(class_summary.loc[class_summary.metric.eq("balanced_accuracy")].groupby("task").head(8).to_string(index=False), flush=True)
    print("\nTOP REGRESSION", flush=True)
    reg_summary = summarize(reg_perf)
    print(reg_summary.loc[reg_summary.metric.eq("mae")].sort_values("mean").head(12).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

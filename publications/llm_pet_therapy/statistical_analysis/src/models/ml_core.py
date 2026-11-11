"""Fixed participant-level random forest analysis.

No discovery, tuning, feature selection, target permutation or full-sample fit.
All output performance is computed from complete OOF vectors per repeat.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold

FEATURES = [
    "smile_prevalence_score", "smile_intensity_score", "au05_prevalence",
    "au07_prevalence", "au12_prevalence", "f0_median_st", "f0_iqr_st",
    "auto_silent_interval_proportion", "auto_periodic_vocalization_proportion",
    "jitter_local_pct", "hnr", "nose_tip_temperature", "left_cheek_temperature",
    "right_cheek_temperature", "mean_hr", "rmssd", "sdnn", "sbp", "dbp",
]
BASICS = ["STAI_pre", "Petting", "Feeding", "Petting_x_Feeding"]
LABELS = ["observation", "petting", "feeding", "combined"]
PROB_COLS = ["p_" + x for x in LABELS]
MODEL_SPECS = {
    "C_RF19": ("main92", "delta", "classification", "forest"),
    "C_DUMMY": ("main92", "dummy", "classification", "dummy"),
    "R_RF19": ("main92", "post", "regression", "forest"),
    "R_MEAN": ("main92", "dummy", "regression", "dummy"),
    "R_BASIC": ("main92", "basic", "regression", "forest"),
    "R_EXTENDED": ("main92", "extended", "regression", "forest"),
    "C_RF19_CC": ("delta_cc", "delta", "classification", "forest"),
    "C_DUMMY_CC": ("delta_cc", "dummy", "classification", "dummy"),
    "R_RF19_CC": ("post_cc", "post", "regression", "forest"),
    "R_MEAN_CC": ("post_cc", "dummy", "regression", "dummy"),
}
COMMON_RF = dict(n_estimators=500, max_depth=3, min_samples_leaf=5,
                 min_samples_split=2, bootstrap=True, max_samples=None,
                 oob_score=False, ccp_alpha=0.0, warm_start=False, n_jobs=1)




def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(x) for x in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def environment():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__, "joblib": joblib.__version__}

def make_model(model_id, seed):
    _, _, task, kind = MODEL_SPECS[model_id]
    if kind == "dummy":
        return DummyClassifier(strategy="most_frequent") if task == "classification" else DummyRegressor(strategy="mean")
    if task == "classification":
        return RandomForestClassifier(**COMMON_RF, criterion="gini", max_features="sqrt", class_weight=None, random_state=seed)
    return RandomForestRegressor(**COMMON_RF, criterion="squared_error", max_features=1.0, random_state=seed)

def model_columns(model_id):
    matrix = MODEL_SPECS[model_id][1]
    return {"delta": FEATURES, "post": FEATURES, "basic": BASICS,
            "extended": BASICS + FEATURES, "dummy": []}[matrix].copy()

def label_values(series):
    strings = series.astype(str)
    if set(strings) <= set(LABELS):
        return strings.map({name: i for i, name in enumerate(LABELS)}).to_numpy(dtype=int)
    numeric = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not np.isin(numeric, np.arange(4)).all():
        raise ValueError("condition must use fixed four labels or integer class codes 0..3")
    return numeric.astype(int)

def expected_splits(data):
    rows, blocks = [], {}
    for version, positions in data["subsets"].items():
        for repeat in range(10):
            seed = 20260924 + repeat
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            for fold, (local_train, local_test) in enumerate(cv.split(positions, data["y_class"][positions])):
                train, test = positions[local_train], positions[local_test]
                blocks[version, repeat, fold] = (train, test)
                for role, indices in [("train", train), ("test", test)]:
                    for idx in indices:
                        rows.append(dict(sample_version=version, repeat=repeat, fold=fold, participant_id=data["ids"][idx], role=role, split_seed=seed))
    return pd.DataFrame(rows), blocks

def metrics(task, y, pred):
    if task == "classification":
        cm = confusion_matrix(y, pred, labels=np.arange(4))
        recalls = cm.diagonal() / cm.sum(axis=1)
        values = dict(balanced_accuracy=float(balanced_accuracy_score(y, pred)), accuracy=float(accuracy_score(y, pred)),
                      macro_f1=float(f1_score(y, pred, labels=np.arange(4), average="macro", zero_division=0)))
        values.update({"recall_" + label: float(value) for label, value in zip(LABELS, recalls)})
        return values, {}
    error = y - pred
    sst = float(np.sum((y - y.mean()) ** 2))
    pearson_valid = np.std(y) > 0 and np.std(pred) > 0
    values = dict(mae=float(mean_absolute_error(y, pred)), rmse=float(np.sqrt(np.mean(error ** 2))),
                  r2=float(1 - np.sum(error ** 2) / sst) if sst > 0 else np.nan,
                  pearson_r=float(np.corrcoef(y, pred)[0, 1]) if pearson_valid else np.nan)
    reasons = {}
    if sst == 0:
        reasons["r2"] = "constant_observed_target"
    if not pearson_valid:
        reasons["pearson_r"] = "constant_observed_or_predicted_values"
    return values, reasons

def fit_fold(model_id, repeat, fold, train, test, data):
    """Only this function fits estimators; exactly one forest/dummy per block."""
    version, matrix, task, kind = MODEL_SPECS[model_id]
    seed = 20270924 + 100 * repeat + fold
    columns = model_columns(model_id)
    y = data["y_class"] if task == "classification" else data["y_reg"]
    imputer_rows = []
    if kind == "dummy":
        # This constant merely satisfies the sklearn estimator API; no feature is read.
        xtrain, xtest = np.zeros((len(train), 1)), np.zeros((len(test), 1))
        imputer = None
    else:
        x = data[matrix]
        if np.isnan(x[train]).all(axis=0).any():
            missing = [columns[i] for i in np.flatnonzero(np.isnan(x[train]).all(axis=0))]
            raise ValueError(f"All-missing training columns: {model_id}/{repeat}/{fold}: {missing}")
        imputer = SimpleImputer(strategy="median", add_indicator=False)
        xtrain = imputer.fit_transform(x[train])
        xtest = imputer.transform(x[test])
        if xtrain.shape[1] != len(columns) or not np.isfinite(xtrain).all() or not np.isfinite(xtest).all():
            raise ValueError("Imputation changed column count or left nonfinite values")
        np.testing.assert_allclose(imputer.statistics_, np.nanmedian(x[train], axis=0), rtol=0, atol=0)
        for j, name in enumerate(columns):
            imputer_rows.append(dict(model_id=model_id, sample_version=version, repeat=repeat, fold=fold,
                                     feature_id=name, column_index=j, median=float(imputer.statistics_[j]),
                                     train_missing_n=int(np.isnan(x[train, j]).sum()), test_missing_n=int(np.isnan(x[test, j]).sum()),
                                     train_n=len(train), test_n=len(test)))
    model = make_model(model_id, seed)
    model.fit(xtrain, y[train])
    pred = model.predict(xtest)
    probabilities = model.predict_proba(xtest) if task == "classification" else np.full((len(test), 4), np.nan)
    if task == "classification":
        if not np.array_equal(model.classes_, np.arange(4)):
            raise ValueError("Probability class mapping mismatch")
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12, rtol=0)
        if not np.array_equal(np.argmax(probabilities, axis=1), pred):
            raise ValueError("Class prediction is inconsistent with fixed-order probabilities")
    result = {"model": model, "imputer": imputer, "model_id": model_id, "repeat": repeat,
              "fold": fold, "train": train, "test": test, "columns": columns, "pred": pred,
              "probabilities": probabilities, "imputer_rows": imputer_rows, "params": model.get_params(deep=True),
              "forest_seed": seed if kind == "forest" else None, "mdi": model.feature_importances_ if kind == "forest" else None}
    if model_id in ("C_RF19", "R_RF19"):
        task_code = 1 if task == "classification" else 2
        orders, seeds, blocks = [], [], []
        # Stable order: shuffle k, feature j, then original ascending test IDs.
        for shuffle in range(10):
            permutation_seed = 20280924 + 100000 * task_code + 1000 * repeat + 100 * fold + shuffle
            order = np.random.Generator(np.random.PCG64(permutation_seed)).permutation(len(test))
            orders.append(order)
            seeds.append(permutation_seed)
            for j in range(19):
                changed = xtest.copy()
                changed[:, j] = xtest[order, j]
                blocks.append(changed)
        batch = np.vstack(blocks)
        batch_pred = model.predict(batch)
        # Real equivalence check against the same fitted model on a single block.
        # No additional fit and no new random draw.
        direct = model.predict(blocks[0])
        np.testing.assert_array_equal(direct, batch_pred[:len(test)])
        result.update(permutation_predictions=batch_pred.reshape(10, 19, len(test)),
                      permutation_orders=np.array(orders), permutation_seeds=seeds,
                      batched_prediction_exact_equivalence=True)
    return result

def write_results(package, checkpoints, data):
    """Aggregate existing fold products, never train, tune, or select models."""
    package = Path(package)
    results = package / "results"
    cv = package / "cv"
    results.mkdir(exist_ok=True)
    cv.mkdir(exist_ok=True)
    oof, imputation, mdi, seeds, parameters, perm_indices = [], [], [], [], [], []
    permutation_frames = []
    for result in checkpoints:
        mid, repeat, fold = result["model_id"], result["repeat"], result["fold"]
        version, _, task, kind = MODEL_SPECS[mid]
        test = result["test"]
        y = data["y_class"] if task == "classification" else data["y_reg"]
        for i, position in enumerate(test):
            row = dict(model_id=mid, sample_version=version, repeat=repeat, fold=fold,
                       participant_id=data["ids"][position], y_true=float(y[position]), y_pred=float(result["pred"][i]))
            row.update(zip(PROB_COLS, result["probabilities"][i].astype(float)))
            oof.append(row)
        imputation.extend(result["imputer_rows"])
        for seed_kind, seed_value in [("split", 20260924 + repeat), ("forest", result["forest_seed"])]:
            if seed_value is not None:
                seeds.append(dict(kind=seed_kind, model_id=mid, sample_version=version, repeat=repeat, fold=fold, shuffle=-1, seed=seed_value))
        parameters.append(dict(model_id=mid, sample_version=version, repeat=repeat, fold=fold,
                               input_columns=result["columns"], input_n_features=len(result["columns"]),
                               parameters=result["params"], imputer_parameters=result["imputer"].get_params() if result["imputer"] is not None else None))
        if mid in ("C_RF19", "R_RF19"):
            for name, importance in zip(FEATURES, result["mdi"]):
                mdi.append(dict(model_id=mid, repeat=repeat, fold=fold, feature_id=name, importance=importance))
            for k in range(10):
                seed = result["permutation_seeds"][k]
                seeds.append(dict(kind="permutation", model_id=mid, sample_version=version, repeat=repeat, fold=fold, shuffle=k, seed=seed))
                for i, source in enumerate(result["permutation_orders"][k]):
                    perm_indices.append(dict(model_id=mid, repeat=repeat, fold=fold, shuffle=k, seed=seed,
                                             participant_id=data["ids"][test[i]], permuted_source_id=data["ids"][test[source]]))
                for j, name in enumerate(FEATURES):
                    permutation_frames.append(pd.DataFrame(dict(model_id=mid, repeat=repeat, feature_id=name, shuffle=k,
                                                              fold=fold, participant_id=data["ids"][test], y_true=y[test],
                                                              y_pred=result["permutation_predictions"][k, j])))
    oof = pd.DataFrame(oof).sort_values(["model_id", "repeat", "participant_id"]).reset_index(drop=True)
    oof.to_csv(results / "oof_predictions.csv", index=False)
    pd.DataFrame(imputation).to_csv(cv / "imputer_statistics.csv", index=False)
    pd.DataFrame(seeds).to_csv(cv / "seeds.csv", index=False)
    pd.DataFrame(perm_indices).to_csv(cv / "permutation_indices.csv", index=False)
    write_json(cv / "model_parameters.json", parameters)
    performance, confusion = [], []
    for (mid, repeat), rows in oof.groupby(["model_id", "repeat"], sort=True):
        version, _, task, _ = MODEL_SPECS[mid]
        expected_ids = data["ids"][data["subsets"][version]]
        if rows.participant_id.duplicated().any() or rows.participant_id.tolist() != expected_ids.tolist():
            raise ValueError(f"Incomplete or duplicate OOF: {mid}/{repeat}")
        y, pred = rows.y_true.to_numpy(float), rows.y_pred.to_numpy(float)
        values, reasons = metrics(task, y, pred)
        for metric, value in values.items():
            performance.append(dict(model_id=mid, sample_version=version, repeat=repeat, n=len(rows), metric=metric,
                                    value=value, undefined_reason=reasons.get(metric, "")))
        if task == "classification":
            cm = confusion_matrix(y, pred, labels=np.arange(4))
            for i in range(4):
                for j in range(4):
                    confusion.append(dict(model_id=mid, repeat=repeat, true_class=i, predicted_class=j,
                                          count=int(cm[i, j]), row_proportion=float(cm[i, j] / cm[i].sum())))
    perf = pd.DataFrame(performance)
    perf.to_csv(results / "performance_by_repeat.csv", index=False)
    summary = perf.groupby(["model_id", "sample_version", "n", "metric"], sort=True).value.agg(mean="mean", sd="std", n_repeats="count").reset_index()
    summary.to_csv(results / "performance_summary.csv", index=False)
    pd.DataFrame(confusion).to_csv(results / "confusion_matrices_by_repeat.csv", index=False)
    perfmap = {(r.model_id, int(r.repeat), r.metric): r.value for r in perf.itertuples()}
    comparisons = []
    contrasts = [
        ("C_RF19_vs_C_DUMMY", "C_RF19", "C_DUMMY", "balanced_accuracy", "RF_minus_dummy"),
        ("R_RF19_vs_R_MEAN", "R_MEAN", "R_RF19", "mae", "reference_MAE_minus_RF_MAE"),
        ("R_EXTENDED_vs_R_BASIC", "R_BASIC", "R_EXTENDED", "mae", "basic_minus_extended"),
        ("R_EXTENDED_vs_R_BASIC", "R_BASIC", "R_EXTENDED", "rmse", "basic_minus_extended"),
        ("R_EXTENDED_vs_R_BASIC", "R_EXTENDED", "R_BASIC", "r2", "extended_minus_basic"),
        ("C_RF19_CC_vs_C_DUMMY_CC", "C_RF19_CC", "C_DUMMY_CC", "balanced_accuracy", "RF_minus_dummy"),
        ("R_RF19_CC_vs_R_MEAN_CC", "R_MEAN_CC", "R_RF19_CC", "mae", "reference_MAE_minus_RF_MAE"),
    ]
    for contrast, first, second, metric, direction in contrasts:
        for repeat in range(10):
            left = oof[(oof.model_id == first) & (oof.repeat == repeat)]
            right = oof[(oof.model_id == second) & (oof.repeat == repeat)]
            if left[["participant_id", "fold"]].values.tolist() != right[["participant_id", "fold"]].values.tolist():
                raise ValueError("Comparison does not share test membership")
            comparisons.append(dict(contrast_id=contrast, repeat=repeat, metric=metric,
                                    difference=perfmap[first, repeat, metric] - perfmap[second, repeat, metric], direction=direction))
    comparisons = pd.DataFrame(comparisons)
    comparisons.to_csv(results / "paired_model_comparisons.csv", index=False)
    comparisons.groupby(["contrast_id", "metric", "direction"], sort=True).difference.agg(mean="mean", sd="std", n_repeats="count").reset_index().to_csv(results / "paired_model_comparisons_summary.csv", index=False)
    permutations = pd.concat(permutation_frames, ignore_index=True)
    permutations.to_csv(results / "permutation_oof_predictions.csv.gz", index=False, compression={"method": "gzip", "compresslevel": 1, "mtime": 0})
    shuffle_importance = []
    for (mid, repeat, name, shuffle), rows in permutations.groupby(["model_id", "repeat", "feature_id", "shuffle"], sort=True):
        if rows.participant_id.duplicated().any() or set(rows.participant_id) != set(data["ids"]):
            raise ValueError("Incomplete permutation OOF vector")
        metric = "balanced_accuracy" if mid == "C_RF19" else "mae"
        base = perfmap[mid, repeat, metric]
        y, pred = rows.y_true.to_numpy(), rows.y_pred.to_numpy()
        permuted = balanced_accuracy_score(y, pred) if mid == "C_RF19" else mean_absolute_error(y, pred)
        importance = base - permuted if mid == "C_RF19" else permuted - base
        shuffle_importance.append(dict(model_id=mid, repeat=repeat, feature_id=name, shuffle=shuffle,
                                       baseline_metric=base, permuted_metric=float(permuted), importance=float(importance)))
    importance = pd.DataFrame(shuffle_importance)
    importance.to_csv(results / "permutation_importance_by_shuffle.csv", index=False)
    rep = importance.groupby(["model_id", "repeat", "feature_id"], sort=True).importance.mean().reset_index()
    rep.to_csv(results / "permutation_importance_by_repeat.csv", index=False)
    summarize_importance(rep).to_csv(results / "permutation_importance_summary.csv", index=False)
    mdi = pd.DataFrame(mdi)
    mdi.to_csv(results / "mdi_importance_by_fold.csv", index=False)
    mdi_rep = mdi.groupby(["model_id", "repeat", "feature_id"], sort=True).importance.mean().reset_index()
    mdi_rep.to_csv(results / "mdi_importance_by_repeat.csv", index=False)
    summarize_importance(mdi_rep).to_csv(results / "mdi_importance_summary.csv", index=False)
    return {"models_completed": list(MODEL_SPECS), "n_fold_fits": len(checkpoints), "n_forest_fits": 300,
            "oof_rows": len(oof), "permutation_oof_rows": len(permutations), "permutation_shuffles": len(importance),
            "n_performance_repeats": 10, "n_folds": 5, "summary_sd_ddof": 1,
            "complete_case_n": {key: len(value) for key, value in data["subsets"].items()},
            "batched_prediction_equivalence_checks": sum("batched_prediction_exact_equivalence" in x for x in checkpoints)}

def summarize_importance(repeat_table):
    table = repeat_table.groupby(["model_id", "feature_id"], sort=True).importance.agg(mean="mean", sd="std", n_repeats="count").reset_index()
    # Stable whitelist tie-breaking affects display only, never fitting.
    table["_feature_order"] = table.feature_id.map({v: i for i, v in enumerate(FEATURES)})
    table = table.sort_values(["model_id", "mean", "_feature_order"], ascending=[True, False, True])
    table["rank"] = table.groupby("model_id").cumcount() + 1
    return table.drop(columns="_feature_order")

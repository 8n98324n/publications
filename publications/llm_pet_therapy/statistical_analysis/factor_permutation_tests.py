from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import model_benchmark as mb

OUT = HERE / "outputs"
N_PERM = 199
N_IMPORTANCE = 20
RNG_SEED = 20261103


def models(feature_ids):
    return {
        "condition4": Pipeline(
            [
                ("modalities", mb.modality_transformer(feature_ids)),
                (
                    "model",
                    LogisticRegression(
                        penalty="l2",
                        solver="lbfgs",
                        C=0.01,
                        max_iter=10000,
                        class_weight="balanced",
                        random_state=RNG_SEED,
                    ),
                ),
            ]
        ),
        "petting": Pipeline(
            [
                ("modalities", mb.modality_transformer(feature_ids)),
                (
                    "model",
                    LogisticRegression(
                        penalty="l2",
                        solver="lbfgs",
                        C=0.01,
                        max_iter=10000,
                        class_weight="balanced",
                        random_state=RNG_SEED,
                    ),
                ),
            ]
        ),
        "feeding": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
            ]
        ),
    }


def permute_target(task, frame, rng):
    if task == "condition4":
        return rng.permutation(frame.condition.astype(str).to_numpy())
    if task == "petting":
        result = frame.Petting.to_numpy(int).copy()
        strata = frame.Feeding.to_numpy(int)
    else:
        result = frame.Feeding.to_numpy(int).copy()
        strata = frame.Petting.to_numpy(int)
    for value in np.unique(strata):
        idx = np.flatnonzero(strata == value)
        result[idx] = rng.permutation(result[idx])
    return result


def cross_validated_mean_ba(estimator, X, y, splits):
    repeat_predictions = {repeat: np.empty(len(y), dtype=y.dtype) for repeat in range(10)}
    fitted = []
    for repeat, fold, train, test in splits:
        model = clone(estimator).fit(X[train], y[train])
        repeat_predictions[repeat][test] = model.predict(X[test])
        fitted.append((repeat, fold, test, model))
    scores = np.array([balanced_accuracy_score(y, repeat_predictions[r]) for r in range(10)])
    return float(scores.mean()), scores, repeat_predictions, fitted


def feature_importance(task, estimator, X, y, splits, feature_ids):
    observed, observed_repeats, _, fitted = cross_validated_mean_ba(estimator, X, y, splits)
    rng = np.random.default_rng(RNG_SEED + 1000 + ["condition4", "petting", "feeding"].index(task))
    rows = []
    by_repeat_fold = {(r, f): (test, model) for r, f, test, model in fitted}
    for feature_index, feature_id in enumerate(feature_ids):
        for shuffle in range(N_IMPORTANCE):
            repeat_predictions = {repeat: np.empty(len(y), dtype=y.dtype) for repeat in range(10)}
            for repeat, fold, train, test in splits:
                test_idx, model = by_repeat_fold[(repeat, fold)]
                X_test = X[test_idx].copy()
                X_test[:, feature_index] = rng.permutation(X_test[:, feature_index])
                repeat_predictions[repeat][test_idx] = model.predict(X_test)
            for repeat in range(10):
                perm_score = balanced_accuracy_score(y, repeat_predictions[repeat])
                rows.append(
                    dict(
                        task=task,
                        feature_id=feature_id,
                        shuffle=shuffle,
                        repeat=repeat,
                        importance=observed_repeats[repeat] - perm_score,
                    )
                )
    return observed, observed_repeats, pd.DataFrame(rows)


def main():
    frame, feature_ids, arrays = mb.load_data()
    splits = mb.split_index(frame)
    X = arrays["delta19"]
    task_targets = {
        "condition4": frame.condition.astype(str).to_numpy(),
        "petting": frame.Petting.to_numpy(int),
        "feeding": frame.Feeding.to_numpy(int),
    }
    task_models = models(feature_ids)
    rng = np.random.default_rng(RNG_SEED)
    test_rows = []
    importance_frames = []
    for task in ["condition4", "petting", "feeding"]:
        y = task_targets[task]
        observed, repeat_scores, importance = feature_importance(task, task_models[task], X, y, splits, feature_ids)
        importance_frames.append(importance)
        null = np.empty(N_PERM)
        for permutation in range(N_PERM):
            yp = permute_target(task, frame, rng)
            null[permutation] = cross_validated_mean_ba(task_models[task], X, yp, splits)[0]
            if (permutation + 1) % 50 == 0:
                print(f"{task}: {permutation + 1}/{N_PERM}", flush=True)
        p = (1 + np.sum(null >= observed)) / (N_PERM + 1)
        test_rows.append(
            dict(
                task=task,
                model=("modality_pca_logit_C0.01" if task != "feeding" else "lda_shrinkage"),
                observed_mean_balanced_accuracy=observed,
                observed_sd_across_repeats=float(repeat_scores.std(ddof=1)),
                permutation_p=float(p),
                null_mean=float(null.mean()),
                null_sd=float(null.std(ddof=1)),
                null_q025=float(np.quantile(null, 0.025)),
                null_q975=float(np.quantile(null, 0.975)),
                n_permutations=N_PERM,
                note="exploratory model-development test; p does not adjust for prior model-family search",
            )
        )
    tests = pd.DataFrame(test_rows)
    tests.to_csv(OUT / "factor_permutation_tests.csv", index=False)
    importance = pd.concat(importance_frames, ignore_index=True)
    importance.to_csv(OUT / "factor_permutation_importance_by_shuffle.csv", index=False)
    summary = (
        importance.groupby(["task", "feature_id"], as_index=False)
        .importance.agg(mean="mean", sd="std", median="median")
        .sort_values(["task", "mean"], ascending=[True, False])
    )
    summary["rank"] = summary.groupby("task").cumcount() + 1
    summary.to_csv(OUT / "factor_permutation_importance_summary.csv", index=False)
    print(tests.to_string(index=False), flush=True)
    print(summary.groupby("task").head(8).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

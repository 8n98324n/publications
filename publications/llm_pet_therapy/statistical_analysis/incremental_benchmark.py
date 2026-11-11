from pathlib import Path
import os
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge

import model_benchmark as mb


def main():
    frame, feature_ids, arrays = mb.load_data()
    splits = mb.split_index(frame)
    specs = [
        mb.Spec(
            "elastic_basic_post",
            "basic4",
            mb.standard_pipeline(ElasticNet(max_iter=20000, random_state=mb.SEED)),
            {"model__alpha": [0.01, 0.1, 1.0], "model__l1_ratio": [0.1, 0.5, 0.9]},
            "regression",
        ),
        mb.Spec(
            "elastic_extended_post",
            "ext_post23",
            mb.standard_pipeline(ElasticNet(max_iter=20000, random_state=mb.SEED)),
            {"model__alpha": [0.01, 0.1, 1.0], "model__l1_ratio": [0.1, 0.5, 0.9]},
            "regression",
        ),
        mb.Spec(
            "ridge_change_basic",
            "basic4",
            mb.standard_pipeline(Ridge()),
            {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
            "regression",
            target_mode="improvement",
        ),
        mb.Spec(
            "ridge_change_delta",
            "ext_delta23",
            mb.standard_pipeline(Ridge()),
            {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
            "regression",
            target_mode="improvement",
        ),
        mb.Spec(
            "elastic_change_basic",
            "basic4",
            mb.standard_pipeline(ElasticNet(max_iter=20000, random_state=mb.SEED)),
            {"model__alpha": [0.01, 0.1, 1.0], "model__l1_ratio": [0.1, 0.5, 0.9]},
            "regression",
            target_mode="improvement",
        ),
        mb.Spec(
            "elastic_change_delta",
            "ext_delta23",
            mb.standard_pipeline(ElasticNet(max_iter=20000, random_state=mb.SEED)),
            {"model__alpha": [0.01, 0.1, 1.0], "model__l1_ratio": [0.1, 0.5, 0.9]},
            "regression",
            target_mode="improvement",
        ),
        mb.Spec(
            "huber_change_basic",
            "basic4",
            mb.standard_pipeline(HuberRegressor(max_iter=5000)),
            {"model__epsilon": [1.1, 1.35, 1.8], "model__alpha": [0.0001, 0.01, 1.0]},
            "regression",
            target_mode="improvement",
        ),
        mb.Spec(
            "huber_change_delta",
            "ext_delta23",
            mb.standard_pipeline(HuberRegressor(max_iter=5000)),
            {"model__epsilon": [1.1, 1.35, 1.8], "model__alpha": [0.0001, 0.01, 1.0]},
            "regression",
            target_mode="improvement",
        ),
    ]
    perf, choices = mb.benchmark_regression(frame, arrays, specs, splits)
    out = ROOT / "work/benchmark_results"
    perf.to_csv(out / "incremental_performance_by_repeat.csv", index=False)
    choices.to_csv(out / "incremental_inner_selection.csv", index=False)
    summary = mb.summarize(perf)
    summary.to_csv(out / "incremental_performance_summary.csv", index=False)
    print(summary.sort_values(["metric", "mean"]).to_string(index=False))


if __name__ == "__main__":
    main()

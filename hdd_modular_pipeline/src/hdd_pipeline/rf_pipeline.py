from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .config import RFConfig
from .metrics import eval_regression
from .utils import clear_memory, ensure_dir, iter_slices, to_numpy_safe


def _get_gpu_rf_classes():
    try:
        from cuml.ensemble import RandomForestClassifier as cuRFClassifier  # type: ignore
        from cuml.ensemble import RandomForestRegressor as cuRFRegressor  # type: ignore
        import cupy as cp  # type: ignore

        return cuRFClassifier, cuRFRegressor, cp
    except Exception:
        return None, None, None


def sample_xy(X, y, sample_n: int | None = None, random_state: int = 42):
    """sample_n=None or sample_n>=len(X) => use all rows."""
    if sample_n is None or len(X) <= sample_n:
        return X, y
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=sample_n, replace=False)
    return X[idx], y[idx]


def train_stage1_random_forest_classifier(X_train, y_train, cfg: RFConfig, sample_n: int | None = None):
    start = time.time()
    X_fit, y_fit = sample_xy(X_train, y_train, sample_n=sample_n, random_state=cfg.random_state)
    mode = "FULL dataset" if (sample_n is None or sample_n >= len(X_train)) else f"sample of {sample_n:,}"
    print(f"Stage 1 RF training on {mode} ({len(X_fit):,} / {len(X_train):,} rows)")

    cu_clf, _, cp = _get_gpu_rf_classes() if cfg.use_gpu_rf else (None, None, None)
    if cu_clf is not None and cp is not None:
        model = cu_clf(n_estimators=cfg.n_estimators, max_depth=cfg.max_depth, random_state=cfg.random_state, n_streams=8)
        X_gpu = cp.asarray(X_fit.astype(np.float32))
        y_gpu = cp.asarray(y_fit.astype(np.int32))
        model.fit(X_gpu, y_gpu)
        del X_gpu, y_gpu
        cp.get_default_memory_pool().free_all_blocks()
        backend = "gpu_cuml"
    else:
        model = RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            n_jobs=cfg.n_jobs,
            random_state=cfg.random_state,
            verbose=cfg.log_rf_trees,
            class_weight="balanced",
        )
        model.fit(X_fit, y_fit)
        backend = "cpu_sklearn"

    del X_fit, y_fit
    clear_memory()
    return model, backend, time.time() - start


def train_stage2_random_forest_regressor(X_train, y_train, cfg: RFConfig, sample_n: int | None = None):
    start = time.time()
    X_fit, y_fit = sample_xy(X_train, y_train, sample_n=sample_n, random_state=cfg.random_state)
    mode = "FULL dataset" if (sample_n is None or sample_n >= len(X_train)) else f"sample of {sample_n:,}"
    print(f"Stage 2 RF training on {mode} ({len(X_fit):,} / {len(X_train):,} active rows)")

    _, cu_reg, cp = _get_gpu_rf_classes() if cfg.use_gpu_rf else (None, None, None)
    if cu_reg is not None and cp is not None:
        model = cu_reg(n_estimators=cfg.n_estimators, max_depth=cfg.max_depth, random_state=cfg.random_state, n_streams=8)
        X_gpu = cp.asarray(X_fit.astype(np.float32))
        y_gpu = cp.asarray(y_fit.astype(np.float32))
        model.fit(X_gpu, y_gpu)
        del X_gpu, y_gpu
        cp.get_default_memory_pool().free_all_blocks()
        backend = "gpu_cuml"
    else:
        model = RandomForestRegressor(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            n_jobs=cfg.n_jobs,
            random_state=cfg.random_state,
            verbose=cfg.log_rf_trees,
        )
        model.fit(X_fit, y_fit)
        backend = "cpu_sklearn"

    del X_fit, y_fit
    clear_memory()
    return model, backend, time.time() - start


def train_stage1_rf_full_safe(X_train, y_train, cfg: RFConfig):
    """Full-row RF training with bounded tree complexity to reduce RAM spikes."""
    start = time.time()
    print(f"Stage 1 RF FULL-SAFE training ({len(X_train):,} rows)")
    model = RandomForestClassifier(
        n_estimators=cfg.full_safe_n_estimators,
        max_depth=cfg.full_safe_max_depth,
        max_leaf_nodes=cfg.full_safe_max_leaf_nodes,
        min_samples_leaf=cfg.full_safe_min_samples_leaf,
        n_jobs=cfg.full_safe_n_jobs,
        random_state=cfg.random_state,
        verbose=cfg.log_rf_trees,
        class_weight="balanced_subsample",
        bootstrap=cfg.full_safe_bootstrap,
    )
    model.fit(X_train, y_train)
    return model, "cpu_sklearn", time.time() - start


def train_stage2_rf_full_safe(X_train, y_train, cfg: RFConfig):
    start = time.time()
    print(f"Stage 2 RF FULL-SAFE training ({len(X_train):,} active rows)")
    model = RandomForestRegressor(
        n_estimators=cfg.full_safe_n_estimators,
        max_depth=cfg.full_safe_max_depth,
        max_leaf_nodes=cfg.full_safe_max_leaf_nodes,
        min_samples_leaf=cfg.full_safe_min_samples_leaf,
        n_jobs=cfg.full_safe_n_jobs,
        random_state=cfg.random_state,
        verbose=cfg.log_rf_trees,
        bootstrap=cfg.full_safe_bootstrap,
    )
    model.fit(X_train, y_train)
    return model, "cpu_sklearn", time.time() - start


def predict_rf(model, backend: str, X):
    if "gpu" in backend.lower():
        import cupy as cp  # type: ignore

        X_gpu = cp.asarray(X.astype(np.float32))
        preds = to_numpy_safe(model.predict(X_gpu))
        del X_gpu
        cp.get_default_memory_pool().free_all_blocks()
        return preds
    return model.predict(X)


def predict_rf_batched(model, backend: str, X, batch_size: int = 500_000):
    preds = []
    for sl in iter_slices(len(X), batch_size):
        batch_pred = predict_rf(model, backend, X[sl])
        preds.append(np.asarray(batch_pred))
        del batch_pred
        clear_memory()
    return np.concatenate(preds) if preds else np.array([])


def combine_stages_batched(s1_pred, X_s1, full_len: int, reg_model, reg_backend: str, batch_size: int = 500_000):
    final_pred = np.zeros(full_len, dtype=np.float32)
    active_idx = np.where(s1_pred == 1)[0]
    for sl in iter_slices(len(active_idx), batch_size):
        idx_batch = active_idx[sl]
        X_batch = X_s1[idx_batch]
        log_pred = predict_rf_batched(reg_model, reg_backend, X_batch, batch_size=batch_size)
        final_pred[idx_batch] = np.expm1(log_pred).astype(np.float32)
        del idx_batch, X_batch, log_pred
        clear_memory()
    return np.clip(final_pred, 0, None)


def run_feature_search(
    train_df: pd.DataFrame,
    train_pos: pd.DataFrame,
    val_df: pd.DataFrame,
    val_pos: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    target: str,
    cfg: RFConfig,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    all_results: list[dict[str, Any]] = []
    for exp_name, train_feats in feature_sets.items():
        print(f"\nEvaluating Feature Set: {exp_name}")
        X_train_s1 = train_df[train_feats].to_numpy(dtype=np.float32, copy=True)
        y_train_s1 = train_df["Has_Throughput"].to_numpy(dtype=np.int32, copy=True)
        X_train_s2 = train_pos[train_feats].to_numpy(dtype=np.float32, copy=True)
        y_train_s2 = np.log1p(train_pos[target].to_numpy(dtype=np.float32, copy=True))

        X_val_s1 = val_df[train_feats].to_numpy(dtype=np.float32, copy=True)
        y_val_s1 = val_df["Has_Throughput"].to_numpy(dtype=np.int32, copy=True)
        X_val_s2 = val_pos[train_feats].to_numpy(dtype=np.float32, copy=True)
        y_val_s2_original = val_pos[target].to_numpy(dtype=np.float32, copy=True)
        y_val_full = val_df[target].to_numpy(dtype=np.float32, copy=True)

        clf, clf_backend, clf_time = train_stage1_random_forest_classifier(X_train_s1, y_train_s1, cfg, sample_n=cfg.sample_n_s1)
        reg, reg_backend, reg_time = train_stage2_random_forest_regressor(X_train_s2, y_train_s2, cfg, sample_n=cfg.sample_n_s2)
        del X_train_s1, y_train_s1, X_train_s2, y_train_s2
        clear_memory()

        val_s1_pred = predict_rf_batched(clf, clf_backend, X_val_s1, cfg.batch_size).astype(int)
        val_s1_acc = accuracy_score(y_val_s1, val_s1_pred)
        val_s1_f1 = f1_score(y_val_s1, val_s1_pred)
        val_s2_pred = np.clip(np.expm1(predict_rf_batched(reg, reg_backend, X_val_s2, cfg.batch_size)), 0, None)
        val_s2_metrics = eval_regression(y_val_s2_original, val_s2_pred, prefix="val_stage2_")
        val_final_pred = combine_stages_batched(val_s1_pred, X_val_s1, len(val_df), reg, reg_backend, cfg.batch_size)
        val_final_metrics = eval_regression(y_val_full, val_final_pred, prefix="val_final_")

        row = {
            "experiment": exp_name,
            "n_features": len(train_feats),
            "features": train_feats,
            "stage1_time_sec": clf_time,
            "stage2_time_sec": reg_time,
            "val_stage1_acc": val_s1_acc,
            "val_stage1_f1": val_s1_f1,
            "stage1_feature_importances": dict(zip(train_feats, to_numpy_safe(clf.feature_importances_))),
            "stage2_feature_importances": dict(zip(train_feats, to_numpy_safe(reg.feature_importances_))),
        }
        row.update(val_s2_metrics)
        row.update(val_final_metrics)
        all_results.append(row)

        del clf, reg, X_val_s1, y_val_s1, X_val_s2, y_val_s2_original, y_val_full, val_s1_pred, val_s2_pred, val_final_pred
        clear_memory()

    compare_df = pd.DataFrame(all_results)
    return compare_df, all_results


def choose_best_clean(compare_df: pd.DataFrame, all_results: list[dict[str, Any]], leaky_cols: list[str]) -> tuple[str, list[str]]:
    display_cols = [
        "experiment",
        "n_features",
        "val_stage1_f1",
        "val_stage2_R2_log",
        "val_final_R2_log",
        "val_final_RMSE_raw",
        "stage1_time_sec",
        "stage2_time_sec",
    ]
    ranked = compare_df[display_cols].sort_values("val_final_R2_log", ascending=False).reset_index(drop=True)
    is_clean = ranked["experiment"].apply(
        lambda exp: not any(f in leaky_cols for f in next(item["features"] for item in all_results if item["experiment"] == exp))
    )
    best_exp_name = ranked[is_clean].sort_values("val_final_R2_log", ascending=False).iloc[0]["experiment"]
    best_feats = next(item["features"] for item in all_results if item["experiment"] == best_exp_name)
    return str(best_exp_name), list(best_feats)


def final_train_and_evaluate_sequential(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    best_exp_name: str,
    best_feats: list[str],
    target: str,
    model_dir: str | Path,
    cfg: RFConfig,
    use_full_safe: bool = True,
) -> dict[str, Any]:
    """Train final RF models sequentially to avoid holding stage1 and stage2 matrices together."""
    model_dir = ensure_dir(model_dir)
    print(f"WINNER CHOSEN: {best_exp_name}")
    print("FINAL RF TRAINING: sequential Stage 1 then Stage 2")

    for c in best_feats:
        train_df[c] = train_df[c].astype(np.float32, copy=False)
        test_df[c] = test_df[c].astype(np.float32, copy=False)

    X_train_s1 = train_df[best_feats].to_numpy(dtype=np.float32, copy=True)
    y_train_s1 = train_df["Has_Throughput"].to_numpy(dtype=np.int32, copy=True)
    if use_full_safe:
        stage1_model, stage1_backend, stage1_time = train_stage1_rf_full_safe(X_train_s1, y_train_s1, cfg)
    else:
        stage1_model, stage1_backend, stage1_time = train_stage1_random_forest_classifier(X_train_s1, y_train_s1, cfg, sample_n=None)
    joblib.dump(stage1_model, model_dir / f"{best_exp_name}_FULL_stage1_clf.joblib", compress=3)
    del X_train_s1, y_train_s1
    clear_memory()

    active_mask = train_df["Has_Throughput"].to_numpy() == 1
    X_train_s2 = train_df.loc[active_mask, best_feats].to_numpy(dtype=np.float32, copy=True)
    y_train_s2 = np.log1p(train_df.loc[active_mask, target].to_numpy(dtype=np.float32, copy=True))
    del active_mask
    clear_memory()
    if use_full_safe:
        stage2_model, stage2_backend, stage2_time = train_stage2_rf_full_safe(X_train_s2, y_train_s2, cfg)
    else:
        stage2_model, stage2_backend, stage2_time = train_stage2_random_forest_regressor(X_train_s2, y_train_s2, cfg, sample_n=None)
    joblib.dump(stage2_model, model_dir / f"{best_exp_name}_FULL_stage2_reg.joblib", compress=3)
    del X_train_s2, y_train_s2
    clear_memory()

    X_test_s1 = test_df[best_feats].to_numpy(dtype=np.float32, copy=True)
    y_test_s1 = test_df["Has_Throughput"].to_numpy(dtype=np.int32, copy=True)
    y_test_full = test_df[target].to_numpy(dtype=np.float32, copy=True)

    test_s1_pred = predict_rf_batched(stage1_model, stage1_backend, X_test_s1, cfg.batch_size).astype(int)
    stage1_report = classification_report(y_test_s1, test_s1_pred, target_names=["Idle", "Active"], output_dict=True)
    stage1_cm = confusion_matrix(y_test_s1, test_s1_pred).tolist()

    active_test = test_df["Has_Throughput"].to_numpy() == 1
    X_test_s2 = test_df.loc[active_test, best_feats].to_numpy(dtype=np.float32, copy=True)
    y_test_s2_original = test_df.loc[active_test, target].to_numpy(dtype=np.float32, copy=True)
    test_s2_pred = np.clip(np.expm1(predict_rf_batched(stage2_model, stage2_backend, X_test_s2, cfg.batch_size)), 0, None)
    stage2_metrics = eval_regression(y_test_s2_original, test_s2_pred, prefix="stage2_test_")
    del X_test_s2, y_test_s2_original, test_s2_pred, active_test
    clear_memory()

    test_final_pred = combine_stages_batched(test_s1_pred, X_test_s1, len(test_df), stage2_model, stage2_backend, cfg.batch_size)
    final_metrics = eval_regression(y_test_full, test_final_pred, prefix="final_test_")

    metrics = {
        "best_exp_name": best_exp_name,
        "stage1_time_sec": stage1_time,
        "stage2_time_sec": stage2_time,
        "stage1_report": stage1_report,
        "stage1_confusion_matrix": stage1_cm,
        **stage2_metrics,
        **final_metrics,
    }

    del stage1_model, stage2_model, X_test_s1, y_test_s1, y_test_full, test_s1_pred, test_final_pred
    clear_memory()
    return metrics

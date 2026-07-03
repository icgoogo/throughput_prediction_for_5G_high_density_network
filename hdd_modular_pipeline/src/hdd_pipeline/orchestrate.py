from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .config import PipelineConfig
from .data import load_or_build_master
from .features import (
    add_has_throughput,
    add_traffic_label,
    apply_physical_filters,
    build_base_features,
    build_feature_sets,
    load_or_engineer_features,
)
from .rf_pipeline import choose_best_clean, final_train_and_evaluate_sequential, run_feature_search
from .sequence_pipeline import run_sequence_pipeline
from .splits import active_frames, fill_missing_from_train, user_train_val_test_split
from .reporting import RunReporter
from .utils import clear_memory, ensure_dir


def run_rf_pipeline(cfg: PipelineConfig, run_final: bool = True, use_full_safe: bool = True) -> dict:
    reporter = RunReporter(cfg.dataset.working_dir / "run_report.json", asdict(cfg))
    try:
        reporter.event("load_master_start")
        master_df = load_or_build_master(cfg.dataset)
        reporter.event("load_master_done", rows=len(master_df), cols=len(master_df.columns))

        feat_df = load_or_engineer_features(cfg.dataset, master_df)
        reporter.event("features_loaded", rows=len(feat_df), cols=len(feat_df.columns))
        feat_df = add_traffic_label(feat_df)
        feat_df = apply_physical_filters(feat_df)
        feat_df = add_has_throughput(feat_df, cfg.target)
        clear_memory()

        feature_sets, leaky_cols = build_feature_sets(feat_df)
        reporter.event("feature_sets_ready", feature_sets=list(feature_sets.keys()))

        train_df, val_df, test_df = user_train_val_test_split(feat_df, cfg.split)
        train_df, val_df, test_df = fill_missing_from_train(train_df, val_df, test_df)
        train_pos, val_pos, test_pos = active_frames(train_df, val_df, test_df)
        reporter.event("split_done", train_rows=len(train_df), val_rows=len(val_df), test_rows=len(test_df))

        compare_df, all_results = run_feature_search(train_df, train_pos, val_df, val_pos, feature_sets, cfg.target, cfg.rf)
        ensure_dir(cfg.dataset.working_dir)
        compare_path = cfg.dataset.working_dir / "rf_compare.csv"
        compare_df.to_csv(compare_path, index=False)
        best_exp_name, best_feats = choose_best_clean(compare_df, all_results, leaky_cols)
        reporter.event("feature_search_done", best_exp_name=best_exp_name, best_feats=best_feats)

        metrics = {"best_exp_name": best_exp_name, "best_feats": best_feats, "rf_compare_path": str(compare_path)}
        if run_final:
            final_metrics = final_train_and_evaluate_sequential(
                train_df=train_df,
                test_df=test_df,
                best_exp_name=best_exp_name,
                best_feats=best_feats,
                target=cfg.target,
                model_dir=cfg.dataset.model_dir,
                cfg=cfg.rf,
                use_full_safe=use_full_safe,
            )
            metrics.update(final_metrics)

        reporter.success(metrics)
        return metrics
    except Exception as exc:
        reporter.fail("run_rf_pipeline", exc)
        raise


def run_full_pipeline(cfg: PipelineConfig, run_final: bool = True, use_full_safe: bool = True):
    metrics = run_rf_pipeline(cfg, run_final=run_final, use_full_safe=use_full_safe)
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--working-dir", type=Path, default=None)
    parser.add_argument("--no-final", action="store_true", help="Run feature search only; skip final full training.")
    parser.add_argument("--unsafe-original-full-rf", action="store_true", help="Use original 100-tree style final RF instead of full-safe RF.")
    args = parser.parse_args(argv)

    cfg = PipelineConfig()
    if args.input_root is not None:
        cfg.dataset.input_root = args.input_root
    if args.working_dir is not None:
        cfg.dataset.working_dir = args.working_dir

    metrics = run_full_pipeline(cfg, run_final=not args.no_final, use_full_safe=not args.unsafe_original_full_rf)
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()

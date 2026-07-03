"""Example entry point for the optional Hybrid LSTM + MLP pipeline.

Normally this is called after RF has selected best_feats. Keep it separate from
RF because PyTorch windows can be slow/memory-heavy.
"""

from hdd_pipeline.config import PipelineConfig
from hdd_pipeline.data import load_or_build_master
from hdd_pipeline.features import add_has_throughput, apply_physical_filters, build_base_features, load_or_engineer_features
from hdd_pipeline.sequence_pipeline import run_sequence_pipeline
from hdd_pipeline.splits import fill_missing_from_train, user_train_val_test_split


def main():
    cfg = PipelineConfig()
    master = load_or_build_master(cfg.dataset)
    df = load_or_engineer_features(cfg.dataset, master)
    df = apply_physical_filters(add_has_throughput(df, cfg.target))
    _, dynamic_cols, static_cols = build_base_features(df)
    train_df, val_df, test_df = user_train_val_test_split(df, cfg.split)
    train_df, val_df, test_df = fill_missing_from_train(train_df, val_df, test_df)
    summary = run_sequence_pipeline(train_df, val_df, test_df, dynamic_cols, static_cols, cfg.target, cfg.sequence)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hdd_pipeline.config import PipelineConfig
from hdd_pipeline.features import add_has_throughput, build_feature_sets
from hdd_pipeline.rf_pipeline import run_feature_search
from hdd_pipeline.splits import active_frames, fill_missing_from_train, user_train_val_test_split


def make_synthetic_df(n_users: int = 20, timesteps: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for uid in range(n_users):
        x, y = rng.normal(0, 10), rng.normal(0, 10)
        for t in range(timesteps):
            state = rng.choice([0, 1, 2, 3, 4, 5], p=[0.02, 0.6, 0.15, 0.05, 0.03, 0.15])
            active = state != 1
            sinr = rng.normal(-10, 3)
            bler = np.clip(rng.beta(1, 10), 0, 1)
            prb = rng.normal(250, 30) if active else rng.normal(5, 2)
            throughput = max(0, prb * max(sinr + 20, 0) / 100 * (1 - bler) + rng.normal(0, 2)) if active else 0
            rows.append(
                {
                    "Timestamp_Sec": t * 15,
                    "User_ID": uid,
                    "X_meters": x + rng.normal(0, 0.2),
                    "Y_meters": y + rng.normal(0, 0.2),
                    "SINR_UL": sinr,
                    "SINR_UL_q90": sinr + rng.normal(1, 0.5),
                    "BLER": bler,
                    "BLER_q10": max(0, bler - 0.05),
                    "RU_Current_Load": rng.integers(10, 200),
                    "Crowd_Density_5m": rng.integers(0, 20),
                    "Log_Effective_Signal_Robust": np.log1p(max(0, 10 ** ((sinr + 1) / 10) / (bler + 1e-4))),
                    "MobileState": state,
                    "Throughput": throughput,
                }
            )
    df = pd.DataFrame(rows)
    for state in range(6):
        df[f"State_{state}"] = (df["MobileState"] == state).astype(np.int8)
    return add_has_throughput(df)


def main():
    cfg = PipelineConfig()
    cfg.split.test_id_start = 16
    cfg.rf.sample_n_s1 = 200
    cfg.rf.sample_n_s2 = 120
    cfg.rf.n_estimators = 5
    cfg.rf.max_depth = 4
    cfg.rf.n_jobs = 1
    cfg.rf.batch_size = 128

    df = make_synthetic_df()
    feature_sets, _ = build_feature_sets(df)
    train_df, val_df, test_df = user_train_val_test_split(df, cfg.split)
    train_df, val_df, test_df = fill_missing_from_train(train_df, val_df, test_df)
    train_pos, val_pos, _ = active_frames(train_df, val_df, test_df)
    compare_df, _ = run_feature_search(train_df, train_pos, val_df, val_pos, feature_sets, cfg.target, cfg.rf)
    print(compare_df[["experiment", "val_final_R2_log"]])
    print("Smoke test passed.")


if __name__ == "__main__":
    main()

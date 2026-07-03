from __future__ import annotations

import random

import pandas as pd

from .config import SplitConfig


def user_train_val_test_split(df: pd.DataFrame, cfg: SplitConfig):
    test_df = df[df["User_ID"] >= cfg.test_id_start].copy()
    train_val_df = df[df["User_ID"] < cfg.test_id_start].copy()

    unique_users = train_val_df["User_ID"].unique().tolist()
    random.seed(cfg.random_seed)
    random.shuffle(unique_users)

    val_size = int(len(unique_users) * cfg.val_frac)
    val_users = set(unique_users[:val_size])
    train_users = set(unique_users[val_size:])

    train_df = train_val_df[train_val_df["User_ID"].isin(train_users)].copy()
    val_df = train_val_df[train_val_df["User_ID"].isin(val_users)].copy()

    if cfg.is_undersampling:
        idle_rows = train_df[train_df["Has_Throughput"] == 0]
        active_rows = train_df[train_df["Has_Throughput"] != 0]
        sampled_idle = idle_rows.sample(n=len(active_rows), random_state=cfg.random_seed)
        train_df = pd.concat([active_rows, sampled_idle], axis=0).sample(frac=1, random_state=cfg.random_seed).reset_index(drop=True)

    print(f"Test users:  {test_df['User_ID'].nunique()} -> {len(test_df):,} rows")
    print(f"Train users: {len(train_users)} -> {len(train_df):,} rows")
    print(f"Val users:   {len(val_users)} -> {len(val_df):,} rows")
    return train_df, val_df, test_df


def fill_missing_from_train(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    fill_vals = train_df.mean(numeric_only=True)
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df.fillna(fill_vals, inplace=True)
    val_df.fillna(fill_vals, inplace=True)
    test_df.fillna(fill_vals, inplace=True)
    return train_df, val_df, test_df


def active_frames(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    return (
        train_df[train_df["Has_Throughput"] == 1].copy(),
        val_df[val_df["Has_Throughput"] == 1].copy(),
        test_df[test_df["Has_Throughput"] == 1].copy(),
    )

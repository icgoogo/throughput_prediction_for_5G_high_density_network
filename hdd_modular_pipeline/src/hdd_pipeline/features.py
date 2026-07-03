from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

from .config import DatasetConfig
from .utils import ensure_dir

TRAFFIC_LABELS = {
    0: "0: Off",
    1: "1: Idle",
    2: "2: Constant Rate",
    3: "3: Video",
    4: "4: Gaming",
    5: "5: HTTP",
}


def add_traffic_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "MobileState" in df.columns:
        df["Traffic_Label"] = df["MobileState"].map(TRAFFIC_LABELS)
    return df


def engineer_acc_arena_features(df: pd.DataFrame, drop_sinr_dl: bool = True) -> pd.DataFrame:
    """Advanced feature engineering from the original notebook.

    Produces:
    - SINR_Linear_q90
    - Effective_Signal_Robust
    - Log_Effective_Signal_Robust
    - RU_Current_Load
    - Velocity
    - Crowd_Density_{2,5,10}m
    - State_* one-hot columns
    - Resource_Efficiency
    """
    print("Initiating Advanced Feature Engineering Pipeline...")
    feat_df = df.copy()
    time_col = "Timestamp_Sec" if "Timestamp_Sec" in feat_df.columns else "Timestamp"
    feat_df = feat_df.sort_values(by=["User_ID", time_col]).reset_index(drop=True)

    if drop_sinr_dl and "SINR_DL" in feat_df.columns:
        feat_df.drop(columns=["SINR_DL"], inplace=True)

    if "SINR_UL_q90" in feat_df.columns and "BLER_q10" in feat_df.columns:
        feat_df["SINR_Linear_q90"] = (10 ** (feat_df["SINR_UL_q90"] / 10.0)).astype(np.float32)
        feat_df["Effective_Signal_Robust"] = np.clip(
            feat_df["SINR_Linear_q90"] / (feat_df["BLER_q10"] + 1e-4), 0, 1e6
        ).astype(np.float32)
        feat_df["Log_Effective_Signal_Robust"] = np.log1p(feat_df["Effective_Signal_Robust"]).astype(np.float32)

    if "RU" in feat_df.columns:
        feat_df["RU_Current_Load"] = feat_df.groupby([time_col, "RU"])["User_ID"].transform("count").astype(np.int16)

    user_groups = feat_df.groupby("User_ID", sort=False)
    feat_df["Delta_X"] = user_groups["X_meters"].diff().fillna(0).astype(np.float32)
    feat_df["Delta_Y"] = user_groups["Y_meters"].diff().fillna(0).astype(np.float32)
    feat_df["Velocity"] = np.sqrt(feat_df["Delta_X"] ** 2 + feat_df["Delta_Y"] ** 2).astype(np.float32)

    radii = [2.0, 5.0, 10.0]
    density_results = {r: np.zeros(len(feat_df), dtype=np.int16) for r in radii}
    for _, group in tqdm(feat_df.groupby(time_col), desc="Mapping Densities"):
        coords = group[["X_meters", "Y_meters"]].to_numpy(dtype=np.float32)
        tree = cKDTree(coords)
        for r in radii:
            counts = np.array([len(neighbors) - 1 for neighbors in tree.query_ball_point(coords, r)], dtype=np.int16)
            density_results[r][group.index] = counts

    feat_df["Crowd_Density_2m"] = density_results[2.0]
    feat_df["Crowd_Density_5m"] = density_results[5.0]
    feat_df["Crowd_Density_10m"] = density_results[10.0]

    if "MobileState" in feat_df.columns:
        state_dummies = pd.get_dummies(feat_df["MobileState"], prefix="State").astype(np.int8)
        feat_df = pd.concat([feat_df, state_dummies], axis=1)

    if "PRB" in feat_df.columns and "MobileState" in feat_df.columns:
        feat_df["Resource_Efficiency"] = (feat_df["PRB"] * (feat_df["MobileState"] != 1)).astype(np.float32)

    feat_df.drop(columns=[c for c in ["Delta_X", "Delta_Y"] if c in feat_df.columns], inplace=True)
    return feat_df


def load_or_engineer_features(cfg: DatasetConfig, master_df: pd.DataFrame) -> pd.DataFrame:
    if cfg.cached_feature_path.exists():
        print(f"Loading cached features: {cfg.cached_feature_path}")
        return pd.read_parquet(cfg.cached_feature_path)
    if cfg.input_feature_path.exists():
        print(f"Loading input features: {cfg.input_feature_path}")
        return pd.read_parquet(cfg.input_feature_path)

    feat_df = engineer_acc_arena_features(master_df)
    ensure_dir(cfg.working_dir)
    feat_df.to_parquet(cfg.cached_feature_path, index=False)
    return feat_df


def apply_physical_filters(df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    checks = [
        ("Throughput", lambda s: s >= 0),
        ("BLER", lambda s: s.between(0, 1)),
        ("BLER_q10", lambda s: s.between(0, 1)),
        ("SINR_UL", lambda s: s.between(-40, 30)),
        ("SINR_UL_q90", lambda s: s.between(-40, 30)),
    ]
    for col, fn in checks:
        if col in df.columns:
            mask &= fn(df[col])
    before = len(df)
    out = df[mask].copy()
    print(f"Removed {before - len(out):,} physically invalid rows")
    return out


def remove_constant_features(df: pd.DataFrame, cols: list[str], threshold: float = 1e-6) -> list[str]:
    kept, dropped = [], []
    for c in cols:
        if c in df.columns and df[c].std(skipna=True) > threshold:
            kept.append(c)
        else:
            dropped.append(c)
    if dropped:
        print("Dropped near-constant features:", dropped)
    return kept


def add_has_throughput(df: pd.DataFrame, target: str = "Throughput") -> pd.DataFrame:
    df = df.copy()
    df["Has_Throughput"] = (df[target] > 0).astype(np.int8)
    return df


def existing_cols(df: pd.DataFrame, cols: Iterable[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def build_feature_sets(df: pd.DataFrame) -> tuple[dict[str, list[str]], list[str]]:
    state_cols = [c for c in df.columns if c.startswith("State_")]
    mean_only_cols = ["SINR_UL", "BLER", "X_meters", "Y_meters", "RU_Current_Load", "Crowd_Density_5m"]
    peak_cols = ["SINR_UL_q90", "BLER_q10", "Log_Effective_Signal_Robust"]
    leaky_cols = ["PRB", "Resource_Efficiency"]

    feature_sets = {
        "B_mean_only_non_leaky": existing_cols(df, mean_only_cols + state_cols),
        "C_peak_enhanced_non_leaky": existing_cols(df, mean_only_cols + peak_cols + state_cols),
    }
    if any(c in df.columns for c in leaky_cols):
        feature_sets["A_leaky_upper_bound"] = existing_cols(df, mean_only_cols + peak_cols + leaky_cols + state_cols)
    return feature_sets, leaky_cols


def build_base_features(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    state_cols = [c for c in df.columns if c.startswith("State_")]
    base_feats = [
        "X_meters",
        "Y_meters",
        "SINR_UL",
        "SINR_UL_q90",
        "BLER",
        "BLER_q10",
        "RU_Current_Load",
        "Velocity",
        "Log_Effective_Signal_Robust",
        "Crowd_Density_5m",
    ] + state_cols
    base_feats = remove_constant_features(df, existing_cols(df, base_feats))

    dynamic_cols = [
        "SINR_UL",
        "BLER",
        "SINR_UL_q90",
        "BLER_q10",
        "Log_Effective_Signal_Robust",
        "RU_Current_Load",
        "X_meters",
        "Y_meters",
        "Velocity",
        "Crowd_Density_5m",
    ]
    dynamic_cols = [c for c in dynamic_cols if c in base_feats]
    static_cols = [c for c in state_cols if c in base_feats]
    return base_feats, dynamic_cols, static_cols


def partition_arena_vfl_channels(df: pd.DataFrame):
    y_target = df["Throughput"].copy()
    network_cols = ["SINR_UL", "PRB", "BLER", "RU", "RU_Current_Load"]
    network_cols = existing_cols(df, network_cols)
    device_cols = [
        "Latitude",
        "Longitude",
        "Altitude",
        "X_meters",
        "Y_meters",
        "Velocity",
        "Crowd_Density_2m",
        "Crowd_Density_5m",
        "Crowd_Density_10m",
        "MobileState",
        "State_0",
        "State_1",
        "State_2",
        "State_3",
        "State_4",
        "State_5",
        "Resource_Efficiency",
    ]
    device_cols = existing_cols(df, device_cols)
    X1_network = df[network_cols].copy()
    X2_device = df[device_cols].copy()
    overlapping = set(X1_network.columns).intersection(set(X2_device.columns))
    assert not overlapping, f"CRITICAL LEAKAGE DETECTED: {overlapping}"
    return X1_network, X2_device, y_target

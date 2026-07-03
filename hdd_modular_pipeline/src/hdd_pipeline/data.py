from __future__ import annotations

import gc
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import DatasetConfig
from .utils import clear_memory, ensure_dir

METRIC_SUBDIRS = {
    "Positions": "Positions_Acc_Arena",
    "Throughput": "Throughput_Acc_Arena",
    "SINRUL": "SINR_Acc_Arena",
    "SINRDL": "SINR_Acc_Arena",
    "PRB": "PRB_Acc_Arena",
    "RU": "RU_Association_Acc_Arena",
    "BLER": "BLER_Acc_Arena",
}

FILE_PATTERNS = {
    "Positions": "Positions_Salt_Tar_UE_Id_",
    "Throughput": "Throughput_UE_Id_",
    "SINRUL": "SINRUL_UE_Id_",
    "SINRDL": "SINRDL_UE_Id_",
    "PRB": "PRBS_UE_Id_",
    "RU": "RU_UE_Id_",
    "BLER": "BLER_UE_Id_",
}


def expected_files(cfg: DatasetConfig) -> list[Path]:
    files: list[Path] = []
    for metric_name, subdir in METRIC_SUBDIRS.items():
        for start_id in range(0, cfg.max_user_id + 1, cfg.chunk_size):
            end_id = min(start_id + cfg.chunk_size - 1, cfg.max_user_id)
            filename = f"{FILE_PATTERNS[metric_name]}{start_id}_{end_id}.csv"
            files.append(cfg.acc_arena_dir / subdir / filename)
    return files


def flatten_metric(file_path: str | Path, feature_name: str) -> pd.DataFrame:
    """Load one wide metric CSV and convert it to Timestamp/User_ID/value rows."""
    df = pd.read_csv(file_path)
    df_long = df.melt(id_vars=["time"], var_name="User_String", value_name=feature_name)
    df_long.rename(columns={"time": "Timestamp"}, inplace=True)
    df_long["User_ID"] = df_long["User_String"].str.replace("entityStats id ", "", regex=False).astype("int32")
    return df_long[["Timestamp", "User_ID", feature_name]]


def _load_positions(pos_path: Path, start_id: int, end_id: int, origin_lat: float | None, origin_lon: float | None):
    df_pos_raw = pd.read_csv(pos_path)
    raw_matrix = df_pos_raw.values
    timestamps = raw_matrix[:, 0].astype(np.int64)
    data_matrix = raw_matrix[:, 1:]

    num_timestamps = len(timestamps)
    num_users_in_chunk = end_id - start_id + 1
    reshaped = data_matrix.reshape(num_timestamps, num_users_in_chunk, 5)

    user_ids = reshaped[:, :, 0].astype(np.int32).reshape(-1)
    lat = reshaped[:, :, 1].astype(np.float32).reshape(-1)
    lon = reshaped[:, :, 2].astype(np.float32).reshape(-1)
    alt = reshaped[:, :, 3].astype(np.float32).reshape(-1)
    mobile = reshaped[:, :, 4].astype(np.int8).reshape(-1)
    ts = np.repeat(timestamps, num_users_in_chunk).astype(np.int64)

    if origin_lat is None:
        origin_lat = float(lat[0])
        origin_lon = float(lon[0])

    lat_to_m = 111_320.0
    lon_to_m = 111_320.0 * np.cos(np.radians(origin_lat))

    pos_c = pd.DataFrame(
        {
            "Timestamp": ts,
            "User_ID": user_ids,
            "Latitude": lat,
            "Longitude": lon,
            "Altitude": alt,
            "MobileState": mobile,
        }
    )
    pos_c["X_meters"] = ((pos_c["Longitude"] - origin_lon) * lon_to_m).astype("float32")
    pos_c["Y_meters"] = ((pos_c["Latitude"] - origin_lat) * lat_to_m).astype("float32")
    pos_c = pos_c.drop_duplicates(subset=["Timestamp", "User_ID"])
    return pos_c, origin_lat, origin_lon


def _aggregate_chunk(chunk_df: pd.DataFrame) -> pd.DataFrame:
    chunk_df["Timestamp_DT"] = pd.to_datetime(chunk_df["Timestamp"], unit="s")
    gb = chunk_df.set_index("Timestamp_DT").groupby("User_ID")

    fast_rules = {
        "Latitude": "mean",
        "Longitude": "mean",
        "Altitude": "mean",
        "X_meters": "mean",
        "Y_meters": "mean",
        "SINR_UL": "mean",
        "SINR_DL": "mean",
        "BLER": "mean",
        "RU": "first",
        "MobileState": "first",
    }
    chunk_agg = gb.resample("15s").agg(fast_rules).dropna().reset_index()

    q_tp = gb.resample("15s")["Throughput"].quantile(0.95).reset_index(name="Throughput")
    q_prb = gb.resample("15s")["PRB"].quantile(0.95).reset_index(name="PRB")
    q_sinr = gb.resample("15s")["SINR_UL"].quantile(0.90).reset_index(name="SINR_UL_q90")
    q_bler = gb.resample("15s")["BLER"].quantile(0.10).reset_index(name="BLER_q10")

    chunk_agg = (
        chunk_agg.merge(q_tp, on=["User_ID", "Timestamp_DT"])
        .merge(q_prb, on=["User_ID", "Timestamp_DT"])
        .merge(q_sinr, on=["User_ID", "Timestamp_DT"])
        .merge(q_bler, on=["User_ID", "Timestamp_DT"])
    )

    float_cols = [
        "Latitude",
        "Longitude",
        "Altitude",
        "X_meters",
        "Y_meters",
        "Throughput",
        "SINR_UL",
        "SINR_DL",
        "PRB",
        "BLER",
        "SINR_UL_q90",
        "BLER_q10",
    ]
    chunk_agg[float_cols] = chunk_agg[float_cols].astype("float32")
    chunk_agg["User_ID"] = chunk_agg["User_ID"].astype("int32")
    chunk_agg["RU"] = chunk_agg["RU"].astype("int16")
    chunk_agg["MobileState"] = chunk_agg["MobileState"].astype("int8")
    return chunk_agg


def clean_master_timeline(master_df: pd.DataFrame) -> pd.DataFrame:
    master_df = master_df.loc[:, ~master_df.columns.duplicated()].copy()
    cols_to_drop = [c for c in ["Timestamp", "index", "level_0"] if c in master_df.columns]
    if cols_to_drop:
        master_df.drop(columns=cols_to_drop, inplace=True)
    master_df["Timestamp_Sec"] = master_df.groupby("User_ID").cumcount() * 15
    cols = ["Timestamp_Sec", "User_ID"] + [c for c in master_df.columns if c not in ["Timestamp_Sec", "User_ID"]]
    return master_df[cols]


def build_master_from_chunks(cfg: DatasetConfig, save: bool = True) -> pd.DataFrame:
    """Build the 15-second master parquet from raw CSV chunks."""
    aggregated_chunks: list[pd.DataFrame] = []
    origin_lat: float | None = None
    origin_lon: float | None = None

    print(f"Starting ingestion for {cfg.max_user_id + 1:,} users...")
    for start_id in range(0, cfg.max_user_id + 1, cfg.chunk_size):
        end_id = min(start_id + cfg.chunk_size - 1, cfg.max_user_id)
        print(f"Processing users {start_id} to {end_id}...")

        tp_path = cfg.acc_arena_dir / "Throughput_Acc_Arena" / f"Throughput_UE_Id_{start_id}_{end_id}.csv"
        sinr_ul_path = cfg.acc_arena_dir / "SINR_Acc_Arena" / f"SINRUL_UE_Id_{start_id}_{end_id}.csv"
        sinr_dl_path = cfg.acc_arena_dir / "SINR_Acc_Arena" / f"SINRDL_UE_Id_{start_id}_{end_id}.csv"
        prb_path = cfg.acc_arena_dir / "PRB_Acc_Arena" / f"PRBS_UE_Id_{start_id}_{end_id}.csv"
        ru_path = cfg.acc_arena_dir / "RU_Association_Acc_Arena" / f"RU_UE_Id_{start_id}_{end_id}.csv"
        bler_path = cfg.acc_arena_dir / "BLER_Acc_Arena" / f"BLER_UE_Id_{start_id}_{end_id}.csv"
        pos_path = cfg.acc_arena_dir / "Positions_Acc_Arena" / f"Positions_Salt_Tar_UE_Id_{start_id}_{end_id}.csv"

        tp_c = flatten_metric(tp_path, "Throughput").drop_duplicates(["Timestamp", "User_ID"])
        sinr_ul_c = flatten_metric(sinr_ul_path, "SINR_UL").drop_duplicates(["Timestamp", "User_ID"])
        sinr_dl_c = flatten_metric(sinr_dl_path, "SINR_DL").drop_duplicates(["Timestamp", "User_ID"])
        prb_c = flatten_metric(prb_path, "PRB").drop_duplicates(["Timestamp", "User_ID"])
        ru_c = flatten_metric(ru_path, "RU").drop_duplicates(["Timestamp", "User_ID"])
        bler_c = flatten_metric(bler_path, "BLER").drop_duplicates(["Timestamp", "User_ID"])
        pos_c, origin_lat, origin_lon = _load_positions(pos_path, start_id, end_id, origin_lat, origin_lon)

        chunk_df = (
            pos_c.merge(tp_c, on=["Timestamp", "User_ID"], how="inner")
            .merge(sinr_ul_c, on=["Timestamp", "User_ID"], how="inner")
            .merge(sinr_dl_c, on=["Timestamp", "User_ID"], how="inner")
            .merge(prb_c, on=["Timestamp", "User_ID"], how="inner")
            .merge(ru_c, on=["Timestamp", "User_ID"], how="inner")
            .merge(bler_c, on=["Timestamp", "User_ID"], how="inner")
        )

        aggregated_chunks.append(_aggregate_chunk(chunk_df))
        del chunk_df, pos_c, tp_c, sinr_ul_c, sinr_dl_c, prb_c, ru_c, bler_c
        clear_memory()

    master_df = pd.concat(aggregated_chunks, ignore_index=True)
    master_df.rename(columns={"Timestamp_DT": "Timestamp"}, inplace=True)
    master_df = clean_master_timeline(master_df)

    if save:
        ensure_dir(cfg.working_dir)
        master_df.to_parquet(cfg.cached_master_path, index=False)
    return master_df


def load_or_build_master(cfg: DatasetConfig) -> pd.DataFrame:
    if cfg.cached_master_path.exists():
        print(f"Loading cached master: {cfg.cached_master_path}")
        return clean_master_timeline(pd.read_parquet(cfg.cached_master_path))
    if cfg.input_master_path.exists():
        print(f"Loading input master: {cfg.input_master_path}")
        return clean_master_timeline(pd.read_parquet(cfg.input_master_path))
    return build_master_from_chunks(cfg, save=True)

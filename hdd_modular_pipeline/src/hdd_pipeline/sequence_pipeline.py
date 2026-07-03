from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .config import SequenceConfig
from .utils import set_seed

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover - torch is optional for RF-only smoke tests
    torch = None
    nn = None
    Dataset = object
    DataLoader = None


def _require_torch():
    if torch is None or nn is None or DataLoader is None:
        raise ImportError("PyTorch is required for sequence_pipeline.py. Install torch or run RF pipeline only.")


class HybridSequenceDataset(Dataset):
    """Sequence dataset for hybrid LSTM + static MLP model."""

    def __init__(
        self,
        df: pd.DataFrame,
        dynamic_cols: list[str],
        static_cols: list[str],
        target_col: str,
        user_col: str,
        seq_len: int,
        stride: int,
        task: str = "classifier",
        active_only: bool = False,
        active_col: str = "Has_Throughput",
    ):
        _require_torch()
        self.df = df.reset_index(drop=True)
        self.dynamic_cols = dynamic_cols
        self.static_cols = static_cols
        self.target_col = target_col
        self.user_col = user_col
        self.seq_len = seq_len
        self.stride = stride
        self.task = task
        self.active_only = active_only

        self.X_dyn = self.df[dynamic_cols].to_numpy(dtype=np.float32)
        self.X_sta = self.df[static_cols].to_numpy(dtype=np.float32)
        self.y_raw = self.df[target_col].to_numpy(dtype=np.float32)
        self.is_active = self.df[active_col].to_numpy(dtype=np.float32)
        self.uid = self.df[user_col].to_numpy()
        self.windows = self._build_windows()

    def _build_windows(self):
        windows = []
        n = len(self.df)
        i = 0
        while i + self.seq_len <= n:
            start = i
            end = i + self.seq_len - 1
            if self.uid[start] == self.uid[end]:
                if self.active_only and self.is_active[end] <= 0:
                    i += self.stride
                    continue
                windows.append((start, end))
            i += self.stride
        return windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start, end = self.windows[idx]
        x_dyn = self.X_dyn[start : end + 1]
        x_sta = self.X_sta[end]
        y_value = self.y_raw[end]
        if self.task == "classifier":
            y = np.float32(self.is_active[end])
        elif self.task == "regressor":
            y = np.float32(np.log1p(y_value))
        else:
            raise ValueError("task must be 'classifier' or 'regressor'")
        return (
            torch.from_numpy(x_dyn),
            torch.from_numpy(x_sta),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(y_value, dtype=torch.float32),
            torch.tensor(end, dtype=torch.long),
        )


class HybridLSTMMLP(nn.Module):
    def __init__(
        self,
        dynamic_dim: int,
        static_dim: int,
        lstm_hidden: int = 32,
        mlp_hidden: int = 16,
        fusion_hidden: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        _require_torch()
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=dynamic_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.static_mlp = nn.Sequential(nn.Linear(static_dim, mlp_hidden), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden + mlp_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(self, x_dynamic, x_static):
        lstm_out, _ = self.lstm(x_dynamic)
        lstm_last = lstm_out[:, -1, :]
        static_emb = self.static_mlp(x_static)
        x = torch.cat([lstm_last, static_emb], dim=1)
        return self.head(x).squeeze(-1)


def make_loader(dataset, cfg: SequenceConfig, device, shuffle: bool = False):
    _require_torch()
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )


def train_model(model, train_loader, val_loader, criterion, cfg: SequenceConfig, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val_loss = float("inf")
    best_state = None
    wait = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for x_dyn, x_sta, y, _, _ in train_loader:
            x_dyn = x_dyn.to(device)
            x_sta = x_sta.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred = model(x_dyn, x_sta)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss_sum += loss.item() * x_dyn.size(0)

        train_loss = train_loss_sum / len(train_loader.dataset)
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for x_dyn, x_sta, y, _, _ in val_loader:
                x_dyn = x_dyn.to(device)
                x_sta = x_sta.to(device)
                y = y.to(device)
                pred = model(x_dyn, x_sta)
                loss = criterion(pred, y)
                val_loss_sum += loss.item() * x_dyn.size(0)
        val_loss = val_loss_sum / len(val_loader.dataset)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                print(f"Early stopping at epoch {epoch}. Best val_loss={best_val_loss:.5f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def predict_model(model, loader, device):
    model.eval()
    preds, ys, y_raws, end_indices = [], [], [], []
    with torch.no_grad():
        for x_dyn, x_sta, y, y_raw, end_idx in loader:
            x_dyn = x_dyn.to(device)
            x_sta = x_sta.to(device)
            pred = model(x_dyn, x_sta).detach().cpu().numpy()
            preds.append(pred)
            ys.append(y.numpy())
            y_raws.append(y_raw.numpy())
            end_indices.append(end_idx.numpy())
    return {
        "pred": np.concatenate(preds),
        "y": np.concatenate(ys),
        "y_raw": np.concatenate(y_raws),
        "end_idx": np.concatenate(end_indices),
    }


def best_threshold_by_f1(y_true, prob):
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        pred = (prob >= t).astype(int)
        score = f1_score(y_true, pred)
        if score > best_f1:
            best_f1 = score
            best_t = t
    return best_t, best_f1


def safe_auc(y_true, prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, prob)


def run_sequence_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dynamic_cols: list[str],
    static_cols: list[str],
    target: str,
    cfg: SequenceConfig,
) -> pd.DataFrame:
    _require_torch()
    set_seed(cfg.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dynamic_scaler = StandardScaler()
    static_scaler = StandardScaler()
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df[dynamic_cols] = dynamic_scaler.fit_transform(train_df[dynamic_cols])
    val_df[dynamic_cols] = dynamic_scaler.transform(val_df[dynamic_cols])
    test_df[dynamic_cols] = dynamic_scaler.transform(test_df[dynamic_cols])

    continuous_static_cols = [c for c in static_cols if not c.startswith("State_")]
    if continuous_static_cols:
        train_df[continuous_static_cols] = static_scaler.fit_transform(train_df[continuous_static_cols])
        val_df[continuous_static_cols] = static_scaler.transform(val_df[continuous_static_cols])
        test_df[continuous_static_cols] = static_scaler.transform(test_df[continuous_static_cols])

    train_cls_ds = HybridSequenceDataset(train_df, dynamic_cols, static_cols, target, "User_ID", cfg.seq_len, cfg.stride, "classifier")
    val_cls_ds = HybridSequenceDataset(val_df, dynamic_cols, static_cols, target, "User_ID", cfg.seq_len, cfg.stride, "classifier")
    train_reg_ds = HybridSequenceDataset(train_df, dynamic_cols, static_cols, target, "User_ID", cfg.seq_len, cfg.stride, "regressor", active_only=True)
    val_reg_ds = HybridSequenceDataset(val_df, dynamic_cols, static_cols, target, "User_ID", cfg.seq_len, cfg.stride, "regressor", active_only=True)

    train_cls_loader = make_loader(train_cls_ds, cfg, device, shuffle=True)
    val_cls_loader = make_loader(val_cls_ds, cfg, device, shuffle=False)
    train_reg_loader = make_loader(train_reg_ds, cfg, device, shuffle=True)
    val_reg_loader = make_loader(val_reg_ds, cfg, device, shuffle=False)

    n_pos = sum(train_cls_ds.is_active[end] > 0 for _, end in train_cls_ds.windows)
    n_neg = len(train_cls_ds) - n_pos
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32).to(device)

    stage1_model = HybridLSTMMLP(len(dynamic_cols), len(static_cols), cfg.lstm_hidden, cfg.mlp_hidden, cfg.fusion_hidden, cfg.num_layers, cfg.dropout).to(device)
    start = time.time()
    stage1_model, _ = train_model(stage1_model, train_cls_loader, val_cls_loader, nn.BCEWithLogitsLoss(pos_weight=pos_weight), cfg, device)
    stage1_time = time.time() - start

    val_cls_out = predict_model(stage1_model, val_cls_loader, device)
    val_prob = 1 / (1 + np.exp(-val_cls_out["pred"]))
    best_threshold, val_best_f1 = best_threshold_by_f1(val_cls_out["y"], val_prob)
    val_auc = safe_auc(val_cls_out["y"], val_prob)

    stage2_model = HybridLSTMMLP(len(dynamic_cols), len(static_cols), cfg.lstm_hidden, cfg.mlp_hidden, cfg.fusion_hidden, cfg.num_layers, cfg.dropout).to(device)
    start = time.time()
    stage2_model, _ = train_model(stage2_model, train_reg_loader, val_reg_loader, nn.MSELoss(), cfg, device)
    stage2_time = time.time() - start

    # Test is touched only after validation threshold selection.
    test_cls_ds = HybridSequenceDataset(test_df, dynamic_cols, static_cols, target, "User_ID", cfg.seq_len, cfg.stride, "classifier")
    test_reg_active_ds = HybridSequenceDataset(test_df, dynamic_cols, static_cols, target, "User_ID", cfg.seq_len, cfg.stride, "regressor", active_only=True)
    test_reg_all_ds = HybridSequenceDataset(test_df, dynamic_cols, static_cols, target, "User_ID", cfg.seq_len, cfg.stride, "regressor", active_only=False)
    test_cls_out = predict_model(stage1_model, make_loader(test_cls_ds, cfg, device), device)
    test_prob = 1 / (1 + np.exp(-test_cls_out["pred"]))
    test_cls_pred = (test_prob >= best_threshold).astype(int)
    test_stage1_f1 = f1_score(test_cls_out["y"], test_cls_pred)
    test_stage1_auc = safe_auc(test_cls_out["y"], test_prob)

    test_active_out = predict_model(stage2_model, make_loader(test_reg_active_ds, cfg, device), device)
    test_pred_raw_active = np.clip(np.expm1(test_active_out["pred"]), 0, None)
    test_true_raw_active = np.expm1(test_active_out["y"])
    test_stage2_rmse = np.sqrt(mean_squared_error(test_true_raw_active, test_pred_raw_active))
    test_stage2_r2_log = r2_score(test_active_out["y"], test_active_out["pred"])

    test_all_out = predict_model(stage2_model, make_loader(test_reg_all_ds, cfg, device), device)
    test_final_pred = np.where(test_prob >= best_threshold, np.clip(np.expm1(test_all_out["pred"]), 0, None), 0.0)
    test_final_r2_log = r2_score(np.log1p(test_cls_out["y_raw"]), np.log1p(test_final_pred))
    test_final_rmse = np.sqrt(mean_squared_error(test_cls_out["y_raw"], test_final_pred))

    return pd.DataFrame(
        [
            {
                "Model": "Hybrid LSTM + MLP",
                "Target": target,
                "Seq_Len": cfg.seq_len,
                "Stride": cfg.stride,
                "Dynamic_Features": len(dynamic_cols),
                "Static_Features": len(static_cols),
                "Val_Stage1_F1": val_best_f1,
                "Val_Stage1_AUC": val_auc,
                "Test_Stage1_F1": test_stage1_f1,
                "Test_Stage1_AUC": test_stage1_auc,
                "Test_Stage2_RMSE_Raw": test_stage2_rmse,
                "Test_Stage2_R2_Log": test_stage2_r2_log,
                "Test_Final_RMSE_Raw": test_final_rmse,
                "Test_Final_R2_Log": test_final_r2_log,
                "Stage1_Time_s": stage1_time,
                "Stage2_Time_s": stage2_time,
                "Total_Time_s": stage1_time + stage2_time,
            }
        ]
    )

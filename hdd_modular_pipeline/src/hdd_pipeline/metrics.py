from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def eval_regression(y_true, y_pred, prefix: str = "") -> dict[str, float]:
    y_true_log = np.log1p(y_true)
    y_pred_log = np.log1p(np.clip(y_pred, 0, None))
    return {
        f"{prefix}MAE_raw": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}RMSE_raw": rmse(y_true, y_pred),
        f"{prefix}R2_raw": float(r2_score(y_true, y_pred)),
        f"{prefix}MAE_log": float(mean_absolute_error(y_true_log, y_pred_log)),
        f"{prefix}RMSE_log": rmse(y_true_log, y_pred_log),
        f"{prefix}R2_log": float(r2_score(y_true_log, y_pred_log)),
    }


def safe_auc(y_true, prob) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, prob))

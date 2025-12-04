import torch
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def compute_metrics(y_true, y_pred):
    """
    y_true, y_pred: arrays of shape (N,)
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)

    if mask.sum() < 2:
        return {
            "corr": np.nan,
            "r2": np.nan,
            "mse": np.nan,
            "mae": np.nan,
        }

    yt = y_true[mask]
    yp = y_pred[mask]

    return {
        "corr": np.corrcoef(yt, yp)[0, 1],
        "r2": r2_score(yt, yp),
        "mse": mean_squared_error(yt, yp),
        "mae": mean_absolute_error(yt, yp),
    }

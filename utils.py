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


def create_train_val_split(df_returns, val_ratio=0.2):
    """
    Split data into train and validation sets (temporal split).
    """
    T = len(df_returns)
    split_idx = int(T * (1 - val_ratio))

    train_df = df_returns.iloc[:split_idx]
    val_df = df_returns.iloc[split_idx:]

    print(f"Train period: {train_df.index[0]} to {train_df.index[-1]}")
    print(f"Val period: {val_df.index[0]} to {val_df.index[-1]}")

    return train_df, val_df

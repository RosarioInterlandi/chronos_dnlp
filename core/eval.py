import torch
import numpy as np
from tqdm import tqdm
from utils import compute_metrics

import warnings

warnings.filterwarnings("ignore", message=".*pin_memory.*")


def chronos_one_step_forecast(
    pipeline, window_df, device, quantile_idx=4  # shape (T, N)  # default = median
):
    """
    Returns:
        y_hat: (N,) predicted next-day returns
    """

    # (T, N) → (N, T) → (1, N, T)
    context = (
        torch.tensor(window_df.values.T, dtype=torch.float32).unsqueeze(0).to(device)
    )

    forecast = pipeline.predict(
        context,
        prediction_length=1,
    )

    # (N, 9, 1) → (N,)
    y_hat = forecast[0][:, quantile_idx, 0].cpu().numpy()

    return y_hat


def run_chronos_sliding_backtest(
    pipeline,
    df_returns,  # full TxN returns DataFrame
    device,
    context_length=200,
    start_idx=None,
    quantile_idx=4,
):
    """
    Walk-forward evaluation of Chronos.

    Returns:
        pred_matrix: (T_eval, N)
        true_matrix: (T_eval, N)
        daily_metrics: list of metric dicts
    """

    df = df_returns.dropna().copy()
    T, N = df.shape

    if start_idx is None:
        start_idx = context_length

    preds = []
    trues = []
    daily_metrics = []
    timestamps = []

    for t in tqdm(range(start_idx, T - 1)):

        window = df.iloc[t - context_length : t]  # (T, N)
        y_true = df.iloc[t].values  # (N,)

        y_pred = chronos_one_step_forecast(
            pipeline=pipeline,
            window_df=window,
            device=device,
            quantile_idx=quantile_idx,
        )

        preds.append(y_pred)
        trues.append(y_true)
        timestamps.append(df.index[t])

        metrics = compute_metrics(y_true, y_pred)
        daily_metrics.append(metrics)

    pred_matrix = np.vstack(preds)
    true_matrix = np.vstack(trues)

    return {
        "preds": pred_matrix,
        "trues": true_matrix,
        "dates": np.array(timestamps),
        "daily_metrics": daily_metrics,
    }


def summarize_backtest_results(results: dict) -> dict:
    """
    Aggregate daily backtest metrics into summary statistics.

    Args:
        results: Output from run_chronos_sliding_backtest.

    Returns:
        Dict with mean correlation, R2, MSE, and MAE.
    """
    daily_metrics = results["daily_metrics"]

    agg = {
        "mean_corr": np.nanmean([m["corr"] for m in daily_metrics]),
        "mean_r2": np.nanmean([m["r2"] for m in daily_metrics]),
        "mean_mse": np.nanmean([m["mse"] for m in daily_metrics]),
        "mean_mae": np.nanmean([m["mae"] for m in daily_metrics]),
    }

    return agg


def compute_quantile_loss(
    y_true: np.ndarray,
    y_pred_quantiles: np.ndarray,
    quantile_levels: list[float] = None,
) -> float:
    """
    Compute the average quantile loss (pinball loss) across all quantiles.

    This is the same loss function Chronos-2 uses during training.

    Args:
        y_true: Array of shape (N,) with actual values.
        y_pred_quantiles: Array of shape (N, n_quantiles) with predicted quantiles.
        quantile_levels: List of quantile levels. Default is Chronos's 9 quantiles:
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    Returns:
        Average quantile loss across all quantiles and samples.

    Example:
        >>> y_true = np.array([0.01, -0.02, 0.005])
        >>> y_pred = np.random.randn(3, 9) * 0.01  # 3 stocks, 9 quantiles
        >>> loss = compute_quantile_loss(y_true, y_pred)
    """
    if quantile_levels is None:
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    total_loss = 0.0
    for i, q in enumerate(quantile_levels):
        errors = y_true - y_pred_quantiles[:, i]
        # Pinball loss: q * max(error, 0) + (1-q) * max(-error, 0)
        total_loss += np.mean(np.where(errors >= 0, q * errors, (q - 1) * errors))

    return total_loss / len(quantile_levels)


def evaluate_model_on_test(
    pipeline,
    df_test,
    context_length: int,
    n_samples: int = 100,
) -> dict:
    """
    Evaluate a Chronos-2 model on test data by computing quantile loss.

    Samples random windows from the test set and computes average loss.

    Args:
        pipeline: Chronos2Pipeline model (zero-shot or fine-tuned).
        df_test: Test DataFrame with shape (T, N).
        context_length: Number of timesteps for context window.
        n_samples: Number of random windows to sample for evaluation.

    Returns:
        Dict with 'mean_quantile_loss', 'std_quantile_loss', 'mean_mse',
        'mean_mae', and 'quantile_losses' list.

    Example:
        >>> results = evaluate_model_on_test(pipeline, df_test, context_length=200)
        >>> print(f"Quantile Loss: {results['mean_quantile_loss']:.4f}")
    """

    data = df_test.values.astype(np.float32)  # (T, N)
    T, N = data.shape

    quantile_losses = []
    mse_losses = []
    mae_losses = []

    # Sample random windows
    np.random.seed(42)  # For reproducibility
    max_start = T - context_length - 1
    start_indices = np.random.choice(
        max_start, size=min(n_samples, max_start), replace=False
    )

    for start in start_indices:
        # Context and target
        context = data[start : start + context_length, :].T  # (N, context_length)
        y_true = data[start + context_length, :]  # (N,)

        # Predict
        forecast = pipeline.predict([{"target": context}], prediction_length=1)
        y_pred_quantiles = forecast[0][:, :, 0].cpu().numpy()  # (N, n_quantiles)
        y_pred_median = y_pred_quantiles[:, 4]  # Median (0.5 quantile)

        # Compute losses
        q_loss = compute_quantile_loss(y_true, y_pred_quantiles)
        mse = np.mean((y_true - y_pred_median) ** 2)
        mae = np.mean(np.abs(y_true - y_pred_median))

        quantile_losses.append(q_loss)
        mse_losses.append(mse)
        mae_losses.append(mae)

    return {
        "mean_quantile_loss": np.mean(quantile_losses),
        "std_quantile_loss": np.std(quantile_losses),
        "mean_mse": np.mean(mse_losses),
        "mean_mae": np.mean(mae_losses),
        "quantile_losses": quantile_losses,
    }


def compare_models(
    baseline_results: dict,
    finetuned_results: dict,
) -> dict:
    """
    Compare baseline and fine-tuned model results.

    Args:
        baseline_results: Output from evaluate_model_on_test for baseline model.
        finetuned_results: Output from evaluate_model_on_test for fine-tuned model.

    Returns:
        Dict with improvement percentages for each metric.
    """
    ql_improvement = (
        (
            baseline_results["mean_quantile_loss"]
            - finetuned_results["mean_quantile_loss"]
        )
        / baseline_results["mean_quantile_loss"]
        * 100
    )
    mse_improvement = (
        (baseline_results["mean_mse"] - finetuned_results["mean_mse"])
        / baseline_results["mean_mse"]
        * 100
    )
    mae_improvement = (
        (baseline_results["mean_mae"] - finetuned_results["mean_mae"])
        / baseline_results["mean_mae"]
        * 100
    )

    return {
        "quantile_loss_improvement": ql_improvement,
        "mse_improvement": mse_improvement,
        "mae_improvement": mae_improvement,
    }

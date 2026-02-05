"""Hyperparameter tuning for fine-tuned Chronos-2 models."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "model"))

from chronos import Chronos2Pipeline  # noqa: E402
from core.data import GICS_LEVEL_1, create_multivariate_windows, prepare_data_for_chronos  # noqa: E402
from core.eval import evaluate_model_on_test  # noqa: E402
from tiingo_data.download_data import get_daily_returns_data_cached  # noqa: E402
from utils import create_train_val_split, get_device  # noqa: E402

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TuningConfig:
    learning_rate: float
    num_steps: int
    batch_size: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "num_steps": self.num_steps,
            "batch_size": self.batch_size,
        }

    def tag(self) -> str:
        lr_tag = f"{self.learning_rate:.1e}".replace("+", "")
        return f"lr_{lr_tag}_steps_{self.num_steps}_bs_{self.batch_size}"


def build_config_grid(grid: dict[str, list[Any]]) -> list[TuningConfig]:
    learning_rates = grid.get("learning_rate", [1e-6])
    num_steps = grid.get("num_steps", [500])
    batch_sizes = grid.get("batch_size", [32])

    return [
        TuningConfig(learning_rate=lr, num_steps=steps, batch_size=batch)
        for lr, steps, batch in itertools.product(learning_rates, num_steps, batch_sizes)
    ]


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_grid(grid_path: str | None) -> dict[str, list[Any]]:
    if grid_path is None:
        return {
            "learning_rate": [1e-6, 5e-6, 1e-5],
            "num_steps": [500, 1000],
            "batch_size": [16, 32],
        }

    grid_file = Path(grid_path)
    with grid_file.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_category_models(path: str | None) -> dict[str, str]:
    if path is None:
        return {}

    model_path = Path(path)
    with model_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_inputs(
    df_train,
    df_val,
    context_length: int,
    prediction_length: int,
    stride: int,
) -> tuple[list[dict[str, np.ndarray]], list[dict[str, np.ndarray]]]:
    train_inputs = create_multivariate_windows(
        df_train,
        context_length=context_length,
        prediction_length=prediction_length,
        stride=stride,
    )
    val_inputs = create_multivariate_windows(
        df_val,
        context_length=context_length,
        prediction_length=prediction_length,
        stride=stride,
    )
    return train_inputs, val_inputs


def tune_single_model(
    model_path: str,
    train_inputs,
    val_inputs,
    df_val,
    configs: list[TuningConfig],
    prediction_length: int,
    context_length: int,
    min_past: int,
    output_dir: Path,
    n_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    device = get_device()

    for config in configs:
        LOGGER.info("Tuning config %s", config.tag())
        set_seeds(seed)

        pipeline = Chronos2Pipeline.from_pretrained(
            model_path,
            device_map=device,
            dtype=torch.float32,
        )

        run_dir = output_dir / config.tag()
        run_dir.mkdir(parents=True, exist_ok=True)

        finetuned = pipeline.fit(
            inputs=train_inputs,
            validation_inputs=val_inputs,
            prediction_length=prediction_length,
            context_length=context_length,
            min_past=min_past,
            num_steps=config.num_steps,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            output_dir=run_dir,
        )

        metrics = evaluate_model_on_test(
            pipeline=finetuned,
            df_test=df_val,
            context_length=context_length,
            n_samples=n_samples,
        )

        results.append(
            {
                "config": config.as_dict(),
                "metrics": metrics,
                "checkpoint_dir": str(run_dir),
            }
        )

    return results


def select_top_configs(results: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    return sorted(results, key=lambda x: x["metrics"]["mean_quantile_loss"])[:top_k]


def filter_available_tickers(df, tickers: list[str]) -> list[str]:
    return [ticker for ticker in tickers if ticker in df.columns]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for fine-tuned Chronos-2 models.")
    parser.add_argument("--general-model-path", type=str, help="Path to the general fine-tuned model checkpoint.")
    parser.add_argument(
        "--category-models-json",
        type=str,
        help="Path to JSON mapping category name -> fine-tuned model checkpoint.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/hyperparameter_tuning")
    parser.add_argument("--grid-json", type=str, help="Path to JSON grid file for hyperparameters.")
    parser.add_argument("--context-length", type=int, default=200)
    parser.add_argument("--prediction-length", type=int, default=1)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--min-past", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=1200)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--tune-general", action="store_true")
    parser.add_argument("--tune-categories", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    if not args.tune_general and not args.tune_categories:
        raise ValueError("Select at least one tuning target: --tune-general and/or --tune-categories")

    set_seeds(args.seed)

    grid = load_grid(args.grid_json)
    configs = build_config_grid(grid)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    df_all = get_daily_returns_data_cached()
    df_train_clean, _ = prepare_data_for_chronos(df_all, test_size=args.test_size)
    df_train_split, df_val_split = create_train_val_split(df_train_clean, val_ratio=args.val_ratio)

    summary: dict[str, Any] = {"general": [], "categories": {}}

    if args.tune_general:
        if not args.general_model_path:
            raise ValueError("--general-model-path is required when --tune-general is set")

        train_inputs, val_inputs = build_inputs(
            df_train_split,
            df_val_split,
            context_length=args.context_length,
            prediction_length=args.prediction_length,
            stride=args.stride,
        )

        general_results = tune_single_model(
            model_path=args.general_model_path,
            train_inputs=train_inputs,
            val_inputs=val_inputs,
            df_val=df_val_split,
            configs=configs,
            prediction_length=args.prediction_length,
            context_length=args.context_length,
            min_past=args.min_past,
            output_dir=output_root / "general",
            n_samples=args.n_samples,
            seed=args.seed,
        )

        summary["general"] = select_top_configs(general_results, args.top_k)

    if args.tune_categories:
        category_models = load_category_models(args.category_models_json)
        if not category_models:
            raise ValueError("--category-models-json is required when --tune-categories is set")

        for category, tickers in GICS_LEVEL_1.items():
            model_path = category_models.get(category)
            if not model_path:
                LOGGER.warning("Skipping category '%s' (missing model path)", category)
                continue

            available_tickers = filter_available_tickers(df_train_split, list(tickers))
            if not available_tickers:
                LOGGER.warning("Skipping category '%s' (no available tickers)", category)
                continue

            train_df = df_train_split[available_tickers]
            val_df = df_val_split[available_tickers]

            train_inputs, val_inputs = build_inputs(
                train_df,
                val_df,
                context_length=args.context_length,
                prediction_length=args.prediction_length,
                stride=args.stride,
            )

            category_results = tune_single_model(
                model_path=model_path,
                train_inputs=train_inputs,
                val_inputs=val_inputs,
                df_val=val_df,
                configs=configs,
                prediction_length=args.prediction_length,
                context_length=args.context_length,
                min_past=args.min_past,
                output_dir=output_root / "categories" / category.replace(" ", "_"),
                n_samples=args.n_samples,
                seed=args.seed,
            )

            summary["categories"][category] = select_top_configs(category_results, args.top_k)

    summary_path = output_root / "top_configs.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Saved top configs to %s", summary_path)


if __name__ == "__main__":
    main()

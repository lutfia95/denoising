from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import AutoGluonTrainingConfig
from src.training.logging_utils import tee_output
from src.training.train_mlp import (
    MLPSpectrumDataset,
    build_confusion_summary,
    compute_metrics,
    fit_feature_normalizer,
    save_confusion_outputs,
)


METADATA_COLUMNS = ("spectrum_index", "peak_index")


def train_autogluon(config: AutoGluonTrainingConfig) -> dict[str, object]:
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        raise ImportError(
            "AutoGluon is not installed. Install `autogluon.tabular` in the training "
            "environment before running the AutoGluon baseline."
        ) from exc

    output_dir = Path(config.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / config.output.log_file_name

    with tee_output(log_path, enabled=config.output.enable_file_logging):
        print(f"[LOG] Writing AutoGluon output to: {log_path}")
        print("[AG] Loading parquet splits...")

        train_df = pd.read_parquet(config.data.train_path)
        val_df = pd.read_parquet(config.data.val_path)
        test_df = pd.read_parquet(config.data.test_path)

        print(
            "[AG] Loaded splits: "
            f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)} spectra"
        )

        training_compatible_config = _make_training_compatible_config(config)
        normalizer = fit_feature_normalizer(train_df, training_compatible_config)

        train_dataset = MLPSpectrumDataset(
            train_df,
            training_compatible_config,
            normalizer,
            split_name="train",
        )
        val_dataset = MLPSpectrumDataset(
            val_df,
            training_compatible_config,
            normalizer,
            split_name="val",
        )
        test_dataset = MLPSpectrumDataset(
            test_df,
            training_compatible_config,
            normalizer,
            split_name="test",
        )

        print("[AG] Flattening per-spectrum samples into peak-level rows...")
        train_table = build_peak_table(train_dataset, config)
        val_table = build_peak_table(val_dataset, config)
        test_table = build_peak_table(test_dataset, config)
        print(
            "[AG] Flattened peaks: "
            f"train={len(train_table)}, val={len(val_table)}, test={len(test_table)}"
        )

        if config.output.save_flattened_tables:
            _save_table(train_table, output_dir / "train_flattened.parquet")
            _save_table(val_table, output_dir / "val_flattened.parquet")
            _save_table(test_table, output_dir / "test_flattened.parquet")

        label = config.autogluon.label
        sample_weight_column = _resolve_sample_weight_column(config)

        train_model_df = _model_input_table(train_table, config)
        val_model_df = _model_input_table(val_table, config)
        test_model_df = _model_input_table(test_table, config)

        predictor_path = _resolve_predictor_path(output_dir, config)
        predictor_kwargs: dict[str, Any] = {
            "label": label,
            "problem_type": config.autogluon.problem_type,
            "eval_metric": config.autogluon.eval_metric,
            "path": str(predictor_path),
            "verbosity": config.autogluon.verbosity,
        }
        if sample_weight_column is not None:
            predictor_kwargs["sample_weight"] = sample_weight_column
            predictor_kwargs["weight_evaluation"] = config.autogluon.weight_evaluation
        if config.autogluon.positive_class is not None:
            predictor_kwargs["positive_class"] = config.autogluon.positive_class

        print(
            "[AG] Fitting TabularPredictor with "
            f"presets={config.autogluon.presets!r}, "
            f"time_limit={config.autogluon.time_limit}, "
            f"predictor_path={predictor_path}"
        )
        predictor = TabularPredictor(**predictor_kwargs).fit(
            train_data=train_model_df,
            tuning_data=val_model_df,
            presets=config.autogluon.presets,
            time_limit=config.autogluon.time_limit,
            dynamic_stacking=config.autogluon.dynamic_stacking,
            num_stack_levels=config.autogluon.num_stack_levels,
            use_bag_holdout=config.autogluon.use_bag_holdout,
            fit_weighted_ensemble=config.autogluon.fit_weighted_ensemble,
            save_bag_folds=config.autogluon.save_bag_folds,
        )

        if config.output.save_fit_summary:
            fit_summary = predictor.fit_summary(verbosity=0)
            with (output_dir / "fit_summary.json").open("w", encoding="utf-8") as handle:
                json.dump(_to_jsonable_obj(fit_summary), handle, indent=2)

        if config.autogluon.save_leaderboard:
            predictor.leaderboard(val_model_df, silent=True).to_csv(
                output_dir / "leaderboard_val.csv",
                index=False,
            )
            predictor.leaderboard(test_model_df, silent=True).to_csv(
                output_dir / "leaderboard_test.csv",
                index=False,
            )

        if config.autogluon.save_feature_importance:
            fi_kwargs: dict[str, Any] = {}
            if config.autogluon.feature_importance_subsample_size is not None:
                fi_kwargs["subsample_size"] = (
                    config.autogluon.feature_importance_subsample_size
                )
            if config.autogluon.feature_importance_num_shuffle_sets is not None:
                fi_kwargs["num_shuffle_sets"] = (
                    config.autogluon.feature_importance_num_shuffle_sets
                )
            if config.autogluon.feature_importance_time_limit is not None:
                fi_kwargs["time_limit"] = config.autogluon.feature_importance_time_limit

            predictor.feature_importance(val_model_df, **fi_kwargs).to_csv(
                output_dir / "feature_importance_val.csv"
            )

        print("[AG] Evaluating validation and test splits...")
        val_eval = evaluate_table(
            predictor=predictor,
            flat_table=val_table,
            model_input_table=val_model_df,
            config=config,
        )
        test_eval = evaluate_table(
            predictor=predictor,
            flat_table=test_table,
            model_input_table=test_model_df,
            config=config,
        )

        threshold = float(config.evaluation.threshold_for_binary_metrics)
        val_conf = build_confusion_summary(val_eval["probs"], val_eval["targets"], threshold)
        test_conf = build_confusion_summary(
            test_eval["probs"], test_eval["targets"], threshold
        )

        if config.output.save_predictions:
            _save_table(val_eval["predictions"], output_dir / "val_predictions.parquet")
            _save_table(test_eval["predictions"], output_dir / "test_predictions.parquet")

        if config.output.save_confusion_matrices:
            save_confusion_outputs(output_dir, "val", val_conf)
            save_confusion_outputs(output_dir, "test", test_conf)

        if config.output.save_metrics_summary:
            summary = {
                "autogluon": asdict(config.autogluon),
                "flattened_rows": {
                    "train": int(len(train_table)),
                    "val": int(len(val_table)),
                    "test": int(len(test_table)),
                },
                "label_column": label,
                "predictor_path": str(predictor_path),
                "best_model": str(predictor.model_best),
                "val_metrics": _to_jsonable_obj(val_eval["metrics"]),
                "test_metrics": _to_jsonable_obj(test_eval["metrics"]),
                "val_confusion_matrix": val_conf,
                "test_confusion_matrix": test_conf,
                "threshold_for_binary_metrics": threshold,
            }
            with (output_dir / "metrics_summary.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(_to_jsonable_obj(summary), handle, indent=2)

        return {
            "predictor": predictor,
            "output_dir": output_dir,
            "val_metrics": val_eval["metrics"],
            "test_metrics": test_eval["metrics"],
            "val_confusion_matrix": val_conf,
            "test_confusion_matrix": test_conf,
        }


def build_peak_table(
    dataset: MLPSpectrumDataset,
    config: AutoGluonTrainingConfig,
) -> pd.DataFrame:
    peak_feature_names = list(config.features.peak_feature_columns)
    if config.features.use_raw_peak_mz:
        peak_feature_names.append("raw_peak_mz")
    if config.features.use_raw_peak_intensity:
        peak_feature_names.append("raw_peak_intensity")

    rows: list[pd.DataFrame] = []
    label = config.autogluon.label
    sample_weight_column = _resolve_sample_weight_column(config)

    for idx in range(len(dataset)):
        sample = dataset[idx]
        peak_features = np.asarray(sample["peak_features"], dtype=np.float32)
        spectrum_features = np.asarray(sample["spectrum_features"], dtype=np.float32)
        targets = np.asarray(sample["targets"], dtype=np.float32).astype(np.int32, copy=False)
        weight = float(sample["weight"])
        spectrum_index = int(sample["row_index"])

        n_peaks = int(peak_features.shape[0])
        table_dict: dict[str, np.ndarray] = {
            "spectrum_index": np.full(n_peaks, spectrum_index, dtype=np.int64),
            "peak_index": np.arange(n_peaks, dtype=np.int64),
            label: targets,
        }

        for col_idx, name in enumerate(peak_feature_names):
            table_dict[name] = peak_features[:, col_idx]

        for col_idx, name in enumerate(config.features.spectrum_feature_columns):
            table_dict[name] = np.full(n_peaks, spectrum_features[col_idx], dtype=np.float32)

        if sample_weight_column is not None:
            table_dict[sample_weight_column] = np.full(
                n_peaks,
                weight,
                dtype=np.float32,
            )

        rows.append(pd.DataFrame(table_dict))

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, axis=0, ignore_index=True)


def evaluate_table(
    predictor: Any,
    flat_table: pd.DataFrame,
    model_input_table: pd.DataFrame,
    config: AutoGluonTrainingConfig,
) -> dict[str, object]:
    probs = _extract_positive_probs(predictor.predict_proba(model_input_table), predictor, config)
    targets = flat_table[config.autogluon.label].to_numpy(dtype=np.int32, copy=False)
    spectrum_indices = flat_table["spectrum_index"].to_numpy(dtype=np.int64, copy=False)
    pred_labels = (probs >= config.evaluation.threshold_for_binary_metrics).astype(np.int32)

    metrics = compute_metrics(
        probs=probs,
        targets=targets,
        spectrum_indices=spectrum_indices,
        config=_make_training_compatible_config(config),
    )

    predictions = flat_table[list(METADATA_COLUMNS)].copy()
    predictions["target"] = targets
    predictions["prob_signal"] = probs
    predictions["pred_label"] = pred_labels

    return {
        "metrics": metrics,
        "probs": probs,
        "targets": targets,
        "spectrum_indices": spectrum_indices,
        "predictions": predictions,
    }


def _resolve_sample_weight_column(
    config: AutoGluonTrainingConfig,
) -> str | None:
    if not config.data.use_training_weights:
        return None
    if config.autogluon.sample_weight_column is not None:
        return config.autogluon.sample_weight_column
    return config.data.weight_column


def _model_input_table(
    flat_table: pd.DataFrame,
    config: AutoGluonTrainingConfig,
) -> pd.DataFrame:
    drop_columns = list(METADATA_COLUMNS)
    return flat_table.drop(columns=drop_columns)


def _extract_positive_probs(
    pred_proba: Any,
    predictor: Any,
    config: AutoGluonTrainingConfig,
) -> np.ndarray:
    if isinstance(pred_proba, pd.Series):
        return pred_proba.to_numpy(dtype=np.float64, copy=False)

    if not isinstance(pred_proba, pd.DataFrame):
        return np.asarray(pred_proba, dtype=np.float64)

    positive_class = config.autogluon.positive_class
    if positive_class is None:
        positive_class = getattr(predictor, "positive_class", None)

    if positive_class in pred_proba.columns:
        return pred_proba[positive_class].to_numpy(dtype=np.float64, copy=False)

    if 1 in pred_proba.columns:
        return pred_proba[1].to_numpy(dtype=np.float64, copy=False)

    return pred_proba.iloc[:, -1].to_numpy(dtype=np.float64, copy=False)


def _make_training_compatible_config(
    config: AutoGluonTrainingConfig,
):
    class _Training:
        cache_dataset_in_memory = False

    class _Config:
        pass

    out = _Config()
    out.data = config.data
    out.features = config.features
    out.training = _Training()
    out.evaluation = config.evaluation
    return out


def _save_table(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
    except Exception:
        fallback = path.with_suffix(".csv")
        df.to_csv(fallback, index=False)


def _resolve_predictor_path(
    output_dir: Path,
    config: AutoGluonTrainingConfig,
) -> Path:
    base_name = config.output.predictor_subdir or "autogluon_predictor"
    if config.output.unique_predictor_subdir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return output_dir / f"{base_name}_{timestamp}"
    return output_dir / base_name


def _to_jsonable_obj(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable_obj(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable_obj(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if not np.isfinite(obj):
            return None
        return float(obj)
    if isinstance(obj, float):
        if not np.isfinite(obj):
            return None
        return obj
    return obj

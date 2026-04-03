from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os
import platform
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from src.config import TrainingConfig
from src.model.mlp import MLPPeakClassifier
from src.training.logging_utils import tee_output


def print_device_info(device: torch.device) -> dict[str, object]:
    info: dict[str, object] = {
        "device_type": device.type,
        "device_str": str(device),
    }

    print("\n=== DEVICE INFO ===")
    print(f"Selected device: {device}")

    if device.type == "cuda":
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()

        props = torch.cuda.get_device_properties(device_index)

        total_memory_gb = props.total_memory / (1024 ** 3)
        allocated_gb = torch.cuda.memory_allocated(device_index) / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved(device_index) / (1024 ** 3)

        info.update(
            {
                "device_name": props.name,
                "device_index": int(device_index),
                "cuda_capability": f"{props.major}.{props.minor}",
                "total_memory_gb": float(total_memory_gb),
                "allocated_memory_gb": float(allocated_gb),
                "reserved_memory_gb": float(reserved_gb),
                "multi_processor_count": int(props.multi_processor_count),
            }
        )

        print(f"Device name          : {props.name}")
        print(f"Device index         : {device_index}")
        print(f"CUDA capability      : {props.major}.{props.minor}")
        print(f"Total GPU memory     : {total_memory_gb:.2f} GB")
        print(f"Allocated GPU memory : {allocated_gb:.2f} GB")
        print(f"Reserved GPU memory  : {reserved_gb:.2f} GB")
        print(f"SM count             : {props.multi_processor_count}")

    elif device.type == "cpu":
        info.update(
            {
                "processor": platform.processor(),
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
                "torch_num_threads": torch.get_num_threads(),
            }
        )

        print(f"Processor            : {platform.processor()}")
        print(f"Platform             : {platform.platform()}")
        print(f"CPU count            : {os.cpu_count()}")
        print(f"Torch threads        : {torch.get_num_threads()}")

        try:
            import psutil

            vm = psutil.virtual_memory()
            total_ram_gb = vm.total / (1024 ** 3)
            available_ram_gb = vm.available / (1024 ** 3)

            info["total_ram_gb"] = float(total_ram_gb)
            info["available_ram_gb"] = float(available_ram_gb)

            print(f"Total RAM            : {total_ram_gb:.2f} GB")
            print(f"Available RAM        : {available_ram_gb:.2f} GB")
        except ImportError:
            print("RAM info             : psutil not installed")

    elif device.type == "mps":
        info.update(
            {
                "processor": platform.processor(),
                "platform": platform.platform(),
            }
        )

        print("Apple Metal (MPS) device selected")
        print(f"Processor            : {platform.processor()}")
        print(f"Platform             : {platform.platform()}")

    else:
        print(f"Unknown device type  : {device.type}")

    print("===================\n")
    return info

@dataclass(slots=True)
class FeatureNormalizer:
    peak_mean: np.ndarray
    peak_std: np.ndarray
    spectrum_mean: np.ndarray
    spectrum_std: np.ndarray

    def normalize_peak_features(self, x: np.ndarray, enabled: bool) -> np.ndarray:
        if not enabled:
            return x.astype(np.float32, copy=False)
        return ((x - self.peak_mean) / self.peak_std).astype(np.float32, copy=False)

    def normalize_spectrum_features(self, x: np.ndarray, enabled: bool) -> np.ndarray:
        if not enabled:
            return x.astype(np.float32, copy=False)
        return ((x - self.spectrum_mean) / self.spectrum_std).astype(np.float32, copy=False)


class MLPSpectrumDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        config: TrainingConfig,
        normalizer: FeatureNormalizer,
        split_name: str,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.config = config
        self.normalizer = normalizer
        self.split_name = split_name

    def __len__(self) -> int:
        return int(len(self.df))

    def __getitem__(self, idx: int) -> dict[str, np.ndarray | float | int]:
        row = self.df.iloc[idx]

        peak_features = self._build_peak_feature_matrix(row)
        spectrum_features = self._build_spectrum_feature_vector(row)
        targets = self._build_target_vector(row)

        n_peaks = peak_features.shape[0]
        if targets.shape[0] != n_peaks:
            raise ValueError(
                f"Target length {targets.shape[0]} does not match number of peaks {n_peaks}"
            )

        weight = 1.0
        if self.config.data.use_training_weights:
            weight = float(row[self.config.data.weight_column])

        return {
            "peak_features": peak_features,
            "spectrum_features": spectrum_features,
            "targets": targets,
            "weight": weight,
            "row_index": int(idx),
        }

    def _build_peak_feature_matrix(self, row: pd.Series) -> np.ndarray:
        cols: list[np.ndarray] = []

        for col_name in self.config.features.peak_feature_columns:
            arr = np.asarray(row[col_name], dtype=np.float32)
            if arr.ndim != 1:
                raise ValueError(f"Peak feature column {col_name!r} must be 1D")
            cols.append(arr)

        raw_peak_cols = self._build_optional_raw_peak_columns(row)
        cols.extend(raw_peak_cols)

        if not cols:
            raise ValueError("No peak feature columns were configured")

        peak_matrix = np.stack(cols, axis=1).astype(np.float32, copy=False)
        peak_matrix = self.normalizer.normalize_peak_features(
            peak_matrix,
            enabled=self.config.features.normalize_peak_features,
        )
        return peak_matrix

    def _build_optional_raw_peak_columns(self, row: pd.Series) -> list[np.ndarray]:
        cols: list[np.ndarray] = []
        if not (
            self.config.features.use_raw_peak_mz
            or self.config.features.use_raw_peak_intensity
        ):
            return cols

        raw_mz = np.asarray(row[self.config.features.raw_peak_mz_column], dtype=np.float32)
        raw_intensity = np.asarray(
            row[self.config.features.raw_peak_intensity_column], dtype=np.float32
        )
        if raw_mz.ndim != 1:
            raise ValueError(
                f"Raw peak m/z column {self.config.features.raw_peak_mz_column!r} must be 1D"
            )
        if raw_intensity.ndim != 1:
            raise ValueError(
                "Raw peak intensity column "
                f"{self.config.features.raw_peak_intensity_column!r} must be 1D"
            )
        if raw_mz.shape[0] != raw_intensity.shape[0]:
            raise ValueError("Raw peak m/z and intensity arrays must have matching length")

        if self.config.features.sort_raw_peak_inputs_by_mz:
            order = np.argsort(raw_mz)
            raw_mz = raw_mz[order]
            raw_intensity = raw_intensity[order]

        if self.config.features.use_raw_peak_mz:
            cols.append(raw_mz.astype(np.float32, copy=False))
        if self.config.features.use_raw_peak_intensity:
            cols.append(raw_intensity.astype(np.float32, copy=False))
        return cols

    def _build_spectrum_feature_vector(self, row: pd.Series) -> np.ndarray:
        values = [
            float(row[col_name])
            for col_name in self.config.features.spectrum_feature_columns
        ]
        x = np.asarray(values, dtype=np.float32)
        x = self.normalizer.normalize_spectrum_features(
            x,
            enabled=self.config.features.normalize_spectrum_features,
        )
        return x

    def _build_target_vector(self, row: pd.Series) -> np.ndarray:
        targets = np.asarray(row[self.config.data.target_column], dtype=np.float32)
        if targets.ndim != 1:
            raise ValueError("Target column must contain 1D arrays")
        return targets.astype(np.float32, copy=False)


def mlp_collate_fn(batch: list[dict[str, np.ndarray | float | int]]) -> dict[str, Tensor]:
    peak_feature_rows: list[np.ndarray] = []
    spectrum_feature_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    weight_rows: list[np.ndarray] = []
    spectrum_index_rows: list[np.ndarray] = []

    for item in batch:
        peak_features = np.asarray(item["peak_features"], dtype=np.float32)
        spectrum_features = np.asarray(item["spectrum_features"], dtype=np.float32)
        targets = np.asarray(item["targets"], dtype=np.float32)
        weight = float(item["weight"])
        row_index = int(item["row_index"])

        n_peaks = peak_features.shape[0]
        repeated_spectrum = np.repeat(spectrum_features[None, :], repeats=n_peaks, axis=0)
        repeated_weight = np.full(shape=(n_peaks,), fill_value=weight, dtype=np.float32)
        repeated_row_index = np.full(shape=(n_peaks,), fill_value=row_index, dtype=np.int64)

        peak_feature_rows.append(peak_features)
        spectrum_feature_rows.append(repeated_spectrum)
        target_rows.append(targets)
        weight_rows.append(repeated_weight)
        spectrum_index_rows.append(repeated_row_index)

    return {
        "peak_features": torch.from_numpy(np.concatenate(peak_feature_rows, axis=0)),
        "spectrum_features": torch.from_numpy(np.concatenate(spectrum_feature_rows, axis=0)),
        "targets": torch.from_numpy(np.concatenate(target_rows, axis=0)),
        "weights": torch.from_numpy(np.concatenate(weight_rows, axis=0)),
        "spectrum_indices": torch.from_numpy(np.concatenate(spectrum_index_rows, axis=0)),
    }


def train_mlp(
    config: TrainingConfig,
    model: MLPPeakClassifier,
    device: str | torch.device | None = None,
) -> dict[str, object]:
    _set_seed(config.training.seed)

    output_dir = Path(config.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / config.output.log_file_name
    with tee_output(log_path, enabled=config.output.enable_file_logging):
        print(f"[LOG] Writing training output to: {log_path}")
        device_verbose = config.output.device_verbose
        used_device = _resolve_device(device)
        device_info = print_device_info(used_device)
        if device_verbose:
            print(device_info)

        train_df = pd.read_parquet(config.data.train_path)
        val_df = pd.read_parquet(config.data.val_path)
        test_df = pd.read_parquet(config.data.test_path)

        normalizer = fit_feature_normalizer(train_df, config)

        train_dataset = MLPSpectrumDataset(
            df=train_df,
            config=config,
            normalizer=normalizer,
            split_name="train",
        )
        val_dataset = MLPSpectrumDataset(
            df=val_df,
            config=config,
            normalizer=normalizer,
            split_name="val",
        )
        test_dataset = MLPSpectrumDataset(
            df=test_df,
            config=config,
            normalizer=normalizer,
            split_name="test",
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.num_workers,
            collate_fn=mlp_collate_fn,
        )
        print("[DEV] Finished train loader!")
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            collate_fn=mlp_collate_fn,
        )
        print("[DEV] Finished val loader!")
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            collate_fn=mlp_collate_fn,
        )
        print("[DEV] Finished test loader!")
        model = model.to(used_device)

        optimizer = _build_optimizer(model, config)
        pos_weight = (
            compute_pos_weight(train_df, config) if config.loss.use_pos_weight else None
        )
        criterion = _build_loss(config, used_device, pos_weight=pos_weight)

        history: list[dict[str, float]] = []
        best_state_dict: dict[str, Tensor] | None = None
        best_epoch = -1
        best_score = -np.inf if config.early_stopping.mode == "max" else np.inf
        bad_epochs = 0

        monitor_name = config.early_stopping.monitor.removeprefix("val_")
        print(f"[DEV] Monitor Name: {monitor_name}")

        for epoch in range(1, config.training.max_epochs + 1):
            train_metrics = run_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                config=config,
                device=used_device,
                training=True,
            )
            val_metrics = run_one_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                criterion=criterion,
                config=config,
                device=used_device,
                training=False,
            )

            epoch_record = {"epoch": float(epoch)}
            for key, value in train_metrics.items():
                epoch_record[f"train_{key}"] = float(value)
            for key, value in val_metrics.items():
                epoch_record[f"val_{key}"] = float(value)
            history.append(epoch_record)

            monitor_value = float(val_metrics[monitor_name])
            improved = _is_improved(
                current=monitor_value,
                best=best_score,
                mode=config.early_stopping.mode,
                min_delta=config.early_stopping.min_delta,
            )

            if improved:
                best_score = monitor_value
                best_epoch = epoch
                best_state_dict = _state_dict_to_cpu(deepcopy(model.state_dict()))
                bad_epochs = 0
            else:
                bad_epochs += 1

            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_metrics['loss']:.6f} | "
                f"val_loss={val_metrics['loss']:.6f} | "
                f"val_pr_auc={val_metrics.get('pr_auc', np.nan):.6f}"
            )

            if (
                config.output.save_epoch_history
                and history
                and (epoch == 1 or epoch % 5 == 0)
            ):
                pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

            if (
                config.early_stopping.enabled
                and bad_epochs >= config.early_stopping.patience
            ):
                print(f"Early stopping at epoch {epoch}. Best epoch was {best_epoch}.")
                break

        if best_state_dict is None:
            best_state_dict = _state_dict_to_cpu(deepcopy(model.state_dict()))
            best_epoch = len(history)

        model.load_state_dict(best_state_dict)
        model = model.to(used_device)

        final_val_eval = evaluate_loader(
            model=model,
            loader=val_loader,
            criterion=criterion,
            config=config,
            device=used_device,
        )
        final_test_eval = evaluate_loader(
            model=model,
            loader=test_loader,
            criterion=criterion,
            config=config,
            device=used_device,
        )

        history_df = pd.DataFrame(history)

        val_conf = build_confusion_summary(
            probs=final_val_eval["probs"],
            targets=final_val_eval["targets"],
            threshold=config.evaluation.threshold_for_binary_metrics,
        )
        test_conf = build_confusion_summary(
            probs=final_test_eval["probs"],
            targets=final_test_eval["targets"],
            threshold=config.evaluation.threshold_for_binary_metrics,
        )

        if config.output.save_best_model:
            save_best_checkpoint(
                output_dir=output_dir,
                model_state_dict=best_state_dict,
                config=config,
                normalizer=normalizer,
                best_epoch=best_epoch,
                best_score=float(best_score),
                pos_weight=pos_weight,
            )

        if config.output.save_epoch_history:
            history_df.to_csv(output_dir / "history.csv", index=False)

        if config.output.save_metrics_summary:
            metrics_summary = {
                "best_epoch": int(best_epoch),
                "best_monitor_value": float(best_score),
                "monitor_name": config.early_stopping.monitor,
                "val_metrics": _to_jsonable(final_val_eval["metrics"]),
                "test_metrics": _to_jsonable(final_test_eval["metrics"]),
                "val_confusion_matrix": val_conf,
                "test_confusion_matrix": test_conf,
                "threshold_for_binary_metrics": float(
                    config.evaluation.threshold_for_binary_metrics
                ),
            }
            with (output_dir / "metrics_summary.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(metrics_summary, handle, indent=2)

        if config.output.save_plots:
            save_training_plots(history_df, output_dir)

        if config.output.save_confusion_matrices:
            save_confusion_outputs(
                output_dir=output_dir,
                name="val",
                confusion_summary=val_conf,
            )
            save_confusion_outputs(
                output_dir=output_dir,
                name="test",
                confusion_summary=test_conf,
            )

        return {
            "model": model,
            "device": str(used_device),
            "device_info": device_info,
            "history": history_df,
            "best_epoch": best_epoch,
            "best_monitor_value": float(best_score),
            "final_val_metrics": final_val_eval["metrics"],
            "final_test_metrics": final_test_eval["metrics"],
            "val_confusion_matrix": val_conf,
            "test_confusion_matrix": test_conf,
            "normalizer": normalizer,
            "pos_weight": None if pos_weight is None else float(pos_weight),
            "output_dir": output_dir,
        }


def run_one_epoch(
    model: MLPPeakClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    config: TrainingConfig,
    device: torch.device,
    training: bool,
) -> dict[str, float]:
    result = _run_loader( #put model in training mode
        model=model,
        loader=loader,
        optimizer=optimizer,
        criterion=criterion,
        config=config,
        device=device,
        training=training,
    )
    return result["metrics"]


def evaluate_loader(
    model: MLPPeakClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, object]:
    return _run_loader(
        model=model,
        loader=loader,
        optimizer=None,
        criterion=criterion,
        config=config,
        device=device,
        training=False,
    )


def _run_loader(
    model: MLPPeakClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    config: TrainingConfig,
    device: torch.device,
    training: bool,
) -> dict[str, object]:
    if training:
        model.train()
    else:
        model.eval()

    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_spectrum_indices: list[np.ndarray] = []

    total_loss = 0.0
    total_peaks = 0

    for batch_idx, batch in enumerate(loader, start=1):
        peak_features = batch["peak_features"].to(device)
        spectrum_features = batch["spectrum_features"].to(device)
        targets = batch["targets"].to(device)
        weights = batch["weights"].to(device)

        with torch.set_grad_enabled(training):
            logits = model(
                peak_features=peak_features,
                spectrum_features=spectrum_features,
            )

            loss_vec = criterion(logits, targets)

            if config.data.use_training_weights:
                loss_vec = loss_vec * weights

            loss = loss_vec.mean()

            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                if config.training.gradient_clip_norm > 0.0:
                    nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=config.training.gradient_clip_norm,
                    )

                optimizer.step()

        batch_size_peaks = int(targets.numel())
        total_loss += float(loss.item()) * batch_size_peaks
        total_peaks += batch_size_peaks
        if batch_idx % 20 == 0 or batch_idx == 1 or batch_idx == len(loader):
            running_loss = total_loss / max(total_peaks, 1)

            msg = (
                f"[{'train' if training else 'eval'}] "
                f"batch {batch_idx}/{len(loader)} | "
                f"batch_loss={loss.item():.6f} | "
                f"running_loss={running_loss:.6f} | "
                f"n_peaks={targets.numel()}"
            )

            if device.type == "cuda":
                allocated_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
                reserved_gb = torch.cuda.memory_reserved(device) / (1024 ** 3)
                msg += (
                    f" | gpu_alloc={allocated_gb:.2f} GB"
                    f" | gpu_reserved={reserved_gb:.2f} GB"
                )

            print(msg)
    
        all_logits.append(logits.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())
        all_spectrum_indices.append(batch["spectrum_indices"].detach().cpu().numpy())

    logits_np = np.concatenate(all_logits, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)
    spectrum_indices_np = np.concatenate(all_spectrum_indices, axis=0)

    probs_np = sigmoid_numpy(logits_np)
    metrics = compute_metrics(
        probs=probs_np,
        targets=targets_np,
        spectrum_indices=spectrum_indices_np,
        config=config,
    )
    metrics["loss"] = total_loss / max(total_peaks, 1)

    return {
        "metrics": metrics,
        "probs": probs_np,
        "targets": targets_np,
        "spectrum_indices": spectrum_indices_np,
    }


def fit_feature_normalizer(
    train_df: pd.DataFrame,
    config: TrainingConfig,
) -> FeatureNormalizer:
    peak_rows: list[np.ndarray] = []
    print(f'[DEV-MLP] Features to normalise:\n {config.features.peak_feature_columns}')
    for col_name in config.features.peak_feature_columns:
        values = [np.asarray(v, dtype=np.float32) for v in train_df[col_name].tolist()]
        peak_rows.append(np.concatenate(values, axis=0))

    if config.features.use_raw_peak_mz or config.features.use_raw_peak_intensity:
        raw_mz_values = [
            np.asarray(v, dtype=np.float32)
            for v in train_df[config.features.raw_peak_mz_column].tolist()
        ]
        raw_intensity_values = [
            np.asarray(v, dtype=np.float32)
            for v in train_df[config.features.raw_peak_intensity_column].tolist()
        ]
        aligned_raw_mz: list[np.ndarray] = []
        aligned_raw_intensity: list[np.ndarray] = []
        for raw_mz, raw_intensity in zip(raw_mz_values, raw_intensity_values, strict=True):
            if config.features.sort_raw_peak_inputs_by_mz:
                order = np.argsort(raw_mz)
                raw_mz = raw_mz[order]
                raw_intensity = raw_intensity[order]
            aligned_raw_mz.append(raw_mz)
            aligned_raw_intensity.append(raw_intensity)

        if config.features.use_raw_peak_mz:
            peak_rows.append(np.concatenate(aligned_raw_mz, axis=0))
        if config.features.use_raw_peak_intensity:
            peak_rows.append(np.concatenate(aligned_raw_intensity, axis=0))

    if not peak_rows:
        raise ValueError("No peak feature columns configured")

    peak_matrix = np.stack(peak_rows, axis=1).astype(np.float64, copy=False)
    peak_mean = peak_matrix.mean(axis=0)
    peak_std = peak_matrix.std(axis=0)
    peak_std[peak_std == 0.0] = 1.0

    spectrum_matrix = train_df[
        config.features.spectrum_feature_columns
    ].to_numpy(dtype=np.float64)
    spectrum_mean = spectrum_matrix.mean(axis=0)
    spectrum_std = spectrum_matrix.std(axis=0)
    spectrum_std[spectrum_std == 0.0] = 1.0

    return FeatureNormalizer(
        peak_mean=peak_mean.astype(np.float32),
        peak_std=peak_std.astype(np.float32),
        spectrum_mean=spectrum_mean.astype(np.float32),
        spectrum_std=spectrum_std.astype(np.float32),
    )


def compute_pos_weight(train_df: pd.DataFrame, config: TrainingConfig) -> float:
    if config.loss.pos_weight is not None:
        return float(config.loss.pos_weight)

    all_targets = np.concatenate(
        [
            np.asarray(v, dtype=np.float32)
            for v in train_df[config.data.target_column].tolist()
        ],
        axis=0,
    )
    n_pos = float((all_targets == 1.0).sum())
    n_neg = float((all_targets == 0.0).sum())

    if n_pos <= 0.0:
        raise ValueError("No positive targets found in the training split")
    if n_neg <= 0.0:
        raise ValueError("No negative targets found in the training split")

    return n_neg / n_pos


def compute_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    spectrum_indices: np.ndarray,
    config: TrainingConfig,
) -> dict[str, float]:
    metrics: dict[str, float] = {}

    threshold = float(config.evaluation.threshold_for_binary_metrics)
    pred_labels = (probs >= threshold).astype(np.int32)
    true_labels = targets.astype(np.int32)

    if "pr_auc" in config.evaluation.report_metrics or config.evaluation.primary_metric == "pr_auc":
        metrics["pr_auc"] = safe_average_precision(true_labels, probs)

    if "roc_auc" in config.evaluation.report_metrics or config.evaluation.primary_metric == "roc_auc":
        metrics["roc_auc"] = safe_roc_auc(true_labels, probs)

    if "f1" in config.evaluation.report_metrics:
        metrics["f1"] = float(f1_score(true_labels, pred_labels, zero_division=0))

    if "mcc" in config.evaluation.report_metrics:
        metrics["mcc"] = safe_mcc(true_labels, pred_labels)

    if "precision" in config.evaluation.report_metrics:
        metrics["precision"] = float(
            precision_score(true_labels, pred_labels, zero_division=0)
        )

    if "recall" in config.evaluation.report_metrics:
        metrics["recall"] = float(
            recall_score(true_labels, pred_labels, zero_division=0)
        )

    for frac in config.evaluation.retained_peak_fractions:
        recall_value = compute_signal_recall_at_fraction(
            probs=probs,
            targets=true_labels,
            spectrum_indices=spectrum_indices,
            retained_fraction=float(frac),
        )
        metrics[f"signal_recall_at_{frac:.2f}"] = recall_value

    return metrics


def compute_signal_recall_at_fraction(
    probs: np.ndarray,
    targets: np.ndarray,
    spectrum_indices: np.ndarray,
    retained_fraction: float,
) -> float:
    total_true_kept = 0.0
    total_true = 0.0

    unique_spectra = np.unique(spectrum_indices)
    for spectrum_id in unique_spectra:
        mask = spectrum_indices == spectrum_id
        spectrum_probs = probs[mask]
        spectrum_targets = targets[mask]

        n_peaks = spectrum_probs.shape[0]
        n_keep = max(1, int(np.ceil(retained_fraction * n_peaks)))

        order = np.argsort(-spectrum_probs)
        keep_idx = order[:n_keep]

        total_true += float(spectrum_targets.sum())
        total_true_kept += float(spectrum_targets[keep_idx].sum())

    if total_true <= 0.0:
        return float("nan")

    return float(total_true_kept / total_true)


def build_confusion_summary(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    preds = (probs >= threshold).astype(np.int32)
    true_labels = targets.astype(np.int32)

    cm = confusion_matrix(true_labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "matrix": cm.astype(int).tolist(),
    }


def save_best_checkpoint(
    output_dir: Path,
    model_state_dict: dict[str, Tensor],
    config: TrainingConfig,
    normalizer: FeatureNormalizer,
    best_epoch: int,
    best_score: float,
    pos_weight: float | None,
) -> None:
    checkpoint = {
        "model_state_dict": model_state_dict,
        "model_config": asdict(config.model),
        "feature_config": {
            "peak_feature_columns": list(config.features.peak_feature_columns),
            "spectrum_feature_columns": list(config.features.spectrum_feature_columns),
            "use_raw_peak_mz": bool(config.features.use_raw_peak_mz),
            "raw_peak_mz_column": str(config.features.raw_peak_mz_column),
            "use_raw_peak_intensity": bool(config.features.use_raw_peak_intensity),
            "raw_peak_intensity_column": str(config.features.raw_peak_intensity_column),
            "sort_raw_peak_inputs_by_mz": bool(
                config.features.sort_raw_peak_inputs_by_mz
            ),
            "normalize_peak_features": bool(config.features.normalize_peak_features),
            "normalize_spectrum_features": bool(
                config.features.normalize_spectrum_features
            ),
        },
        "normalizer": {
            "peak_mean": normalizer.peak_mean,
            "peak_std": normalizer.peak_std,
            "spectrum_mean": normalizer.spectrum_mean,
            "spectrum_std": normalizer.spectrum_std,
        },
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "pos_weight": None if pos_weight is None else float(pos_weight),
        "threshold_for_binary_metrics": float(
            config.evaluation.threshold_for_binary_metrics
        ),
    }
    torch.save(checkpoint, output_dir / "best_model.pt")


def save_training_plots(history_df: pd.DataFrame, output_dir: Path) -> None:
    if history_df.empty:
        return

    plot_metric(
        history_df=history_df,
        train_col="train_loss",
        val_col="val_loss",
        ylabel="Loss",
        output_path=output_dir / "loss_curve.png",
    )

    if "train_pr_auc" in history_df.columns and "val_pr_auc" in history_df.columns:
        plot_metric(
            history_df=history_df,
            train_col="train_pr_auc",
            val_col="val_pr_auc",
            ylabel="PR-AUC",
            output_path=output_dir / "pr_auc_curve.png",
        )

    if "train_f1" in history_df.columns and "val_f1" in history_df.columns:
        plot_metric(
            history_df=history_df,
            train_col="train_f1",
            val_col="val_f1",
            ylabel="F1",
            output_path=output_dir / "f1_curve.png",
        )


def plot_metric(
    history_df: pd.DataFrame,
    train_col: str,
    val_col: str,
    ylabel: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df[train_col], label=train_col)
    plt.plot(history_df["epoch"], history_df[val_col], label=val_col)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} over epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_confusion_outputs(
    output_dir: Path,
    name: str,
    confusion_summary: dict[str, object],
) -> None:
    matrix = np.asarray(confusion_summary["matrix"], dtype=np.int64)

    cm_df = pd.DataFrame(
        matrix,
        index=["true_0", "true_1"],
        columns=["pred_0", "pred_1"],
    )
    cm_df.to_csv(output_dir / f"{name}_confusion_matrix.csv")

    plt.figure(figsize=(5, 4))
    plt.imshow(matrix)
    plt.xticks([0, 1], ["pred_0", "pred_1"])
    plt.yticks([0, 1], ["true_0", "true_1"])
    plt.title(f"{name} confusion matrix")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center")

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_confusion_matrix.png", dpi=150)
    plt.close()


def _build_optimizer(
    model: nn.Module,
    config: TrainingConfig,
) -> torch.optim.Optimizer:
    name = config.training.optimizer.name.lower()

    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.training.optimizer.learning_rate,
            weight_decay=config.training.optimizer.weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {config.training.optimizer.name}")


def _build_loss(
    config: TrainingConfig,
    device: torch.device,
    pos_weight: float | None,
) -> nn.Module:
    name = config.loss.name.lower()

    if name != "bce_with_logits":
        raise ValueError(f"Unsupported loss: {config.loss.name}")

    pos_weight_tensor = None
    if pos_weight is not None:
        pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)

    return nn.BCEWithLogitsLoss(
        reduction=config.loss.reduction,
        pos_weight=pos_weight_tensor,
    )


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _is_improved(
    current: float,
    best: float,
    mode: str,
    min_delta: float,
) -> bool:
    if mode == "max":
        return current > (best + min_delta)
    if mode == "min":
        return current < (best - min_delta)
    raise ValueError(f"Unsupported early stopping mode: {mode}")


def _state_dict_to_cpu(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    return {k: v.detach().cpu() for k, v in state_dict.items()}


def _to_jsonable(obj: dict[str, float]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key, value in obj.items():
        if isinstance(value, float) and not np.isfinite(value):
            out[key] = None
        else:
            out[key] = float(value)
    return out


def sigmoid_numpy(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return float("nan")


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def safe_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(matthews_corrcoef(y_true, y_pred))
    except ValueError:
        return float("nan")

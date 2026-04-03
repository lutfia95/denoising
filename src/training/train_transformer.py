from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.config import TrainingConfig
from src.model.transformer import PeakTransformerClassifier
from src.training.logging_utils import tee_output
from src.training.train_mlp import (
    MLPSpectrumDataset,
    _build_loss,
    _build_optimizer,
    _is_improved,
    _resolve_device,
    _set_seed,
    _to_jsonable,
    build_confusion_summary,
    compute_metrics,
    compute_pos_weight,
    fit_feature_normalizer,
    print_device_info,
    save_confusion_outputs,
    save_training_plots,
)


def transformer_collate_fn(
    batch: list[dict[str, np.ndarray | float | int]],
) -> dict[str, Tensor]:
    batch_size = len(batch)
    peak_lengths = [
        int(np.asarray(item["peak_features"], dtype=np.float32).shape[0]) for item in batch
    ]
    max_peaks = max(peak_lengths)

    first_peak = np.asarray(batch[0]["peak_features"], dtype=np.float32)
    first_spectrum = np.asarray(batch[0]["spectrum_features"], dtype=np.float32)
    peak_dim = int(first_peak.shape[1])
    spectrum_dim = int(first_spectrum.shape[0])

    # Pad each spectrum to the batch max so the transformer can process a dense tensor.
    peak_features = np.zeros((batch_size, max_peaks, peak_dim), dtype=np.float32)
    spectrum_features = np.zeros((batch_size, spectrum_dim), dtype=np.float32)
    targets = np.zeros((batch_size, max_peaks), dtype=np.float32)
    weights = np.ones((batch_size, max_peaks), dtype=np.float32)
    padding_mask = np.ones((batch_size, max_peaks), dtype=bool)
    spectrum_indices = np.full((batch_size, max_peaks), fill_value=-1, dtype=np.int64)

    for row_idx, item in enumerate(batch):
        peak_arr = np.asarray(item["peak_features"], dtype=np.float32)
        spectrum_arr = np.asarray(item["spectrum_features"], dtype=np.float32)
        target_arr = np.asarray(item["targets"], dtype=np.float32)
        weight = float(item["weight"])
        spectrum_index = int(item["row_index"])
        n_peaks = peak_arr.shape[0]

        peak_features[row_idx, :n_peaks] = peak_arr
        spectrum_features[row_idx] = spectrum_arr
        targets[row_idx, :n_peaks] = target_arr
        weights[row_idx, :n_peaks] = weight
        padding_mask[row_idx, :n_peaks] = False
        spectrum_indices[row_idx, :n_peaks] = spectrum_index

    return {
        "peak_features": torch.from_numpy(peak_features),
        "spectrum_features": torch.from_numpy(spectrum_features),
        "targets": torch.from_numpy(targets),
        "weights": torch.from_numpy(weights),
        "padding_mask": torch.from_numpy(padding_mask),
        "spectrum_indices": torch.from_numpy(spectrum_indices),
    }


def train_transformer(
    config: TrainingConfig,
    model: PeakTransformerClassifier,
    device: str | torch.device | None = None,
) -> dict[str, object]:
    _set_seed(config.training.seed)
    print(f'[DEV] Setting seed with {config.training.seed}')

    output_dir = Path(config.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / config.output.log_file_name
    with tee_output(log_path, enabled=config.output.enable_file_logging):
        print(f"[LOG] Writing training output to: {log_path}")
        used_device = _resolve_device(device)
        device_info = print_device_info(used_device)
        if config.output.device_verbose:
            print(device_info)
        print(f'[DEV] Reading training data...')
        train_df = pd.read_parquet(config.data.train_path)
        val_df = pd.read_parquet(config.data.val_path)
        test_df = pd.read_parquet(config.data.test_path)
        print(f'[DEV] Loaded train set with {len(train_df)} data points\n')
        print(f'[DEV] Loaded test set with {len(test_df)} data points\n')
        print(f'[DEV] Loaded val set with {len(val_df)} data points\n')
        print(f'[DEV] Normalizing peak features...\n')
        normalizer = fit_feature_normalizer(train_df, config)
        train_dataset = MLPSpectrumDataset(train_df, config, normalizer, split_name="train")
        val_dataset = MLPSpectrumDataset(val_df, config, normalizer, split_name="val")
        test_dataset = MLPSpectrumDataset(test_df, config, normalizer, split_name="test")

        loader_kwargs = _build_loader_kwargs(config.training.num_workers)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            collate_fn=transformer_collate_fn,
            **loader_kwargs,
        )
        print("[DEV] Finished train loader!")
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            collate_fn=transformer_collate_fn,
            **loader_kwargs,
        )
        print("[DEV] Finished val loader!")
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            collate_fn=transformer_collate_fn,
            **loader_kwargs,
        )
        print("[DEV] Finished test loader!")

        model = model.to(used_device)
        optimizer = _build_optimizer(model, config)
        print(f"[DEV] Finished building optimizer: {optimizer}")
        pos_weight = (
            compute_pos_weight(train_df, config) if config.loss.use_pos_weight else None
        )
        print(f"[DEV] Computed the positive weights only from training set: {pos_weight}")
        criterion = _build_loss(config, used_device, pos_weight=pos_weight)
        print(f"[DEV] Computed the positive weights only from training set: {pos_weight}")
        #exit()
        history: list[dict[str, float]] = []
        best_state_dict: dict[str, Tensor] | None = None
        best_epoch = -1
        best_score = -np.inf if config.early_stopping.mode == "max" else np.inf
        bad_epochs = 0

        monitor_name = config.early_stopping.monitor.removeprefix("val_")
        print(f"[DEV] Monitor Name: {monitor_name}")
        print(f"[DEV] Pos weight: {pos_weight}")

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
            _save_best_checkpoint(
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
    model: PeakTransformerClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    config: TrainingConfig,
    device: torch.device,
    training: bool,
) -> dict[str, float]:
    result = _run_loader(
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
    model: PeakTransformerClassifier,
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
    model: PeakTransformerClassifier,
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
    total_valid_peaks = 0

    for batch_idx, batch in enumerate(loader, start=1):
        peak_features = batch["peak_features"].to(device)
        spectrum_features = batch["spectrum_features"].to(device)
        targets = batch["targets"].to(device)
        weights = batch["weights"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        valid_mask = ~padding_mask

        with torch.set_grad_enabled(training):
            logits = model(
                peak_features=peak_features,
                spectrum_features=spectrum_features,
                padding_mask=padding_mask,
            )
            loss_matrix = criterion(logits, targets)

            if config.data.use_training_weights:
                loss_matrix = loss_matrix * weights

            # Ignore padded peak positions in both optimization and reported loss.
            loss_matrix = loss_matrix * valid_mask.to(loss_matrix.dtype)
            valid_count = int(valid_mask.sum().item())
            loss = loss_matrix.sum() / max(valid_count, 1)

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

        total_loss += float(loss.item()) * valid_count
        total_valid_peaks += valid_count

        if batch_idx % 20 == 0 or batch_idx == 1 or batch_idx == len(loader):
            running_loss = total_loss / max(total_valid_peaks, 1)
            msg = (
                f"[{'train' if training else 'eval'}] "
                f"batch {batch_idx}/{len(loader)} | "
                f"batch_loss={loss.item():.6f} | "
                f"running_loss={running_loss:.6f} | "
                f"n_peaks={valid_count}"
            )
            if device.type == "cuda":
                allocated_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
                reserved_gb = torch.cuda.memory_reserved(device) / (1024 ** 3)
                msg += (
                    f" | gpu_alloc={allocated_gb:.2f} GB"
                    f" | gpu_reserved={reserved_gb:.2f} GB"
                )
            print(msg)

        all_logits.append(logits[valid_mask].detach().cpu().numpy())
        all_targets.append(targets[valid_mask].detach().cpu().numpy())
        all_spectrum_indices.append(batch["spectrum_indices"][valid_mask.cpu()].numpy())

    logits_np = np.concatenate(all_logits, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)
    spectrum_indices_np = np.concatenate(all_spectrum_indices, axis=0)

    probs_np = _sigmoid_numpy(logits_np)
    metrics = compute_metrics(
        probs=probs_np,
        targets=targets_np,
        spectrum_indices=spectrum_indices_np,
        config=config,
    )
    metrics["loss"] = total_loss / max(total_valid_peaks, 1)
    return {
        "metrics": metrics,
        "probs": probs_np,
        "targets": targets_np,
        "spectrum_indices": spectrum_indices_np,
    }


def _build_loader_kwargs(num_workers: int) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return kwargs


def _state_dict_to_cpu(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    return {key: value.detach().cpu() for key, value in state_dict.items()}


def _save_best_checkpoint(
    output_dir: Path,
    model_state_dict: dict[str, Tensor],
    config: TrainingConfig,
    normalizer: object,
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


def _sigmoid_numpy(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

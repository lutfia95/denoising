from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Sampler

from src.config import TrainingConfig
from src.model.transformer_imp import PeakTransformerImpClassifier
from src.training.logging_utils import tee_output
from src.training.train_mlp import (
    MLPSpectrumDataset,
    _autocast_context,
    _build_grad_scaler,
    _build_loader_kwargs,
    _build_loss,
    _build_optimizer,
    _configure_runtime,
    _is_improved,
    _maybe_compile_model,
    _resolve_device,
    _set_seed,
    _to_jsonable,
    _use_non_blocking_transfers,
    build_confusion_summary,
    compute_metrics,
    compute_pos_weight,
    fit_feature_normalizer,
    print_device_info,
    save_confusion_outputs,
    save_training_plots,
)
from src.training.train_transformer import (
    _resolve_resume_checkpoint,
    _save_best_checkpoint,
    _sigmoid_numpy,
    _state_dict_to_cpu,
    transformer_collate_fn,
)


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Batch spectra with similar peak counts together to reduce padding waste."""

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        shuffle: bool,
        seed: int,
        bucket_size_multiplier: int = 32,
    ) -> None:
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.bucket_size = max(self.batch_size, self.batch_size * bucket_size_multiplier)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(indices)

        batches: list[list[int]] = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            bucket.sort(key=lambda idx: (self.lengths[idx], idx))
            for batch_start in range(0, len(bucket), self.batch_size):
                batch = bucket[batch_start : batch_start + self.batch_size]
                if batch:
                    batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)

        yield from batches

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size)


def build_length_bucketed_loader(
    dataset: MLPSpectrumDataset,
    config: TrainingConfig,
    device: torch.device,
    shuffle: bool,
) -> tuple[DataLoader, LengthBucketBatchSampler]:
    lengths = dataset.df["num_peaks"].astype(int).tolist()
    batch_sampler = LengthBucketBatchSampler(
        lengths=lengths,
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        seed=config.training.seed,
    )
    loader_kwargs = _build_loader_kwargs(config, device)
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=transformer_collate_fn,
        **loader_kwargs,
    )
    return loader, batch_sampler


class WarmupCosineScheduler:
    """Simple per-step warmup + cosine decay scheduler for transformer training."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        total_steps: int,
        warmup_steps: int,
        min_lr_scale: float = 0.05,
    ) -> None:
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.total_steps = max(int(total_steps), 1)
        self.warmup_steps = max(int(warmup_steps), 0)
        self.min_lr_scale = float(min_lr_scale)
        self.step_count = 0
        self._set_lr(0.0 if self.warmup_steps > 0 else self.base_lr)

    def state_dict(self) -> dict[str, float | int]:
        return {
            "step_count": int(self.step_count),
            "base_lr": float(self.base_lr),
            "total_steps": int(self.total_steps),
            "warmup_steps": int(self.warmup_steps),
            "min_lr_scale": float(self.min_lr_scale),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.step_count = int(state.get("step_count", 0))
        self._set_lr(self.get_lr_for_step(self.step_count))

    def get_lr_for_step(self, step_count: int) -> float:
        if self.warmup_steps > 0 and step_count < self.warmup_steps:
            return self.base_lr * float(step_count + 1) / float(self.warmup_steps)

        if self.total_steps <= self.warmup_steps:
            return self.base_lr

        progress = float(step_count - self.warmup_steps) / float(
            self.total_steps - self.warmup_steps
        )
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_lr = self.base_lr * self.min_lr_scale
        return min_lr + (self.base_lr - min_lr) * cosine

    def step(self) -> float:
        lr = self.get_lr_for_step(self.step_count)
        self._set_lr(lr)
        self.step_count += 1
        return lr

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr


def train_transformer_imp(
    config: TrainingConfig,
    model: PeakTransformerImpClassifier,
    device: str | torch.device | None = None,
) -> dict[str, object]:
    _set_seed(config.training.seed)

    output_dir = Path(config.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / config.output.log_file_name
    latest_checkpoint_path = output_dir / "last_checkpoint.pt"

    with tee_output(log_path, enabled=config.output.enable_file_logging):
        print(f"[LOG] Writing training output to: {log_path}")
        _configure_runtime(config)
        used_device = _resolve_device(device)
        device_info = print_device_info(used_device)
        if config.output.device_verbose:
            print(device_info)

        train_df = pd.read_parquet(config.data.train_path)
        val_df = pd.read_parquet(config.data.val_path)
        test_df = pd.read_parquet(config.data.test_path)

        normalizer = fit_feature_normalizer(train_df, config)
        train_dataset = MLPSpectrumDataset(train_df, config, normalizer, split_name="train")
        val_dataset = MLPSpectrumDataset(val_df, config, normalizer, split_name="val")
        test_dataset = MLPSpectrumDataset(test_df, config, normalizer, split_name="test")

        train_loader, train_sampler = build_length_bucketed_loader(
            dataset=train_dataset,
            config=config,
            device=used_device,
            shuffle=True,
        )
        val_loader, _ = build_length_bucketed_loader(
            dataset=val_dataset,
            config=config,
            device=used_device,
            shuffle=False,
        )
        test_loader, _ = build_length_bucketed_loader(
            dataset=test_dataset,
            config=config,
            device=used_device,
            shuffle=False,
        )

        model = model.to(used_device)
        resume_checkpoint = _resolve_resume_checkpoint(
            config=config,
            output_dir=output_dir,
            default_checkpoint_path=latest_checkpoint_path,
        )
        resume_state: dict[str, object] | None = None
        if resume_checkpoint is not None:
            print(f"[DEV] Resuming improved transformer training from: {resume_checkpoint}")
            resume_state = torch.load(resume_checkpoint, map_location="cpu")
            checkpoint_model_state = resume_state.get("model_state_dict")
            if not isinstance(checkpoint_model_state, dict):
                raise ValueError(
                    f"Checkpoint {resume_checkpoint} is missing model_state_dict"
                )
            model.load_state_dict(checkpoint_model_state)

        model = _maybe_compile_model(model, config, used_device)
        optimizer = _build_optimizer(model, config)
        pos_weight = (
            compute_pos_weight(train_df, config) if config.loss.use_pos_weight else None
        )
        criterion = _build_loss(config, used_device, pos_weight=pos_weight)
        grad_scaler = _build_grad_scaler(config, used_device)

        scheduler: WarmupCosineScheduler | None = None
        if config.training.scheduler.enabled:
            total_steps = config.training.max_epochs * len(train_loader)
            warmup_steps = max(1, int(0.1 * total_steps))
            scheduler = WarmupCosineScheduler(
                optimizer=optimizer,
                base_lr=config.training.optimizer.learning_rate,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                min_lr_scale=0.05,
            )
            print(
                f"[DEV] Enabled warmup+cosine scheduler: total_steps={total_steps}, "
                f"warmup_steps={warmup_steps}"
            )

        history: list[dict[str, float]] = []
        best_state_dict: dict[str, Tensor] | None = None
        best_epoch = -1
        best_score = -np.inf if config.early_stopping.mode == "max" else np.inf
        bad_epochs = 0
        start_epoch = 1

        if resume_state is not None:
            checkpoint_optimizer_state = resume_state.get("optimizer_state_dict")
            if isinstance(checkpoint_optimizer_state, dict):
                optimizer.load_state_dict(checkpoint_optimizer_state)

            checkpoint_grad_scaler_state = resume_state.get("grad_scaler_state_dict")
            if grad_scaler is not None and isinstance(checkpoint_grad_scaler_state, dict):
                grad_scaler.load_state_dict(checkpoint_grad_scaler_state)

            checkpoint_scheduler_state = resume_state.get("scheduler_state_dict")
            if scheduler is not None and isinstance(checkpoint_scheduler_state, dict):
                scheduler.load_state_dict(checkpoint_scheduler_state)

            checkpoint_history = resume_state.get("history", [])
            if isinstance(checkpoint_history, list):
                history = checkpoint_history

            checkpoint_best_state = resume_state.get("best_state_dict")
            if isinstance(checkpoint_best_state, dict):
                best_state_dict = checkpoint_best_state

            checkpoint_best_epoch = resume_state.get("best_epoch")
            if checkpoint_best_epoch is not None:
                best_epoch = int(checkpoint_best_epoch)

            checkpoint_best_score = resume_state.get("best_score")
            if checkpoint_best_score is not None:
                best_score = float(checkpoint_best_score)

            checkpoint_bad_epochs = resume_state.get("bad_epochs")
            if checkpoint_bad_epochs is not None:
                bad_epochs = int(checkpoint_bad_epochs)

            checkpoint_epoch = resume_state.get("epoch")
            if checkpoint_epoch is not None:
                start_epoch = int(checkpoint_epoch) + 1

        monitor_name = config.early_stopping.monitor.removeprefix("val_")

        for epoch in range(start_epoch, config.training.max_epochs + 1):
            train_sampler.set_epoch(epoch)
            train_metrics = run_one_epoch_imp(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                config=config,
                device=used_device,
                training=True,
                grad_scaler=grad_scaler,
                scheduler=scheduler,
            )
            val_metrics = run_one_epoch_imp(
                model=model,
                loader=val_loader,
                optimizer=None,
                criterion=criterion,
                config=config,
                device=used_device,
                training=False,
                grad_scaler=None,
                scheduler=None,
            )

            epoch_record = {"epoch": float(epoch)}
            for key, value in train_metrics.items():
                epoch_record[f"train_{key}"] = float(value)
            for key, value in val_metrics.items():
                epoch_record[f"val_{key}"] = float(value)
            epoch_record["learning_rate"] = float(optimizer.param_groups[0]["lr"])
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
            else:
                bad_epochs += 1

            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_metrics['loss']:.6f} | "
                f"val_loss={val_metrics['loss']:.6f} | "
                f"val_pr_auc={val_metrics.get('pr_auc', np.nan):.6f} | "
                f"lr={optimizer.param_groups[0]['lr']:.6e}"
            )

            if config.output.save_epoch_history and history and (epoch == 1 or epoch % 5 == 0):
                pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

            if config.training.save_latest_checkpoint:
                _save_resume_checkpoint_imp(
                    checkpoint_path=latest_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    grad_scaler=grad_scaler,
                    scheduler=scheduler,
                    config=config,
                    normalizer=normalizer,
                    history=history,
                    epoch=epoch,
                    best_epoch=best_epoch,
                    best_score=float(best_score),
                    bad_epochs=bad_epochs,
                    best_state_dict=best_state_dict,
                    pos_weight=pos_weight,
                )

            if config.early_stopping.enabled and bad_epochs >= config.early_stopping.patience:
                print(f"Early stopping at epoch {epoch}. Best epoch was {best_epoch}.")
                break

        if best_state_dict is None:
            best_state_dict = _state_dict_to_cpu(deepcopy(model.state_dict()))
            best_epoch = len(history)

        optimizer.zero_grad(set_to_none=True)
        del optimizer
        if grad_scaler is not None:
            del grad_scaler
        if used_device.type == "cuda":
            torch.cuda.empty_cache()

        model.load_state_dict(best_state_dict)
        model = model.to(used_device)

        final_val_eval = evaluate_loader_imp(
            model=model,
            loader=val_loader,
            criterion=criterion,
            config=config,
            device=used_device,
        )
        final_test_eval = evaluate_loader_imp(
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
            with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as handle:
                import json

                json.dump(metrics_summary, handle, indent=2)

        if config.output.save_plots:
            save_training_plots(history_df, output_dir)

        if config.output.save_confusion_matrices:
            save_confusion_outputs(output_dir=output_dir, name="val", confusion_summary=val_conf)
            save_confusion_outputs(output_dir=output_dir, name="test", confusion_summary=test_conf)

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


def run_one_epoch_imp(
    model: PeakTransformerImpClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    config: TrainingConfig,
    device: torch.device,
    training: bool,
    grad_scaler: torch.cuda.amp.GradScaler | None,
    scheduler: WarmupCosineScheduler | None,
) -> dict[str, float]:
    result = _run_loader_imp(
        model=model,
        loader=loader,
        optimizer=optimizer,
        criterion=criterion,
        config=config,
        device=device,
        training=training,
        grad_scaler=grad_scaler,
        scheduler=scheduler,
    )
    return result["metrics"]


def evaluate_loader_imp(
    model: PeakTransformerImpClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, object]:
    return _run_loader_imp(
        model=model,
        loader=loader,
        optimizer=None,
        criterion=criterion,
        config=config,
        device=device,
        training=False,
        grad_scaler=None,
        scheduler=None,
    )


def _run_loader_imp(
    model: PeakTransformerImpClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    config: TrainingConfig,
    device: torch.device,
    training: bool,
    grad_scaler: torch.cuda.amp.GradScaler | None,
    scheduler: WarmupCosineScheduler | None,
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
        non_blocking = _use_non_blocking_transfers(config, device)
        peak_features = batch["peak_features"].to(device, non_blocking=non_blocking)
        spectrum_features = batch["spectrum_features"].to(device, non_blocking=non_blocking)
        targets = batch["targets"].to(device, non_blocking=non_blocking)
        weights = batch["weights"].to(device, non_blocking=non_blocking)
        padding_mask = batch["padding_mask"].to(device, non_blocking=non_blocking)
        valid_mask = ~padding_mask

        if training:
            with torch.set_grad_enabled(True):
                with _autocast_context(config, device):
                    logits = model(
                        peak_features=peak_features,
                        spectrum_features=spectrum_features,
                        padding_mask=padding_mask,
                    )
                    loss_matrix = criterion(logits, targets)

                if config.data.use_training_weights:
                    loss_matrix = loss_matrix * weights

                loss_matrix = loss_matrix * valid_mask.to(loss_matrix.dtype)
                valid_count = int(valid_mask.sum().item())
                loss = loss_matrix.sum() / max(valid_count, 1)

                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                if grad_scaler is not None and grad_scaler.is_enabled():
                    grad_scaler.scale(loss).backward()
                    if config.training.gradient_clip_norm > 0.0:
                        grad_scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_norm=config.training.gradient_clip_norm,
                        )
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    loss.backward()
                    if config.training.gradient_clip_norm > 0.0:
                        nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_norm=config.training.gradient_clip_norm,
                        )
                    optimizer.step()

                if scheduler is not None:
                    scheduler.step()
        else:
            with torch.inference_mode():
                with _autocast_context(config, device):
                    logits = model(
                        peak_features=peak_features,
                        spectrum_features=spectrum_features,
                        padding_mask=padding_mask,
                    )
                    loss_matrix = criterion(logits, targets)

                if config.data.use_training_weights:
                    loss_matrix = loss_matrix * weights

                loss_matrix = loss_matrix * valid_mask.to(loss_matrix.dtype)
                valid_count = int(valid_mask.sum().item())
                loss = loss_matrix.sum() / max(valid_count, 1)

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
            if training and optimizer is not None:
                msg += f" | lr={optimizer.param_groups[0]['lr']:.6e}"
            if device.type == "cuda":
                allocated_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
                reserved_gb = torch.cuda.memory_reserved(device) / (1024 ** 3)
                msg += f" | gpu_alloc={allocated_gb:.2f} GB | gpu_reserved={reserved_gb:.2f} GB"
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


def _save_resume_checkpoint_imp(
    checkpoint_path: Path,
    model: PeakTransformerImpClassifier,
    optimizer: torch.optim.Optimizer,
    grad_scaler: torch.cuda.amp.GradScaler | None,
    scheduler: WarmupCosineScheduler | None,
    config: TrainingConfig,
    normalizer: object,
    history: list[dict[str, float]],
    epoch: int,
    best_epoch: int,
    best_score: float,
    bad_epochs: int,
    best_state_dict: dict[str, Tensor] | None,
    pos_weight: float | None,
) -> None:
    checkpoint = {
        "epoch": int(epoch),
        "model_state_dict": _state_dict_to_cpu(model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(),
        "grad_scaler_state_dict": None if grad_scaler is None else grad_scaler.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "history": history,
        "best_state_dict": best_state_dict,
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "bad_epochs": int(bad_epochs),
        "model_config": asdict(config.model),
        "feature_config": {
            "peak_feature_columns": list(config.features.peak_feature_columns),
            "spectrum_feature_columns": list(config.features.spectrum_feature_columns),
            "use_raw_peak_mz": bool(config.features.use_raw_peak_mz),
            "raw_peak_mz_column": str(config.features.raw_peak_mz_column),
            "use_raw_peak_intensity": bool(config.features.use_raw_peak_intensity),
            "raw_peak_intensity_column": str(config.features.raw_peak_intensity_column),
            "sort_raw_peak_inputs_by_mz": bool(config.features.sort_raw_peak_inputs_by_mz),
            "normalize_peak_features": bool(config.features.normalize_peak_features),
            "normalize_spectrum_features": bool(config.features.normalize_spectrum_features),
            "use_instrument_label": bool(config.features.use_instrument_label),
            "instrument_names": list(config.features.instrument_names),
            "instrument_label_source_column": str(
                config.features.instrument_label_source_column
            ),
        },
        "normalizer": {
            "peak_mean": normalizer.peak_mean,
            "peak_std": normalizer.peak_std,
            "spectrum_mean": normalizer.spectrum_mean,
            "spectrum_std": normalizer.spectrum_std,
        },
        "pos_weight": None if pos_weight is None else float(pos_weight),
        "threshold_for_binary_metrics": float(config.evaluation.threshold_for_binary_metrics),
    }
    torch.save(checkpoint, checkpoint_path)

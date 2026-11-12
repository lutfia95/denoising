"""Train XL using the shared store and manifests produced by split_data_xl.py."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import random

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.model.xl import XLConfig, XLPeakClassifier, amp_context, model_inputs
from src.training.logging_utils import tee_output
from src.xl_data import (SPECTRUM_FEATURES, XLBatchSampler, XLFeatureConfig,
                         XLNormalizer, XLSpectrumDataset, feature_names, model_feature_names,
                         read_config, resolve_path, write_json, xl_collate)


def select_threshold(samples, objective="min_source_recall", target_recall=0.95):
    if objective == "min_source_recall":
        if not 0 < target_recall <= 1:
            raise ValueError("target_recall must be in (0, 1]")
        limits = []
        for source, (probs, labels) in samples.items():
            positives = probs[labels.astype(bool)]
            if not len(positives):
                raise ValueError(f"No sampled validation positives in {source}")
            # >= threshold is signal. The largest permissible order statistic
            # excludes at most floor((1 - recall) * positives) positives.
            index = min(int(math.floor((1 - target_recall) * len(positives) + 1e-10)), len(positives) - 1)
            limits.append(float(np.partition(positives, index)[index]))
        return min(limits)
    if objective == "f1":
        probs = np.concatenate([x[0] for x in samples.values()])
        labels = np.concatenate([x[1] for x in samples.values()])
        precision, recall, thresholds = precision_recall_curve(labels, probs)
        f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
        return float(thresholds[np.argmax(f1)])
    raise ValueError(f"Unsupported threshold objective: {objective}")


def confusion_metrics(counts):
    tn, fp, fn, tp = [int(x) for x in counts]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    denom = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "mcc": (tp * tn - fp * fn) / denom if denom else 0.0,
            "retained_fraction": (tp + fp) / max(sum(counts), 1),
            "confusion_matrix": [[tn, fp], [fn, tp]]}


@torch.inference_mode()
def evaluate(model, loader, device, config, threshold=0.5, intensity_baseline=False):
    if model is not None:
        model.eval()
    frame = loader.dataset.frame
    total = frame.groupby("source").num_peaks.sum().to_dict()
    cap = config["evaluation"].get("max_sampled_peaks_per_source", 2000000)
    rates = {s: min(1.0, cap / n) if cap else 1.0 for s, n in total.items()}
    sampled_p, sampled_y = {s: [] for s in total}, {s: [] for s in total}
    counts = {s: np.zeros(4, dtype=np.int64) for s in total}
    retained = {s: [] for s in total}
    retained_intensity = {s: 0.0 for s in total}
    spectrum_counts = {s: 0 for s in total}
    fractions = config["evaluation"].get("retained_peak_fractions", [0.3, 0.5, 0.7])
    loss_sum, spectra = 0.0, 0
    for batch in loader:
        if intensity_baseline:
            # sqrt(relative intensity) is monotonic in original intensity.
            probabilities = batch["intensity_features"][..., 1].numpy()
        else:
            with amp_context(device, config["training"]):
                logits = model(**model_inputs(batch, device))
            probabilities = logits.float().sigmoid().cpu().numpy()
            losses = F.binary_cross_entropy_with_logits(logits.float(), batch["targets"].to(device), reduction="none")
            valid = ~batch["padding_mask"].to(device)
            loss_sum += float(((losses * valid).sum(1) / valid.sum(1)).sum().cpu())
        for k, index in enumerate(batch["indices"].tolist()):
            source = frame.iloc[index].source
            n = int((~batch["padding_mask"][k]).sum())
            p = probabilities[k, :n]
            y = batch["targets"][k, :n].numpy().astype(bool)
            pred = p >= threshold
            relative_intensity = batch["intensity_features"][k, :n, 1].numpy() ** 2
            retained_intensity[source] += float(relative_intensity[pred].sum() / max(float(relative_intensity.sum()), 1e-12))
            spectrum_counts[source] += 1
            counts[source] += [np.sum(~pred & ~y), np.sum(pred & ~y), np.sum(~pred & y), np.sum(pred & y)]
            # Same label-independent sample at every epoch, independent of batch
            # ordering; all peaks still contribute to exact confusion counts.
            rng = np.random.default_rng(np.random.SeedSequence([config["training"].get("seed", 42), index, 917]))
            selected = rng.random(n) < rates[source] if rates[source] < 1 else np.ones(n, dtype=bool)
            sampled_p[source].append(p[selected])
            sampled_y[source].append(y[selected])
            if y.any():
                recalls = []
                for fraction in fractions:
                    keep = max(1, math.ceil(n * fraction))
                    top = np.argpartition(-p, keep - 1)[:keep]
                    recalls.append(float(y[top].sum() / y.sum()))
                retained[source].append(recalls)
            spectra += 1
    samples = {s: (np.concatenate(sampled_p[s]), np.concatenate(sampled_y[s])) for s in total}
    metrics = {}
    for source, (p, y) in samples.items():
        both = len(np.unique(y)) == 2
        stats = {**confusion_metrics(counts[source]),
                 "average_precision": float(average_precision_score(y, p)) if both else None,
                 "roc_auc": float(roc_auc_score(y, p)) if both else None,
                 "sampled_peaks": len(y), "total_peaks": int(total[source]),
                 "ranking_metrics_approximate": rates[source] < 1,
                 "sampled_positive_fraction": float(y.mean()) if len(y) else None}
        stats["spectrum_macro_retained_intensity_fraction"] = retained_intensity[source] / max(spectrum_counts[source], 1)
        means = np.mean(retained[source], axis=0) if retained[source] else [None] * len(fractions)
        stats["spectrum_macro_signal_recall_at_fraction"] = dict(zip(map(str, fractions), means))
        metrics[source] = stats
    ap = [v["average_precision"] for v in metrics.values()]
    if any(x is None for x in ap):
        raise ValueError("Each evaluation source must have both sampled classes; increase evaluation sample cap")
    result = {"sources": metrics, "source_macro_average_precision": float(np.mean(ap)),
              "pooled": confusion_metrics(sum(counts.values())), "threshold": float(threshold)}
    if not intensity_baseline:
        result["loss"] = loss_sum / max(spectra, 1)
    return result, samples


def sample_weights(dataset, training):
    confidence = dataset.frame.weight.to_numpy(dtype=np.float32) if training.get("use_training_weights", True) else np.ones(len(dataset))
    source = dataset.source_weights if training.get("balance_sources", True) else np.ones(len(dataset))
    weights = confidence * source
    return weights / weights.mean()


class WeightEMA:
    """Exponential moving average of weights; evaluated and saved instead of raw weights.

    The decay ramps up as (1 + n) / (10 + n) so early averages are not dominated
    by the random initialization.
    """
    def __init__(self, model, decay):
        if not 0 < decay < 1:
            raise ValueError("ema_decay must be in (0, 1), or null to disable")
        self.decay, self.updates = decay, 0
        self.model = deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for average, current in zip(self.model.parameters(), model.parameters()):
            average.lerp_(current.detach(), 1 - decay)
        for average, current in zip(self.model.buffers(), model.buffers()):
            average.copy_(current)


def save_checkpoint(path, state):
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    temp.replace(path)


def train(config, root):
    settings = config["training"]
    seed = int(settings.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(settings.get("cpu_num_threads", 8)))
    device_name = settings.get("device", "auto")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    split_dir = resolve_path(root, config["data"]["split_dir"])
    output = resolve_path(root, config["output"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    features = XLFeatureConfig(**config.get("features", {}))
    split_meta = json.loads((split_dir / "split_metadata.json").read_text())
    for field in ("histogram_bins", "histogram_max_mz"):
        if split_meta["feature_config"][field] != getattr(features, field):
            raise ValueError(f"Persisted summary setting {field} changed: rebuild the split in a new directory")
    data_signature = hashlib.sha256()
    for name in ("train.parquet", "val.parquet", "test.parquet", "split_metadata.json"):
        data_signature.update((split_dir / name).read_bytes())
    signature = data_signature.hexdigest()
    datasets = {name: XLSpectrumDataset(split_dir / f"{name}.parquet", features,
                training=name == "train", max_train_peaks=settings.get("max_train_peaks", 8192), seed=seed,
                augment_ppm_std=settings.get("augment_ppm_std", 0.0) or 0.0,
                augment_intensity_log_std=settings.get("augment_intensity_log_std", 0.0) or 0.0)
                for name in ("train", "val", "test")}
    for name, dataset in datasets.items():
        groups = dataset.frame.groupby("source")[["num_peaks", "num_positive"]].sum()
        if (groups.num_positive <= 0).any() or (groups.num_positive >= groups.num_peaks).any():
            raise ValueError(f"{name}: both label classes are required in every source")
    # Recheck manifest isolation, including when users supply their own manifests.
    for field in ("group_id", "run_id"):
        seen = set()
        for name, dataset in datasets.items():
            values = set(dataset.frame[field])
            if seen.intersection(values):
                raise ValueError(f"Cross-split leakage in {field} ({name})")
            seen.update(values)
    resume_path = settings.get("resume_from_checkpoint")
    resume = torch.load(resolve_path(root, resume_path), map_location="cpu", weights_only=False) if resume_path else None
    if resume:
        previous_config = {**resume["config"], "training": {**resume["config"]["training"], "resume_from_checkpoint": None}}
        current_config = {**config, "training": {**settings, "resume_from_checkpoint": None}}
        if resume.get("data_signature") != signature or previous_config != current_config:
            raise ValueError("Resume requires the same data manifests and configuration")
        if not (output / "best_model.pt").exists():
            raise FileNotFoundError("Resume requires best_model.pt in the original output directory")
        normalizer = XLNormalizer(**resume["normalizer"])
    else:
        if (output / "best_model.pt").exists() or (output / "last_checkpoint.pt").exists():
            raise FileExistsError("Output already contains checkpoints; set resume_from_checkpoint or use a new output directory")
        normalizer = XLNormalizer.fit(datasets["train"])
    for dataset in datasets.values():
        dataset.normalizer = normalizer
    write_json(output / "normalizer.json", asdict(normalizer))
    model_config = (XLConfig.from_checkpoint(resume["model_config"]) if resume else
                    XLConfig(context_input_dim=4 + features.histogram_bins,
                             max_context_tokens=1 + features.previous_spectra,
                             peak_input_dim=len(model_feature_names(features)), **config.get("model", {})))
    cap = settings.get("max_train_peaks", 8192)
    if cap is not None and (not isinstance(cap, int) or cap < 1):
        raise ValueError("max_train_peaks must be a positive integer or null")
    if (cap and cap < model_config.full_attention_max_peaks
            and datasets["train"].frame.num_peaks.max() > model_config.full_attention_max_peaks):
        raise ValueError("max_train_peaks must not move long spectra below the full-attention cutoff")
    model = XLPeakClassifier(model_config).to(device)
    print(f"Device: {device}; parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"Architecture: width={model_config.d_model}, layers={model_config.num_layers}, "
          f"heads={model_config.num_heads}, global_slots={model_config.global_slots}, "
          f"mass_bias={model_config.use_pairwise_mass_bias}, "
          f"peak_landmarks={model_config.mass_attention_landmarks}, "
          f"pairwise_chunk={model_config.pairwise_query_chunk_size}, "
          f"complement_bias={model_config.use_complement_bias}, "
          f"complement_neighbors={model_config.complement_neighbors}, "
          f"complement_tolerance_da={model_config.complement_tolerance_da}, "
          f"peak_features={features.peak_feature_set} ({model_config.peak_input_dim})", flush=True)
    samplers, loaders = {}, {}
    workers = int(settings.get("num_workers", 0))
    for name, dataset in datasets.items():
        samplers[name] = XLBatchSampler(dataset, settings.get("batch_size", 4),
                        settings.get("batch_peak_budget", 16384), name == "train", seed)
        # Workers restart each epoch so epoch-dependent uniform sampling is fresh.
        loaders[name] = DataLoader(dataset, batch_sampler=samplers[name], collate_fn=xl_collate,
                       num_workers=workers, pin_memory=device.type == "cuda", persistent_workers=False)
    weights = torch.tensor(sample_weights(datasets["train"], settings), dtype=torch.float32)
    pos_weight = None
    if settings.get("use_pos_weight", False):
        frame = datasets["train"].frame
        positive_fraction = float(np.average(frame.num_positive / frame.num_peaks, weights=weights.numpy()))
        ratio = min(settings.get("max_pos_weight", 10.0), (1 - positive_fraction) / max(positive_fraction, 1e-12))
        pos_weight = torch.tensor(ratio, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.get("learning_rate", 0.0003),
                                  weight_decay=settings.get("weight_decay", 0.01))
    accumulation = int(settings.get("gradient_accumulation_steps", 4))
    if accumulation < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    steps_per_epoch = math.ceil(len(samplers["train"]) / accumulation)
    total_steps = steps_per_epoch * int(settings.get("max_epochs", 60))
    warmup = max(1, int(total_steps * settings.get("warmup_fraction", 0.05)))

    def lr_scale(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min((step - warmup) / max(total_steps - warmup, 1), 1.0)
        return 0.05 + 0.95 * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    ema_decay = settings.get("ema_decay")
    ema = WeightEMA(model, float(ema_decay)) if ema_decay else None
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and settings.get("enable_amp", True) and not torch.cuda.is_bf16_supported())
    history, best_score, bad_epochs, first_epoch = [], -1.0, 0, 1
    common = {"format": "xl-v2", "config": config, "model_config": asdict(model_config),
              "feature_config": asdict(features), "peak_feature_names": feature_names(features),
              "spectrum_feature_names": SPECTRUM_FEATURES, "normalizer": asdict(normalizer),
              "data_signature": signature, "pos_weight": float(pos_weight) if pos_weight is not None else None}
    if resume:
        model.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        scaler.load_state_dict(resume["scaler_state_dict"])
        history, best_score, bad_epochs = resume["history"], resume["best_score"], resume["bad_epochs"]
        first_epoch = resume["epoch"] + 1
        torch.set_rng_state(resume["torch_rng_state"])
        if ema is not None:
            if "ema_state_dict" not in resume:
                raise ValueError("Resume checkpoint has no EMA weights but ema_decay is set")
            ema.model.load_state_dict(resume["ema_state_dict"])
            ema.updates = resume["ema_updates"]
        if device.type == "cuda" and resume.get("cuda_rng_states") is not None:
            torch.cuda.set_rng_state_all(resume["cuda_rng_states"])
    elif config["evaluation"].get("intensity_baseline", True):
        baseline, _ = evaluate(None, loaders["val"], device, config, intensity_baseline=True)
        write_json(output / "intensity_baseline_val.json", baseline)
        print(f"Validation intensity baseline macro AP: {baseline['source_macro_average_precision']:.5f}", flush=True)
    for epoch in range(first_epoch, settings.get("max_epochs", 60) + 1):
        model.train()
        datasets["train"].epoch = epoch
        samplers["train"].epoch = epoch
        optimizer.zero_grad(set_to_none=True)
        train_sum, train_count, pending = 0.0, 0, 0
        number_batches = len(samplers["train"])
        for step, batch in enumerate(loaders["train"], 1):
            with amp_context(device, settings):
                logits = model(**model_inputs(batch, device))
            targets = batch["targets"].to(device)
            valid = ~batch["padding_mask"].to(device)
            matrix = F.binary_cross_entropy_with_logits(logits.float(), targets, pos_weight=pos_weight, reduction="none")
            per_spectrum = (matrix * valid).sum(1) / valid.sum(1)
            loss_sum = (per_spectrum * weights[batch["indices"]].to(device)).sum()
            if not torch.isfinite(loss_sum):
                raise FloatingPointError("Nonfinite training loss")
            scaler.scale(loss_sum).backward()
            pending += len(targets)
            train_sum += float(loss_sum.detach())
            train_count += len(targets)
            if step % accumulation == 0 or step == number_batches:
                scaler.unscale_(optimizer)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(pending)
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings.get("gradient_clip_norm", 1.0))
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                    if ema is not None:
                        ema.update(model)
                optimizer.zero_grad(set_to_none=True)
                pending = 0
            if step == 1 or step % settings.get("log_every_batches", 100) == 0:
                print(f"Epoch {epoch} batch {step}/{number_batches}: loss={train_sum / train_count:.5f}", flush=True)
        selected = ema.model if ema is not None else model
        validation, _ = evaluate(selected, loaders["val"], device, config)
        score = validation["source_macro_average_precision"]
        history.append({"epoch": epoch, "train_loss": train_sum / train_count,
                        "val_loss": validation["loss"], "val_source_macro_ap": score,
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        if score > best_score + settings.get("min_delta", 0.0001):
            best_score, bad_epochs = score, 0
            save_checkpoint(output / "best_model.pt", {**common, "model_state_dict": selected.state_dict(),
                            "epoch": epoch, "best_score": best_score, "threshold": 0.5,
                            "threshold_tuned": False})
        else:
            bad_epochs += 1
        save_checkpoint(output / "last_checkpoint.pt", {**common, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict(), "epoch": epoch, "history": history,
                        "best_score": best_score, "bad_epochs": bad_epochs,
                        **({"ema_state_dict": ema.model.state_dict(), "ema_updates": ema.updates} if ema else {}),
                        "torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else None})
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        write_json(output / "latest_validation.json", validation)
        print(f"Epoch {epoch}: train={train_sum / train_count:.5f}, val macro AP={score:.5f}, best={best_score:.5f}", flush=True)
        if bad_epochs >= settings.get("patience", 10):
            break
    best = torch.load(output / "best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    _, samples = evaluate(model, loaders["val"], device, config)
    threshold = select_threshold(samples, config["evaluation"].get("threshold_objective", "min_source_recall"),
                                 config["evaluation"].get("target_recall", 0.95))
    validation, _ = evaluate(model, loaders["val"], device, config, threshold)
    # The test set is opened for scoring only after model and threshold selection.
    test, _ = evaluate(model, loaders["test"], device, config, threshold)
    best.update(threshold=threshold, threshold_tuned=True, threshold_selection=config["evaluation"], validation_metrics=validation)
    save_checkpoint(output / "best_model.pt", best)
    write_json(output / "metrics_summary.json", {"best_epoch": best["epoch"], "threshold": threshold,
               "threshold_fit_on": "validation rank-metric sample", "validation": validation, "test": test})
    print(f"Saved {output / 'best_model.pt'}; validation threshold={threshold:.6f}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_xl.yml")
    args = parser.parse_args()
    config, root = read_config(args.config)
    with tee_output(resolve_path(root, config["output"]["output_dir"]) / "training.log", enabled=True):
        train(config, root)


if __name__ == "__main__":
    main()

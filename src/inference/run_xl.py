"""Stream MGF spectra through an XL checkpoint without feature overrides."""
from __future__ import annotations

import argparse
from collections import deque
import csv
from pathlib import Path

import numpy as np
import torch

from src.model.xl import XLConfig, XLPeakClassifier, amp_context, model_inputs
from src.xl_data import (XLFeatureConfig, XLNormalizer, make_features, prepare_arrays,
                         read_config, resolve_path, write_json, xl_collate)


def iter_mgf(path):
    """Yield metadata and original peak arrays, retaining double precision."""
    params, peaks = None, []
    with Path(path).open() as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line or line.startswith(("#", ";", "!")):
                continue
            if line.upper() == "BEGIN IONS":
                if params is not None:
                    raise ValueError(f"{path}:{line_number}: nested BEGIN IONS")
                params, peaks = [], []
            elif line.upper() == "END IONS":
                if params is None:
                    raise ValueError(f"{path}:{line_number}: unmatched END IONS")
                yield params, np.asarray(peaks, dtype=np.float64).reshape(-1, 2)
                params, peaks = None, []
            elif params is not None:
                if "=" in line:
                    params.append(tuple(line.split("=", 1)))
                else:
                    fields = line.split()
                    if len(fields) < 2:
                        raise ValueError(f"{path}:{line_number}: invalid peak line")
                    peaks.append((float(fields[0]), float(fields[1])))
        if params is not None:
            raise ValueError(f"{path}: missing END IONS")


def build_sample(peaks, params, features, normalizer, history, last_scan):
    metadata = {key.upper(): value for key, value in params}
    precursor = float(metadata["PEPMASS"].split()[0])
    charge_value = metadata["CHARGE"].strip()
    # Reject ambiguous precursor charges rather than silently choosing a charge.
    if not charge_value.rstrip("+").isdigit():
        raise ValueError(f"Expected one positive CHARGE, got {charge_value!r}")
    charge = int(charge_value.rstrip("+"))
    mz, intensity, _, order = prepare_arrays(peaks[:, 0], peaks[:, 1])
    peak, spectrum, histogram = make_features(mz, intensity, charge, precursor, features)
    scan = int(metadata["SCANS"]) if "SCANS" in metadata else None
    if features.previous_spectra and scan is None:
        raise ValueError("XL history requires integer SCANS in MGF. Train with previous_spectra: 0 if scan IDs are unavailable.")
    if features.previous_spectra and last_scan is not None and scan < last_scan:
        raise ValueError("MGF spectra must be ordered by SCANS for causal history")
    context = np.zeros((features.previous_spectra + 1, 4 + features.histogram_bins), dtype=np.float32)
    context[0] = np.r_[normalizer.spectrum(spectrum, features), histogram]
    mask = np.ones(features.previous_spectra + 1, dtype=bool)
    mask[0] = False
    valid_history = [(s, summary) for s, summary in history if scan is not None and s < scan and
                     (features.max_scan_gap is None or scan - s <= features.max_scan_gap)]
    for i, (_, summary) in enumerate(valid_history[-features.previous_spectra:][::-1] if features.previous_spectra else [], 1):
        context[i], mask[i] = summary, False
    normalized = normalizer.peak(peak, features)
    sample = {"peak_features": normalized, "mz": mz.astype(np.float32),
              "delta_mz": (precursor - mz).astype(np.float32),
              "intensity_features": np.column_stack([normalized[:, 0], np.sqrt(intensity / max(float(intensity.max()), 1e-12))]).astype(np.float32),
              "context": context, "context_mask": mask, "targets": np.zeros(len(mz), dtype=np.float32),
              "weight": 1.0, "source_weight": 1.0, "index": 0}
    # Match training: one representative per scan (the first charge variant).
    if features.previous_spectra and (not history or history[-1][0] != scan):
        history.append((scan, context[0].copy()))
    return sample, order, scan


def write_spectrum(handle, params, peaks):
    handle.write("BEGIN IONS\n")
    for key, value in params:
        handle.write(f"{key}={value}\n")
    for mz, intensity in peaks:
        handle.write(f"{mz:.10g} {intensity:.10g}\n")
    handle.write("END IONS\n\n")


@torch.inference_mode()
def run_inference(config, root):
    if config.get("features"):
        raise ValueError("Feature overrides are unsupported; XL features come from the checkpoint")
    checkpoint_path = resolve_path(root, config["model"]["checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") not in ("xl-v1", "xl-v2"):
        raise ValueError("Expected an XL checkpoint; legacy transformer checkpoints are incompatible")
    features = XLFeatureConfig(**checkpoint["feature_config"])
    normalizer = XLNormalizer(**checkpoint["normalizer"])
    model = XLPeakClassifier(XLConfig.from_checkpoint(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    runtime = config.get("runtime", {})
    name = runtime.get("device", "auto")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)
    torch.set_num_threads(int(runtime.get("cpu_num_threads", 8)))
    model.to(device).eval()
    override = config.get("evaluation", {}).get("threshold")
    threshold = float(override) if override is not None else float(checkpoint["threshold"])
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    if override is None and not checkpoint.get("threshold_tuned", False):
        raise ValueError("Checkpoint threshold is not tuned yet; finish training or explicitly set evaluation.threshold")
    output = resolve_path(root, config["output"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    paths = [resolve_path(root, p) for p in config["input"]["mgf_paths"]]
    if not paths or len({p.stem for p in paths}) != len(paths):
        raise ValueError("Provide MGF paths with distinct stems")
    summaries = []
    for path in paths:
        destination = output / f"{path.stem}.xl.mgf"
        report = output / f"{path.stem}.spectra.csv"
        score_path = output / f"{path.stem}.peaks.csv"
        if destination.exists() or report.exists() or score_path.exists():
            raise FileExistsError(f"Inference output already exists for {path.stem}")
        history = deque(maxlen=max(1, features.previous_spectra + 1))
        last_scan, total_peaks, kept_peaks, n_spectra = None, 0, 0, 0
        score_handle = score_path.open("w", newline="") if config["output"].get("write_peak_scores", False) else None
        try:
            score_writer = csv.writer(score_handle) if score_handle else None
            if score_writer:
                score_writer.writerow(["spectrum_index", "peak_index", "mz", "intensity", "score", "kept"])
            with destination.open("w") as mgf, report.open("w", newline="") as report_handle:
                report_writer = csv.writer(report_handle)
                report_writer.writerow(["spectrum_index", "title", "scan", "input_peaks", "kept_peaks", "retained_intensity_fraction"])
                for params, peaks in iter_mgf(path):
                    metadata = {key.upper(): value for key, value in params}
                    if not len(peaks):
                        write_spectrum(mgf, params, peaks)
                        report_writer.writerow([n_spectra, metadata.get("TITLE", ""), metadata.get("SCANS", ""), 0, 0, 0])
                        n_spectra += 1
                        continue
                    sample, order, last_scan = build_sample(peaks, params, features, normalizer, history, last_scan)
                    batch = xl_collate([sample])
                    with amp_context(device, checkpoint["config"]["training"]):
                        logits = model(**model_inputs(batch, device))
                    scores_sorted = logits.float().sigmoid()[0].cpu().numpy()
                    scores = np.empty_like(scores_sorted)
                    scores[order] = scores_sorted
                    keep = scores >= threshold
                    # Neither minimum-kept-peaks nor an intensity fallback changes
                    # the validated decision rule without being evaluated first.
                    write_spectrum(mgf, params, peaks[keep])
                    report_writer.writerow([n_spectra, metadata.get("TITLE", ""), last_scan,
                         len(peaks), int(keep.sum()), float(peaks[keep, 1].sum() / max(float(peaks[:, 1].sum()), 1e-12))])
                    if score_writer:
                        for j, ((mz, intensity), score, selected) in enumerate(zip(peaks, scores, keep)):
                            score_writer.writerow([n_spectra, j, mz, intensity, float(score), bool(selected)])
                    total_peaks += len(peaks)
                    kept_peaks += int(keep.sum())
                    n_spectra += 1
                    if n_spectra % 100 == 0:
                        print(f"{path.name}: {n_spectra} spectra", flush=True)
        finally:
            if score_handle:
                score_handle.close()
        summaries.append({"input": str(path), "output": str(destination), "spectra": n_spectra,
                          "input_peaks": total_peaks, "kept_peaks": kept_peaks})
    write_json(output / "inference_summary.json", {"checkpoint": str(checkpoint_path), "threshold": threshold,
               "feature_config": checkpoint["feature_config"], "files": summaries})
    print(f"Wrote filtered MGF and reports to {output}", flush=True)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/infer_xl.yml")
    args = parser.parse_args()
    config, root = read_config(args.config)
    run_inference(config, root)


if __name__ == "__main__":
    main()

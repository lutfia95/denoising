from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.peak_features import PeakFeatureComputer, PeakFeatureConfig
from src.features.spectrum_features import SpectrumFeatureComputer, SpectrumFeatureConfig
from src.model.transformer_imp import PeakTransformerImpClassifier, PeakTransformerImpConfig
from src.training.train_mlp import FeatureNormalizer
from src.training.train_transformer import transformer_collate_fn
from src.types import SpectrumRecord


PEAK_FEATURE_NAME_TO_ATTR = {
    "peak_feature_mz": "mz",
    "peak_feature_log_intensity": "log_intensity",
    "peak_feature_relative_intensity": "relative_intensity",
    "peak_feature_mz_over_precursor": "mz_over_precursor",
    "peak_feature_delta_to_precursor": "delta_to_precursor",
    "peak_feature_delta_prev": "delta_prev",
    "peak_feature_delta_next": "delta_next",
}

SPECTRUM_FEATURE_NAME_TO_GETTER = {
    "Charge": lambda record, spectrum_features: float(record.charge),
    "exp m/z": lambda record, spectrum_features: float(record.precursor_mz),
    "num_peaks": lambda record, spectrum_features: float(spectrum_features.num_peaks),
    "tic": lambda record, spectrum_features: float(spectrum_features.tic),
}


@dataclass(slots=True)
class MGFSpectrum:
    params: list[tuple[str, str]]
    mz_arr: np.ndarray
    int_arr: np.ndarray
    title: str
    rtinseconds: float | None
    pepmass: float
    charge: int
    scans: int | None


def load_inference_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("Inference config must be a mapping")
    return config


def load_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint at {checkpoint_path} is not a dictionary")
    required = {"model_state_dict", "model_config", "feature_config", "normalizer"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(
            f"Checkpoint at {checkpoint_path} is missing required keys: {sorted(missing)}"
        )
    return checkpoint


def build_normalizer(raw: dict[str, Any]) -> FeatureNormalizer:
    return FeatureNormalizer(
        peak_mean=np.asarray(raw["peak_mean"], dtype=np.float32),
        peak_std=np.asarray(raw["peak_std"], dtype=np.float32),
        spectrum_mean=np.asarray(raw["spectrum_mean"], dtype=np.float32),
        spectrum_std=np.asarray(raw["spectrum_std"], dtype=np.float32),
    )


def read_mgf(path: str | Path) -> list[MGFSpectrum]:
    spectra: list[MGFSpectrum] = []
    current_params: list[tuple[str, str]] | None = None
    current_peaks: list[tuple[float, float]] = []

    def finalize_current() -> None:
        nonlocal current_params, current_peaks
        if current_params is None:
            return

        params_dict = {key.upper(): value for key, value in current_params}
        pepmass_raw = params_dict.get("PEPMASS")
        if pepmass_raw is None:
            raise ValueError("Encountered an MGF spectrum without PEPMASS")
        pepmass = float(str(pepmass_raw).split()[0])

        charge_raw = params_dict.get("CHARGE")
        if charge_raw is None:
            raise ValueError("Encountered an MGF spectrum without CHARGE")
        charge = _parse_charge(charge_raw)

        title = str(params_dict.get("TITLE", f"spectrum_{len(spectra)}"))
        rtinseconds = (
            None
            if "RTINSECONDS" not in params_dict
            else float(params_dict["RTINSECONDS"])
        )
        scans = None if "SCANS" not in params_dict else int(params_dict["SCANS"])

        if current_peaks:
            peak_array = np.asarray(current_peaks, dtype=np.float32)
            mz_arr = peak_array[:, 0]
            int_arr = peak_array[:, 1]
        else:
            mz_arr = np.zeros((0,), dtype=np.float32)
            int_arr = np.zeros((0,), dtype=np.float32)

        spectra.append(
            MGFSpectrum(
                params=list(current_params),
                mz_arr=mz_arr,
                int_arr=int_arr,
                title=title,
                rtinseconds=rtinseconds,
                pepmass=pepmass,
                charge=charge,
                scans=scans,
            )
        )
        current_params = None
        current_peaks = []

    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.upper() == "BEGIN IONS":
                finalize_current()
                current_params = []
                current_peaks = []
                continue
            if line.upper() == "END IONS":
                finalize_current()
                continue
            if current_params is None:
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                current_params.append((key, value))
                continue

            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid peak line in MGF: {line!r}")
            current_peaks.append((float(parts[0]), float(parts[1])))

    finalize_current()
    return spectra


def write_mgf(path: str | Path, spectra: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    with output_path.open("w", encoding="utf-8") as handle:
        for spectrum in spectra:
            handle.write("BEGIN IONS\n")
            for key, value in spectrum["params"]:
                handle.write(f"{key}={value}\n")
            mz_arr = np.asarray(spectrum["mz_arr"], dtype=np.float32)
            int_arr = np.asarray(spectrum["int_arr"], dtype=np.float32)
            for mz, intensity in zip(mz_arr, int_arr, strict=True):
                handle.write(f"{float(mz):.6f} {float(intensity):.6f}\n")
            handle.write("END IONS\n")


def _parse_charge(raw_charge: str) -> int:
    token = str(raw_charge).strip()
    if token.endswith("+") or token.endswith("-"):
        token = token[:-1]
    return int(token)


def _build_instrument_one_hot(
    source_value: str,
    feature_config: dict[str, Any],
) -> np.ndarray:
    if not bool(feature_config.get("use_instrument_label", False)):
        return np.zeros((0,), dtype=np.float32)

    instrument_names = list(feature_config.get("instrument_names", []))
    if not instrument_names:
        raise ValueError(
            "use_instrument_label is true, but instrument_names is empty in feature config"
        )

    matches = [
        idx for idx, instrument_name in enumerate(instrument_names) if instrument_name in source_value
    ]
    if not matches:
        raise ValueError(
            f"Could not match instrument label using source value {source_value!r} "
            f"and names {instrument_names!r}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous instrument label using source value {source_value!r} "
            f"and names {instrument_names!r}"
        )

    instrument_vec = np.zeros((len(instrument_names),), dtype=np.float32)
    instrument_vec[matches[0]] = 1.0
    return instrument_vec


def _compare_feature_configs(config_features: dict[str, Any], checkpoint_features: dict[str, Any]) -> None:
    keys_to_compare = [
        "peak_feature_columns",
        "spectrum_feature_columns",
        "use_raw_peak_mz",
        "raw_peak_mz_column",
        "use_raw_peak_intensity",
        "raw_peak_intensity_column",
        "sort_raw_peak_inputs_by_mz",
        "normalize_peak_features",
        "normalize_spectrum_features",
        "use_instrument_label",
        "instrument_names",
        "instrument_label_source_column",
    ]
    mismatches: list[str] = []
    for key in keys_to_compare:
        if key == "use_instrument_label":
            config_value = bool(config_features.get(key, False))
            checkpoint_value = bool(checkpoint_features.get(key, False))
        elif key == "instrument_names":
            config_value = list(config_features.get(key, []))
            checkpoint_value = list(checkpoint_features.get(key, []))
        elif key == "instrument_label_source_column":
            config_value = config_features.get(key)
            checkpoint_value = checkpoint_features.get(key)
            if not config_features.get("use_instrument_label", False):
                config_value = None
            if not checkpoint_features.get("use_instrument_label", False):
                checkpoint_value = None
        else:
            config_value = config_features.get(key)
            checkpoint_value = checkpoint_features.get(key)
        if config_value != checkpoint_value:
            mismatches.append(
                f"{key}: config={config_value!r}, checkpoint={checkpoint_value!r}"
            )
    if mismatches:
        joined = "\n".join(mismatches)
        raise ValueError(
            "Inference feature config does not match the trained checkpoint.\n"
            f"{joined}"
        )


def build_record(
    spectrum: MGFSpectrum,
    mgf_path: Path,
    index: int,
    input_cfg: dict[str, Any],
) -> SpectrumRecord:
    peak_list_file_name = str(
        input_cfg.get("peak_list_file_name") or mgf_path.name
    )
    scan = spectrum.scans if spectrum.scans is not None else index
    scan_id: str | int = spectrum.title if spectrum.title else scan

    return SpectrumRecord(
        search_id=mgf_path.stem,
        peak_list_file_name=peak_list_file_name,
        scan=int(scan),
        mz_arr=spectrum.mz_arr.astype(np.float32, copy=False),
        int_arr=spectrum.int_arr.astype(np.float32, copy=False),
        charge=int(spectrum.charge),
        precursor_mz=float(spectrum.pepmass),
        annotation_mask=None,
        fdr=None,
        scan_id=scan_id,
    )


def build_model_inputs(
    record: SpectrumRecord,
    feature_config: dict[str, Any],
    normalizer: FeatureNormalizer,
) -> tuple[np.ndarray, np.ndarray, SpectrumRecord]:
    peak_computer = PeakFeatureComputer(
        PeakFeatureConfig(
            use_log_intensity=True,
            use_relative_intensity=True,
            use_mz_over_precursor=True,
            use_delta_to_precursor=True,
            use_delta_neighbors=True,
            sort_by_mz=bool(feature_config.get("sort_raw_peak_inputs_by_mz", True)),
        )
    )
    spectrum_computer = SpectrumFeatureComputer(
        SpectrumFeatureConfig(use_tic=True, use_num_peaks=True)
    )

    processed_record, peak_feature_set = peak_computer.compute(record)
    spectrum_feature_set = spectrum_computer.compute(processed_record)

    peak_columns: list[np.ndarray] = []
    for feature_name in feature_config["peak_feature_columns"]:
        attr_name = PEAK_FEATURE_NAME_TO_ATTR.get(feature_name)
        if attr_name is None:
            raise ValueError(f"Unsupported peak feature name: {feature_name!r}")
        feature_values = getattr(peak_feature_set, attr_name)
        if feature_values is None:
            raise ValueError(f"Peak feature {feature_name!r} was not computed")
        peak_columns.append(np.asarray(feature_values, dtype=np.float32))

    if bool(feature_config.get("use_raw_peak_mz", False)):
        peak_columns.append(processed_record.mz_arr.astype(np.float32, copy=False))
    if bool(feature_config.get("use_raw_peak_intensity", False)):
        peak_columns.append(processed_record.int_arr.astype(np.float32, copy=False))

    if not peak_columns:
        raise ValueError("No peak features were configured for inference")

    peak_matrix = np.stack(peak_columns, axis=1).astype(np.float32, copy=False)
    peak_matrix = normalizer.normalize_peak_features(
        peak_matrix,
        enabled=bool(feature_config.get("normalize_peak_features", True)),
    )

    spectrum_values = []
    for feature_name in feature_config["spectrum_feature_columns"]:
        getter = SPECTRUM_FEATURE_NAME_TO_GETTER.get(feature_name)
        if getter is None:
            raise ValueError(f"Unsupported spectrum feature name: {feature_name!r}")
        spectrum_values.append(getter(processed_record, spectrum_feature_set))

    spectrum_vector = np.asarray(spectrum_values, dtype=np.float32)
    spectrum_vector = normalizer.normalize_spectrum_features(
        spectrum_vector,
        enabled=bool(feature_config.get("normalize_spectrum_features", True)),
    )

    instrument_source_value = str(
        feature_config.get("instrument_source_value")
        or processed_record.peak_list_file_name
    )
    instrument_vector = _build_instrument_one_hot(instrument_source_value, feature_config)
    if instrument_vector.size > 0:
        spectrum_vector = np.concatenate([spectrum_vector, instrument_vector], axis=0)

    return peak_matrix, spectrum_vector, processed_record


def run_inference(config_path: str | Path) -> dict[str, Any]:
    raw_config = load_inference_config(config_path)
    input_cfg = dict(raw_config.get("input", {}))
    model_cfg = dict(raw_config.get("model", {}))
    feature_cfg = dict(raw_config.get("features", {}))
    evaluation_cfg = dict(raw_config.get("evaluation", {}))
    runtime_cfg = dict(raw_config.get("runtime", {}))
    output_cfg = dict(raw_config.get("output", {}))

    checkpoint_path = Path(model_cfg["checkpoint_path"])
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_feature_config = dict(checkpoint["feature_config"])
    _compare_feature_configs(feature_cfg, checkpoint_feature_config)

    threshold = float(
        checkpoint["threshold_for_binary_metrics"]
        if evaluation_cfg.get("use_checkpoint_threshold", True)
        else evaluation_cfg["threshold_for_binary_metrics"]
    )

    feature_cfg["instrument_source_value"] = str(
        input_cfg.get("instrument_source_value")
        or input_cfg.get("peak_list_file_name")
        or Path(input_cfg["mgf_path"]).name
    )

    model_config = PeakTransformerImpConfig(**dict(checkpoint["model_config"]))
    normalizer = build_normalizer(dict(checkpoint["normalizer"]))

    device_name = str(runtime_cfg.get("device", "cpu"))
    device = torch.device(device_name)
    batch_size = int(runtime_cfg.get("batch_size", 8))

    model = PeakTransformerImpClassifier(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    mgf_path = Path(input_cfg["mgf_path"])
    spectra = read_mgf(mgf_path)
    if not spectra:
        raise ValueError(f"No spectra found in MGF file: {mgf_path}")

    batch_items: list[dict[str, Any]] = []
    spectrum_payloads: list[dict[str, Any]] = []
    all_peak_rows: list[dict[str, Any]] = []
    dropped_peak_rows: list[dict[str, Any]] = []
    kept_spectra: list[dict[str, Any]] = []

    def flush_batch() -> None:
        if not batch_items:
            return

        batch = transformer_collate_fn(batch_items)
        with torch.inference_mode():
            logits = model(
                peak_features=batch["peak_features"].to(device),
                spectrum_features=batch["spectrum_features"].to(device),
                padding_mask=batch["padding_mask"].to(device),
            )
            probs = torch.sigmoid(logits).detach().cpu().numpy()

        for payload, spectrum_probs in zip(spectrum_payloads[: len(batch_items)], probs, strict=True):
            n_peaks = payload["n_peaks"]
            peak_probs = np.asarray(spectrum_probs[:n_peaks], dtype=np.float32)
            pred_mask = peak_probs >= threshold

            processed_record = payload["processed_record"]
            mgf_spectrum = payload["mgf_spectrum"]
            title = payload["title"]
            rtinseconds = payload["rtinseconds"]

            kept_count = int(pred_mask.sum())
            dropped_indices = np.flatnonzero(~pred_mask)
            dropped_peaks = [
                {
                    "peak_rank_by_mz": int(idx),
                    "mz": float(processed_record.mz_arr[idx]),
                    "intensity": float(processed_record.int_arr[idx]),
                    "prob_signal": float(peak_probs[idx]),
                }
                for idx in dropped_indices
            ]
            all_peak_rows.extend(
                {
                    "spectrum_index": int(payload["spectrum_index"]),
                    "title": title,
                    "scan": int(processed_record.scan),
                    "peak_rank_by_mz": int(idx),
                    "mz": float(processed_record.mz_arr[idx]),
                    "intensity": float(processed_record.int_arr[idx]),
                    "prob_signal": float(peak_probs[idx]),
                    "pred_label": int(pred_mask[idx]),
                    "threshold": float(threshold),
                    "charge": int(processed_record.charge),
                    "precursor_mz": float(processed_record.precursor_mz),
                    "rtinseconds": None if rtinseconds is None else float(rtinseconds),
                }
                for idx in range(n_peaks)
            )
            dropped_peak_rows.append(
                {
                    "spectrum_index": int(payload["spectrum_index"]),
                    "title": title,
                    "scan": int(processed_record.scan),
                    "scan_id": str(processed_record.scan_id),
                    "charge": int(processed_record.charge),
                    "precursor_mz": float(processed_record.precursor_mz),
                    "rtinseconds": None if rtinseconds is None else float(rtinseconds),
                    "threshold": float(threshold),
                    "total_peaks": int(n_peaks),
                    "dropped_peak_count": int(len(dropped_peaks)),
                    "dropped_peaks": json.dumps(dropped_peaks),
                }
            )

            filtered_mode = str(output_cfg.get("filtered_mgf_mode", "drop_below_threshold"))
            if filtered_mode == "drop_below_threshold":
                filtered_mz = processed_record.mz_arr[pred_mask]
                filtered_int = processed_record.int_arr[pred_mask]
            elif filtered_mode == "zero_below_threshold":
                filtered_mz = processed_record.mz_arr
                filtered_int = processed_record.int_arr.copy()
                filtered_int[~pred_mask] = 0.0
            else:
                raise ValueError(
                    "filtered_mgf_mode must be one of "
                    "'drop_below_threshold' or 'zero_below_threshold'"
                )

            kept_spectra.append(
                {
                    "params": list(mgf_spectrum.params),
                    "mz_arr": filtered_mz,
                    "int_arr": filtered_int,
                }
            )
            payload["kept_count"] = kept_count
            payload["total_count"] = n_peaks

        del batch_items[:]
        del spectrum_payloads[: len(spectrum_payloads)]

    for spectrum_index, mgf_spectrum in enumerate(spectra):
        record = build_record(mgf_spectrum, mgf_path, spectrum_index, input_cfg)
        peak_matrix, spectrum_vector, processed_record = build_model_inputs(
            record=record,
            feature_config=feature_cfg,
            normalizer=normalizer,
        )
        if peak_matrix.shape[0] == 0:
            continue

        batch_items.append(
            {
                "peak_features": peak_matrix,
                "spectrum_features": spectrum_vector,
                "targets": np.zeros((peak_matrix.shape[0],), dtype=np.float32),
                "weight": 1.0,
                "row_index": spectrum_index,
            }
        )
        spectrum_payloads.append(
            {
                "spectrum_index": spectrum_index,
                "mgf_spectrum": mgf_spectrum,
                "processed_record": processed_record,
                "n_peaks": int(peak_matrix.shape[0]),
                "title": mgf_spectrum.title,
                "rtinseconds": mgf_spectrum.rtinseconds,
            }
        )

        if len(batch_items) >= batch_size:
            flush_batch()

    flush_batch()

    output_dir = Path(output_cfg.get("output_dir", "outputs/inference_transformer_imp"))
    output_dir.mkdir(parents=True, exist_ok=True)

    peak_predictions = pd.DataFrame(all_peak_rows)
    peak_predictions.to_csv(output_dir / "peak_predictions.csv", index=False)

    spectrum_summary = (
        peak_predictions.groupby(["spectrum_index", "title", "scan", "charge", "precursor_mz"], dropna=False)
        .agg(
            total_peaks=("pred_label", "size"),
            predicted_signal_peaks=("pred_label", "sum"),
            mean_prob_signal=("prob_signal", "mean"),
            max_prob_signal=("prob_signal", "max"),
        )
        .reset_index()
    )
    spectrum_summary["threshold"] = threshold
    spectrum_summary.to_csv(output_dir / "spectrum_summary.csv", index=False)

    if bool(output_cfg.get("write_dropped_peaks_report", False)):
        pd.DataFrame(dropped_peak_rows).to_csv(
            output_dir / str(output_cfg.get("dropped_peaks_report_name", "dropped_peaks.csv")),
            index=False,
        )

    if bool(output_cfg.get("write_filtered_mgf", True)):
        write_mgf(output_dir / "filtered_predictions.mgf", kept_spectra)

    summary = {
        "mgf_path": str(mgf_path),
        "checkpoint_path": str(checkpoint_path),
        "num_spectra": int(len(spectra)),
        "num_predicted_peak_rows": int(len(peak_predictions)),
        "threshold_for_binary_predictions": float(threshold),
        "filtered_mgf_mode": str(output_cfg.get("filtered_mgf_mode", "drop_below_threshold")),
        "output_dir": str(output_dir),
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference for the improved transformer denoising model on an MGF file."
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        help="Path to the inference YAML config.",
    )
    parser.add_argument(
        "--config",
        help="Path to the inference YAML config.",
    )
    args = parser.parse_args()
    args.config = args.config or args.config_path
    if not args.config:
        parser.error("the following arguments are required: --config")
    return args


def main() -> None:
    args = parse_args()
    summary = run_inference(args.config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.features.fdr_weights import FDRWeightConfig
from src.features.peak_features import PeakFeatureConfig
from src.features.spectrum_features import SpectrumFeatureConfig
from src.splitting.splitter import SplitConfig


@dataclass(slots=True)
class AppConfig:
    peak_features: PeakFeatureConfig
    spectrum_features: SpectrumFeatureConfig
    fdr: FDRWeightConfig
    split: SplitConfig


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    feature_cfg = raw.get("feature_engineering", {})
    fdr_cfg = raw.get("fdr", {})
    split_cfg = raw.get("split", {})

    peak_features = PeakFeatureConfig(
        use_log_intensity=bool(feature_cfg.get("use_log_intensity", True)),
        use_relative_intensity=bool(feature_cfg.get("use_relative_intensity", True)),
        use_mz_over_precursor=bool(feature_cfg.get("use_mz_over_precursor", True)),
        use_delta_to_precursor=bool(feature_cfg.get("use_delta_to_precursor", True)),
        use_delta_neighbors=bool(feature_cfg.get("use_delta_neighbors", True)),
        sort_by_mz=bool(feature_cfg.get("sort_by_mz", True)),
        eps=float(feature_cfg.get("eps", 1.0e-8)),
    )

    spectrum_features = SpectrumFeatureConfig(
        use_tic=bool(feature_cfg.get("use_tic", True)),
        use_num_peaks=bool(feature_cfg.get("use_num_peaks", True)),
    )

    fdr = FDRWeightConfig(
        enabled=bool(fdr_cfg.get("enabled", True)),
        clip_min=float(fdr_cfg.get("clip_min", 0.0)),
        clip_max=float(fdr_cfg.get("clip_max", 0.01)),
        weight_min=float(fdr_cfg.get("weight_min", 0.2)),
        mode=str(fdr_cfg.get("mode", "linear")),
    )

    split = SplitConfig(
        split_method=str(split_cfg.get("split_method", "PeakListFileName")),
        train_fraction=float(split_cfg.get("train_fraction", 0.70)),
        val_fraction=float(split_cfg.get("val_fraction", 0.15)),
        test_fraction=float(split_cfg.get("test_fraction", 0.15)),
        random_seed=int(split_cfg.get("random_seed", 42)),
    )

    return AppConfig(
        peak_features=peak_features,
        spectrum_features=spectrum_features,
        fdr=fdr,
        split=split,
    )
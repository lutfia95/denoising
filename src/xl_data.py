"""The XL feature contract, shared by splitting, training and MGF inference."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, Sampler
import yaml


PEAK_FEATURES = (
    "log_intensity", "relative_intensity", "mz_over_precursor",
    "delta_to_precursor", "delta_prev", "delta_next",
)
# v2 drops the two intensity duplicates of intensity_features and adds rank,
# local relative intensity and b/y complement evidence. Column 0 (log
# intensity) feeds only the intensity embedding, not the peak projection.
PEAK_FEATURES_V2 = (
    "log_intensity", "mz_over_precursor", "delta_to_precursor", "delta_prev", "delta_next",
    "intensity_rank", "local_relative_intensity", "complement_distance",
)
SPECTRUM_FEATURES = ("Charge", "exp m/z", "log_num_peaks", "log_tic")
PROTON_MASS = 1.007276466812
COMPLEMENT_MAX_DISTANCE = 5.0  # Da; farther partners all count as "no complement"
COMPLEMENT_SCALE = 0.01  # Da; resolution of the log-compressed complement distance


def feature_names(config):
    """All normalized peak columns, in make_features order."""
    return PEAK_FEATURES_V2 if config.peak_feature_set == "v2" else PEAK_FEATURES


def model_feature_names(config):
    """Columns that enter the peak projection (peak_input_dim of the model)."""
    names = feature_names(config)
    return names[1:] if config.peak_feature_set == "v2" else names


def complement_target(precursor_mz, charge):
    """Singly charged b + y m/z sum: neutral precursor mass plus two protons."""
    return (float(precursor_mz) - PROTON_MASS) * float(charge) + 2 * PROTON_MASS


def window_max(values, mz, half_width):
    """Maximum of values over mz +- half_width for sorted mz, via a sparse table."""
    n = len(values)
    lo = np.searchsorted(mz, mz - half_width, side="left")
    hi = np.searchsorted(mz, mz + half_width, side="right") - 1
    table = [values]
    while (1 << len(table)) <= n:
        prev, step = table[-1], 1 << (len(table) - 1)
        table.append(np.maximum(prev[:-step], prev[step:]))
    level = np.floor(np.log2(hi - lo + 1)).astype(int)
    out = np.empty(n, dtype=np.float64)
    for k in np.unique(level):
        rows = level == k
        out[rows] = np.maximum(table[k][lo[rows]], table[k][hi[rows] - (1 << k) + 1])
    return out


def complement_distance(mz, target):
    """Distance from each peak to its nearest other peak at target - mz (sorted mz)."""
    partner = target - mz
    pos = np.searchsorted(mz, partner)
    best = np.full(len(mz), np.inf)
    index = np.arange(len(mz))
    # Self-matches (mz near target/2) are excluded, so check two neighbours per side.
    for offset in (-2, -1, 0, 1):
        j = np.clip(pos + offset, 0, len(mz) - 1)
        distance = np.where(j == index, np.inf, np.abs(mz[j] - partner))
        best = np.minimum(best, distance)
    return np.minimum(best, COMPLEMENT_MAX_DISTANCE)


def augment_spectrum(mz, intensity, precursor_mz, rng, ppm_std=0.0, intensity_log_std=0.0):
    """Global ppm calibration shift (order preserving) and per-peak intensity jitter."""
    if ppm_std:
        factor = 1.0 + rng.normal(0.0, ppm_std) * 1e-6
        mz, precursor_mz = mz * factor, precursor_mz * factor
    if intensity_log_std:
        intensity = intensity * np.exp(rng.normal(0.0, intensity_log_std, len(intensity)))
    return mz, intensity, precursor_mz


def model_peak_arrays(peak, intensity, normalizer, config):
    """Split normalized make_features output into model peak and intensity inputs."""
    normalized = normalizer.peak(peak, config)
    peak_features = normalized[:, 1:] if config.peak_feature_set == "v2" else normalized
    intensity_features = np.column_stack([
        normalized[:, 0], np.sqrt(intensity / max(float(np.max(intensity)), 1e-12)),
    ]).astype(np.float32)
    return np.ascontiguousarray(peak_features), intensity_features


def arrow_rows(table):
    """Expose list columns as NumPy views instead of millions of Python floats."""
    columns = {}
    for name in table.schema.names:
        array = table.column(name)
        if isinstance(array, pa.ChunkedArray):
            array = array.chunk(0) if array.num_chunks == 1 else array.combine_chunks()
        if pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
            offsets = array.offsets.to_numpy(zero_copy_only=False)
            values = array.values.to_numpy(zero_copy_only=False)
            # Retain null lists as None so validation can reject them explicitly.
            valid = array.is_valid().to_numpy(zero_copy_only=False)
            columns[name] = [values[offsets[i]:offsets[i + 1]] if valid[i] else None
                             for i in range(len(array))]
        else:
            columns[name] = array.to_numpy(zero_copy_only=False)
    return [{name: values[i] for name, values in columns.items()} for i in range(table.num_rows)]


@dataclass
class XLFeatureConfig:
    # Absolute intensity is retained after log compression, never z-scored raw.
    histogram_bins: int = 64
    histogram_max_mz: float = 4000.0
    normalization_clip: float = 10.0
    previous_spectra: int = 2
    # None means the previous available annotated spectra, regardless of scan gap.
    max_scan_gap: int | None = None
    # v1 = the original six features (default, so older checkpoints load as-is).
    peak_feature_set: str = "v1"
    local_intensity_window: float = 50.0  # Da half width for local_relative_intensity

    def __post_init__(self):
        if self.histogram_bins < 1 or self.histogram_max_mz <= 0:
            raise ValueError("Invalid histogram settings")
        if self.previous_spectra not in (0, 1, 2):
            raise ValueError("previous_spectra must be 0, 1 or 2")
        if self.normalization_clip <= 0 or (self.max_scan_gap is not None and self.max_scan_gap < 1):
            raise ValueError("Invalid normalization_clip or max_scan_gap")
        if self.peak_feature_set not in ("v1", "v2"):
            raise ValueError("peak_feature_set must be v1 or v2")
        if self.local_intensity_window <= 0:
            raise ValueError("local_intensity_window must be positive")


def read_config(path):
    path = Path(path).resolve()
    seen = set()
    while True:
        if path in seen:
            raise ValueError("Circular config forwarding")
        seen.add(path)
        with path.open() as handle:
            config = yaml.safe_load(handle)
        if not isinstance(config, dict):
            raise ValueError("Config must be a mapping")
        if set(config) != {"extends"}:
            break
        path = (path.parent / config["extends"]).resolve()
    root = (path.parent / config.get("project_root", "..")).resolve()
    return config, root


def resolve_path(root, path):
    return (Path(root) / path).resolve()


def canonical_run(value):
    """Merge explicit recalibrated/export filename variants, preserving run identity."""
    name = str(value).replace("\\", "/").rsplit("/", 1)[-1].lower()
    name = re.sub(r"\.(mgf|raw|mzml|mzxml)(\.gz)?$", "", name)
    name = re.sub(r"^(recali|recalibrated|recal)[_-]+", "", name)
    return re.sub(r"[_-]+(recali|recalibrated|recal)$", "", name)


def prepare_arrays(mz, intensity, labels=None):
    mz = np.asarray(mz, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    if mz.ndim != 1 or intensity.ndim != 1 or mz.shape != intensity.shape or not mz.size:
        raise ValueError("Peak arrays must be nonempty, aligned 1D arrays")
    if not np.isfinite(mz).all() or not np.isfinite(intensity).all():
        raise ValueError("Nonfinite peak input")
    if (mz <= 0).any() or (intensity < 0).any():
        raise ValueError("m/z must be positive and intensity nonnegative")
    order = np.argsort(mz, kind="stable")
    target = None
    if labels is not None:
        target = np.asarray(labels)
        if target.shape != mz.shape or not np.isin(target, [0, 1]).all():
            raise ValueError("Labels must be binary and aligned with peaks")
        target = target[order].astype(np.float32)
    return mz[order], intensity[order], target, order


def make_features(mz, intensity, charge, precursor_mz, config):
    """Inputs are validated and sorted together before entering this function."""
    if not np.isfinite(precursor_mz) or precursor_mz <= 0 or charge < 1 or int(charge) != charge:
        raise ValueError("A positive precursor m/z and integer charge are required")
    mz = np.asarray(mz, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    relative = intensity / max(float(intensity.max()), 1e-12)
    gaps = np.diff(mz)
    # Gap distributions are heavy-tailed. Signed log compresses precursor delta.
    delta = precursor_mz - mz
    signed_delta = np.sign(delta) * np.log1p(np.abs(delta))
    if config.peak_feature_set == "v2":
        # Average ranks keep tied intensities identical; a single peak ranks 1.
        from scipy.stats import rankdata
        rank = (rankdata(intensity) - 1) / max(len(mz) - 1, 1) if len(mz) > 1 else np.ones(1)
        local = intensity / np.maximum(window_max(intensity, mz, config.local_intensity_window), 1e-12)
        complement = complement_distance(mz, complement_target(precursor_mz, charge))
        peak = np.column_stack([
            np.log1p(intensity), mz / precursor_mz, signed_delta,
            np.log1p(np.r_[0.0, gaps]), np.log1p(np.r_[gaps, 0.0]),
            rank, local, np.log1p(complement / COMPLEMENT_SCALE),
        ]).astype(np.float32)
    else:
        peak = np.column_stack([
            np.log1p(intensity), relative, mz / precursor_mz, signed_delta,
            np.log1p(np.r_[0.0, gaps]), np.log1p(np.r_[gaps, 0.0]),
        ]).astype(np.float32)
    spectrum = np.array([charge, precursor_mz, np.log1p(len(mz)),
                         np.log1p(intensity.sum())], dtype=np.float32)
    # Hellinger-style histogram: scale invariant; overflow remains in the last bin.
    bins = np.minimum((mz / config.histogram_max_mz * config.histogram_bins).astype(int),
                      config.histogram_bins - 1)
    hist = np.bincount(bins, weights=intensity, minlength=config.histogram_bins)
    hist = np.sqrt(hist / max(float(hist.sum()), 1e-12)).astype(np.float32)
    if not np.isfinite(peak).all() or not np.isfinite(spectrum).all() or not np.isfinite(hist).all():
        raise ValueError("Nonfinite derived features")
    return peak, spectrum, hist


@dataclass
class XLNormalizer:
    peak_mean: list
    peak_std: list
    spectrum_mean: list
    spectrum_std: list

    def peak(self, x, config):
        return np.clip((x - np.asarray(self.peak_mean)) / np.asarray(self.peak_std),
                       -config.normalization_clip, config.normalization_clip).astype(np.float32)

    def spectrum(self, x, config):
        return np.clip((x - np.asarray(self.spectrum_mean)) / np.asarray(self.spectrum_std),
                       -config.normalization_clip, config.normalization_clip).astype(np.float32)

    @classmethod
    def fit(cls, dataset):
        # Equal source mass, then equal spectrum mass within each source.
        # Float64 moments avoid the catastrophic cancellation of raw-intensity stats.
        total = np.zeros(len(feature_names(dataset.features)), dtype=np.float64)
        total2 = total.copy()
        weights = dataset.source_weights
        for i in range(len(dataset)):
            row = dataset.raw_row(i)
            peak, _, _ = make_features(row["mz_arr"], row["int_arr"], row["Charge"],
                                       row["exp m/z"], dataset.features)
            total += weights[i] * peak.mean(axis=0, dtype=np.float64)
            total2 += weights[i] * np.square(peak.astype(np.float64)).mean(axis=0)
            if (i + 1) % 10000 == 0:
                print(f"Normalizer: {i + 1}/{len(dataset)} spectra", flush=True)
        mean = total / weights.sum()
        std = np.sqrt(np.maximum(total2 / weights.sum() - mean ** 2, 1e-8))
        spec = np.stack(dataset.frame.spectrum_features).astype(np.float64)
        spec_mean = np.average(spec, weights=weights, axis=0)
        spec_std = np.sqrt(np.average((spec - spec_mean) ** 2, weights=weights, axis=0))
        return cls(mean.tolist(), std.tolist(), spec_mean.tolist(),
                   np.maximum(spec_std, 1e-4).tolist())


def previous_indices(frame, config):
    result = np.full((len(frame), config.previous_spectra), -1, dtype=np.int64)
    if not config.previous_spectra:
        return result
    # Histories are confined to this manifest and this physical run. Alternate
    # precursor charges from the same scan are never considered previous scans.
    for _, part in frame.groupby("run_id", sort=False):
        history = []
        for scan, same_scan in part.groupby("scan", sort=True):
            valid = [j for s, j in history if config.max_scan_gap is None or scan - s <= config.max_scan_gap]
            prev = valid[-config.previous_spectra:][::-1]
            for i in same_scan.index:
                result[i, :len(prev)] = prev
            history.append((scan, int(same_scan.index[0])))
            history = history[-config.previous_spectra:]
    return result


class XLSpectrumDataset(Dataset):
    def __init__(self, manifest, features, normalizer=None, training=False,
                 max_train_peaks=None, seed=42, cache_row_groups=2,
                 augment_ppm_std=0.0, augment_intensity_log_std=0.0):
        manifest = Path(manifest).resolve()
        self.frame = pd.read_parquet(manifest).reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"Empty manifest: {manifest}")
        self.path = manifest.parent / "spectra.parquet"
        self.features = features
        self.normalizer = normalizer
        self.training = training
        self.max_train_peaks = max_train_peaks
        if augment_ppm_std < 0 or augment_intensity_log_std < 0:
            raise ValueError("Augmentation standard deviations must be nonnegative")
        self.augment_ppm_std = float(augment_ppm_std)
        self.augment_intensity_log_std = float(augment_intensity_log_std)
        self.seed = seed
        self.epoch = 0
        self.cache_row_groups = cache_row_groups
        self._file = None
        self._cache = OrderedDict()
        self.previous = previous_indices(self.frame, features)
        self._spectrum = np.stack(self.frame.spectrum_features).astype(np.float32)
        self._histogram = np.stack(self.frame.histogram).astype(np.float32)
        counts = self.frame.source.value_counts()
        # A fixed dataset-level scale, not a per-batch denominator that would
        # cancel source balancing for batches drawn from one source.
        self.source_weights = self.frame.source.map(len(self.frame) / (len(counts) * counts)).to_numpy()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        state["_cache"] = OrderedDict()
        return state

    def __len__(self):
        return len(self.frame)

    def raw_row(self, index):
        item = self.frame.iloc[index]
        rg = int(item.row_group)
        if self._file is None:
            self._file = pq.ParquetFile(self.path)
        if rg not in self._cache:
            self._cache[rg] = arrow_rows(self._file.read_row_group(rg))
            if len(self._cache) > self.cache_row_groups:
                self._cache.popitem(last=False)
        self._cache.move_to_end(rg)
        return self._cache[rg][int(item.row_offset)]

    def __getitem__(self, index):
        row = self.raw_row(index)
        mz = np.asarray(row["mz_arr"], dtype=np.float64)
        intensity = np.asarray(row["int_arr"], dtype=np.float64)
        precursor, charge = float(row["exp m/z"]), row["Charge"]
        if self.training and (self.augment_ppm_std or self.augment_intensity_log_std):
            # Labels stay attached: a few-ppm shift does not change an annotation.
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index, 1]))
            mz, intensity, precursor = augment_spectrum(mz, intensity, precursor, rng,
                                                        self.augment_ppm_std, self.augment_intensity_log_std)
        peak, _, _ = make_features(mz, intensity, charge, precursor, self.features)
        peak_features, intensity_features = model_peak_arrays(peak, intensity, self.normalizer, self.features)
        target = np.asarray(row["annotation_mask"], dtype=np.float32)
        # Uniform sampling during training only. Features and summaries describe
        # the full spectrum; validation and inference score every original peak.
        if self.training and self.max_train_peaks and len(mz) > self.max_train_peaks:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
            keep = np.sort(rng.choice(len(mz), self.max_train_peaks, replace=False))
            mz, peak_features, intensity_features, target = (
                mz[keep], peak_features[keep], intensity_features[keep], target[keep])
        indices = [index, *self.previous[index].tolist()]
        context = np.zeros((len(indices), 4 + self.features.histogram_bins), dtype=np.float32)
        mask = np.ones(len(indices), dtype=bool)
        for k, j in enumerate(indices):
            if j >= 0:
                context[k] = np.r_[self.normalizer.spectrum(self._spectrum[j], self.features), self._histogram[j]]
                mask[k] = False
        return {
            "peak_features": peak_features,
            "mz": mz.astype(np.float32),
            "intensity_features": intensity_features,
            "delta_mz": (precursor - mz).astype(np.float32),
            "complement_mz": (complement_target(precursor, charge) - mz).astype(np.float32),
            "context": context, "context_mask": mask, "targets": target,
            "weight": float(self.frame.iloc[index].weight),
            "source_weight": float(self.source_weights[index]),
            "index": index,
        }


def xl_collate(samples):
    batch, length = len(samples), max(len(x["mz"]) for x in samples)
    output = {}
    for key in ("peak_features", "mz", "delta_mz", "complement_mz", "intensity_features", "targets"):
        shape = (batch, length, *samples[0][key].shape[1:])
        values = np.zeros(shape, dtype=np.float32)
        for i, sample in enumerate(samples):
            values[i, :len(sample[key])] = sample[key]
        output[key] = torch.from_numpy(values)
    output["padding_mask"] = torch.arange(length)[None, :] >= torch.tensor([len(x["mz"]) for x in samples])[:, None]
    for key in ("context", "context_mask"):
        output[key] = torch.from_numpy(np.stack([x[key] for x in samples]))
    for key in ("weight", "source_weight"):
        output[key] = torch.tensor([x[key] for x in samples], dtype=torch.float32)
    output["indices"] = torch.tensor([x["index"] for x in samples])
    return output


class XLBatchSampler(Sampler):
    """Shuffle row groups, then bucket nearby lengths; enforce padded peak budget."""
    def __init__(self, dataset, batch_size=4, peak_budget=16384, shuffle=True, seed=42):
        self.dataset, self.batch_size, self.peak_budget = dataset, batch_size, peak_budget
        self.shuffle, self.seed, self.epoch = shuffle, seed, 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        groups = list(self.dataset.frame.groupby("row_group", sort=False).indices.values())
        if self.shuffle:
            rng.shuffle(groups)
        lengths = self.dataset.frame.num_peaks.to_numpy()
        cap = self.dataset.max_train_peaks if self.dataset.training else None
        for group in groups:
            indices = list(group)
            if self.shuffle:
                rng.shuffle(indices)
            indices.sort(key=lambda j: min(lengths[j], cap) if cap else lengths[j])
            batches, batch, longest = [], [], 0
            for j in indices:
                size = min(lengths[j], cap) if cap else lengths[j]
                if batch and (len(batch) >= self.batch_size or max(longest, size) * (len(batch) + 1) > self.peak_budget):
                    batches.append(batch)
                    batch, longest = [], 0
                batch.append(int(j))
                longest = max(longest, size)
            if batch:
                batches.append(batch)
            if self.shuffle:
                rng.shuffle(batches)
            yield from batches

    def __len__(self):
        return sum(1 for _ in self)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

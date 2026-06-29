from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any
import pandas as pd

from src.processing.spectrum_processor import ProcessedSpectrum


@dataclass(slots=True)
class SplitConfig:
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    random_seed: int = 42
    split_method: str = "PeakListFileName"
    stratify_by_recali: bool = False
    length_weight: bool = False
    length_weight_eps: float = 1.0
    length_weight_min: float = 0.5
    length_weight_max: float = 2.0

    def validate(self) -> None:
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(total - 1.0) > 1.0e-8:
            raise ValueError(
                "train_fraction + val_fraction + test_fraction must sum to 1.0"
            )
        if self.train_fraction <= 0.0:
            raise ValueError("train_fraction must be > 0")
        if self.val_fraction <= 0.0:
            raise ValueError("val_fraction must be > 0")
        if self.test_fraction <= 0.0:
            raise ValueError("test_fraction must be > 0")

        allowed = {"PeakListFileName", "SearchID", "peak_list_file_name", "search_id"}
        if self.split_method not in allowed:
            raise ValueError(
                f"split_method must be one of {sorted(allowed)}, got {self.split_method!r}"
            )
        if self.length_weight_eps <= 0.0:
            raise ValueError("length_weight_eps must be > 0")
        if self.length_weight_min <= 0.0:
            raise ValueError("length_weight_min must be > 0")
        if self.length_weight_max < self.length_weight_min:
            raise ValueError(
                "length_weight_max must be greater than or equal to length_weight_min"
            )


@dataclass(slots=True)
class SplitResult:
    train: list[ProcessedSpectrum]
    val: list[ProcessedSpectrum]
    test: list[ProcessedSpectrum]
    group_to_split_df: pd.DataFrame
    summary_df: pd.DataFrame

class GroupedSpectrumSplitter:
    def __init__(self, config: SplitConfig) -> None:
        self.config = config
        self.config.validate()

    def _get_group_key(self, ps: ProcessedSpectrum) -> str | int:
        method = self.config.split_method

        if method in {"PeakListFileName", "peak_list_file_name"}:
            return ps.record.peak_list_file_name

        if method in {"SearchID", "search_id"}:
            return ps.record.search_id

        raise ValueError(f"Unsupported split_method: {method}")

    def split(self, spectra: list[ProcessedSpectrum]) -> SplitResult:
        if not spectra:
            raise ValueError("spectra list is empty")

        unique_groups = sorted({self._get_group_key(ps) for ps in spectra}, key=str)

        if len(unique_groups) < 3:
            raise ValueError(
                f"Need at least 3 unique groups for train/val/test split using "
                f"split_method={self.config.split_method!r}"
            )

        n_groups = len(unique_groups)
        n_train = max(1, int(round(n_groups * self.config.train_fraction)))
        n_val = max(1, int(round(n_groups * self.config.val_fraction)))
        n_test = n_groups - n_train - n_val

        if n_test < 1:
            n_test = 1
            if n_train > n_val and n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1

        rng = random.Random(self.config.random_seed)
        shuffled_groups = unique_groups[:]
        rng.shuffle(shuffled_groups)

        if self.config.stratify_by_recali and any(
            ps.record.recali is not None for ps in spectra
        ):
            train_groups, val_groups, test_groups = self._assign_recali_stratified_groups(
                spectra=spectra,
                groups=shuffled_groups,
                group_capacities=(n_train, n_val, n_test),
                rng=rng,
            )
        else:
            train_groups = set(shuffled_groups[:n_train])
            val_groups = set(shuffled_groups[n_train:n_train + n_val])
            test_groups = set(shuffled_groups[n_train + n_val:])

        if not train_groups or not val_groups or not test_groups:
            raise RuntimeError("One of the splits is empty after group assignment")

        train: list[ProcessedSpectrum] = []
        val: list[ProcessedSpectrum] = []
        test: list[ProcessedSpectrum] = []

        group_to_split_rows: list[dict[str, Any]] = []

        for group_key in shuffled_groups:
            if group_key in train_groups:
                split_name = "train"
            elif group_key in val_groups:
                split_name = "val"
            else:
                split_name = "test"

            group_to_split_rows.append(
                {
                    "split_method": self.config.split_method,
                    "group_key": group_key,
                    "split": split_name,
                }
            )

        for ps in spectra:
            group_key = self._get_group_key(ps)

            if group_key in train_groups:
                train.append(ps)
            elif group_key in val_groups:
                val.append(ps)
            elif group_key in test_groups:
                test.append(ps)
            else:
                raise RuntimeError(f"Group key {group_key!r} was not assigned to any split")

        group_to_split_df = pd.DataFrame(group_to_split_rows)
        summary_df = self._build_summary_df(
            train=train,
            val=val,
            test=test,
            group_to_split_df=group_to_split_df,
        )

        return SplitResult(
            train=train,
            val=val,
            test=test,
            group_to_split_df=group_to_split_df,
            summary_df=summary_df,
        )

    def _assign_recali_stratified_groups(
        self,
        spectra: list[ProcessedSpectrum],
        groups: list[str | int],
        group_capacities: tuple[int, int, int],
        rng: random.Random,
    ) -> tuple[set[str | int], set[str | int], set[str | int]]:
        """Balance recali row counts while preserving group boundaries."""
        split_names = ("train", "val", "test")
        fractions = {
            "train": self.config.train_fraction,
            "val": self.config.val_fraction,
            "test": self.config.test_fraction,
        }
        capacities = dict(zip(split_names, group_capacities))
        assignments: dict[str, set[str | int]] = {
            split_name: set() for split_name in split_names
        }
        counts = {
            split_name: {"total": 0, "true": 0, "false": 0}
            for split_name in split_names
        }
        group_counts = {
            group: {"total": 0, "true": 0, "false": 0}
            for group in groups
        }
        for ps in spectra:
            stats = group_counts[self._get_group_key(ps)]
            stats["total"] += 1
            if ps.record.recali is True:
                stats["true"] += 1
            elif ps.record.recali is False:
                stats["false"] += 1

        global_counts = {
            metric: sum(stats[metric] for stats in group_counts.values())
            for metric in ("total", "true", "false")
        }
        targets = {
            split_name: {
                metric: fractions[split_name] * global_counts[metric]
                for metric in global_counts
            }
            for split_name in split_names
        }

        # Randomization resolves exact ties reproducibly. Sorting large and strongly
        # single-class groups first gives the greedy allocator the hardest choices first.
        ordered_groups = groups[:]
        rng.shuffle(ordered_groups)
        ordered_groups.sort(
            key=lambda group: (
                max(group_counts[group]["true"], group_counts[group]["false"]),
                group_counts[group]["total"],
            ),
            reverse=True,
        )

        for group in ordered_groups:
            candidates = [
                split_name
                for split_name in split_names
                if len(assignments[split_name]) < capacities[split_name]
            ]
            rng.shuffle(candidates)
            chosen = min(
                candidates,
                key=lambda split_name: self._recali_assignment_cost(
                    candidate_split=split_name,
                    group_stats=group_counts[group],
                    current_counts=counts,
                    targets=targets,
                ),
            )
            assignments[chosen].add(group)
            for metric in ("total", "true", "false"):
                counts[chosen][metric] += group_counts[group][metric]

        return (
            assignments["train"],
            assignments["val"],
            assignments["test"],
        )

    @staticmethod
    def _recali_assignment_cost(
        candidate_split: str,
        group_stats: dict[str, int],
        current_counts: dict[str, dict[str, int]],
        targets: dict[str, dict[str, float]],
    ) -> float:
        cost = 0.0
        for split_name, split_counts in current_counts.items():
            for metric in ("total", "true", "false"):
                value = split_counts[metric]
                if split_name == candidate_split:
                    value += group_stats[metric]
                target = targets[split_name][metric]
                if target > 0.0:
                    cost += ((value - target) / target) ** 2
        return cost

    def write_split_parquets(
        self,
        split_result: SplitResult,
        output_dir: str | Path,
    ) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        train_df = self._processed_spectra_to_df(split_result.train, split_name="train")
        val_df = self._processed_spectra_to_df(split_result.val, split_name="val")
        test_df = self._processed_spectra_to_df(split_result.test, split_name="test")

        train_df.to_parquet(output_path / "train.parquet", index=False)
        val_df.to_parquet(output_path / "val.parquet", index=False)
        test_df.to_parquet(output_path / "test.parquet", index=False)

        split_result.summary_df.to_parquet(
            output_path / "split_summary.parquet",
            index=False,
        )
        split_result.group_to_split_df.to_parquet(
            output_path / "group_to_split.parquet",
            index=False,
        )

    def _build_summary_df(
        self,
        train: list[ProcessedSpectrum],
        val: list[ProcessedSpectrum],
        test: list[ProcessedSpectrum],
        group_to_split_df: pd.DataFrame,
    ) -> pd.DataFrame:
        rows = [
            self._build_one_summary_row("train", train, group_to_split_df),
            self._build_one_summary_row("val", val, group_to_split_df),
            self._build_one_summary_row("test", test, group_to_split_df),
        ]
        return pd.DataFrame(rows)

    def _build_one_summary_row(
        self,
        split_name: str,
        spectra: list[ProcessedSpectrum],
        group_to_split_df: pd.DataFrame,
    ) -> dict[str, Any]:
        total_rows = len(spectra)
        unique_groups = int((group_to_split_df["split"] == split_name).sum())

        unique_spectra_keys = {
            (ps.record.peak_list_file_name, ps.record.scan_id)
            for ps in spectra
        }

        return {
            "split": split_name,
            "split_method": self.config.split_method,
            "n_rows": total_rows,
            "n_unique_spectra": len(unique_spectra_keys),
            "n_unique_groups": unique_groups,
            "n_recali_true": sum(ps.record.recali is True for ps in spectra),
            "n_recali_false": sum(ps.record.recali is False for ps in spectra),
            "n_recali_missing": sum(ps.record.recali is None for ps in spectra),
            "recali_true_fraction": self._recali_true_fraction(spectra),
        }

    @staticmethod
    def _recali_true_fraction(spectra: list[ProcessedSpectrum]) -> float | None:
        n_true = sum(ps.record.recali is True for ps in spectra)
        n_false = sum(ps.record.recali is False for ps in spectra)
        n_known = n_true + n_false
        return None if n_known == 0 else n_true / n_known

    def _processed_spectra_to_df(
        self,
        spectra: list[ProcessedSpectrum],
        split_name: str,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []

        for ps in spectra:
            length_weight = self._compute_length_weight(ps)
            fdr_weight = float(ps.fdr_weight.weight)
            final_weight = fdr_weight * length_weight

            rows.append(
                {
                    "split": split_name, # train / test / val
                    "group_split_method": self.config.split_method, # PeakListFileName or SearchID
                    "group_split_key": self._get_group_key(ps),
                    "SearchID": ps.record.search_id,
                    "PeakListFileName": ps.record.peak_list_file_name,
                    "scan": ps.record.scan, # Scan number of the spectrum in the source fil
                    "ScanId": ps.record.scan_id, # Spectrum identifier used for tracking and uniqueness
                    "mz_arr": ps.record.mz_arr.tolist(),
                    "int_arr": ps.record.int_arr.tolist(),
                    "Charge": ps.record.charge,  #precursor charge state of the spectrums
                    "exp m/z": ps.record.precursor_mz, #experimental precursor m/z
                    "annotation_mask": (
                        None
                        if ps.record.annotation_mask is None
                        else ps.record.annotation_mask.tolist()
                    ),
                    "fdr": ps.record.fdr, #Original FDR
                    "recali": ps.record.recali,
                    "clipped_fdr": ps.fdr_weight.clipped_fdr, # FDR after clipping into the configured allowed range
                    "fdr_weight": fdr_weight, # Weight derived only from the configured FDR rule
                    "length_weight": length_weight, # Optional per-spectrum true/false balance factor
                    "weight": final_weight, # Final training weight after combining the active components
                    "num_peaks": ps.spectrum_features.num_peaks, # Number of peaks in the processed spectrum
                    "tic": ps.spectrum_features.tic, # Total ion current, which is the sum of spectrum' intensities
                    "peak_feature_mz": ps.peak_features.mz.tolist(), # Peak m/z values used in the peak feature matrix
                    "peak_feature_log_intensity": ( # log(1+intensity)
                        None
                        if ps.peak_features.log_intensity is None
                        else ps.peak_features.log_intensity.tolist()
                    ),
                    #peak intensities normalized by the maximum intensity within the same spectrum
                    "peak_feature_relative_intensity": (
                        None
                        if ps.peak_features.relative_intensity is None
                        else ps.peak_features.relative_intensity.tolist()
                    ),#peak m/z values scaled relative to the precursor m/z
                    "peak_feature_mz_over_precursor": (
                        None
                        if ps.peak_features.mz_over_precursor is None
                        else ps.peak_features.mz_over_precursor.tolist()
                    ),# Difference between precursor m/z and each peak m/z value
                    "peak_feature_delta_to_precursor": (
                        None
                        if ps.peak_features.delta_to_precursor is None
                        else ps.peak_features.delta_to_precursor.tolist()
                    ), ## m/z difference between each peak and the previous peak after sorting by m/z
                    "peak_feature_delta_prev": (
                        None
                        if ps.peak_features.delta_prev is None
                        else ps.peak_features.delta_prev.tolist()
                    ), # m/z difference between the next peak and the current peak after sorting by m/z
                    "peak_feature_delta_next": (
                        None
                        if ps.peak_features.delta_next is None
                        else ps.peak_features.delta_next.tolist()
                    ),
                }
            )

        return pd.DataFrame(rows)

    def _compute_length_weight(self, ps: ProcessedSpectrum) -> float:
        if not self.config.length_weight:
            return 1.0

        annotation_mask = ps.record.annotation_mask
        if annotation_mask is None:
            return 1.0

        n_true = int(annotation_mask.sum())
        n_false = int(annotation_mask.shape[0] - n_true)
        eps = float(self.config.length_weight_eps)
        raw_weight = (n_true + eps) / (n_false + eps)

        return float(
            min(
                max(raw_weight, self.config.length_weight_min),
                self.config.length_weight_max,
            )
        )

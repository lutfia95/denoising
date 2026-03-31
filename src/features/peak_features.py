from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.types import PeakFeatureSet, SpectrumRecord


@dataclass(slots=True)
class PeakFeatureConfig:
    use_log_intensity: bool = True
    use_relative_intensity: bool = True
    use_mz_over_precursor: bool = True
    use_delta_to_precursor: bool = True
    use_delta_neighbors: bool = True
    sort_by_mz: bool = True
    eps: float = 1.0e-8


class PeakFeatureComputer:
    def __init__(self, config: PeakFeatureConfig) -> None:
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        if self.config.eps <= 0.0:
            raise ValueError("eps must be > 0")

    def compute(self, record: SpectrumRecord) -> tuple[SpectrumRecord, PeakFeatureSet]:
        record.validate()

        processed_record = self._prepare_record(record)

        mz_arr = processed_record.mz_arr.astype(np.float32, copy=False)
        int_arr = processed_record.int_arr.astype(np.float32, copy=False)
        precursor_mz = float(processed_record.precursor_mz)

        log_intensity = None
        if self.config.use_log_intensity:
            clipped_int_arr = np.clip(int_arr, a_min=0.0, a_max=None)
            log_intensity = np.log1p(clipped_int_arr).astype(np.float32, copy=False)

        relative_intensity = None
        if self.config.use_relative_intensity:
            max_intensity = float(np.max(int_arr)) if int_arr.size > 0 else 0.0
            denom = max(max_intensity, self.config.eps)
            relative_intensity = (int_arr / denom).astype(np.float32, copy=False)

        mz_over_precursor = None
        if self.config.use_mz_over_precursor:
            denom = max(precursor_mz, self.config.eps)
            mz_over_precursor = (mz_arr / denom).astype(np.float32, copy=False)

        delta_to_precursor = None
        if self.config.use_delta_to_precursor:
            delta_to_precursor = (precursor_mz - mz_arr).astype(np.float32, copy=False)

        delta_prev = None
        delta_next = None
        if self.config.use_delta_neighbors:
            delta_prev, delta_next = self._compute_neighbor_deltas(mz_arr)

        feature_set = PeakFeatureSet(
            mz=mz_arr,
            log_intensity=log_intensity,
            relative_intensity=relative_intensity,
            mz_over_precursor=mz_over_precursor,
            delta_to_precursor=delta_to_precursor,
            delta_prev=delta_prev,
            delta_next=delta_next,
        )
        return processed_record, feature_set

    def _prepare_record(self, record: SpectrumRecord) -> SpectrumRecord:
        mz_arr = record.mz_arr.astype(np.float32, copy=True)
        int_arr = record.int_arr.astype(np.float32, copy=True)
        annotation_mask = None

        if record.annotation_mask is not None:
            annotation_mask = record.annotation_mask.copy()

        if self.config.sort_by_mz:
            order = np.argsort(mz_arr)
            mz_arr = mz_arr[order]
            int_arr = int_arr[order]
            if annotation_mask is not None:
                annotation_mask = annotation_mask[order]

        return SpectrumRecord(
            search_id=record.search_id,
            peak_list_file_name=record.peak_list_file_name,
            scan=record.scan,
            mz_arr=mz_arr,
            int_arr=int_arr,
            charge=record.charge,
            precursor_mz=float(record.precursor_mz),
            annotation_mask=annotation_mask,
            fdr=record.fdr,
            scan_id=record.scan_id,
        )

    def _compute_neighbor_deltas(self, mz_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        #after sorting peaks per spectrum! 
        delta_prev = np.zeros_like(mz_arr, dtype=np.float32)
        delta_next = np.zeros_like(mz_arr, dtype=np.float32)

        if mz_arr.size > 1:
            diffs = np.diff(mz_arr).astype(np.float32, copy=False)
            delta_prev[1:] = diffs
            delta_next[:-1] = diffs

        return delta_prev, delta_next
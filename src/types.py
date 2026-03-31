from __future__ import annotations

from dataclasses import dataclass
import numpy as np


#https://docs.python.org/3/library/dataclasses.html
# slots: memory usage can be lower
@dataclass(slots=True)
class SpectrumRecord:
    search_id: str | int
    peak_list_file_name: str
    scan: int
    mz_arr: np.ndarray
    int_arr: np.ndarray
    charge: int
    precursor_mz: float
    annotation_mask: np.ndarray | None
    fdr: float | None
    scan_id: str | int  # maybe useful later for augmentation / tracking

    def validate(self) -> None:
        if self.mz_arr.ndim != 1:
            raise ValueError("mz_arr must be 1D")
        if self.int_arr.ndim != 1:
            raise ValueError("int_arr must be 1D")
        if self.mz_arr.shape[0] != self.int_arr.shape[0]:
            raise ValueError("mz_arr and int_arr must have same length")

        if self.annotation_mask is not None:
            if self.annotation_mask.ndim != 1:
                raise ValueError("annotation_mask must be 1D")
            if self.annotation_mask.shape[0] != self.mz_arr.shape[0]:
                raise ValueError("annotation_mask length must match mz_arr")

    @property
    def num_peaks(self) -> int:
        # as number of peaks can be computed, we can use it! 
        return int(self.mz_arr.shape[0])
    @property
    def tic(self) -> float:
        return float(np.sum(self.int_arr, dtype=np.float64))

@dataclass(slots=True)
class PeakFeatureSet:
    mz: np.ndarray
    log_intensity: np.ndarray | None
    relative_intensity: np.ndarray | None
    mz_over_precursor: np.ndarray | None
    delta_to_precursor: np.ndarray | None
    delta_prev: np.ndarray | None
    delta_next: np.ndarray | None

    def as_matrix(self) -> np.ndarray:
        cols: list[np.ndarray] = [self.mz.astype(np.float32, copy=False)]

        for arr in (
            self.log_intensity,
            self.relative_intensity,
            self.mz_over_precursor,
            self.delta_to_precursor,
            self.delta_prev,
            self.delta_next,
        ):
            if arr is not None:
                cols.append(arr.astype(np.float32, copy=False))

        return np.stack(cols, axis=1).astype(np.float32, copy=False)


@dataclass(slots=True)
class SpectrumFeatureSet:
    charge: int
    precursor_mz: float
    num_peaks: int
    tic: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                float(self.charge), float(self.precursor_mz), float(self.num_peaks),
                float(self.tic),
            ],
            dtype=np.float32,
        )


@dataclass(slots=True)
class FDRWeightResult:
    raw_fdr: float | None
    #clipped_fdr=min(max(fdr,clip_min),clip_max)
    clipped_fdr: float | None # the FDR value after forcing it into an allowed range
    weight: float
    
@dataclass(slots=True)
class ProcessedSpectrum:
    #this should hold now all the data converted
    record: SpectrumRecord
    peak_features: PeakFeatureSet
    spectrum_features: SpectrumFeatureSet
    fdr_weight: FDRWeightResult
from __future__ import annotations

from dataclasses import dataclass
from src.types import SpectrumFeatureSet, SpectrumRecord


@dataclass(slots=True)
class SpectrumFeatureConfig:
    use_tic: bool = True
    use_num_peaks: bool = True


class SpectrumFeatureComputer:
    def __init__(self, config: SpectrumFeatureConfig) -> None:
        self.config = config

    def compute(self, record: SpectrumRecord) -> SpectrumFeatureSet:
        record.validate()

        num_peaks = record.num_peaks if self.config.use_num_peaks else 0
        tic = record.tic if self.config.use_tic else 0.0

        return SpectrumFeatureSet(
            charge=int(record.charge),
            precursor_mz=float(record.precursor_mz),
            num_peaks=int(num_peaks),
            tic=float(tic),
        )
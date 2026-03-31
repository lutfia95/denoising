from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from src.features.fdr_weights import FDRWeightComputer
from src.features.peak_features import PeakFeatureComputer
from src.features.spectrum_features import SpectrumFeatureComputer
from src.types import FDRWeightResult, PeakFeatureSet, SpectrumFeatureSet, SpectrumRecord


@dataclass(slots=True)
class ProcessedSpectrum:
    record: SpectrumRecord
    peak_features: PeakFeatureSet
    spectrum_features: SpectrumFeatureSet
    fdr_weight: FDRWeightResult


class SpectrumProcessor:
    def __init__(
        self,
        peak_feature_computer: PeakFeatureComputer,
        spectrum_feature_computer: SpectrumFeatureComputer,
        fdr_weight_computer: FDRWeightComputer,
    ) -> None:
        self.peak_feature_computer = peak_feature_computer
        self.spectrum_feature_computer = spectrum_feature_computer
        self.fdr_weight_computer = fdr_weight_computer

    def process_record(self, record: SpectrumRecord) -> ProcessedSpectrum:
        processed_record, peak_features = self.peak_feature_computer.compute(record)
        spectrum_features = self.spectrum_feature_computer.compute(processed_record)
        fdr_weight = self.fdr_weight_computer.compute(processed_record.fdr)

        return ProcessedSpectrum(
            record=processed_record,
            peak_features=peak_features,
            spectrum_features=spectrum_features,
            fdr_weight=fdr_weight,
        )

    def process_dataframe(self, df: pd.DataFrame) -> list[ProcessedSpectrum]:
        processed: list[ProcessedSpectrum] = []

        for _, row in df.iterrows():
            record = self.row_to_record(row)
            processed.append(self.process_record(record))

        return processed

    @staticmethod
    def row_to_record(row: pd.Series) -> SpectrumRecord:
        annotation_mask = row["annotation_mask"]
        if annotation_mask is not None:
            annotation_mask = SpectrumProcessor._to_bool_array(annotation_mask)

        fdr = row["fdr"]
        if pd.isna(fdr):
            fdr = None
        else:
            fdr = float(fdr)

        return SpectrumRecord(
            search_id=row["SearchID"],
            peak_list_file_name=str(row["PeakListFileName"]),
            scan=int(row["scan"]),
            mz_arr=SpectrumProcessor._to_float_array(row["mz_arr"]),
            int_arr=SpectrumProcessor._to_float_array(row["int_arr"]),
            charge=int(row["Charge"]),
            precursor_mz=float(row["exp m/z"]),
            annotation_mask=annotation_mask,
            fdr=fdr,
            scan_id=row["ScanId"],
        )

    @staticmethod
    def _to_float_array(values: object) -> "np.ndarray":
        import numpy as np

        return np.asarray(values, dtype=np.float32)

    @staticmethod
    def _to_bool_array(values: object) -> "np.ndarray":
        import numpy as np

        return np.asarray(values, dtype=bool)
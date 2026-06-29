import sys

import pandas as pd
from pathlib import Path
from dataclasses import asdict
from pprint import pprint
import numpy as np
sys.path.append(str(Path.cwd().parent))

from src.data_loader import ParquetDataLoader

from src.features.fdr_weights import FDRWeightComputer
from src.features.peak_features import PeakFeatureComputer
from src.features.spectrum_features import SpectrumFeatureComputer
from src.processing.spectrum_processor import SpectrumProcessor
from src.config import load_config, load_training_config
from src.splitting.splitter import GroupedSpectrumSplitter, SplitConfig
#from src.model.mlp import MLPConfig, MLPPeakClassifier
from src.training.train_mlp import train_mlp

from src.model.mlp import MLPPeakClassifier
import matplotlib.pyplot as plt



FDR_6pct_PATH = "../data/labeled_new_29962926_5pct.parquet"
#FDR_6pct_PATH = "../data/unique_FDR6pct_filtered.parquet"
# (55598, 10)
# ['SearchID', 'PeakListFileName', 'scan', 'mz_arr', 'int_arr', 'Charge', 'exp m/z', 'annotation_mask', 'fdr', 'ScanId']
#    split      split_method  n_rows  n_unique_spectra  n_unique_groups
# 0  train  PeakListFileName   38694             29884              162
# 1    val  PeakListFileName    9161              6992               35
# 2   test  PeakListFileName    7743              6703               34
FDR_6pct_LOADER = ParquetDataLoader(FDR_6pct_PATH)
FDR_6pct_DF = FDR_6pct_LOADER.load()
FDR_6pct_DF.columns
print(FDR_6pct_LOADER.counts())
print(FDR_6pct_DF.shape)
print(FDR_6pct_DF.columns.tolist())

CONFIG_PATH = Path.cwd().parent / "configs" / "config.yml"
APP_CONFIG = load_config(CONFIG_PATH)

peak_feature_computer = PeakFeatureComputer(APP_CONFIG.peak_features)
spectrum_feature_computer = SpectrumFeatureComputer(APP_CONFIG.spectrum_features)
fdr_weight_computer = FDRWeightComputer(APP_CONFIG.fdr)

processor = SpectrumProcessor(
    peak_feature_computer=peak_feature_computer,
    spectrum_feature_computer=spectrum_feature_computer,
    fdr_weight_computer=fdr_weight_computer,
)

processed_spectra = processor.process_dataframe(FDR_6pct_DF)

splitter = GroupedSpectrumSplitter(APP_CONFIG.split)
split_result = splitter.split(processed_spectra)

OUTPUT_DIR = Path.cwd().parent / "data" / "splits_5pct_filtered_new"
#OUTPUT_DIR = Path.cwd().parent / "data" / "splits_6pct_filtered"
splitter.write_split_parquets(split_result, OUTPUT_DIR)

split_summary = split_result.summary_df.copy()
split_summary.insert(
    3,
    "row_fraction",
    split_summary["n_rows"] / split_summary["n_rows"].sum(),
)

print("\nGenerated split summary")
print(
    split_summary.to_string(
        index=False,
        formatters={
            "row_fraction": lambda value: f"{value:.2%}",
            "recali_true_fraction": lambda value: f"{value:.2%}",
        },
    )
)
print(f"\nSplit parquet files written to: {OUTPUT_DIR.resolve()}")

# {'rows': 60127, 'columns': 11, 'duplicate_rows': 14538}
# (60127, 11)
# ['SearchID', 'PeakListFileName', 'scan', 'mz_arr', 'int_arr', 'Charge', 'exp m/z', 'annotation_mask', 'fdr', 'raw_raptor', 'recali']

# Generated split summary
# split     split_method  n_rows row_fraction  n_unique_spectra  n_unique_groups  n_recali_true  n_recali_false  n_recali_missing recali_true_fraction
# train PeakListFileName   41703       69.36%             29783              206          30837           10866                 0               73.94%
#   val PeakListFileName    9212       15.32%              7422               44           6837            2375                 0               74.22%
#  test PeakListFileName    9212       15.32%              8314               44           6837            2375                 0               74.22%

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



FDR_6pct_PATH = "../data/unique_FDR6pct.parquet"
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

print(split_result.summary_df)

OUTPUT_DIR = Path.cwd().parent / "data" / "splits_6pct"
splitter.write_split_parquets(split_result, OUTPUT_DIR)
import sys

import pandas as pd
from pathlib import Path
from dataclasses import asdict
from pprint import pprint

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

TRAIN_CONFIG_PATH = Path.cwd().parent / "configs" / "train_mlp.yml"
TRAIN_CONFIG = load_training_config(TRAIN_CONFIG_PATH)

pprint(asdict(TRAIN_CONFIG.model), sort_dicts=False)

model = MLPPeakClassifier(TRAIN_CONFIG.model)
print(model)

results = train_mlp(
    config=TRAIN_CONFIG,
    model=model,
)


print("Best epoch:", results["best_epoch"])
print("Validation metrics:", results["final_val_metrics"])
print("Test metrics:", results["final_test_metrics"])
print("Validation confusion matrix:", results["val_confusion_matrix"])
print("Test confusion matrix:", results["test_confusion_matrix"])
print("Artifacts saved to:", results["output_dir"])

results["history"].tail()

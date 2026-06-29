import sys

from pathlib import Path

sys.path.append(str(Path.cwd().parent))

from src.config import load_autogluon_training_config
from src.training.train_autogluon import train_autogluon


TRAIN_CONFIG_PATH = Path.cwd().parent / "configs" / "train_autogluon.yml"
TRAIN_CONFIG = load_autogluon_training_config(TRAIN_CONFIG_PATH)

results = train_autogluon(TRAIN_CONFIG)

print("Validation metrics:", results["val_metrics"])
print("Test metrics:", results["test_metrics"])
print("Validation confusion matrix:", results["val_confusion_matrix"])
print("Test confusion matrix:", results["test_confusion_matrix"])
print("Artifacts saved to:", results["output_dir"])

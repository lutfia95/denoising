import sys

from dataclasses import asdict
from pathlib import Path
from pprint import pprint

sys.path.append(str(Path.cwd().parent))

from src.config import load_transformer_training_config
from src.model.transformer import PeakTransformerClassifier
from src.training.train_transformer import train_transformer


TRAIN_CONFIG_PATH = Path.cwd().parent / "configs" / "train_transformer.yml"
TRAIN_CONFIG = load_transformer_training_config(TRAIN_CONFIG_PATH)

pprint(asdict(TRAIN_CONFIG.model), sort_dicts=False)

model = PeakTransformerClassifier(TRAIN_CONFIG.model)
print(model)

results = train_transformer(
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

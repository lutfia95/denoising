import sys

from dataclasses import asdict
from pathlib import Path
from pprint import pprint

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.config import load_transformer_imp_training_config
from src.model.transformer_imp import PeakTransformerImpClassifier
from src.training.train_transformer_imp import train_transformer_imp


TRAIN_CONFIG_PATH = ROOT / "configs" / "train_transformer_imp.yml"
TRAIN_CONFIG = load_transformer_imp_training_config(TRAIN_CONFIG_PATH)

pprint(asdict(TRAIN_CONFIG.model), sort_dicts=False)

model = PeakTransformerImpClassifier(TRAIN_CONFIG.model)
print(model)

results = train_transformer_imp(
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

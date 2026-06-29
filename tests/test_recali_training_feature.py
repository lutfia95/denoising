import unittest

import pandas as pd

from src.config import load_transformer_imp_training_config
from src.features.recali_label import build_recali_label
from src.model.transformer_imp import PeakTransformerImpClassifier


class RecaliTrainingFeatureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_transformer_imp_training_config(
            "configs/train_transformer_imp.yml"
        )

    def test_config_increases_model_spectrum_input_width(self) -> None:
        self.assertTrue(self.config.features.use_recali_label)
        self.assertEqual(self.config.features.recali_label_source_column, "recali")
        self.assertEqual(self.config.model.spectrum_input_dim, 8)

        model = PeakTransformerImpClassifier(self.config.model)
        self.assertEqual(model.spectrum_projection.in_features, 8)

    def test_binary_recali_encoding(self) -> None:
        recali_true_vector = build_recali_label(
            pd.Series({"recali": True}),
            enabled=True,
            source_column="recali",
        )
        recali_false_vector = build_recali_label(
            pd.Series({"recali": False}),
            enabled=True,
            source_column="recali",
        )

        self.assertEqual(recali_true_vector.shape, (1,))
        self.assertEqual(recali_false_vector.shape, (1,))
        self.assertEqual(float(recali_true_vector[-1]), 1.0)
        self.assertEqual(float(recali_false_vector[-1]), 0.0)


if __name__ == "__main__":
    unittest.main()

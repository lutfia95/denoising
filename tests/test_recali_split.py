from types import SimpleNamespace
import unittest

import pandas as pd

from src.processing.spectrum_processor import SpectrumProcessor
from src.splitting.splitter import GroupedSpectrumSplitter, SplitConfig


class RecaliSplitTest(unittest.TestCase):
    def test_processor_preserves_recali(self) -> None:
        row = pd.Series(
            {
                "SearchID": "search-1",
                "PeakListFileName": "file-1",
                "scan": 7,
                "mz_arr": [100.0],
                "int_arr": [10.0],
                "Charge": 2,
                "exp m/z": 500.0,
                "annotation_mask": [True],
                "fdr": 0.001,
                "recali": True,
            }
        )

        record = SpectrumProcessor.row_to_record(row)

        self.assertIs(record.recali, True)

    def test_grouped_split_balances_recali_ratio(self) -> None:
        spectra = [
            SimpleNamespace(
                record=SimpleNamespace(
                    peak_list_file_name=f"group-{index}",
                    search_id=index,
                    scan_id=index,
                    recali=index < 14,
                )
            )
            for index in range(20)
        ]
        splitter = GroupedSpectrumSplitter(
            SplitConfig(
                train_fraction=0.70,
                val_fraction=0.15,
                test_fraction=0.15,
                random_seed=42,
                split_method="PeakListFileName",
                stratify_by_recali=True,
            )
        )

        result = splitter.split(spectra)

        summary = result.summary_df.set_index("split")
        self.assertEqual(summary.loc["train", "n_unique_groups"], 14)
        self.assertEqual(summary.loc["val", "n_unique_groups"], 3)
        self.assertEqual(summary.loc["test", "n_unique_groups"], 3)
        self.assertTrue((summary["n_recali_true"] > 0).all())
        self.assertTrue((summary["n_recali_false"] > 0).all())
        self.assertLessEqual(
            float(summary["recali_true_fraction"].max())
            - float(summary["recali_true_fraction"].min()),
            0.05,
        )

        group_sets = [
            set(result.group_to_split_df.loc[
                result.group_to_split_df["split"] == split_name, "group_key"
            ])
            for split_name in ("train", "val", "test")
        ]
        self.assertTrue(group_sets[0].isdisjoint(group_sets[1]))
        self.assertTrue(group_sets[0].isdisjoint(group_sets[2]))
        self.assertTrue(group_sets[1].isdisjoint(group_sets[2]))


if __name__ == "__main__":
    unittest.main()

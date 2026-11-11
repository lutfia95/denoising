"""Exercise the XL contracts that can silently corrupt denoising experiments."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from split_data_xl import split_data
from src.inference.run_xl import build_sample, iter_mgf, run_inference
from src.model.xl import (MassAwarePeakAttention, PairwiseMassBias, XLConfig,
                          XLPeakClassifier, model_inputs)
from src.training.train_xl import sample_weights, select_threshold, train
from src.xl_data import (XLFeatureConfig, XLNormalizer, XLSpectrumDataset,
                         make_features, prepare_arrays, previous_indices, read_config, xl_collate)


def make_fixture(root):
    rng = np.random.default_rng(123)
    legacy, annotations = [], []
    for source in range(2):
        for project in range(12):
            for scan in range(4):
                # Unique peak fingerprints, unsorted input and a clear label rule.
                mz = rng.uniform(50, 1800, 17)
                intensity = rng.uniform(0, 100, 17)
                row = {"SearchID": f"project{source}_{project}", "annotation_mask": intensity > 50,
                       "fdr": -0.01 if source == 0 and project == 0 else 0.005}
                if source == 0:
                    row.update(PeakListFileName=f"run{source}_{project}.mgf", scan=scan,
                               mz_arr=mz, int_arr=intensity, Charge=3, **{"exp m/z": 600})
                    legacy.append(row)
                else:
                    row.update(dataset_id=f"PXD{project}", run_name=f"run{source}_{project}",
                               scan_number=scan, peak_mz=mz, peak_intensity=intensity,
                               precursor_charge=3.0, precursor_mz=600.0)
                    annotations.append(row)
    duplicate = deepcopy(legacy[0])
    duplicate["PeakListFileName"] = "recal_run0_0.raw"
    duplicate["SearchID"] = "alternate_search"
    duplicate["annotation_mask"] = ~duplicate["annotation_mask"]
    duplicate["fdr"] = 0.04
    legacy.append(duplicate)
    pd.DataFrame(legacy).to_parquet(root / "legacy.parquet")
    pd.DataFrame(annotations).to_parquet(root / "annotations.parquet")
    return {"data": {"sources": [{"name": "five", "schema": "legacy", "path": "legacy.parquet"},
                                  {"name": "one", "schema": "annotations", "path": "annotations.parquet"}],
                     "split_dir": "splits"},
            "split": {"group_by": "project", "fractions": [0.7, 0.15, 0.15], "seed": 42, "row_group_size": 8},
            "features": {"histogram_bins": 8, "previous_spectra": 2},
            "model": {"d_model": 24, "num_heads": 3, "num_layers": 2, "local_window": 4,
                      "global_slots": 2, "mz_frequencies": 4, "dropout": 0.0},
            "training": {"seed": 42, "device": "cpu", "cpu_num_threads": 1, "max_epochs": 1,
                         "batch_size": 4, "gradient_accumulation_steps": 2, "learning_rate": 0.003,
                         "max_train_peaks": 12, "num_workers": 0},
            "evaluation": {"max_sampled_peaks_per_source": None, "intensity_baseline": True},
            "output": {"output_dir": "model"}}


class XLTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_sort_label_alignment_and_validation(self):
        mz, intensity, target, order = prepare_arrays([300, 100, 200], [3, 1, 2], [1, 0, 1])
        np.testing.assert_array_equal(target, [0, 1, 1])
        np.testing.assert_array_equal(intensity, [1, 2, 3])
        np.testing.assert_array_equal(order, [1, 2, 0])
        feature = XLFeatureConfig()
        p, s, h = make_features(mz, intensity, 3, 600, feature)
        self.assertTrue(np.isfinite(p).all())
        self.assertAlmostEqual(float((h ** 2).sum()), 1.0, places=6)
        self.assertAlmostEqual(float(s[-1]), np.log1p(6), places=6)
        for arrays in [([1], [-1]), ([float("nan")], [1]), ([1, 2], [1])]:
            with self.assertRaises(ValueError):
                prepare_arrays(*arrays)
        with self.assertRaises(ValueError):
            prepare_arrays([1], [2], [2])

    def test_history_boundaries_and_gaps(self):
        frame = pd.DataFrame({"run_id": ["a", "a", "b", "a", "a"], "scan": [20, 10, 5, 20, 40]})
        history = previous_indices(frame, XLFeatureConfig(previous_spectra=2))
        np.testing.assert_array_equal(history[0], [1, -1])
        np.testing.assert_array_equal(history[3], [1, -1])
        np.testing.assert_array_equal(history[4], [0, 1])
        np.testing.assert_array_equal(history[2], [-1, -1])
        limited = previous_indices(frame, XLFeatureConfig(previous_spectra=2, max_scan_gap=15))
        np.testing.assert_array_equal(limited[4], [-1, -1])

    def test_large_default_and_legacy_config(self):
        config, _ = read_config("config/train_xl.yml")
        model_config = XLConfig(**config["model"])
        self.assertEqual((model_config.d_model, model_config.num_heads, model_config.num_layers,
                          model_config.global_slots), (256, 8, 6, 32))
        model = XLPeakClassifier(model_config)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 8232481)
        legacy = XLConfig.from_checkpoint({"d_model": 192, "num_heads": 6,
                                           "num_layers": 4, "global_slots": 16})
        self.assertFalse(legacy.use_pairwise_mass_bias)
        self.assertEqual(legacy.mass_attention_landmarks, 0)
        old = XLPeakClassifier(legacy)
        self.assertEqual(sum(p.numel() for p in old.parameters()), 3161473)
        self.assertFalse(any("mass_bias" in name for name in old.state_dict()))

    def test_pairwise_bias_physical_differences_and_gradients(self):
        config = XLConfig(d_model=24, num_heads=3, pairwise_mass_frequencies=4,
                          pairwise_mass_hidden_dim=8)
        bias = PairwiseMassBias(config)
        mz = torch.tensor([[100.0, 137.5, 200.0]])
        self.assertTrue(torch.equal(bias(mz, mz), torch.zeros(1, 3, 3, 3)))
        torch.nn.init.normal_(bias.network[-1].weight, std=0.1)
        before = bias(mz, mz)
        torch.testing.assert_close(before, bias(mz + 2000, mz + 2000))
        self.assertFalse(torch.allclose(before, before.transpose(-1, -2)))
        before.square().sum().backward()
        self.assertGreater(float(bias.network[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(bias.network[-1].weight.grad.abs().sum()), 0)

    def test_sparse_distant_peaks_and_duplicate_masking(self):
        config = XLConfig(d_model=4, num_heads=1, local_window=2, full_attention_max_peaks=0,
                          mass_attention_landmarks=1, dropout=0, gradient_checkpointing=False)
        attention = MassAwarePeakAttention(config).eval()
        with torch.no_grad():
            attention.q.weight.zero_()
            attention.q.bias.zero_()
            attention.kv.weight.zero_()
            attention.kv.bias.zero_()
            attention.kv.weight[4:].copy_(torch.eye(4))
            attention.out.weight.copy_(torch.eye(4))
            attention.out.bias.zero_()
        tokens = torch.zeros(1, 6, 4)
        tokens[0, 5, 0] = 9
        mz = torch.tensor([[100.0, 101, 200, 201, 300, 1000]])
        intensity = torch.tensor([[0.0, 0, 0, 0, 0, 1]])
        padding = torch.zeros(1, 6, dtype=torch.bool)
        result = attention.forward_peaks(tokens, mz, padding, intensity, shift=0)
        # Peak 0 sees local peaks 0/1 and distant landmark 5, with equal logits.
        self.assertAlmostEqual(float(result[0, 0, 0].detach()), 3.0, places=6)
        # Peak 4 already sees peak 5 locally: the landmark must not be counted twice.
        self.assertAlmostEqual(float(result[0, 4, 0].detach()), 4.5, places=6)
        for scores in (intensity, torch.zeros_like(intensity)):
            expected = attention.forward_peaks(tokens, mz, padding, scores, shift=1)
            padded = attention.forward_peaks(
                torch.nn.functional.pad(tokens, (0, 0, 0, 4), value=10000),
                torch.nn.functional.pad(mz, (0, 4), value=10000),
                torch.nn.functional.pad(padding, (0, 4), value=True),
                torch.nn.functional.pad(scores, (0, 4), value=10000), shift=1)
            torch.testing.assert_close(expected, padded[:, :6])
        # A learned preference for mass gaps changes attention even with Q=K=0.
        with torch.no_grad():
            network = attention.mass_bias.network
            network[0].weight.zero_()
            network[0].bias.zero_()
            network[0].weight[0, -2] = 1
            network[-1].weight.zero_()
            network[-1].weight[0, 0] = 5
        changed = attention.forward_peaks(tokens, mz, padding, intensity, shift=0)
        self.assertLess(float(changed[0, 0, 0].detach()), 3.0)

    def test_chunked_attention_and_checkpoint_gradients_match(self):
        torch.manual_seed(72)
        config = XLConfig(d_model=24, num_heads=3, num_layers=2, local_window=4,
                          full_attention_max_peaks=0, global_slots=2, context_input_dim=12,
                          mz_frequencies=4, dropout=0, pairwise_query_chunk_size=4,
                          mass_attention_landmarks=3, gradient_checkpointing=True)
        chunked = XLPeakClassifier(config).train()
        for block in chunked.blocks:
            torch.nn.init.normal_(block.local_attn.mass_bias.network[-1].weight, std=0.1)
        other = deepcopy(config)
        other.pairwise_query_chunk_size = 64
        other.gradient_checkpointing = False
        unchunked = XLPeakClassifier(other).train()
        unchunked.load_state_dict(chunked.state_dict())
        batch = {"peak_features": torch.randn(2, 13, 6), "mz": torch.rand(2, 13) * 1000,
                 "delta_mz": torch.rand(2, 13) * 1000, "intensity_features": torch.rand(2, 13, 2),
                 "context": torch.rand(2, 3, 12), "context_mask": torch.tensor([[False, True, True], [False, False, True]]),
                 "padding_mask": torch.tensor([[False] * 7 + [True] * 6, [False] * 13])}
        first, second = chunked(**batch), unchunked(**batch)
        torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-5)
        target = torch.rand_like(first)
        valid = ~batch["padding_mask"]
        torch.nn.functional.binary_cross_entropy_with_logits(first[valid], target[valid]).backward()
        torch.nn.functional.binary_cross_entropy_with_logits(second[valid], target[valid]).backward()
        for (name, a), (_, b) in zip(chunked.named_parameters(), unchunked.named_parameters()):
            if a.grad is not None:
                self.assertIsNotNone(b.grad, name)
                torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-4, msg=name)
        self.assertGreater(float(chunked.blocks[0].local_attn.mass_bias.network[0].weight.grad.abs().sum()), 0)

        # Query chunking also preserves full short-spectrum attention.
        attention = chunked.blocks[0].local_attn.eval()
        attention.full_limit = 32
        reference = deepcopy(attention)
        reference.chunk_size = 64
        with torch.no_grad():
            a = attention.forward_peaks(batch["peak_features"].new_zeros(2, 13, 24), batch["mz"],
                                        batch["padding_mask"], batch["intensity_features"][..., 1], 0)
            b = reference.forward_peaks(batch["peak_features"].new_zeros(2, 13, 24), batch["mz"],
                                        batch["padding_mask"], batch["intensity_features"][..., 1], 0)
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)

    def test_padding_masks_and_learning(self):
        torch.manual_seed(123)
        config = XLConfig(d_model=24, num_heads=3, num_layers=2, local_window=4,
                          global_slots=2, mz_frequencies=4, context_input_dim=12, dropout=0,
                          full_attention_max_peaks=0)
        model = XLPeakClassifier(config)
        batch = {"peak_features": torch.randn(2, 9, 6), "mz": torch.rand(2, 9) * 1000,
                 "delta_mz": torch.rand(2, 9) * 1000, "intensity_features": torch.rand(2, 9, 2),
                 "context": torch.rand(2, 3, 12), "context_mask": torch.tensor([[False, True, True], [False, False, True]]),
                 "padding_mask": torch.tensor([[False] * 5 + [True] * 4, [False] * 9])}
        model.eval()
        original = model(**batch)
        changed = {k: v.clone() for k, v in batch.items()}
        for key in ("peak_features", "mz", "delta_mz", "intensity_features"):
            changed[key][0, 5:] = 10000
        changed["context"][changed["context_mask"]] = 10000
        torch.testing.assert_close(model(**changed)[0, :5], original[0, :5], atol=1e-6, rtol=1e-6)
        short = {k: (v[:1, :5] if k not in ("context", "context_mask") else v[:1]) for k, v in batch.items()}
        torch.testing.assert_close(model(**short)[0], original[0, :5], atol=1e-6, rtol=1e-6)
        model.train()
        labels = batch["intensity_features"][..., 0] > 0.5
        valid = ~batch["padding_mask"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        first = float(torch.nn.functional.binary_cross_entropy_with_logits(model(**batch)[valid], labels[valid].float()).detach())
        for _ in range(20):
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(**batch)[valid], labels[valid].float())
            loss.backward()
            optimizer.step()
        self.assertLess(float(loss.detach()), first * 0.5)

        # The short-spectrum full-attention branch must obey the same masks.
        config.full_attention_max_peaks = 32
        full = XLPeakClassifier(config).eval()
        torch.testing.assert_close(full(**changed)[0, :5], full(**batch)[0, :5], atol=1e-6, rtol=1e-6)
        config.full_attention_max_peaks = 6
        mixed = XLPeakClassifier(config).eval()
        torch.testing.assert_close(mixed(**short)[0], mixed(**batch)[0, :5], atol=1e-6, rtol=1e-6)

    def test_nested_checkpoint_preserves_attention_dropout(self):
        torch.manual_seed(19)
        config = XLConfig(d_model=16, num_heads=2, num_layers=2, local_window=4,
                          full_attention_max_peaks=6, global_slots=2, context_input_dim=12,
                          mz_frequencies=4, dropout=0.15, pairwise_query_chunk_size=4,
                          mass_attention_landmarks=2, gradient_checkpointing=True)
        checkpointed = XLPeakClassifier(config).train()
        ordinary = deepcopy(checkpointed)
        ordinary.config.gradient_checkpointing = False
        for block in ordinary.blocks:
            block.local_attn.checkpoint_attention = False
        batch = {"peak_features": torch.randn(2, 9, 6), "mz": torch.rand(2, 9) * 1000,
                 "delta_mz": torch.rand(2, 9) * 1000, "intensity_features": torch.rand(2, 9, 2),
                 "context": torch.rand(2, 3, 12), "context_mask": torch.tensor([[False, True, True], [False, False, True]]),
                 "padding_mask": torch.tensor([[False] * 5 + [True] * 4, [False] * 9])}
        state = torch.get_rng_state()
        first = checkpointed(**batch)
        torch.set_rng_state(state)
        second = ordinary(**batch)
        torch.testing.assert_close(first, second)
        first.square().sum().backward()
        second.square().sum().backward()
        for (name, a), (_, b) in zip(checkpointed.named_parameters(), ordinary.named_parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, atol=1e-5, rtol=1e-4, msg=name)

    def test_threshold_has_source_recall_constraint(self):
        samples = {"one": (np.array([0.1, 0.9, 0.8]), np.array([0, 1, 1])),
                   "two": (np.array([0.2, 0.4, 0.3]), np.array([0, 1, 1]))}
        threshold = select_threshold(samples, target_recall=1.0)
        self.assertEqual(threshold, 0.3)
        for p, y in samples.values():
            self.assertEqual(float((p[y.astype(bool)] >= threshold).mean()), 1.0)

    def test_full_split_train_inference_and_resume(self):
        with tempfile.TemporaryDirectory(prefix="xl-test-") as folder:
            root = Path(folder)
            config = make_fixture(root)
            split_data(config, root)
            metadata = json.loads((root / "splits/split_metadata.json").read_text())
            self.assertEqual(metadata["audit"]["deduplicated_rows"], 1)
            self.assertEqual(metadata["audit"]["duplicate_label_disagreements"], 1)
            conflict = pd.read_parquet(root / "splits/annotation_conflict_examples.parquet")
            self.assertEqual(len(conflict), 1)
            self.assertEqual(int(conflict.iloc[0].disagreeing_peaks), 17)
            manifests = [pd.read_parquet(root / f"splits/{s}.parquet") for s in ("train", "val", "test")]
            for i, left in enumerate(manifests):
                self.assertEqual(set(left.source), {"one", "five"})
                for right in manifests[i + 1:]:
                    self.assertFalse(set(left.group_id) & set(right.group_id))
                    self.assertFalse(set(left.run_id) & set(right.run_id))
            train(config, root)
            checkpoint = torch.load(root / "model/best_model.pt", weights_only=False)
            self.assertEqual(checkpoint["format"], "xl-v2")
            self.assertTrue(checkpoint["threshold_tuned"])
            features = XLFeatureConfig(**checkpoint["feature_config"])
            normalizer = XLNormalizer(**checkpoint["normalizer"])
            dataset = XLSpectrumDataset(root / "splits/val.parquet", features, normalizer)
            weights = sample_weights(dataset, {"use_training_weights": False, "balance_sources": True})
            totals = pd.Series(weights).groupby(dataset.frame.source).sum()
            self.assertAlmostEqual(totals.iloc[0], totals.iloc[1], places=5)
            raw = dataset.raw_row(0)
            peaks = np.column_stack([raw["mz_arr"], raw["int_arr"]])
            params = [("PEPMASS", str(raw["exp m/z"])), ("CHARGE", f"{raw['Charge']}+"), ("SCANS", "1")]
            from collections import deque
            sample, _, _ = build_sample(peaks, params, features, normalizer, deque(maxlen=3), None)
            # Find a first-in-run sample to compare identical missing-history masks.
            first = np.flatnonzero((dataset.previous == -1).all(1))[0]
            raw = dataset.raw_row(int(first))
            peaks = np.column_stack([raw["mz_arr"], raw["int_arr"]])
            sample, _, _ = build_sample(peaks, params, features, normalizer, deque(maxlen=3), None)
            for key in ("peak_features", "mz", "delta_mz", "intensity_features", "context", "context_mask"):
                np.testing.assert_allclose(sample[key], dataset[int(first)][key], atol=1e-6)
            with (root / "example.mgf").open("w") as handle:
                from src.inference.run_xl import write_spectrum
                write_spectrum(handle, params, peaks[::-1])
            inference = {"input": {"mgf_paths": ["example.mgf"]},
                         "model": {"checkpoint_path": "model/best_model.pt"},
                         "output": {"output_dir": "inference", "write_peak_scores": True}}
            result = run_inference(inference, root)
            self.assertEqual(result[0]["input_peaks"], len(peaks))
            self.assertEqual(len(list(iter_mgf(root / "inference/example.xl.mgf"))), 1)
            scores = pd.read_csv(root / "inference/example.peaks.csv")
            np.testing.assert_allclose(scores.mz, peaks[::-1, 0])
            model = XLPeakClassifier(XLConfig(**checkpoint["model_config"]))
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            with torch.no_grad():
                expected = model(**model_inputs(xl_collate([sample]), "cpu")).sigmoid()[0].numpy()[::-1]
            np.testing.assert_allclose(scores.score, expected, atol=1e-6)
            # Original xl-v1 checkpoints lack the new mass-attention settings.
            legacy_values = dict(checkpoint["model_config"])
            for key in ("use_pairwise_mass_bias", "mass_attention_landmarks", "pairwise_mass_frequencies",
                        "pairwise_mass_hidden_dim", "pairwise_query_chunk_size"):
                legacy_values.pop(key)
            legacy_model = XLPeakClassifier(XLConfig.from_checkpoint(legacy_values)).eval()
            legacy_checkpoint = {**checkpoint, "format": "xl-v1", "model_config": legacy_values,
                                 "model_state_dict": legacy_model.state_dict()}
            torch.save(legacy_checkpoint, root / "model/legacy_model.pt")
            old_inference = deepcopy(inference)
            old_inference["model"]["checkpoint_path"] = "model/legacy_model.pt"
            old_inference["output"]["output_dir"] = "legacy_inference"
            run_inference(old_inference, root)
            legacy_scores = pd.read_csv(root / "legacy_inference/example.peaks.csv")
            with torch.no_grad():
                expected = legacy_model(**model_inputs(xl_collate([sample]), "cpu")).sigmoid()[0].numpy()[::-1]
            np.testing.assert_allclose(legacy_scores.score, expected, atol=1e-6)
            # No epochs remain, but loading all training state and final evaluation
            # must still work with the resume path added to the same config.
            config["training"]["resume_from_checkpoint"] = "model/last_checkpoint.pt"
            train(config, root)
            # Changing only targets cannot change input features or normalization.
            self.assertEqual(len(checkpoint["normalizer"]["peak_mean"]), 6)
            self.assertEqual(asdict(features)["previous_spectra"], 2)
            ablation = deepcopy(config)
            ablation["features"]["previous_spectra"] = 0
            ablation["training"]["resume_from_checkpoint"] = None
            ablation["output"]["output_dir"] = "model_no_history"
            train(ablation, root)
            without_history = torch.load(root / "model_no_history/best_model.pt", weights_only=False)
            self.assertEqual(without_history["model_config"]["max_context_tokens"], 1)


if __name__ == "__main__":
    unittest.main()

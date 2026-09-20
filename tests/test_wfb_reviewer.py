from __future__ import annotations

import json
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data import _window_delta_t
from src.wfb_review_data import ReviewDataset, audit_and_prepare, motion_from_positions
from src.wfb_review_models import CORE, SEQUENCE, build_review_model
from src.wfb_review_training import train_epoch
from scripts.check_wfb_gradients import check
from scripts.run_wfb_reviewer_experiment import parse_args, run
from scripts.summarize_wfb_reviewer import holm, paired_statistics, summarize
from scripts.evaluate_wfb_reviewer_robustness import PerturbedObservations


def fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    for si, split in enumerate(["train", "val", "test"]):
        rows = []
        for j in range(8):
            frames = np.array([200000 + j, 200001 + j, 200004 + j, 200005 + j, 200006 + j])
            times = frames / 2.5
            row = {"source": f"scene_{si}_{j % 2}", "agent_id": str(j), "start_index": 0,
                   "irregular_sample_id": 0}
            old_dt = np.r_[0.4, np.diff(times[:3].astype(np.float32))]
            for i in range(3):
                for name, value in {"x": .01 * j + .002 * i, "y": .005 * j - .003 * i,
                                    "frame": frames[i], "time": times[i], "dt": old_dt[i]}.items():
                    row[f"obs_{i}_{name}"] = value
            for i in range(2):
                for name, value in {"x": .01 * j + .002 * (i + 3), "y": .005 * j - .003 * (i + 3),
                                    "frame": frames[i + 3], "time": times[i + 3]}.items():
                    row[f"target_{i}_{name}"] = value
            rows.append(row)
        pd.DataFrame(rows).to_csv(root / f"windows_{split}.csv", index=False)


class ReviewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_large_timestamp_subtraction(self):
        times = np.array([200000, 200001, 200002], dtype=np.float64) / 2.5
        result = _window_delta_t(times, np.arange(3), .4, 1e-4)
        np.testing.assert_allclose(result, .4, atol=1e-7, rtol=0)
        self.assertGreater(np.max(np.abs(np.diff(times.astype(np.float32)) - .4)), .001)

    def test_motion_uses_actual_observed_intervals(self):
        xy = np.array([[[0, 0], [2, 0], [8, 0]]], dtype=float)
        out = motion_from_positions(xy, np.array([[0, 1, 2.]]))
        np.testing.assert_allclose(out[0, :, 2], [0, 2, 3])
        np.testing.assert_allclose(out[0, :, 4], [0, 0, .5])

    def test_all_models_forward_and_capacity(self):
        x, dt = torch.randn(2, 8, 2), torch.randn(2, 8)
        for name in CORE + SEQUENCE:
            model, metadata = build_review_model(name, obs_len=8, pred_len=12, feature_dim=2)
            output = model(x, dt)
            output = output[0] if isinstance(output, tuple) else output
            self.assertEqual(output.shape, (2, 12, 2))
            self.assertTrue(torch.isfinite(output).all())
            if name.endswith("_matched"):
                self.assertLess(abs(metadata["parameter_difference_pct"]), 1.1)
            output.sum().backward()

    def test_audit_cache_interventions_and_train_only_norm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root / "csv")
            report = audit_and_prepare(root / "csv", root / "cache", obs_len=3, pred_len=2,
                                       time_basis="frames", fps=2.5, prepare=True)
            self.assertEqual(report["total_windows"], 24)
            self.assertEqual(report["splits"]["train"]["legacy_float32_dt_match_rate"], 1)
            real = ReviewDataset(root / "cache", "test")
            constant = ReviewDataset(root / "cache", "test", mode="constant")
            shuffle = ReviewDataset(root / "cache", "test", mode="shuffled", seed=12)
            np.testing.assert_allclose(constant.dt, 0, atol=1e-6)
            np.testing.assert_allclose(shuffle.dt[:, 0], 0)
            np.testing.assert_allclose(np.sort(real.dt[:, 1:], 1), np.sort(shuffle.dt[:, 1:], 1))
            torch.testing.assert_close(real[2]["x"], constant[2]["x"])
            np.testing.assert_allclose(report["normalization"]["feature_mean"], np.load(root / "cache/train_x.npy").mean((0, 1)), atol=1e-7)
            noise = PerturbedObservations(real, "gaussian", 0)
            torch.testing.assert_close(noise[2]["x"], real[2]["x"])
            corrupted = PerturbedObservations(real, "missing", .5)
            torch.testing.assert_close(corrupted[2]["target"], real[2]["target"])
            torch.testing.assert_close(corrupted[2]["x"], corrupted[2]["x"])
            del real, constant, shuffle, noise, corrupted

    def test_track_leakage_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root / "csv")
            df = pd.read_csv(root / "csv/windows_train.csv")
            df.to_csv(root / "csv/windows_test.csv", index=False)
            with self.assertRaisesRegex(ValueError, "Track overlap"):
                audit_and_prepare(root / "csv", root / "cache", obs_len=3, pred_len=2, time_basis="frames", fps=2.5)

    def test_numerical_gradients(self):
        self.assertTrue(check().passed.all())

    def test_combined_zero_matches_standard_update(self):
        base, _ = build_review_model("wfb_real", obs_len=3, pred_len=2, feature_dim=2,
                                     hidden_dim=8, wave_dim=4, depth=2, dropout=0)
        batch = [{"x": torch.randn(2, 3, 2), "delta_t": torch.randn(2, 3), "target": torch.randn(2, 2, 2)}]
        standard, combined = copy.deepcopy(base), copy.deepcopy(base)
        for model, variant in [(standard, "standard"), (combined, "combined")]:
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            diagnostics = train_epoch(model, batch, optimizer, torch.device("cpu"), variant, 0, 1)
            self.assertEqual(diagnostics["correction_to_task_gradient_ratio"], 0)
        for key, value in standard.state_dict().items():
            torch.testing.assert_close(value, combined.state_dict()[key], rtol=0, atol=0)
        optimizer = torch.optim.AdamW(base.parameters(), lr=.001)
        diagnostics = train_epoch(base, batch, optimizer, torch.device("cpu"), "laplacian", .1, 1)
        self.assertGreater(diagnostics["weighted_correction_gradient_norm"], 0)

    def test_small_sample_statistics(self):
        r = paired_statistics([1, 1, 1, 1, 1], [.5, .6, .7, .8, .9])
        self.assertEqual(r["sign_flip_p"], .0625)
        self.assertLess(r["ci95_high"], 0)
        self.assertIsNone(paired_statistics([1], [.5])["ci95_low"])
        np.testing.assert_allclose(holm([.01, .04, .03]), [.03, .06, .06])

    def test_training_selection_resume_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root / "csv")
            audit_and_prepare(root / "csv", root / "cache", obs_len=3, pred_len=2, time_basis="frames", fps=2.5, prepare=True)
            for model in ["wfb_real", "wfb_shuffled", "ffn_dt", "siren_dt_matched"]:
                args = parse_args(["--cache_dir", str(root / "cache"), "--output_dir", str(root / "runs" / model),
                                   "--model", model, "--device", "cpu", "--seeds", "1", "2", "--epochs", "2",
                                   "--lrs", ".001", ".0001", "--hidden_dim", "16", "--wave_dim", "4", "--depth", "2",
                                   "--batch_size", "4", "--threads", "1", "--benchmark_seconds", ".001", "--benchmark_trials", "1",
                                   "--sine_omegas", "1", "3"])
                run(args)
                selection = json.loads((args.output_dir / "selection.json").read_text())
                val = pd.read_csv(args.output_dir / "validation_candidates.csv")
                self.assertEqual(selection["candidate"], val.groupby("candidate").best_val_ADE.mean().idxmin())
                result = pd.read_csv(args.output_dir / "results.csv")
                self.assertEqual(len(result), 2)
                self.assertTrue(np.allclose(result.RMSE**2, result.MSE))
                args.resume = True
                run(args)
            summarize(root / "runs")
            self.assertTrue((root / "runs/summary/ade_position.pdf").exists())
            self.assertEqual(len(pd.read_csv(root / "runs/summary/paired_ADE.csv")), 3)


if __name__ == "__main__":
    unittest.main()

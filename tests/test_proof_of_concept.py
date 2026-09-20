from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data import FEATURE_COLUMNS, TrajectoryWindowDataset
from src.metrics import metrics_from_totals, trajectory_metric_totals
from src.models import WFBTrajectoryPredictor, build_model, laplacian_penalty
from src.training import temporal_correction_grads


class ProofOfConceptTests(unittest.TestCase):
    def test_additive_metrics_are_exact_across_unequal_batches(self) -> None:
        pred = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 1.0]],
                [[2.0, 0.0], [0.0, 2.0]],
                [[3.0, 4.0], [1.0, 0.0]],
            ]
        )
        target = torch.zeros_like(pred)
        totals = {key: 0.0 for key in trajectory_metric_totals(pred[:1], target[:1])}
        for batch_pred, batch_target in [(pred[:2], target[:2]), (pred[2:], target[2:])]:
            for key, value in trajectory_metric_totals(batch_pred, batch_target).items():
                totals[key] += value
        metrics = metrics_from_totals(totals)
        expected_mse = float((pred - target).pow(2).mean())
        self.assertAlmostEqual(metrics["MSE"], expected_mse)
        self.assertAlmostEqual(metrics["RMSE"], expected_mse**0.5)
        self.assertAlmostEqual(metrics["ADE"], float(torch.linalg.norm(pred, dim=-1).mean()))

    def test_temporal_correction_matches_autograd_and_fixes_k(self) -> None:
        torch.manual_seed(3)
        model = WFBTrajectoryPredictor(3, 2, feature_dim=2, wave_dim=4, hidden_dim=8, depth=2)
        x = torch.randn(2, 3, 2)
        dt = torch.randn(2, 3)
        _, aux = model(x, dt)
        parameters, gradients = temporal_correction_grads(model, aux, lambda_laplacian=0.25)
        expected = torch.autograd.grad(0.25 * laplacian_penalty(aux), parameters)
        for actual, reference in zip(gradients, expected):
            torch.testing.assert_close(actual, reference)
        self.assertIsNone(model.wfb.k_raw.grad)

    def test_feed_forward_control_models_accept_expected_inputs(self) -> None:
        x = torch.randn(4, 3, 2)
        dt = torch.randn(4, 3)
        for name, include_dt in [("ffn", False), ("ffn", True), ("sine_ffn", True)]:
            model = build_model(
                name,
                obs_len=3,
                pred_len=2,
                hidden_dim=16,
                wave_dim=8,
                depth=3,
                dropout=0.0,
                feature_dim=2,
                include_delta_t=include_dt,
            )
            output = model(x, dt)
            self.assertEqual(output.shape, (4, 2, 2))
            self.assertTrue(torch.isfinite(output).all())

    def test_temporal_controls_are_deterministic_and_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            obs_len, pred_len = 3, 2
            row: dict[str, float | str] = {"source": "scene", "agent_id": "agent"}
            xy = [(0.0, 0.0), (1.0, 0.0), (3.0, 0.0)]
            dt = [1.0, 2.0, 3.0]
            for index in range(obs_len):
                values = [xy[index][0], xy[index][1], 9.0, 9.0, 9.0, 9.0]
                for name, value in zip(FEATURE_COLUMNS, values):
                    row[f"obs_{index}_{name}"] = value
                row[f"obs_{index}_dt"] = dt[index]
            for index in range(pred_len):
                row[f"target_{index}_x"] = 0.0
                row[f"target_{index}_y"] = 0.0
            csv_path = root / "windows.csv"
            pd.DataFrame([row]).to_csv(csv_path, index=False)
            normalization_path = root / "normalization.json"
            normalization_path.write_text(
                json.dumps(
                    {
                        "feature_mean": [0.0] * 6,
                        "feature_std": [1.0] * 6,
                        "dt_mean": [2.0],
                        "dt_std": [1.0],
                        "feature_columns": FEATURE_COLUMNS,
                    }
                ),
                encoding="utf-8",
            )

            shuffled = TrajectoryWindowDataset(
                csv_path, normalization_path, obs_len, pred_len, t_mode="shuffled", t_seed=17
            )
            torch.testing.assert_close(shuffled[0]["delta_t"], shuffled[0]["delta_t"])
            torch.testing.assert_close(
                shuffled[0]["delta_t_original_raw"], torch.tensor([1.0, 2.0, 3.0])
            )

            constant = TrajectoryWindowDataset(
                csv_path, normalization_path, obs_len, pred_len, t_mode="constant"
            )
            torch.testing.assert_close(constant[0]["delta_t"], torch.zeros(obs_len))

            recomputed = TrajectoryWindowDataset(
                csv_path,
                normalization_path,
                obs_len,
                pred_len,
                t_mode="constant",
                input_features="motion",
                motion_dt_policy="recompute",
            )
            self.assertFalse(torch.allclose(recomputed[0]["x"][:, 2:4], torch.full((obs_len, 2), 9.0)))


if __name__ == "__main__":
    unittest.main()

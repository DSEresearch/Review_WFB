"""Paired inference-only perturbation tests for validation-selected checkpoints."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.wfb_review_data import ReviewDataset, digest, dump, motion_from_positions
from src.wfb_review_models import build_review_model
from src.wfb_review_training import evaluate, loader


class PerturbedObservations(Dataset):
    def __init__(self, base, kind, level, seed=501):
        self.base, self.kind, self.level, self.seed = base, kind, level, seed

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + index)
        xy = self.base.x[index, :, :2].astype(np.float64).copy()
        dt = self.base.dt[index].astype(np.float64) * self.base.norm["dt_std"] + self.base.norm["dt_mean"]
        dt[0] = 0
        if self.kind == "gaussian":
            xy += rng.normal(0, self.level, xy.shape)
        elif self.kind == "missing":
            # Causal last-observation carry-forward; first observation always retained.
            missing = rng.random(len(xy)) < self.level
            for i in range(1, len(xy)):
                if missing[i]:
                    xy[i] = xy[i - 1]
        elif self.kind == "time_jitter":
            dt[1:] *= np.exp(rng.normal(-0.5 * self.level**2, self.level, len(dt) - 1))
        else:
            raise ValueError(self.kind)
        x = motion_from_positions(xy[None], dt[None])[0, :, :self.base.dims]
        standardized_dt = (dt - self.base.norm["dt_mean"]) / self.base.norm["dt_std"]
        standardized_dt[0] = 0
        return {"x": torch.from_numpy((x - self.base.mean) / self.base.std),
                "delta_t": torch.from_numpy(standardized_dt.astype(np.float32)),
                "target": torch.from_numpy(self.base.target[index].copy())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_dir", type=Path, required=True)
    p.add_argument("--cache_dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--noise_std", nargs="+", type=float, default=[0.005, 0.01, 0.02])
    p.add_argument("--missing_rates", nargs="+", type=float, default=[0.1, 0.25, 0.5])
    p.add_argument("--jitter_std", nargs="+", type=float, default=[0.05, 0.1, 0.25])
    args = p.parse_args()
    if any(v < 0 for v in args.noise_std + args.jitter_std) or any(v < 0 or v >= 1 for v in args.missing_rates):
        p.error("Noise/jitter must be nonnegative; missing rates must be in [0, 1)")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    cache = json.loads((args.cache_dir / "cache_manifest.json").read_text())
    groups = pd.read_csv(args.cache_dir / "test_groups.csv", dtype=str)
    conditions = [("clean", 0)] + [("gaussian", v) for v in args.noise_std] + [("missing", v) for v in args.missing_rates] + [("time_jitter", v) for v in args.jitter_std]
    results = []
    output = args.results_dir / "robustness"
    output.mkdir(exist_ok=True)
    for file in sorted(args.results_dir.glob("*/results.csv")):
        config = json.loads((file.parent / "run_config.json").read_text())
        if config["model"] in {"wfb_shuffled", "wfb_constant"}:
            continue
        if config["cache_manifest_sha256"] != digest(args.cache_dir / "cache_manifest.json"):
            raise ValueError("Cache differs from the training cache")
        options = {k: config[k] for k in ["hidden_dim", "wave_dim", "depth", "dropout", "decoder", "wave_ablation", "fourier_scale", "sine_omega_0", "layers"]}
        options.update(obs_len=cache["obs_len"], pred_len=cache["pred_len"], feature_dim=2 if config["features"] == "position" else 6)
        base = ReviewDataset(args.cache_dir, "test", config["features"])
        for run in pd.read_csv(file).itertuples():
            options.update(fourier_scale=run.fourier_scale, sine_omega_0=run.sine_omega_0)
            model, _ = build_review_model(config["model"], **options)
            checkpoint = file.parent / "candidates" / f"c{int(run.candidate):03d}" / f"seed_{int(run.seed)}" / "best.pt"
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            model.to(device)
            for kind, level in conditions:
                dataset = base if kind == "clean" else PerturbedObservations(base, kind, level)
                metrics, track_metrics = evaluate(model, loader(dataset, args.batch_size, False, 0), device, groups)
                destination = output / file.parent.name / f"seed_{int(run.seed)}"
                destination.mkdir(parents=True, exist_ok=True)
                track_metrics.to_csv(destination / f"{kind}_{level:g}_tracks.csv", index=False)
                results.append({"experiment": file.parent.name, "seed": int(run.seed), "condition": kind,
                                "level": level, "coordinate_unit": cache["coordinate_unit"], **metrics})
                print(file.parent.name, run.seed, kind, level, metrics["ADE"], flush=True)
    if not results:
        raise FileNotFoundError("No eligible selected checkpoints")
    table = pd.DataFrame(results)
    table.to_csv(output / "results.csv", index=False)
    table.groupby(["experiment", "condition", "level"])[["ADE", "FDE", "MSE", "RMSE"]].agg(["mean", "std", "count"]).to_csv(output / "summary.csv")
    dump(output / "protocol.json", {"noise_units": cache["coordinate_unit"],
         "missing": "within-window causal carry-forward, not removal of timestamps or a new prediction horizon",
         "time_jitter": "positive multiplicative interval noise; mean-one lognormal",
         "motion": "derivatives recomputed from corrupted observations and intervals",
         "pairing": "same deterministic perturbations for each window across all seeds and models",
         "selection": "no retraining or hyperparameter selection using these perturbed test results"})


if __name__ == "__main__":
    main()

"""Train candidates on train/validation; test only the selected hyperparameters."""
from __future__ import annotations

import argparse
import itertools
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import set_seed
from src.wfb_review_data import ReviewDataset, digest, dump
from src.wfb_review_models import CORE, SEQUENCE, build_review_model
from src.wfb_review_training import benchmark, evaluate, loader, train_epoch


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--model", choices=CORE + SEQUENCE, required=True)
    p.add_argument("--features", choices=["position", "motion"], default="position")
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--lrs", nargs="+", type=float, default=[1e-4, 3e-4, 1e-3])
    p.add_argument("--lambdas", nargs="+", type=float, default=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3])
    p.add_argument("--variant", choices=["standard", "combined", "laplacian"], default="standard")
    p.add_argument("--decoder", choices=["mlp", "linear"], default="mlp")
    p.add_argument("--wave_ablation", choices=["full", "without_A", "without_k", "without_omega", "without_theta"], default="full")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--wave_dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1)
    p.add_argument("--fourier_scale", type=float, default=1)
    p.add_argument("--sine_omega_0", type=float, default=30)
    p.add_argument("--fourier_scales", nargs="+", type=float, help="Validation-only scale candidates for Fourier models")
    p.add_argument("--sine_omegas", nargs="+", type=float, help="Validation-only omega_0 candidates for SIREN-style model")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--benchmark_seconds", type=float, default=1)
    p.add_argument("--benchmark_trials", type=int, default=5)
    p.add_argument("--resume", action="store_true")
    return p.parse_args(argv)


def run(args):
    for key in ("epochs", "patience", "batch_size", "threads", "benchmark_trials"):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    if len(set(args.seeds)) != len(args.seeds) or any(lr <= 0 for lr in args.lrs):
        raise ValueError("Seeds must be unique and learning rates positive")
    if args.benchmark_seconds <= 0 or any(lam < 0 for lam in args.lambdas):
        raise ValueError("Benchmark duration must be positive; lambdas must be nonnegative")
    if any(value <= 0 for value in (args.fourier_scales or [args.fourier_scale]) + (args.sine_omegas or [args.sine_omega_0])):
        raise ValueError("Periodic scale/frequency candidates must be positive")
    if not args.model.startswith("wfb") and (args.variant != "standard" or args.decoder != "mlp" or args.wave_ablation != "full"):
        raise ValueError("WFB update, decoder and wave controls require a WFB model")
    if args.wave_ablation != "full" and args.variant != "standard":
        raise ValueError("Wave-parameter ablations use standard gradients; removed parameters have no correction gradient")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cache = json.loads((args.cache_dir / "cache_manifest.json").read_text())
    for filename, expected in cache["cache_sha256"].items():
        if digest(args.cache_dir / filename) != expected:
            raise ValueError(f"Prepared cache changed after audit: {filename}")
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in {"resume", "device"}}
    config["cache_manifest_sha256"] = digest(args.cache_dir / "cache_manifest.json")
    source_paths = [Path(__file__), *[Path(__file__).resolve().parents[1] / "src" / f for f in (
        "wfb_review_data.py", "wfb_review_models.py", "wfb_review_training.py", "models.py", "training.py", "sequence_models.py")]]
    config["source_sha256"] = {p.name: digest(p) for p in source_paths}
    config_path = root / "run_config.json"
    if config_path.exists():
        if not args.resume or json.loads(config_path.read_text()) != config:
            raise ValueError("Existing output has different configuration/source, or --resume was not provided; use a new output directory")
        if (root / "results.csv").exists():
            print(f"Already complete: {root}", flush=True)
            return
    else:
        dump(config_path, config)
    dump(root / "environment.json", {"python": sys.version, "torch": torch.__version__,
         "platform": platform.platform(), "device": str(device), "cuda_runtime": torch.version.cuda,
         "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None})
    mode = args.model.removeprefix("wfb_") if args.model.startswith("wfb_") else "real"
    # Identical interval interventions across model seeds and learning-rate candidates.
    datasets = {split: ReviewDataset(args.cache_dir, split, args.features, mode, seed=9100 + i)
                for i, split in enumerate(["train", "val"])}
    model_options = {k: getattr(args, k) for k in ["hidden_dim", "wave_dim", "depth", "dropout", "decoder", "wave_ablation", "fourier_scale", "sine_omega_0", "layers"]}
    model_options.update(obs_len=cache["obs_len"], pred_len=cache["pred_len"], feature_dim=2 if args.features == "position" else 6)
    lambdas = [0.0] if args.variant == "standard" else sorted(set(([0.0] if args.variant == "combined" else []) + args.lambdas))
    scales = sorted(set(args.fourier_scales or [args.fourier_scale])) if "fourier" in args.model or args.model == "rff_dt_matched" else [args.fourier_scale]
    omegas = sorted(set(args.sine_omegas or [args.sine_omega_0])) if args.model == "siren_dt_matched" else [args.sine_omega_0]
    candidates = list(itertools.product(sorted(set(args.lrs)), lambdas, scales, omegas))
    candidate_rows = []
    for candidate_index, (lr, lam, scale, omega) in enumerate(candidates):
        model_options.update(fourier_scale=scale, sine_omega_0=omega)
        for seed in args.seeds:
            run_dir = root / "candidates" / f"c{candidate_index:03d}" / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            metric_path = run_dir / "validation.json"
            if args.resume and metric_path.exists() and (run_dir / "best.pt").exists():
                candidate_rows.append(json.loads(metric_path.read_text()))
                continue
            set_seed(seed)
            torch.backends.cudnn.benchmark = False
            model, architecture = build_review_model(args.model, **model_options)
            model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
            train_loader = loader(datasets["train"], args.batch_size, True, seed, args.workers, device.type == "cuda")
            val_loader = loader(datasets["val"], args.batch_size, False, seed, args.workers, device.type == "cuda")
            best, stale, history = float("inf"), 0, []
            start = time.perf_counter()
            for epoch in range(1, args.epochs + 1):
                diagnostics = train_epoch(model, train_loader, optimizer, device, args.variant, lam, args.grad_clip)
                val, _ = evaluate(model, val_loader, device)
                history.append({"epoch": epoch, **diagnostics, **{f"val_{k}": v for k, v in val.items()}})
                if val["ADE"] < best:
                    best, best_epoch, stale = val["ADE"], epoch, 0
                    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    torch.save(state, run_dir / "best.pt")
                else:
                    stale += 1
                pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
                print(f"{args.model}/{args.features} seed={seed} lr={lr:g} lambda={lam:g} epoch={epoch} val_ADE={val['ADE']:.6g}", flush=True)
                if stale >= args.patience:
                    break
            row = {"candidate": candidate_index, "seed": seed, "lr": lr, "lambda": lam,
                   "fourier_scale": scale, "sine_omega_0": omega,
                   "best_val_ADE": best, "best_epoch": best_epoch, "epochs_trained": len(history),
                   "training_seconds": time.perf_counter() - start, **architecture}
            dump(metric_path, row)
            candidate_rows.append(row)
            del model, optimizer, train_loader, val_loader
    tuning = pd.DataFrame(candidate_rows)
    tuning.to_csv(root / "validation_candidates.csv", index=False)
    ranking = tuning.groupby("candidate", as_index=False).agg(mean_val_ADE=("best_val_ADE", "mean"), seeds=("seed", "size"))
    if not (ranking.seeds == len(args.seeds)).all():
        raise ValueError("Incomplete candidate seed grid")
    selected = int(ranking.sort_values(["mean_val_ADE", "candidate"]).iloc[0].candidate)
    dump(root / "selection.json", {"candidate": selected, "lr": candidates[selected][0],
         "lambda": candidates[selected][1], "fourier_scale": candidates[selected][2],
         "sine_omega_0": candidates[selected][3], "criterion": "mean validation ADE across the same seeds; test never used",
         "candidate_count": len(candidates), "seeds": args.seeds})
    # Only after validation selection is frozen do we access the test arrays.
    test = ReviewDataset(args.cache_dir, "test", args.features, mode, seed=9102)
    test_loader = loader(test, args.batch_size, False, 0, args.workers, device.type == "cuda")
    groups = pd.read_csv(args.cache_dir / "test_groups.csv", dtype=str)
    model_options.update(fourier_scale=candidates[selected][2], sine_omega_0=candidates[selected][3])
    rows = []
    for seed in args.seeds:
        run_dir = root / "candidates" / f"c{selected:03d}" / f"seed_{seed}"
        model, architecture = build_review_model(args.model, **model_options)
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True))
        model.to(device)
        metrics, grouped = evaluate(model, test_loader, device, groups)
        selected_dir = root / f"seed_{seed}"
        selected_dir.mkdir(exist_ok=True)
        grouped.to_csv(selected_dir / "test_track_metrics.csv", index=False)
        pd.read_csv(run_dir / "history.csv").to_csv(selected_dir / "history.csv", index=False)
        timing = benchmark(model, next(iter(test_loader)), device, args.benchmark_seconds, args.benchmark_trials)
        train_row = tuning[(tuning.candidate == selected) & (tuning.seed == seed)].iloc[0].to_dict()
        for field in ["candidate", "seed", "best_epoch", "epochs_trained", "parameters", "reference_parameters", "effective_hidden_dim"]:
            train_row[field] = int(train_row[field])
        row = {"model": args.model, "features": args.features, "variant": args.variant, "decoder": args.decoder,
               "wave_ablation": args.wave_ablation, **train_row, **metrics, **timing,
               "coordinate_unit": cache["coordinate_unit"], "checkpoint": str(run_dir / "best.pt"),
               "cache_manifest_sha256": config["cache_manifest_sha256"]}
        dump(selected_dir / "metrics.json", row)
        rows.append(row)
        del model
    pd.DataFrame(rows).to_csv(root / "results.csv", index=False)
    print(pd.DataFrame(rows)[["model", "seed", "ADE", "FDE", "parameters", "latency_mean_ms"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    run(parse_args())

"""Training, diagnostics and evaluation shared by the reviewer experiments."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .models import WFBTrajectoryPredictor
from .training import forward_model, temporal_correction_grads


def loader(dataset, batch_size, shuffle, seed, workers=0, cuda=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      persistent_workers=workers > 0, pin_memory=cuda,
                      generator=torch.Generator().manual_seed(seed))


def train_epoch(model, batches, optimizer, device, variant, weight, clip):
    model.train()
    totals = torch.zeros(5, device=device, dtype=torch.float64)
    diagnostic_batches = 0
    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        pred, target, aux = forward_model(model, batch, device)
        loss = (pred - target).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        correction = None
        if isinstance(model, WFBTrajectoryPredictor) and variant != "standard":
            parameters, correction = temporal_correction_grads(model, aux, weight)
        loss.backward()
        if correction is not None:
            task = torch.cat([p.grad.flatten() for p in parameters])
            corr = torch.cat([g.detach().flatten() for g in correction])
            task_norm, correction_norm = task.norm(), corr.norm()
            totals[2] += task_norm
            totals[3] += correction_norm
            totals[4] += correction_norm / task_norm.clamp_min(1e-12)
            diagnostic_batches += 1
            for parameter, gradient in zip(parameters, correction):
                if variant == "laplacian":
                    parameter.grad = gradient.detach().clone()
                else:
                    parameter.grad.add_(gradient.detach())
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
        optimizer.step()
        totals[0] += loss.detach() * len(target)
        totals[1] += len(target)
    values = totals.cpu().tolist()
    return {"train_MSE": values[0] / values[1],
            "temporal_task_gradient_norm": values[2] / max(1, diagnostic_batches),
            "weighted_correction_gradient_norm": values[3] / max(1, diagnostic_batches),
            "correction_to_task_gradient_ratio": values[4] / max(1, diagnostic_batches)}


@torch.inference_mode()
def evaluate(model, batches, device, groups=None):
    model.eval()
    errors = []
    for batch in batches:
        pred, target, _ = forward_model(model, batch, device)
        diff = (pred - target).double()
        distances = torch.linalg.vector_norm(diff, dim=-1)
        errors.append(torch.stack([distances.mean(1), distances[:, -1], diff.square().mean((1, 2))], 1).cpu().numpy())
    errors = np.concatenate(errors)
    if not np.isfinite(errors).all():
        raise FloatingPointError("Nonfinite evaluation predictions")
    ade, fde, mse = errors.mean(0)
    metrics = {"ADE": float(ade), "FDE": float(fde), "MSE": float(mse), "RMSE": float(np.sqrt(mse))}
    grouped = None
    if groups is not None:
        if len(groups) != len(errors):
            raise ValueError("Prediction/group alignment mismatch")
        per_window = groups.reset_index(drop=True).copy()
        per_window[["ADE", "FDE", "MSE"]] = errors
        grouped = per_window.groupby(["source", "agent_id"], sort=True).agg(
            ADE=("ADE", "mean"), FDE=("FDE", "mean"), MSE=("MSE", "mean"), windows=("ADE", "size")
        ).reset_index()
    return metrics, grouped


@torch.inference_mode()
def benchmark(model, batch, device, seconds=1.0, trials=5, warmup=30):
    """Resident-input FP32 inference; each timed block is synchronized on CUDA."""
    model.eval()
    x = batch["x"].to(device)
    dt = batch["delta_t"].to(device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(warmup):
        model(x, dt)
    sync()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    latencies = []
    for trial in range(trials):
        elapsed, calls = 0.0, 0
        while elapsed < seconds:
            sync()
            start = time.perf_counter()
            for _ in range(20):
                model(x, dt)
            sync()
            elapsed += time.perf_counter() - start
            calls += 20
        latencies.append(1000 * elapsed / calls)
    mean = float(np.mean(latencies))
    return {"latency_mean_ms": mean, "latency_trial_std_ms": float(np.std(latencies, ddof=1)) if trials > 1 else None,
            "throughput_samples_s": len(x) * 1000 / mean, "benchmark_batch_size": len(x),
            "benchmark_min_seconds_per_trial": seconds, "benchmark_trials": trials,
            "inference_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
            "timing_scope": "resident-input FP32 forward pass; no data loading or transfer; synchronized CUDA"}

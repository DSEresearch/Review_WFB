"""Numerically check task and temporal-correction gradients in the existing WFB."""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models import WFBTrajectoryPredictor, laplacian_penalty


def check(seed=1):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = WFBTrajectoryPredictor(3, 2, feature_dim=2, wave_dim=4, hidden_dim=8, depth=1, dropout=0).double()
    x, dt, target = torch.randn(2, 3, 2, dtype=torch.float64), torch.randn(2, 3, dtype=torch.float64), torch.randn(2, 2, 2, dtype=torch.float64)
    rows = []
    for objective in ["task_MSE", "temporal_correction"]:
        for name, parameter in model.named_parameters():
            if not name.startswith("wfb."):
                continue
            if objective == "temporal_correction" and name not in {"wfb.At_raw", "wfb.omega_raw", "wfb.theta_t"}:
                continue
            def loss():
                pred, aux = model(x, dt)
                return (pred - target).square().mean() if objective == "task_MSE" else laplacian_penalty(aux)
            gradient = torch.autograd.grad(loss(), parameter)[0].flatten()[0].item()
            original = parameter.flatten()[0].item()
            for step in [1e-4, 1e-5, 1e-6]:
                with torch.no_grad():
                    parameter.flatten()[0] = original + step
                    plus = loss().item()
                    parameter.flatten()[0] = original - step
                    minus = loss().item()
                    parameter.flatten()[0] = original
                numeric = (plus - minus) / (2 * step)
                error = abs(numeric - gradient)
                rows.append({"objective": objective, "parameter": name, "coordinate": 0,
                             "step": step, "autograd": gradient, "central_difference": numeric,
                             "absolute_error": error, "passed": error <= 1e-7 + 1e-4 * abs(gradient)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", type=Path, default=Path("outputs/wfb_gradient_check"))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = check()
    result.to_csv(args.output_dir / "finite_difference_checks.csv", index=False)
    print(result.groupby("objective").agg(max_absolute_error=("absolute_error", "max"), all_passed=("passed", "all")))
    if not result.passed.all():
        raise SystemExit("Gradient verification failed")

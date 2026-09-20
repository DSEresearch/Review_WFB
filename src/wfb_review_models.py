"""Trajectory controls for WFB review, using the original WFB unchanged."""
from __future__ import annotations

import math

import torch
from torch import nn

from .models import FFNTrajectoryPredictor, MLP, SineTrajectoryPredictor, WFBTrajectoryPredictor, count_parameters
from .sequence_models import TransformerTrajectoryPredictor


class FourierPredictor(nn.Module):
    def __init__(self, obs_len, pred_len, feature_dim, hidden_dim, depth, dropout,
                 wave_dim, scale=1.0, learned=False):
        super().__init__()
        self.pred_len = pred_len
        basis = torch.randn(feature_dim + 1, wave_dim) * scale
        if learned:
            self.basis = nn.Parameter(basis)
        else:
            self.register_buffer("basis", basis)
        self.head = MLP(obs_len * 2 * wave_dim, hidden_dim, 2 * pred_len, depth, dropout)

    def forward(self, x, delta_t):
        phase = 2 * math.pi * (torch.cat([x, delta_t.unsqueeze(-1)], dim=-1) @ self.basis)
        features = torch.cat([phase.cos(), phase.sin()], dim=-1)
        return self.head(features.flatten(1)).view(len(x), self.pred_len, 2)


class Time2VecPredictor(nn.Module):
    def __init__(self, obs_len, pred_len, feature_dim, hidden_dim, depth, dropout, wave_dim):
        super().__init__()
        self.pred_len = pred_len
        self.time = nn.Linear(1, wave_dim)
        self.head = MLP(obs_len * (feature_dim + wave_dim), hidden_dim, 2 * pred_len, depth, dropout)

    def forward(self, x, delta_t):
        encoded = self.time(delta_t.unsqueeze(-1))
        encoded = torch.cat([encoded[..., :1], encoded[..., 1:].sin()], dim=-1)
        return self.head(torch.cat([x, encoded], dim=-1).flatten(1)).view(len(x), self.pred_len, 2)


class RecurrentPredictor(nn.Module):
    def __init__(self, pred_len, feature_dim, hidden_dim, layers, dropout, include_dt, kind):
        super().__init__()
        self.include_dt, self.pred_len = include_dt, pred_len
        core = nn.LSTM if kind == "lstm" else nn.GRU
        self.encoder = core(feature_dim + int(include_dt), hidden_dim, num_layers=layers,
                            dropout=dropout if layers > 1 else 0, batch_first=True)
        self.head = nn.Linear(hidden_dim, pred_len * 2)

    def forward(self, x, delta_t):
        if self.include_dt:
            x = torch.cat([x, delta_t.unsqueeze(-1)], dim=-1)
        hidden, _ = self.encoder(x)
        return self.head(hidden[:, -1]).view(len(x), self.pred_len, 2)


class TimeTransformer(TransformerTrajectoryPredictor):
    def forward(self, x, delta_t):
        return super().forward(torch.cat([x, delta_t.unsqueeze(-1)], dim=-1))


CORE = ["ffn_x", "ffn_dt", "ffn_x_matched", "ffn_dt_matched",
        "siren_dt_matched", "rff_dt_matched", "learned_fourier_dt_matched", "time2vec_dt_matched",
        "wfb_real", "wfb_shuffled", "wfb_constant"]
SEQUENCE = ["gru_x_matched", "gru_dt_matched", "lstm_dt_matched", "transformer_dt_matched"]


def build_review_model(name, *, obs_len, pred_len, feature_dim, hidden_dim=256, wave_dim=128,
                       depth=4, dropout=0.1, decoder="mlp", wave_ablation="full",
                       fourier_scale=1.0, sine_omega_0=30.0, layers=2):
    reference = lambda: WFBTrajectoryPredictor(obs_len, pred_len, feature_dim, wave_dim,
                                               hidden_dim, depth, dropout, decoder=decoder)

    def factory(width):
        common = dict(obs_len=obs_len, pred_len=pred_len, feature_dim=feature_dim,
                      hidden_dim=width, depth=depth)
        if name.startswith("wfb_"):
            return WFBTrajectoryPredictor(**common, wave_dim=wave_dim, dropout=dropout,
                                          decoder=decoder, wave_ablation=wave_ablation)
        if name.startswith("ffn_"):
            return FFNTrajectoryPredictor(**common, dropout=dropout, include_delta_t="_dt" in name)
        if name == "siren_dt_matched":
            return SineTrajectoryPredictor(**common, include_delta_t=True, omega_0=sine_omega_0)
        if name in {"rff_dt_matched", "learned_fourier_dt_matched"}:
            return FourierPredictor(**common, dropout=dropout, wave_dim=wave_dim,
                                    scale=fourier_scale, learned=name.startswith("learned"))
        if name == "time2vec_dt_matched":
            return Time2VecPredictor(**common, dropout=dropout, wave_dim=wave_dim)
        if name.startswith(("gru_", "lstm_")):
            return RecurrentPredictor(pred_len, feature_dim, width, layers, dropout,
                                      "_dt" in name, name.split("_")[0])
        if name == "transformer_dt_matched":
            return TimeTransformer(obs_len, pred_len, feature_dim + 1, d_model=width,
                                   nhead=4, num_layers=layers, ffn_dim=4 * width, dropout=dropout)
        raise ValueError(f"Unknown review model: {name}")

    # Count on the meta device to avoid allocating candidate networks or consuming RNG state.
    with torch.device("meta"):
        target = count_parameters(reference())
        width = hidden_dim
        if name.endswith("_matched"):
            unit = 4 if name == "transformer_dt_matched" else 1
            low, high = 1, 2048 // unit
            while low < high:
                mid = (low + high) // 2
                if count_parameters(factory(mid * unit)) < target:
                    low = mid + 1
                else:
                    high = mid
            candidates = {max(1, low - 1) * unit, low * unit}
            width = min(candidates, key=lambda w: abs(count_parameters(factory(w)) - target))
    model = factory(width)
    count = count_parameters(model)
    mismatch = abs(count - target) / target
    if name.endswith("_matched") and mismatch > 0.05:
        raise ValueError(f"{name}: parameter mismatch {mismatch:.1%}; increase reference capacity")
    return model, {"parameters": count, "reference_parameters": target,
                   "parameter_difference_pct": 100 * (count - target) / target,
                   "effective_hidden_dim": width}

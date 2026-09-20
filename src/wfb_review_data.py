"""Auditable review protocol; never rewrites the historical window CSVs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def motion_from_positions(xy: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Only observed samples contribute; the first derivatives are unknown/zero."""
    if not np.isfinite(dt).all() or np.any(dt[:, 1:] <= 0):
        raise ValueError("Observation intervals after the first sample must be finite and positive")
    x = np.zeros((*xy.shape[:2], 6), dtype=np.float64)
    x[..., :2] = xy
    x[:, 1:, 2:4] = np.diff(xy, axis=1) / dt[:, 1:, None]
    # Acceleration at the second sample is unknown without a preceding velocity.
    x[:, 2:, 4:6] = np.diff(x[:, 1:, 2:4], axis=1) / dt[:, 2:, None]
    return x.astype(np.float32)


def extract_intervals(df, obs_len, time_basis, fps=None, fps_by_source=None):
    if time_basis == "frames":
        frames = df[[f"obs_{i}_frame" for i in range(obs_len)]].to_numpy(np.float64)
        if not np.isfinite(frames).all() or not np.allclose(frames, np.rint(frames), rtol=0, atol=1e-7):
            raise ValueError("Frame-based timing requires finite integer frame indices")
        rates = df.source.map(fps_by_source).to_numpy(float) if fps_by_source else np.full(len(df), fps or np.nan)
        if not np.isfinite(rates).all() or np.any(rates <= 0):
            raise ValueError("Provide a verified positive FPS for every source")
        gaps = np.diff(frames, axis=1) / rates[:, None]
    elif time_basis == "timestamps":
        times = df[[f"obs_{i}_time" for i in range(obs_len)]].to_numpy(np.float64)
        gaps = np.diff(times, axis=1)
    else:
        raise ValueError("time_basis must be frames or timestamps")
    if not np.isfinite(gaps).all() or np.any(gaps <= 0):
        raise ValueError("Nonpositive/nonfinite intervals; audit the tracks before training")
    return np.column_stack([np.zeros(len(df)), gaps])


def audit_and_prepare(processed: Path, output: Path, *, obs_len=8, pred_len=12,
                      time_basis="timestamps", fps=None, fps_by_source=None,
                      coordinate_unit="unspecified", prepare=False,
                      require_source_disjoint=False):
    output.mkdir(parents=True, exist_ok=True)
    if prepare and (output / "cache_manifest.json").exists():
        raise FileExistsError("Use a new cache directory; an existing cache must not be overwritten")
    report = {"schema": 1, "processed_dir": str(processed.resolve()), "obs_len": obs_len,
              "pred_len": pred_len, "time_basis": time_basis, "fps": fps,
              "fps_by_source": fps_by_source, "coordinate_unit": coordinate_unit,
              "splits": {}, "warnings": [], "errors": []}
    if coordinate_unit == "unspecified":
        report["warnings"].append("ADE units are unspecified; normalized image coordinates are not meters.")
    summary_path = processed / "preprocess_summary.json"
    if summary_path.exists():
        report["original_preprocessing"] = json.loads(summary_path.read_text())
        report["warnings"].append("FPS and original agent identities require dataset provenance verification; this audit cannot validate them.")
    track_sets, source_sets = {}, {}
    by_source, norm = [], None
    for split in ("train", "val", "test"):
        path = processed / f"windows_{split}.csv"
        print(f"Reading {split}: {path}", flush=True)
        header = pd.read_csv(path, nrows=0).columns
        columns = ["source", "agent_id"]
        columns += [f"obs_{i}_{v}" for i in range(obs_len) for v in ("x", "y", "dt", "frame", "time")]
        columns += [f"target_{i}_{v}" for i in range(pred_len) for v in ("x", "y", "frame", "time")]
        optional = ["start_index", "irregular_sample_id", "start_frame"]
        df = pd.read_csv(path, usecols=[c for c in columns + optional if c in header],
                         dtype={"source": str, "agent_id": str})
        if df.empty or df[["source", "agent_id"]].isna().any().any():
            raise ValueError(f"Empty split or missing source/agent identity: {split}")
        tracks = set(map(tuple, df[["source", "agent_id"]].drop_duplicates().to_numpy()))
        track_sets[split], source_sets[split] = tracks, set(df.source)
        dt = extract_intervals(df, obs_len, time_basis, fps, fps_by_source)
        xy = df[[f"obs_{i}_{v}" for i in range(obs_len) for v in ("x", "y")]].to_numpy(float).reshape(-1, obs_len, 2)
        target = df[[f"target_{i}_{v}" for i in range(pred_len) for v in ("x", "y")]].to_numpy(np.float32).reshape(-1, pred_len, 2)
        if not np.isfinite(xy).all() or not np.isfinite(target).all():
            raise ValueError(f"Nonfinite positions/targets in {split}")
        old = df[[f"obs_{i}_dt" for i in range(1, obs_len)]].to_numpy(float)
        error = np.abs(old - dt[:, 1:])
        stats = {"windows": len(df), "tracks": len(tracks), "sources": len(source_sets[split]),
                 "sha256": digest(path), "dt_min_seconds": float(dt[:, 1:].min()),
                 "dt_max_seconds": float(dt[:, 1:].max()), "dt_mean_seconds": float(dt[:, 1:].mean()),
                 "dt_std_seconds": float(dt[:, 1:].std()),
                 "stored_dt_mismatch_rate_at_1e_6": float((error > 1e-6).mean()),
                 "stored_dt_max_abs_error_seconds": float(error.max()),
                 "nonconstant_window_rate": float((np.ptp(dt[:, 1:], axis=1) > 1e-6).mean())}
        frame_cols = [f"obs_{i}_frame" for i in range(obs_len)]
        if all(c in df for c in frame_cols):
            frames = df[frame_cols].to_numpy(float)
            rates = df.source.map(fps_by_source).to_numpy(float) if fps_by_source else np.full(len(df), fps or np.nan)
            if np.isfinite(rates).all():
                exact = np.diff(frames, axis=1) / rates[:, None]
                simulated = np.diff((frames / rates[:, None]).astype(np.float32), axis=1)
                stats["legacy_float32_dt_match_rate"] = float(np.isclose(old, simulated, atol=1e-7, rtol=0).mean())
                stats["frame_based_max_abs_error_seconds"] = float(np.abs(old - exact).max())
        time_suffix = "frame" if time_basis == "frames" else "time"
        chronology = [f"obs_{i}_{time_suffix}" for i in range(obs_len)] + [f"target_{i}_{time_suffix}" for i in range(pred_len)]
        if not all(c in df for c in chronology):
            report["errors"].append(f"{split}: target timing unavailable; cannot verify future chronology")
        else:
            ordered = df[chronology].to_numpy(float)
            if not np.isfinite(ordered).all() or (np.diff(ordered, axis=1) <= 0).any():
                report["errors"].append(f"{split}: target/observation times are not strictly increasing")
            stats["duplicate_windows"] = int(df.duplicated(["source", "agent_id"] + chronology).sum())
            # Horizons describe physical prediction duration, not an extra model input.
            horizon = ordered[:, -1] - ordered[:, obs_len - 1]
            if time_basis == "frames":
                horizon /= rates
            stats["prediction_horizon_seconds_min_max"] = [float(horizon.min()), float(horizon.max())]
        if "start_index" in df:
            starts = df[["source", "agent_id", "start_index"]].drop_duplicates()
            stats["distinct_start_positions"] = len(starts)
            stats["windows_per_start"] = len(df) / len(starts)
        report["splits"][split] = stats
        for source, group in df.groupby("source", sort=True):
            by_source.append({"split": split, "source": source, "windows": len(group),
                              "tracks": group.agent_id.nunique()})
        if prepare:
            x = motion_from_positions(xy, dt)
            np.save(output / f"{split}_x.npy", x)
            np.save(output / f"{split}_dt.npy", dt.astype(np.float32))
            np.save(output / f"{split}_target.npy", target)
            df[["source", "agent_id"]].to_csv(output / f"{split}_groups.csv", index=False)
            if split == "train":
                mean = x.mean(axis=(0, 1), dtype=np.float64)
                std = x.std(axis=(0, 1), dtype=np.float64)
                norm = {"feature_mean": mean.tolist(), "feature_std": np.maximum(std, 1e-6).tolist(),
                        "dt_mean": float(dt[:, 1:].mean()), "dt_std": max(float(dt[:, 1:].std()), 1e-6)}
        print(f"{split}: {len(df):,} windows; interval mismatch {stats['stored_dt_mismatch_rate_at_1e_6']:.2%}", flush=True)
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if track_sets[left] & track_sets[right]:
            report["errors"].append(f"Track overlap between {left} and {right}")
        overlap = source_sets[left] & source_sets[right]
        if overlap:
            message = f"{left}/{right}: {len(overlap)} shared sources; this is not a scene-disjoint benchmark"
            report["errors" if require_source_disjoint else "warnings"].append(message)
    report["total_windows"] = sum(s["windows"] for s in report["splits"].values())
    original = report.get("original_preprocessing", {})
    if original.get("windows") is not None and original["windows"] != report["total_windows"]:
        report["errors"].append("Split window total differs from preprocessing summary")
    report["raw_observation_count_verification"] = (
        "Historical raw_rows is metadata only. Re-run preprocessing to obtain track_window_counts.csv."
    )
    pd.DataFrame(by_source).to_csv(output / "source_counts.csv", index=False)
    dump(output / "data_audit.json", report)
    if report["errors"]:
        raise ValueError("Audit failed: " + "; ".join(report["errors"]))
    if prepare:
        dump(output / "normalization.json", norm)
        report["normalization"] = norm
        report["first_interval"] = "zero sentinel; excluded from fitting dt normalization and from shuffling"
        report["motion_features"] = "recomputed using only observed positions and real intervals; held fixed in dt interventions"
        report["cache_sha256"] = {p.name: digest(p) for p in sorted(output.glob("*_*.npy"))}
        report["cache_sha256"].update({p.name: digest(p) for p in sorted(output.glob("*_groups.csv"))})
        report["cache_sha256"]["normalization.json"] = digest(output / "normalization.json")
        dump(output / "cache_manifest.json", report)
    return report


class ReviewDataset(Dataset):
    def __init__(self, root, split, features="position", mode="real", seed=0):
        root = Path(root)
        self.x = np.load(root / f"{split}_x.npy", mmap_mode="r")
        self.target = np.load(root / f"{split}_target.npy", mmap_mode="r")
        dt = np.load(root / f"{split}_dt.npy").copy()
        norm = json.loads((root / "normalization.json").read_text())
        self.norm = norm
        self.dims = 2 if features == "position" else 6
        self.mean = np.array(norm["feature_mean"][:self.dims], dtype=np.float32)
        self.std = np.array(norm["feature_std"][:self.dims], dtype=np.float32)
        rng = np.random.default_rng(seed)
        if mode == "shuffled":
            indices = np.argsort(rng.random(dt[:, 1:].shape), axis=1)
            dt[:, 1:] = np.take_along_axis(dt[:, 1:], indices, axis=1)
        elif mode == "constant":
            dt[:, 1:] = norm["dt_mean"]
        elif mode != "real":
            raise ValueError(mode)
        self.dt = (dt - norm["dt_mean"]) / norm["dt_std"]
        self.dt[:, 0] = 0

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return {"x": torch.from_numpy((self.x[index, :, :self.dims] - self.mean) / self.std),
                "delta_t": torch.from_numpy(self.dt[index]),
                "target": torch.from_numpy(self.target[index].copy())}

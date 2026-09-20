"""Summarize selected WFB review models without treating windows as seed replicates."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.wfb_review_data import dump


def paired_statistics(reference, candidate):
    """Negative candidate-reference difference favors the candidate."""
    d = np.asarray(candidate, float) - np.asarray(reference, float)
    n = len(d)
    result = {"paired_seeds": n, "mean_difference": float(d.mean()),
              "relative_improvement_pct": float(100 * (np.mean(reference) - np.mean(candidate)) / np.mean(reference))}
    if n < 2:
        return {**result, "ci95_low": None, "ci95_high": None, "paired_t_p": None,
                "paired_effect_dz": None, "sign_flip_p": None}
    sd = float(d.std(ddof=1))
    margin = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    if sd == 0:
        p = 1.0 if d.mean() == 0 else 0.0
    else:
        p = float(stats.ttest_1samp(d, 0).pvalue)
    flip_p = None
    if n <= 16:
        signs = np.array(list(itertools.product([-1, 1], repeat=n)))
        flip_p = float((np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-14).mean())
    return {**result, "ci95_low": float(d.mean() - margin), "ci95_high": float(d.mean() + margin),
            "paired_t_p": p, "paired_effect_dz": float(d.mean() / sd) if sd else None,
            "sign_flip_p": flip_p}


def holm(values):
    p = np.asarray(values, float)
    result = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    ordered = valid[np.argsort(p[valid])]
    adjusted = np.maximum.accumulate((len(ordered) - np.arange(len(ordered))) * p[ordered])
    result[ordered] = np.minimum(adjusted, 1)
    return result


def source_bootstrap(reference_dir, candidate_dir, seeds, samples=2000):
    differences = []
    for seed in seeds:
        a = pd.read_csv(reference_dir / f"seed_{seed}/test_track_metrics.csv", dtype={"source": str, "agent_id": str})
        b = pd.read_csv(candidate_dir / f"seed_{seed}/test_track_metrics.csv", dtype={"source": str, "agent_id": str})
        merged = a.merge(b, on=["source", "agent_id"], suffixes=("_a", "_b"), validate="one_to_one")
        if len(merged) != len(a) or len(merged) != len(b) or not (merged.windows_a == merged.windows_b).all():
            raise ValueError("Mismatched test tracks in paired comparison")
        merged["weighted_diff"] = (merged.ADE_b - merged.ADE_a) * merged.windows_a
        grouped = merged.groupby("source").agg(error_sum=("weighted_diff", "sum"), windows=("windows_a", "sum"))
        differences.append(grouped)
    mean = pd.concat(differences).groupby(level=0).mean()
    if len(mean) < 2:
        return {"source_clusters": len(mean), "source_bootstrap_ci95_low": None, "source_bootstrap_ci95_high": None}
    rng = np.random.default_rng(501)
    estimates = []
    for _ in range(samples):
        draw = mean.iloc[rng.integers(0, len(mean), len(mean))]
        estimates.append(draw.error_sum.sum() / draw.windows.sum())
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {"source_clusters": len(mean), "source_bootstrap_ci95_low": float(low), "source_bootstrap_ci95_high": float(high)}


def savefig(fig, output, name):
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def summarize(root: Path):
    files = [p for p in sorted(root.glob("*/results.csv")) if (p.parent / "run_config.json").exists()]
    if not files:
        raise FileNotFoundError("No completed reviewer experiments")
    output = root / "summary"
    output.mkdir(exist_ok=True)
    frames, dirs = [], {}
    for path in files:
        df = pd.read_csv(path)
        if df.seed.duplicated().any():
            raise ValueError(f"Duplicate seeds: {path}")
        df["experiment"] = path.parent.name
        frames.append(df)
        dirs[path.parent.name] = path.parent
    runs = pd.concat(frames, ignore_index=True)
    if runs.cache_manifest_sha256.nunique() != 1:
        raise ValueError("Different datasets/normalizations cannot be pooled into one review table")
    runs.to_csv(output / "all_seed_results.csv", index=False)
    keys = ["features", "experiment"]
    summary = runs.groupby(keys)[["ADE", "FDE", "MSE", "RMSE", "parameters", "latency_mean_ms", "throughput_samples_s"]].agg(["mean", "std", "count"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(output / "performance.csv", index=False)
    paired = []
    for feature, feature_runs in runs.groupby("features"):
        reference = feature_runs[(feature_runs.model == "wfb_real") & (feature_runs.variant == "standard") &
                                 (feature_runs.decoder == "mlp") & (feature_runs.wave_ablation == "full")]
        if reference.empty:
            continue
        ref_id = reference.experiment.iloc[0]
        reference = reference.set_index("seed")
        for name, candidate in feature_runs.groupby("experiment"):
            if name == ref_id:
                continue
            candidate = candidate.set_index("seed")
            if set(candidate.index) != set(reference.index):
                raise ValueError(f"Unequal seed sets: {ref_id} and {name}")
            seeds = sorted(reference.index.astype(int))
            statistics = paired_statistics(reference.loc[seeds, "ADE"], candidate.loc[seeds, "ADE"])
            clustered = source_bootstrap(dirs[ref_id], dirs[name], seeds)
            paired.append({"features": feature, "reference": ref_id, "candidate": name, **statistics, **clustered})
    paired_df = pd.DataFrame(paired)
    if not paired_df.empty:
        paired_df["holm_p"] = holm(paired_df.paired_t_p)
    paired_df.to_csv(output / "paired_ADE.csv", index=False)
    for feature, data in summary.groupby("features"):
        data = data.sort_values("ADE_mean")
        fig, ax = plt.subplots(figsize=(10, max(4, 0.34 * len(data))))
        labels = data.experiment.str.replace(f"__{feature}", "", regex=False)
        ax.barh(labels, data.ADE_mean, xerr=data.ADE_std.fillna(0), color="#218c82", capsize=3)
        ax.invert_yaxis()
        ax.set_xlabel("Test ADE (coordinate units; mean +/- seed SD)")
        ax.set_title(f"{feature.capitalize()} inputs; validation-selected configurations")
        fig.tight_layout()
        savefig(fig, output, f"ade_{feature}")
    for name, directory in dirs.items():
        data = pd.read_csv(directory / "validation_candidates.csv")
        if data["lambda"].nunique() > 1:
            aggregate = data.groupby(["lambda", "lr"]).best_val_ADE.agg(["mean", "std"]).reset_index()
            chosen = aggregate.loc[aggregate.groupby("lambda")["mean"].idxmin()].sort_values("lambda")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.errorbar(chosen["lambda"], chosen["mean"], yerr=chosen["std"].fillna(0), marker="o", capsize=3)
            ax.set_xscale("symlog", linthresh=1e-8)
            ax.set_xlabel("Laplacian weight (0 = no correction in combined variant)")
            ax.set_ylabel("Validation ADE (mean +/- seed SD)")
            ax.set_title(name.replace("__", " / "), fontsize=8)
            fig.tight_layout()
            savefig(fig, output, f"lambda_{name}")
        histories = [(p.parent.name, pd.read_csv(p)) for p in sorted(directory.glob("seed_*/history.csv"))]
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
        for seed, history in histories:
            for ax, column in zip(axes, ["train_MSE", "val_MSE", "val_ADE"]):
                ax.plot(history.epoch, history[column], label=seed, linewidth=1)
                ax.set(xlabel="Epoch", ylabel=column)
        axes[-1].legend(fontsize=7)
        fig.tight_layout()
        savefig(fig, output, f"convergence_{name}")
    lines = [r"% Requires graphicx. Errors are in the coordinate units recorded in the cache.",
             r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{lrrr}", r"\hline", r"Model & ADE & FDE & Parameters \\", r"\hline"]
    for row in summary.itertuples():
        label = row.experiment.replace("_", r"\_")
        def formatted(mean, std):
            return f"${mean:.6f} \\pm {std:.6f}$" if np.isfinite(std) else f"${mean:.6f}$"
        lines.append(f"{label} & {formatted(row.ADE_mean, row.ADE_std)} & {formatted(row.FDE_mean, row.FDE_std)} & {row.parameters_mean:.0f} " + r"\\")
    lines += [r"\hline", r"\end{tabular}", "}"]
    (output / "performance_table.tex").write_text("\n".join(lines), encoding="utf-8")
    dump(output / "interpretation_notes.json", {
        "negative_paired_difference": "candidate has lower error than real-time standard WFB",
        "confidence_interval": "paired Student t interval across independently trained seeds, not across windows",
        "source_bootstrap": "resamples whole sources, including overlapping windows; averages errors over seeds first; conditional on trained models",
        "multiple_testing": "Holm correction across all reported ADE t-tests",
        "five_seed_limit": "An exact two-sided sign-flip test with five seeds has minimum p=0.0625",
        "selection": "Hyperparameters selected using mean validation ADE across seeds; test evaluated only for selected candidates",
        "lambda_curves": "Validation curves, not test performance curves",
        "claim": "If real time does not beat shuffled and constant time, correct interval assignment is not established as the cause of gains"
    })
    print(summary[["experiment", "ADE_mean", "ADE_std", "parameters_mean"]].to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_dir", type=Path, required=True)
    summarize(p.parse_args().results_dir)

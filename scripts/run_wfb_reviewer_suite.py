"""Distribute independent WFB reviewer experiments over distinct GPUs."""
from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.wfb_review_data import dump
from src.wfb_review_models import CORE, SEQUENCE


def jobs(suites):
    output = []
    def add(model, features="position", variant="standard", decoder="mlp", wave_ablation="full"):
        job = dict(model=model, features=features, variant=variant, decoder=decoder, wave_ablation=wave_ablation)
        if job not in output:
            output.append(job)
    if "core" in suites:
        for name in CORE:
            add(name)
    if "motion" in suites:
        for name in CORE:
            add(name, features="motion")
    if "sequence" in suites:
        for name in ["wfb_real", *SEQUENCE]:
            add(name)
    if "laplacian" in suites:
        for variant in ["standard", "combined", "laplacian"]:
            add("wfb_real", variant=variant)
    if "decoder" in suites:
        for decoder in ["mlp", "linear"]:
            add("wfb_real", decoder=decoder)
        add("ffn_dt_matched")
    if "wave" in suites:
        for ablation in ["full", "without_A", "without_k", "without_omega", "without_theta"]:
            add("wfb_real", wave_ablation=ablation)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, default=Path("outputs/wfb_reviewer"))
    p.add_argument("--suites", nargs="+", choices=["core", "motion", "sequence", "laplacian", "decoder", "wave"], default=["core", "sequence", "laplacian", "decoder"])
    p.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--lrs", nargs="+", type=float, default=[1e-4, 3e-4, 1e-3])
    p.add_argument("--lambdas", nargs="+", type=float, default=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3])
    p.add_argument("--fourier_scales", nargs="+", type=float, default=[1.0])
    p.add_argument("--sine_omegas", nargs="+", type=float, default=[30.0])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--benchmark_seconds", type=float, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    if len(set(args.devices)) != len(args.devices):
        p.error("Each GPU must be listed once; timing requires one worker per GPU")
    if "cuda" in args.devices and "cuda:0" in args.devices:
        p.error("cuda and cuda:0 refer to the same GPU")
    if not (args.cache_dir / "cache_manifest.json").exists():
        p.error("Prepare the audited cache first with prepare_wfb_reviewer_data.py --prepare")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for job in jobs(args.suites):
        name = "__".join(job.values())
        cmd = [sys.executable, str(ROOT / "scripts/run_wfb_reviewer_experiment.py"),
               "--cache_dir", str(args.cache_dir.resolve()), "--output_dir", str((args.output_dir / name).resolve())]
        for key, value in job.items():
            cmd += [f"--{key}", value]
        for key in ["seeds", "lrs", "lambdas", "fourier_scales", "sine_omegas"]:
            cmd += [f"--{key}", *map(str, getattr(args, key))]
        for key in ["epochs", "patience", "batch_size", "threads", "workers", "benchmark_seconds"]:
            cmd += [f"--{key}", str(getattr(args, key))]
        if args.resume:
            cmd.append("--resume")
        nlambda = 1 if job["variant"] == "standard" else len(set(args.lambdas + ([0.0] if job["variant"] == "combined" else [])))
        periodic_count = len(set(args.fourier_scales)) if job["model"] in {"rff_dt_matched", "learned_fourier_dt_matched"} else len(set(args.sine_omegas)) if job["model"] == "siren_dt_matched" else 1
        manifest.append({"id": name, **job, "command": cmd, "candidate_fits": len(args.seeds) * len(set(args.lrs)) * nlambda * periodic_count})
    dump(args.output_dir / "suite_manifest.json", {"jobs": manifest, "devices": args.devices})
    print(f"{len(manifest)} jobs, {sum(j['candidate_fits'] for j in manifest)} candidate fits; test evaluation follows validation selection.", flush=True)
    if args.dry_run:
        for job in manifest:
            print(job["id"], job["candidate_fits"], "fits")
        return
    pending = collections.deque(manifest)
    active, failures = {}, []
    try:
        while pending or active:
            for device in args.devices:
                if device not in active and pending:
                    job = pending.popleft()
                    log = (args.output_dir / f"{job['id']}.log").open("a", encoding="utf-8")
                    process = subprocess.Popen(job["command"] + ["--device", device], cwd=ROOT,
                                               stdout=log, stderr=subprocess.STDOUT)
                    active[device] = (process, log, job)
                    print(f"Started {job['id']} on {device}; log: {log.name}", flush=True)
            for device, (process, log, job) in list(active.items()):
                status = process.poll()
                if status is not None:
                    log.close()
                    del active[device]
                    print(f"Finished {job['id']} exit={status}", flush=True)
                    if status:
                        failures.append({"job": job["id"], "exit_code": status})
            if active:
                time.sleep(1)
    finally:
        for process, log, _ in active.values():
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()
    dump(args.output_dir / "suite_status.json", {"failures": failures, "complete": not failures})
    if failures:
        raise RuntimeError(f"{len(failures)} failed jobs; inspect logs and use --resume")
    subprocess.run([sys.executable, str(ROOT / "scripts/summarize_wfb_reviewer.py"),
                    "--results_dir", str(args.output_dir.resolve())], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

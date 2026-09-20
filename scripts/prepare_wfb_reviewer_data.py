"""Audit historical windows; optionally prepare a separate corrected array cache."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.wfb_review_data import audit_and_prepare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processed_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--time_basis", choices=["frames", "timestamps"], required=True)
    p.add_argument("--fps", type=float, help="Verified FPS in units of the stored frame indices")
    p.add_argument("--fps_json", type=Path, help="JSON mapping exact source names to verified FPS")
    p.add_argument("--obs_len", type=int, default=8)
    p.add_argument("--pred_len", type=int, default=12)
    p.add_argument("--coordinate_unit", choices=["unspecified", "normalized_image", "pixels", "meters"], default="unspecified")
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--require_source_disjoint", action="store_true")
    args = vars(p.parse_args())
    mapping = args.pop("fps_json")
    args["fps_by_source"] = json.loads(mapping.read_text()) if mapping else None
    args["processed"] = args.pop("processed_dir")
    args["output"] = args.pop("output_dir")
    if args["fps"] is not None and mapping:
        p.error("Choose --fps or --fps_json, not both")
    report = audit_and_prepare(**args)
    print(json.dumps({k: report[k] for k in ["total_windows", "warnings", "errors"]}, indent=2))


if __name__ == "__main__":
    main()

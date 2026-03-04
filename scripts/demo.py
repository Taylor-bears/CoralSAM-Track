#!/usr/bin/env python
"""
demo.py – Single-sequence coral video segmentation demo.

Usage
-----
# With drift correction (default)
python scripts/demo.py --seq video102

# Disable drift correction (baseline)
python scripts/demo.py --seq video102 --no_drift_correction

# Custom config and output dir
python scripts/demo.py --seq video75 --config configs/default.yaml --output outputs/my_run

# Run all sequences
python scripts/demo.py --all
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tracker import CoralTracker
from src.utils import load_config, list_sequences

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("demo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CoralSAM-Track: single-sequence demo"
    )
    parser.add_argument(
        "--seq",
        type=str,
        default=None,
        help="Sequence name to process (e.g. video102). Required unless --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all sequences in the dataset.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--no_drift_correction",
        action="store_true",
        help="Disable drift detection (run baseline).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output base directory from config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device ('cuda' or 'cpu'). Auto-detected if not set.",
    )
    return parser.parse_args()


def print_summary(seq_name: str, result) -> None:
    timing = result.timing
    reinits = result.reinit_frames
    print(f"\n{'='*60}")
    print(f"  Sequence  : {seq_name}")
    print(f"  Frames    : {timing.get('n_frames', len(result.masks))}")
    print(f"  Total time: {timing.get('total_s', 0):.2f} s")
    print(f"  FPS       : {timing.get('fps', 0):.1f}")
    print(f"  ms/frame  : {timing.get('mean_ms_per_frame', 0):.1f}")
    print(f"  Re-inits  : {len(reinits)}" + (f"  @ frames {reinits[:5]}" if reinits else ""))
    print(f"{'='*60}\n")


def main() -> None:
    args = parse_args()

    if not args.seq and not args.all:
        print("Error: specify --seq <name> or --all")
        sys.exit(1)

    # Load config
    cfg = load_config(args.config)
    if args.output:
        cfg.setdefault("output", {})["base_dir"] = args.output

    data_root = cfg.get("data_root", "partial_coralvos/partial")

    # Determine sequences to run
    if args.all:
        sequences = list_sequences(data_root)
        log.info("Running all %d sequences: %s", len(sequences), sequences)
    else:
        sequences = [args.seq]

    # Build tracker
    tracker = CoralTracker(
        cfg=cfg,
        device=args.device,
        use_drift_correction=not args.no_drift_correction,
    )

    all_timing = {}
    for seq in sequences:
        log.info("--- Starting sequence: %s ---", seq)
        try:
            result = tracker.run_sequence(
                data_root=data_root,
                seq_name=seq,
                save_output=True,
            )
            print_summary(seq, result)
            all_timing[seq] = result.timing
        except Exception as exc:
            log.error("Failed on %s: %s", seq, exc, exc_info=True)

    # Print aggregate summary
    if len(all_timing) > 1:
        from statistics import mean
        valid = [v for v in all_timing.values() if v]
        if valid:
            avg_fps = mean(v["fps"] for v in valid if "fps" in v)
            print(f"\nAggregate FPS (mean over {len(valid)} sequences): {avg_fps:.1f}")

    # Save timing JSON
    out_base = cfg.get("output", {}).get("base_dir", "outputs")
    drift_tag = "baseline" if args.no_drift_correction else "with_drift_corr"
    timing_path = Path(out_base) / drift_tag / "timing.json"
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    with open(timing_path, "w") as f:
        json.dump(all_timing, f, indent=2)
    log.info("Timing saved to %s", timing_path)


if __name__ == "__main__":
    main()

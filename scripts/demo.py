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
import csv
import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean

import torch

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


def _setup_file_logging(log_path: Path) -> None:
    """Add a file handler so all log output is also written to disk."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logging.getLogger().addHandler(fh)
    log.info("Logging to file: %s", log_path)


def _write_csv_rows(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_run_summary(mode_label: str, all_results: dict) -> dict:
    per_sequence_rows = []
    per_sequence = {}

    for seq_name, result in sorted(all_results.items()):
        summary = result.summary or result.build_summary()
        timing = summary.get("timing", {})
        source_counts = summary.get("reinit_source_counts", {})
        gate_counts = summary.get("reinit_gate_counts", {})
        row = {
            "seq_name": seq_name,
            "n_frames": summary.get("n_frames", 0),
            "fps": timing.get("fps", 0.0),
            "total_s": timing.get("total_s", 0.0),
            "mean_ms_per_frame": timing.get("mean_ms_per_frame", 0.0),
            "init_frame_idx": summary.get("init_frame_idx"),
            "reinit_count": summary.get("reinit_count", 0),
            "keyframe_count": summary.get("keyframe_count", 0),
            "drift_checks": summary.get("drift_checks", 0),
            "drift_events": summary.get("drift_events", 0),
            "mean_confidence": summary.get("mean_confidence", 0.0),
            "median_confidence": summary.get("median_confidence", 0.0),
            "mean_area_pixels": summary.get("mean_area_pixels", 0.0),
            "memory_flush": source_counts.get("memory_flush", 0),
            "memory_flush_fallback": source_counts.get("memory_flush_fallback", 0),
            "keyframe_rewind": source_counts.get("keyframe_rewind", 0),
            "auto_init": source_counts.get("auto_init", 0),
            "gate_accepted": gate_counts.get("accepted", 0),
            "gate_rejected": gate_counts.get("rejected", 0),
            "gate_skipped": gate_counts.get("skipped", 0) + gate_counts.get("skipped_flush", 0),
        }
        per_sequence_rows.append(row)
        per_sequence[seq_name] = summary

    overall = {
        "mode": mode_label,
        "n_sequences": len(per_sequence_rows),
        "mean_fps": round(mean(row["fps"] for row in per_sequence_rows), 4) if per_sequence_rows else 0.0,
        "mean_ms_per_frame": round(
            mean(row["mean_ms_per_frame"] for row in per_sequence_rows), 4
        ) if per_sequence_rows else 0.0,
        "total_runtime_s": round(sum(row["total_s"] for row in per_sequence_rows), 4),
        "total_reinits": int(sum(row["reinit_count"] for row in per_sequence_rows)),
        "total_drift_checks": int(sum(row["drift_checks"] for row in per_sequence_rows)),
        "total_drift_events": int(sum(row["drift_events"] for row in per_sequence_rows)),
        "reinit_source_counts": dict(
            Counter(
                key
                for row in per_sequence_rows
                for key, value in {
                    "memory_flush": row["memory_flush"],
                    "memory_flush_fallback": row["memory_flush_fallback"],
                    "keyframe_rewind": row["keyframe_rewind"],
                    "auto_init": row["auto_init"],
                }.items()
                for _ in range(int(value))
            )
        ),
    }

    return {
        "mode": mode_label,
        "overall": overall,
        "per_sequence": per_sequence,
        "table_rows": per_sequence_rows,
    }


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
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help=(
            "Unique run identifier appended to log filenames and output directories "
            "(e.g. '20260305_120000'). Auto-generated from current timestamp if omitted."
        ),
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

    # Determine run identifier (used for timestamped logs + output dirs)
    run_id: str = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    # Short tag used for log filenames; "with_drift_corr" is the folder name used by the tracker
    log_tag = "baseline" if args.no_drift_correction else "drift_corr"
    log_path = Path("logs") / f"{log_tag}_{run_id}.log"
    _setup_file_logging(log_path)
    log.info("Run ID: %s", run_id)

    # Load config
    cfg = load_config(args.config)
    if args.output:
        cfg.setdefault("output", {})["base_dir"] = args.output
    else:
        # Default: place outputs under outputs/<run_id>/
        cfg.setdefault("output", {})["base_dir"] = str(Path("outputs") / run_id)

    log.info("Output base dir: %s", cfg["output"]["base_dir"])

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

    all_results = {}
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
            all_results[seq] = result
            all_timing[seq] = result.timing
        except Exception as exc:
            log.error("Failed on %s: %s", seq, exc, exc_info=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Print aggregate summary
    if len(all_timing) > 1:
        valid = [v for v in all_timing.values() if v]
        if valid:
            avg_fps = mean(v["fps"] for v in valid if "fps" in v)
            print(f"\nAggregate FPS (mean over {len(valid)} sequences): {avg_fps:.1f}")

    # Save timing JSON  (folder name matches what CoralTracker uses)
    out_base = cfg.get("output", {}).get("base_dir", "outputs")
    folder_tag = "baseline" if args.no_drift_correction else "with_drift_corr"
    timing_path = Path(out_base) / folder_tag / "timing.json"
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    with open(timing_path, "w") as f:
        json.dump(all_timing, f, indent=2)
    log.info("Timing saved to %s", timing_path)

    run_summary = _build_run_summary(folder_tag, all_results)
    run_summary_json = Path(out_base) / folder_tag / "run_summary.json"
    run_summary_csv = Path(out_base) / folder_tag / "run_summary.csv"
    with open(run_summary_json, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, ensure_ascii=False)
    _write_csv_rows(run_summary_csv, run_summary["table_rows"])
    log.info("Run summary saved to %s and %s", run_summary_json, run_summary_csv)


if __name__ == "__main__":
    main()

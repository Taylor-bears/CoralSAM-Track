#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import logging
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("revision_experiments")


PRESET_ORDER = ["baseline", "soft_only", "no_gate", "full"]
PRESET_META = {
    "baseline": {
        "title": "A0 Baseline",
        "description": "SAM2 propagation without drift correction",
    },
    "soft_only": {
        "title": "A1 Soft-only",
        "description": "Drift detection with memory refresh only",
    },
    "no_gate": {
        "title": "A2 No-gate",
        "description": "Two-tier correction with hard reinit gate disabled",
    },
    "full": {
        "title": "A3 Full",
        "description": "Full CoralSAM-Track",
    },
}


def _load_eval_module():
    eval_path = Path(__file__).with_name("eval.py")
    spec = importlib.util.spec_from_file_location("revision_eval_module", eval_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load evaluation module from {eval_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVAL = None
CoralTracker = None
load_config = None
list_sequences = None


def _load_runtime_modules() -> None:
    global EVAL, CoralTracker, load_config, list_sequences
    if EVAL is None:
        EVAL = _load_eval_module()
    if CoralTracker is None or load_config is None or list_sequences is None:
        from src.tracker import CoralTracker as _CoralTracker
        from src.utils import load_config as _load_config, list_sequences as _list_sequences

        CoralTracker = _CoralTracker
        load_config = _load_config
        list_sequences = _list_sequences


def _write_csv_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = []
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


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _parse_extra_methods(raw_items: List[str]) -> List[Tuple[str, Path]]:
    parsed: List[Tuple[str, Path]] = []
    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"Invalid --extra_method '{item}'. Use LABEL=PATH")
        label, path_str = item.split("=", 1)
        label = label.strip()
        path = Path(path_str.strip())
        if not label:
            raise ValueError(f"Invalid --extra_method '{item}': empty label")
        if not path.exists():
            raise FileNotFoundError(f"Extra method directory not found: {path}")
        parsed.append((label, path))
    return parsed


def _apply_preset(base_cfg: dict, preset: str, output_base: Path, save_vis: bool) -> Tuple[dict, bool]:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("output", {})["base_dir"] = str(output_base)
    cfg["output"]["mode_tag"] = preset
    cfg["output"]["save_vis"] = bool(save_vis)
    cfg["output"]["save_masks"] = True

    drift_cfg = cfg.setdefault("drift", {})
    use_drift_correction = preset != "baseline"

    if preset == "baseline":
        drift_cfg["enabled"] = False
    elif preset == "soft_only":
        drift_cfg["enabled"] = True
        drift_cfg["hard_reinit_enabled"] = False
        drift_cfg["reinit_gate_enabled"] = False
    elif preset == "no_gate":
        drift_cfg["enabled"] = True
        drift_cfg["hard_reinit_enabled"] = True
        drift_cfg["reinit_gate_enabled"] = False
    elif preset == "full":
        drift_cfg["enabled"] = True
        drift_cfg["hard_reinit_enabled"] = True
        drift_cfg["reinit_gate_enabled"] = True
    else:
        raise ValueError(f"Unknown preset: {preset}")

    return cfg, use_drift_correction


def _evaluate_method(pred_root: Path, gt_root: Path, sequences: Optional[List[str]], label: str):
    _load_runtime_modules()
    return EVAL.evaluate_dataset(
        pred_root=pred_root,
        gt_root=gt_root,
        sequences=sequences,
        label=label,
    )


def _build_ablation_overview(
    mode_results: Dict[str, Dict[str, object]],
    run_summaries: Dict[str, dict],
    presets: List[str],
) -> List[Dict[str, object]]:
    anchor_mode = "baseline" if "baseline" in mode_results else presets[0]
    anchor_jf = float(mode_results[anchor_mode]["overall"]["JF_mean"])
    rows: List[Dict[str, object]] = []
    for preset in presets:
        eval_dict = mode_results[preset]
        overall = eval_dict["overall"]
        runtime = (run_summaries.get(preset) or {}).get("overall", {})
        rows.append(
            {
                "mode": preset,
                "title": PRESET_META[preset]["title"],
                "description": PRESET_META[preset]["description"],
                "J_mean": overall["J_mean"],
                "F_mean": overall["F_mean"],
                "JF_mean": overall["JF_mean"],
                "delta_JF_vs_baseline": round(float(overall["JF_mean"]) - anchor_jf, 4),
                "mean_tail_minus_head_JF": overall["mean_tail_minus_head_JF"],
                "mean_success_rate_50": overall["mean_success_rate_50"],
                "mean_fps": runtime.get("mean_fps"),
                "mean_ms_per_frame": runtime.get("mean_ms_per_frame"),
                "total_reinits": runtime.get("total_reinits"),
                "total_drift_events": runtime.get("total_drift_events"),
            }
        )
    return rows


def _build_ablation_per_sequence(
    mode_results: Dict[str, Dict[str, object]],
    presets: List[str],
) -> List[Dict[str, object]]:
    seqs = sorted(
        {
            seq
            for result in mode_results.values()
            for seq in result["per_sequence"].keys()
        }
    )
    rows: List[Dict[str, object]] = []
    for seq in seqs:
        row: Dict[str, object] = {"seq_name": seq}
        anchor_mode = "baseline" if "baseline" in mode_results else presets[0]
        anchor_metrics = mode_results[anchor_mode]["per_sequence"].get(seq, {})
        anchor_jf = float(anchor_metrics.get("JF_mean", 0.0))
        for preset in presets:
            metrics = mode_results[preset]["per_sequence"].get(seq, {})
            row[f"{preset}_JF_mean"] = metrics.get("JF_mean")
            row[f"{preset}_tail_minus_head_JF"] = metrics.get("tail_minus_head_JF")
            if preset != "baseline":
                row[f"{preset}_delta_vs_baseline"] = round(
                    float(metrics.get("JF_mean", 0.0)) - anchor_jf,
                    4,
                )
        rows.append(row)
    return rows


def _build_method_overview(
    method_results: Dict[str, Dict[str, object]],
    run_summaries: Dict[str, Optional[dict]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for label, eval_dict in method_results.items():
        runtime = (run_summaries.get(label) or {}).get("overall", {})
        overall = eval_dict["overall"]
        rows.append(
            {
                "method": label,
                "J_mean": overall["J_mean"],
                "F_mean": overall["F_mean"],
                "JF_mean": overall["JF_mean"],
                "mean_tail_minus_head_JF": overall["mean_tail_minus_head_JF"],
                "mean_success_rate_50": overall["mean_success_rate_50"],
                "mean_fps": runtime.get("mean_fps"),
                "mean_ms_per_frame": runtime.get("mean_ms_per_frame"),
                "total_reinits": runtime.get("total_reinits"),
                "total_drift_events": runtime.get("total_drift_events"),
            }
        )
    return rows


def _build_method_per_sequence(method_results: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    seqs = sorted(
        {
            seq
            for result in method_results.values()
            for seq in result["per_sequence"].keys()
        }
    )
    labels = list(method_results.keys())
    rows: List[Dict[str, object]] = []
    for seq in seqs:
        row: Dict[str, object] = {"seq_name": seq}
        for label in labels:
            metrics = method_results[label]["per_sequence"].get(seq, {})
            slug = _slug(label)
            row[f"{slug}_JF_mean"] = metrics.get("JF_mean")
            row[f"{slug}_tail_minus_head_JF"] = metrics.get("tail_minus_head_JF")
        rows.append(row)
    return rows


def _build_revision_markdown(
    output_base: Path,
    presets: List[str],
    ablation_rows: List[Dict[str, object]],
    method_rows: List[Dict[str, object]],
    extra_methods: List[Tuple[str, Path]],
) -> str:
    lines = ["# Revision Experiment Report", ""]
    lines.append(f"- Output base: `{output_base}`")
    lines.append(f"- Ablation presets: {', '.join(presets)}")
    if extra_methods:
        lines.append("- Extra methods:")
        for label, path in extra_methods:
            lines.append(f"  - {label}: `{path}`")
    lines.append("")
    lines.append("## Ablation Overview")
    lines.append("")
    lines.append("| Mode | Description | J&F | Delta vs Baseline | FPS | Reinits |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in ablation_rows:
        fps = row["mean_fps"]
        lines.append(
            "| {title} | {description} | {JF_mean:.4f} | {delta_JF_vs_baseline:+.4f} | {fps} | {reinits} |".format(
                title=row["title"],
                description=row["description"],
                JF_mean=float(row["JF_mean"]),
                delta_JF_vs_baseline=float(row["delta_JF_vs_baseline"]),
                fps="-" if fps is None else f"{float(fps):.2f}",
                reinits="-" if row["total_reinits"] is None else int(row["total_reinits"]),
            )
        )
    lines.append("")
    lines.append("## Method Overview")
    lines.append("")
    lines.append("| Method | J&F | Tail-Head | FPS |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in method_rows:
        fps = row["mean_fps"]
        lines.append(
            "| {method} | {JF_mean:.4f} | {tail:.4f} | {fps} |".format(
                method=row["method"],
                JF_mean=float(row["JF_mean"]),
                tail=float(row["mean_tail_minus_head_JF"]),
                fps="-" if fps is None else f"{float(fps):.2f}",
            )
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run minimal revision ablations and comparison reports for CoralSAM-Track."
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default=None, help="Base output directory. Defaults to outputs/<run_id>")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--seq", nargs="*", default=None, help="Specific sequences to process.")
    parser.add_argument("--all", action="store_true", help="Process all sequences in the dataset.")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=PRESET_ORDER,
        choices=PRESET_ORDER,
        help="Ablation presets to run.",
    )
    parser.add_argument(
        "--extra_method",
        action="append",
        default=[],
        help="Extra prediction masks to evaluate as LABEL=PATH.",
    )
    parser.add_argument(
        "--save_vis",
        action="store_true",
        help="Save overlay visualisations for the new ablation runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _load_runtime_modules()

    if not args.seq and not args.all:
        log.info("No sequence subset provided; defaulting to all dataset sequences.")
        args.all = True

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(args.output) if args.output else Path("outputs") / f"revision_{run_id}"
    output_base.mkdir(parents=True, exist_ok=True)
    report_dir = output_base / "revision_stats"
    report_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    data_root = cfg.get("data_root", "partial_coralvos/partial")
    gt_root = Path(data_root) / "masks"

    sequences = args.seq or list_sequences(data_root)
    log.info("Running revision experiments on %d sequences: %s", len(sequences), sequences)
    log.info("Output base: %s", output_base)

    mode_results: Dict[str, Dict[str, object]] = {}
    run_summaries: Dict[str, dict] = {}

    for preset in args.modes:
        log.info("=== Preset %s (%s) ===", preset, PRESET_META[preset]["description"])
        preset_cfg, use_drift_correction = _apply_preset(cfg, preset, output_base, args.save_vis)
        tracker = CoralTracker(
            cfg=preset_cfg,
            device=args.device,
            use_drift_correction=use_drift_correction,
        )

        all_results = {}
        for seq_name in sequences:
            log.info("Running %s on %s", preset, seq_name)
            result = tracker.run_sequence(
                data_root=data_root,
                seq_name=seq_name,
                save_output=True,
            )
            all_results[seq_name] = result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        run_summary = _build_run_summary(preset, all_results)
        run_summaries[preset] = run_summary

        mode_dir = output_base / preset
        mode_dir.mkdir(parents=True, exist_ok=True)
        with open(mode_dir / "run_summary.json", "w", encoding="utf-8") as f:
            json.dump(run_summary, f, indent=2, ensure_ascii=False)
        _write_csv_rows(mode_dir / "run_summary.csv", run_summary["table_rows"])

        pred_root = output_base / preset / "masks"
        eval_dict = _evaluate_method(
            pred_root=pred_root,
            gt_root=gt_root,
            sequences=sequences,
            label=PRESET_META[preset]["title"],
        )
        mode_results[preset] = eval_dict
        EVAL._export_eval_tables(report_dir, preset, eval_dict)

    if "baseline" in mode_results and "full" in mode_results:
        baseline_vs_full = EVAL.build_comparison(mode_results["baseline"], mode_results["full"])
        _write_csv_rows(report_dir / "full_vs_baseline_per_sequence.csv", baseline_vs_full["per_sequence"])
        _write_csv_rows(report_dir / "full_vs_baseline_per_frame.csv", baseline_vs_full["per_frame"])
        with open(report_dir / "full_vs_baseline_summary.json", "w", encoding="utf-8") as f:
            json.dump(baseline_vs_full, f, indent=2, ensure_ascii=False)

    ablation_overview = _build_ablation_overview(mode_results, run_summaries, args.modes)
    ablation_per_sequence = _build_ablation_per_sequence(mode_results, args.modes)
    _write_csv_rows(report_dir / "ablation_overview.csv", ablation_overview)
    _write_csv_rows(report_dir / "ablation_per_sequence.csv", ablation_per_sequence)

    extra_methods = _parse_extra_methods(args.extra_method)
    method_results: Dict[str, Dict[str, object]] = {}
    method_run_summaries: Dict[str, Optional[dict]] = {}

    if "baseline" in mode_results:
        method_results["SAM2 baseline"] = mode_results["baseline"]
        method_run_summaries["SAM2 baseline"] = run_summaries.get("baseline")
    if "full" in mode_results:
        method_results["CoralSAM-Track"] = mode_results["full"]
        method_run_summaries["CoralSAM-Track"] = run_summaries.get("full")

    for label, path in extra_methods:
        eval_dict = _evaluate_method(
            pred_root=path,
            gt_root=gt_root,
            sequences=sequences,
            label=label,
        )
        method_results[label] = eval_dict
        method_run_summaries[label] = EVAL._load_run_summary(path)
        slug = _slug(label)
        EVAL._export_eval_tables(report_dir, slug, eval_dict)

    if method_results:
        method_overview = _build_method_overview(method_results, method_run_summaries)
        method_per_sequence = _build_method_per_sequence(method_results)
        _write_csv_rows(report_dir / "comparison_overview.csv", method_overview)
        _write_csv_rows(report_dir / "comparison_per_sequence.csv", method_per_sequence)
    else:
        method_overview = []

    revision_report = _build_revision_markdown(
        output_base=output_base,
        presets=args.modes,
        ablation_rows=ablation_overview,
        method_rows=method_overview,
        extra_methods=extra_methods,
    )
    (report_dir / "revision_report.md").write_text(revision_report, encoding="utf-8")

    with open(report_dir / "revision_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "output_base": str(output_base),
                "modes": args.modes,
                "extra_methods": [{"label": label, "path": str(path)} for label, path in extra_methods],
                "ablation_overview": ablation_overview,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    log.info("Revision outputs saved to %s", report_dir)
    log.info("Ablation overview: %s", report_dir / "ablation_overview.csv")
    log.info("Method comparison: %s", report_dir / "comparison_overview.csv")


if __name__ == "__main__":
    main()

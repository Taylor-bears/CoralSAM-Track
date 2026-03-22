#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    def tqdm(iterable, **_kwargs):
        return iterable

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, read_mask

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval")


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = (pred_b & gt_b).sum()
    union = (pred_b | gt_b).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def _boundary_map(mask: np.ndarray, bound_th: float = 0.008) -> np.ndarray:
    from scipy.ndimage import binary_erosion

    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)

    h, w = mask.shape
    bound_pix = max(1, int(round(bound_th * np.sqrt(h**2 + w**2))))
    mask_bool = mask.astype(bool)
    eroded = binary_erosion(mask_bool, iterations=bound_pix)
    return mask_bool ^ eroded


def compute_f_measure(pred: np.ndarray, gt: np.ndarray, bound_th: float = 0.008) -> float:
    from scipy.ndimage import binary_dilation

    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)

    if gt_b.sum() == 0 and pred_b.sum() == 0:
        return 1.0
    if gt_b.sum() == 0 or pred_b.sum() == 0:
        return 0.0

    pred_bound = _boundary_map(pred_b, bound_th)
    gt_bound = _boundary_map(gt_b, bound_th)

    h, w = pred.shape
    bound_pix = max(1, int(round(bound_th * np.sqrt(h**2 + w**2))))
    pred_bound_dil = binary_dilation(pred_bound, iterations=bound_pix)
    gt_bound_dil = binary_dilation(gt_bound, iterations=bound_pix)

    p_num = (pred_bound & gt_bound_dil).sum()
    p_den = pred_bound.sum()
    r_num = (gt_bound & pred_bound_dil).sum()
    r_den = gt_bound.sum()

    precision = float(p_num) / float(p_den) if p_den > 0 else 0.0
    recall = float(r_num) / float(r_den) if r_den > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_std(values: List[float]) -> float:
    return float(np.std(values)) if values else 0.0


def _safe_median(values: List[float]) -> float:
    return float(np.median(values)) if values else 0.0


def _safe_percentile(values: List[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def _mean_for_indices(values: List[float], idxs: np.ndarray) -> float:
    if len(idxs) == 0:
        return 0.0
    return float(np.mean([values[int(i)] for i in idxs]))


def _sequence_summary(
    seq_name: str,
    frame_records: List[Dict[str, object]],
    missing_pred: int,
) -> Dict[str, object]:
    if not frame_records:
        return {
            "seq_name": seq_name,
            "J_mean": 0.0,
            "F_mean": 0.0,
            "JF_mean": 0.0,
            "n_frames": 0,
            "n_missing_predictions": 0,
        }

    j_scores = [float(row["J"]) for row in frame_records]
    f_scores = [float(row["F"]) for row in frame_records]
    jf_scores = [float(row["JF"]) for row in frame_records]
    n = len(frame_records)
    thirds = np.array_split(np.arange(n), 3)

    worst_k = max(1, int(np.ceil(n * 0.1)))
    sorted_jf = sorted(jf_scores)
    best_idx = int(np.argmax(jf_scores))
    worst_idx = int(np.argmin(jf_scores))

    for label, idxs in zip(("head", "mid", "tail"), thirds):
        for idx in idxs:
            frame_records[int(idx)]["segment"] = label

    return {
        "seq_name": seq_name,
        "J_mean": _round(_safe_mean(j_scores)),
        "F_mean": _round(_safe_mean(f_scores)),
        "JF_mean": _round(_safe_mean(jf_scores)),
        "J_std": _round(_safe_std(j_scores)),
        "F_std": _round(_safe_std(f_scores)),
        "JF_std": _round(_safe_std(jf_scores)),
        "J_median": _round(_safe_median(j_scores)),
        "F_median": _round(_safe_median(f_scores)),
        "JF_median": _round(_safe_median(jf_scores)),
        "J_min": _round(min(j_scores)),
        "F_min": _round(min(f_scores)),
        "JF_min": _round(min(jf_scores)),
        "J_max": _round(max(j_scores)),
        "F_max": _round(max(f_scores)),
        "JF_max": _round(max(jf_scores)),
        "head_JF": _round(_mean_for_indices(jf_scores, thirds[0])),
        "mid_JF": _round(_mean_for_indices(jf_scores, thirds[1])),
        "tail_JF": _round(_mean_for_indices(jf_scores, thirds[2])),
        "tail_minus_head_JF": _round(
            _mean_for_indices(jf_scores, thirds[2]) - _mean_for_indices(jf_scores, thirds[0])
        ),
        "success_rate_50": _round(sum(v >= 0.50 for v in jf_scores) / n, 4),
        "success_rate_75": _round(sum(v >= 0.75 for v in jf_scores) / n, 4),
        "failure_rate_25": _round(sum(v < 0.25 for v in jf_scores) / n, 4),
        "worst_10pct_JF": _round(_safe_mean(sorted_jf[:worst_k])),
        "best_frame": frame_records[best_idx]["frame_name"],
        "best_frame_JF": _round(jf_scores[best_idx]),
        "worst_frame": frame_records[worst_idx]["frame_name"],
        "worst_frame_JF": _round(jf_scores[worst_idx]),
        "mean_pred_area_ratio": _round(
            _safe_mean([float(row["pred_area_ratio"]) for row in frame_records]), 6
        ),
        "mean_gt_area_ratio": _round(
            _safe_mean([float(row["gt_area_ratio"]) for row in frame_records]), 6
        ),
        "n_missing_predictions": int(missing_pred),
        "missing_prediction_rate": _round(missing_pred / n, 4),
        "n_frames": n,
    }


def evaluate_sequence(
    pred_dir: Path,
    gt_dir: Path,
    seq_name: str,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    gt_files = sorted(gt_dir.glob("*.png"))
    if not gt_files:
        gt_files = sorted(gt_dir.glob("*.jpg"))
    if not gt_files:
        log.warning("No GT masks found in %s", gt_dir)
        return _sequence_summary(seq_name, [], 0), []

    frame_records: List[Dict[str, object]] = []
    missing_pred = 0

    for order_idx, gt_path in enumerate(tqdm(gt_files, desc=seq_name, leave=False)):
        frame_name = gt_path.stem
        pred_path = pred_dir / f"{frame_name}.png"
        if not pred_path.exists():
            pred_path = pred_dir / f"{frame_name}.jpg"

        gt_mask = read_mask(str(gt_path))
        gt_area = int(gt_mask.sum())
        image_pixels = int(gt_mask.shape[0] * gt_mask.shape[1])

        if not pred_path.exists():
            missing_pred += 1
            pred_mask = np.zeros_like(gt_mask, dtype=bool)
            pred_area = 0
            j_score = 0.0
            f_score = 0.0
            prediction_missing = True
        else:
            pred_mask = read_mask(str(pred_path))
            if pred_mask.shape != gt_mask.shape:
                import cv2

                pred_mask = cv2.resize(
                    pred_mask.astype(np.uint8),
                    (gt_mask.shape[1], gt_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            pred_area = int(pred_mask.sum())
            j_score = compute_iou(pred_mask, gt_mask)
            f_score = compute_f_measure(pred_mask, gt_mask)
            prediction_missing = False

        frame_number = int(frame_name) if frame_name.isdigit() else order_idx
        jf_score = (j_score + f_score) / 2.0
        frame_records.append(
            {
                "seq_name": seq_name,
                "frame_name": frame_name,
                "frame_number": frame_number,
                "order_idx": order_idx,
                "frame_position_pct": _round(order_idx / max(len(gt_files) - 1, 1), 6),
                "prediction_missing": prediction_missing,
                "pred_area_pixels": pred_area,
                "gt_area_pixels": gt_area,
                "pred_area_ratio": _round(pred_area / image_pixels, 6),
                "gt_area_ratio": _round(gt_area / image_pixels, 6),
                "J": _round(j_score),
                "F": _round(f_score),
                "JF": _round(jf_score),
            }
        )

    if missing_pred:
        log.warning(
            "Sequence '%s': %d GT frames had no prediction (scored as 0).",
            seq_name,
            missing_pred,
        )

    return _sequence_summary(seq_name, frame_records, missing_pred), frame_records


def evaluate_dataset(
    pred_root: Path,
    gt_root: Path,
    sequences: Optional[List[str]] = None,
    label: str = "Method",
) -> Dict[str, object]:
    if sequences is None:
        sequences = sorted([d.name for d in pred_root.iterdir() if d.is_dir()])

    per_sequence: Dict[str, object] = {}
    frame_records: List[Dict[str, object]] = []
    for seq in sequences:
        pred_dir = pred_root / seq
        gt_dir = gt_root / seq
        if not pred_dir.exists():
            log.warning("Prediction dir not found: %s", pred_dir)
            continue
        if not gt_dir.exists():
            log.warning("GT dir not found: %s - skipping", gt_dir)
            continue

        seq_summary, seq_frames = evaluate_sequence(pred_dir, gt_dir, seq)
        per_sequence[seq] = seq_summary
        frame_records.extend(seq_frames)

    valid = [v for v in per_sequence.values() if v.get("n_frames", 0) > 0]
    if valid:
        jf_means = [float(v["JF_mean"]) for v in valid]
        overall = {
            "J_mean": _round(_safe_mean([float(v["J_mean"]) for v in valid])),
            "F_mean": _round(_safe_mean([float(v["F_mean"]) for v in valid])),
            "JF_mean": _round(_safe_mean(jf_means)),
            "frame_weighted_J_mean": _round(_safe_mean([float(r["J"]) for r in frame_records])),
            "frame_weighted_F_mean": _round(_safe_mean([float(r["F"]) for r in frame_records])),
            "frame_weighted_JF_mean": _round(_safe_mean([float(r["JF"]) for r in frame_records])),
            "median_sequence_JF": _round(_safe_median(jf_means)),
            "sequence_JF_std": _round(_safe_std(jf_means)),
            "mean_tail_minus_head_JF": _round(
                _safe_mean([float(v["tail_minus_head_JF"]) for v in valid])
            ),
            "mean_success_rate_50": _round(
                _safe_mean([float(v["success_rate_50"]) for v in valid])
            ),
            "mean_success_rate_75": _round(
                _safe_mean([float(v["success_rate_75"]) for v in valid])
            ),
            "mean_worst_10pct_JF": _round(
                _safe_mean([float(v["worst_10pct_JF"]) for v in valid])
            ),
            "total_frames": int(sum(int(v["n_frames"]) for v in valid)),
            "n_sequences": len(valid),
        }
    else:
        overall = {
            "J_mean": 0.0,
            "F_mean": 0.0,
            "JF_mean": 0.0,
            "frame_weighted_J_mean": 0.0,
            "frame_weighted_F_mean": 0.0,
            "frame_weighted_JF_mean": 0.0,
            "median_sequence_JF": 0.0,
            "sequence_JF_std": 0.0,
            "mean_tail_minus_head_JF": 0.0,
            "mean_success_rate_50": 0.0,
            "mean_success_rate_75": 0.0,
            "mean_worst_10pct_JF": 0.0,
            "total_frames": 0,
            "n_sequences": 0,
        }

    return {
        "label": label,
        "per_sequence": per_sequence,
        "overall": overall,
        "frame_records": frame_records,
    }


def print_results(eval_dict: Dict[str, object]) -> None:
    label = str(eval_dict["label"])
    overall = eval_dict["overall"]
    per_seq = eval_dict["per_sequence"]

    print(f"\n{'=' * 80}")
    print(f"  Results for: {label}")
    print(f"{'=' * 80}")
    print(f"  {'Sequence':<18} {'J':>8} {'F':>8} {'J&F':>8} {'Tail-Head':>10} {'S@0.5':>8} {'Frames':>8}")
    print(f"  {'-' * 72}")
    for seq, m in sorted(per_seq.items()):
        print(
            f"  {seq:<18} {float(m['J_mean']):>8.4f} {float(m['F_mean']):>8.4f} "
            f"{float(m['JF_mean']):>8.4f} {float(m['tail_minus_head_JF']):>10.4f} "
            f"{float(m['success_rate_50']):>8.4f} {int(m['n_frames']):>8}"
        )
    print(f"  {'=' * 72}")
    print(
        f"  {'OVERALL':<18} {float(overall['J_mean']):>8.4f} {float(overall['F_mean']):>8.4f} "
        f"{float(overall['JF_mean']):>8.4f} {float(overall['mean_tail_minus_head_JF']):>10.4f} "
        f"{float(overall['mean_success_rate_50']):>8.4f} {int(overall['total_frames']):>8}"
    )
    print(f"{'=' * 80}\n")


def build_comparison(res_a: Dict[str, object], res_b: Dict[str, object]) -> Dict[str, object]:
    per_sequence_rows: List[Dict[str, object]] = []
    seqs = sorted(set(res_a["per_sequence"]) | set(res_b["per_sequence"]))
    for seq in seqs:
        ma = res_a["per_sequence"].get(seq, {})
        mb = res_b["per_sequence"].get(seq, {})
        per_sequence_rows.append(
            {
                "seq_name": seq,
                "baseline_J_mean": ma.get("J_mean", 0.0),
                "method_J_mean": mb.get("J_mean", 0.0),
                "delta_J_mean": _round(float(mb.get("J_mean", 0.0)) - float(ma.get("J_mean", 0.0))),
                "baseline_JF_mean": ma.get("JF_mean", 0.0),
                "method_JF_mean": mb.get("JF_mean", 0.0),
                "delta_JF_mean": _round(float(mb.get("JF_mean", 0.0)) - float(ma.get("JF_mean", 0.0))),
                "baseline_tail_minus_head_JF": ma.get("tail_minus_head_JF", 0.0),
                "method_tail_minus_head_JF": mb.get("tail_minus_head_JF", 0.0),
                "delta_tail_minus_head_JF": _round(
                    float(mb.get("tail_minus_head_JF", 0.0)) - float(ma.get("tail_minus_head_JF", 0.0))
                ),
                "baseline_success_rate_50": ma.get("success_rate_50", 0.0),
                "method_success_rate_50": mb.get("success_rate_50", 0.0),
                "delta_success_rate_50": _round(
                    float(mb.get("success_rate_50", 0.0)) - float(ma.get("success_rate_50", 0.0))
                ),
            }
        )

    frame_map_a = {
        (str(row["seq_name"]), str(row["frame_name"])): row for row in res_a["frame_records"]
    }
    frame_map_b = {
        (str(row["seq_name"]), str(row["frame_name"])): row for row in res_b["frame_records"]
    }
    frame_rows: List[Dict[str, object]] = []
    for key in sorted(set(frame_map_a) | set(frame_map_b)):
        a = frame_map_a.get(key)
        b = frame_map_b.get(key)
        base_jf = float(a["JF"]) if a else None
        method_jf = float(b["JF"]) if b else None
        delta_jf = None if base_jf is None or method_jf is None else method_jf - base_jf
        frame_rows.append(
            {
                "seq_name": key[0],
                "frame_name": key[1],
                "segment": (b or a or {}).get("segment"),
                "baseline_J": None if not a else a["J"],
                "method_J": None if not b else b["J"],
                "delta_J": None if not (a and b) else _round(float(b["J"]) - float(a["J"])),
                "baseline_F": None if not a else a["F"],
                "method_F": None if not b else b["F"],
                "delta_F": None if not (a and b) else _round(float(b["F"]) - float(a["F"])),
                "baseline_JF": None if base_jf is None else _round(base_jf),
                "method_JF": None if method_jf is None else _round(method_jf),
                "delta_JF": None if delta_jf is None else _round(delta_jf),
                "frame_position_pct": (b or a or {}).get("frame_position_pct"),
            }
        )

    delta_values = [float(row["delta_JF"]) for row in frame_rows if row["delta_JF"] is not None]
    tail_deltas = [
        float(row["delta_JF"]) for row in frame_rows
        if row["delta_JF"] is not None and row.get("segment") == "tail"
    ]
    head_deltas = [
        float(row["delta_JF"]) for row in frame_rows
        if row["delta_JF"] is not None and row.get("segment") == "head"
    ]

    improved_sequences = sum(float(row["delta_JF_mean"]) > 0 for row in per_sequence_rows)
    worsened_sequences = sum(float(row["delta_JF_mean"]) < 0 for row in per_sequence_rows)
    improved_frames = sum(delta > 0 for delta in delta_values)
    worsened_frames = sum(delta < 0 for delta in delta_values)
    improved_frames_1pp = sum(delta > 0.01 for delta in delta_values)
    worsened_frames_1pp = sum(delta < -0.01 for delta in delta_values)

    best_seq = max(per_sequence_rows, key=lambda row: float(row["delta_JF_mean"]), default=None)
    worst_seq = min(per_sequence_rows, key=lambda row: float(row["delta_JF_mean"]), default=None)

    summary = {
        "baseline_label": res_a["label"],
        "method_label": res_b["label"],
        "delta_J_mean": _round(float(res_b["overall"]["J_mean"]) - float(res_a["overall"]["J_mean"])),
        "delta_F_mean": _round(float(res_b["overall"]["F_mean"]) - float(res_a["overall"]["F_mean"])),
        "delta_JF_mean": _round(float(res_b["overall"]["JF_mean"]) - float(res_a["overall"]["JF_mean"])),
        "delta_frame_weighted_JF_mean": _round(
            float(res_b["overall"]["frame_weighted_JF_mean"]) - float(res_a["overall"]["frame_weighted_JF_mean"])
        ),
        "improved_sequences": improved_sequences,
        "worsened_sequences": worsened_sequences,
        "improved_frames": improved_frames,
        "worsened_frames": worsened_frames,
        "improved_frames_gt_1pp": improved_frames_1pp,
        "worsened_frames_gt_1pp": worsened_frames_1pp,
        "mean_delta_JF": _round(_safe_mean(delta_values)),
        "median_delta_JF": _round(_safe_median(delta_values)),
        "p25_delta_JF": _round(_safe_percentile(delta_values, 25)),
        "p75_delta_JF": _round(_safe_percentile(delta_values, 75)),
        "head_mean_delta_JF": _round(_safe_mean(head_deltas)),
        "tail_mean_delta_JF": _round(_safe_mean(tail_deltas)),
        "largest_gain_sequence": None if not best_seq else best_seq["seq_name"],
        "largest_gain_delta_JF": None if not best_seq else best_seq["delta_JF_mean"],
        "largest_drop_sequence": None if not worst_seq else worst_seq["seq_name"],
        "largest_drop_delta_JF": None if not worst_seq else worst_seq["delta_JF_mean"],
    }

    return {
        "summary": summary,
        "per_sequence": per_sequence_rows,
        "per_frame": frame_rows,
    }


def print_comparison(comparison: Dict[str, object]) -> None:
    rows = comparison["per_sequence"]
    summary = comparison["summary"]
    print(f"\n{'=' * 96}")
    print(f"  Comparison: {summary['baseline_label']}  vs  {summary['method_label']}")
    print(f"{'=' * 96}")
    print(
        f"  {'Sequence':<18} {'Base JF':>10} {'Method JF':>10} {'Delta':>10} "
        f"{'Base Tail-H':>12} {'Method Tail-H':>14} {'Delta Tail-H':>14}"
    )
    print(f"  {'-' * 90}")
    for row in rows:
        print(
            f"  {row['seq_name']:<18} {float(row['baseline_JF_mean']):>10.4f} "
            f"{float(row['method_JF_mean']):>10.4f} {float(row['delta_JF_mean']):>10.4f} "
            f"{float(row['baseline_tail_minus_head_JF']):>12.4f} "
            f"{float(row['method_tail_minus_head_JF']):>14.4f} "
            f"{float(row['delta_tail_minus_head_JF']):>14.4f}"
        )
    print(f"  {'=' * 90}")
    print(
        f"  OVERALL delta J&F: {float(summary['delta_JF_mean']):+.4f} | "
        f"frame-weighted delta J&F: {float(summary['delta_frame_weighted_JF_mean']):+.4f}"
    )
    print(
        f"  Improved seq/frame: {summary['improved_sequences']} / {summary['improved_frames']} | "
        f"Worsened seq/frame: {summary['worsened_sequences']} / {summary['worsened_frames']}"
    )
    print(
        f"  Tail mean delta J&F: {float(summary['tail_mean_delta_JF']):+.4f} | "
        f"Head mean delta J&F: {float(summary['head_mean_delta_JF']):+.4f}"
    )
    print(f"{'=' * 96}\n")


def _write_csv_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
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


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_run_summary(mask_root: Path) -> Optional[dict]:
    return _load_json(mask_root.parent / "run_summary.json")


def _method_overview_row(eval_dict: Dict[str, object], run_summary: Optional[dict]) -> Dict[str, object]:
    overall = eval_dict["overall"]
    runtime = (run_summary or {}).get("overall", {})
    return {
        "method": eval_dict["label"],
        "J_mean": overall["J_mean"],
        "F_mean": overall["F_mean"],
        "JF_mean": overall["JF_mean"],
        "frame_weighted_JF_mean": overall["frame_weighted_JF_mean"],
        "mean_tail_minus_head_JF": overall["mean_tail_minus_head_JF"],
        "mean_success_rate_50": overall["mean_success_rate_50"],
        "mean_worst_10pct_JF": overall["mean_worst_10pct_JF"],
        "n_sequences": overall["n_sequences"],
        "total_frames": overall["total_frames"],
        "mean_fps": runtime.get("mean_fps"),
        "mean_ms_per_frame": runtime.get("mean_ms_per_frame"),
        "total_runtime_s": runtime.get("total_runtime_s"),
        "total_reinits": runtime.get("total_reinits"),
        "total_drift_events": runtime.get("total_drift_events"),
    }


def _export_eval_tables(report_dir: Path, prefix: str, eval_dict: Dict[str, object]) -> None:
    seq_rows = [{"seq_name": seq, **metrics} for seq, metrics in sorted(eval_dict["per_sequence"].items())]
    _write_csv_rows(report_dir / f"{prefix}_per_sequence.csv", seq_rows)
    _write_csv_rows(report_dir / f"{prefix}_per_frame.csv", eval_dict["frame_records"])
    with open(report_dir / f"{prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump(eval_dict, f, indent=2, ensure_ascii=False)


def _build_markdown_report(
    overview_rows: List[Dict[str, object]],
    comparison: Optional[Dict[str, object]],
    drift_run_summary: Optional[dict],
) -> str:
    lines = ["# CoralSAM-Track Evaluation Report", ""]
    lines.append("## Overall Metrics")
    lines.append("")
    lines.append("| Method | J | F | J&F | Weighted J&F | Tail-Head | S@0.5 | FPS | ms/frame | Re-inits |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in overview_rows:
        lines.append(
            "| {method} | {J_mean:.4f} | {F_mean:.4f} | {JF_mean:.4f} | {frame_weighted_JF_mean:.4f} | "
            "{mean_tail_minus_head_JF:.4f} | {mean_success_rate_50:.4f} | {fps} | {ms} | {reinits} |".format(
                method=row["method"],
                J_mean=float(row["J_mean"]),
                F_mean=float(row["F_mean"]),
                JF_mean=float(row["JF_mean"]),
                frame_weighted_JF_mean=float(row["frame_weighted_JF_mean"]),
                mean_tail_minus_head_JF=float(row["mean_tail_minus_head_JF"]),
                mean_success_rate_50=float(row["mean_success_rate_50"]),
                fps="-" if row["mean_fps"] is None else f"{float(row['mean_fps']):.2f}",
                ms="-" if row["mean_ms_per_frame"] is None else f"{float(row['mean_ms_per_frame']):.2f}",
                reinits="-" if row["total_reinits"] is None else int(row["total_reinits"]),
            )
        )
    lines.append("")

    if comparison is not None:
        summary = comparison["summary"]
        lines.append("## Baseline vs Drift-Corrected Summary")
        lines.append("")
        lines.append(f"- Overall delta J&F: {float(summary['delta_JF_mean']):+.4f}")
        lines.append(f"- Frame-weighted delta J&F: {float(summary['delta_frame_weighted_JF_mean']):+.4f}")
        lines.append(
            f"- Improved sequences / worsened sequences: {summary['improved_sequences']} / {summary['worsened_sequences']}"
        )
        lines.append(
            f"- Improved frames / worsened frames: {summary['improved_frames']} / {summary['worsened_frames']}"
        )
        lines.append(
            f"- Tail mean delta J&F / head mean delta J&F: {float(summary['tail_mean_delta_JF']):+.4f} / {float(summary['head_mean_delta_JF']):+.4f}"
        )
        lines.append(
            f"- Largest gain sequence: {summary['largest_gain_sequence']} ({float(summary['largest_gain_delta_JF']):+.4f})"
            if summary["largest_gain_sequence"]
            else "- Largest gain sequence: -"
        )
        lines.append(
            f"- Largest drop sequence: {summary['largest_drop_sequence']} ({float(summary['largest_drop_delta_JF']):+.4f})"
            if summary["largest_drop_sequence"]
            else "- Largest drop sequence: -"
        )
        lines.append("")
        lines.append("## Per-Sequence Comparison")
        lines.append("")
        lines.append("| Sequence | Base J&F | Drift J&F | Delta | Base Tail-Head | Drift Tail-Head | Delta Tail-Head | Re-inits |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        drift_seq_summary = (drift_run_summary or {}).get("per_sequence", {})
        for row in comparison["per_sequence"]:
            reinits = drift_seq_summary.get(row["seq_name"], {}).get("reinit_count", "-")
            lines.append(
                "| {seq} | {base:.4f} | {method:.4f} | {delta:+.4f} | {bth:.4f} | {mth:.4f} | {dth:+.4f} | {reinits} |".format(
                    seq=row["seq_name"],
                    base=float(row["baseline_JF_mean"]),
                    method=float(row["method_JF_mean"]),
                    delta=float(row["delta_JF_mean"]),
                    bth=float(row["baseline_tail_minus_head_JF"]),
                    mth=float(row["method_tail_minus_head_JF"]),
                    dth=float(row["delta_tail_minus_head_JF"]),
                    reinits=reinits,
                )
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CoralSAM-Track predictions")
    parser.add_argument(
        "--pred_dir",
        type=str,
        required=True,
        help="Directory with per-sequence prediction masks (e.g. outputs/with_drift_corr/masks).",
    )
    parser.add_argument(
        "--baseline_dir",
        type=str,
        default=None,
        help="Baseline prediction masks directory for comparison (optional).",
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--seq",
        nargs="*",
        default=None,
        help="Specific sequence names to evaluate (default: all in pred_dir).",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="If set, save evaluation results as JSON.",
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        default=None,
        help="Directory for CSV/Markdown report artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_root = cfg.get("data_root", "partial_coralvos/partial")
    gt_root = Path(data_root) / "masks"

    pred_root = Path(args.pred_dir)
    if not pred_root.exists():
        log.error("Prediction directory not found: %s", pred_root)
        sys.exit(1)

    report_dir = (
        Path(args.report_dir)
        if args.report_dir
        else Path(args.output_json).parent / "paper_stats"
        if args.output_json
        else pred_root.parent / "paper_stats"
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    res = evaluate_dataset(
        pred_root=pred_root,
        gt_root=gt_root,
        sequences=args.seq,
        label="w/ drift correction" if "with_drift" in str(pred_root) else pred_root.parent.name,
    )
    print_results(res)
    _export_eval_tables(report_dir, "method", res)

    pred_run_summary = _load_run_summary(pred_root)
    overview_rows = [_method_overview_row(res, pred_run_summary)]
    combined_output: Dict[str, object] = {
        "method": res,
        "method_overview": overview_rows,
    }

    if args.baseline_dir:
        baseline_root = Path(args.baseline_dir)
        log.info("Evaluating baseline in %s", baseline_root)
        res_baseline = evaluate_dataset(
            pred_root=baseline_root,
            gt_root=gt_root,
            sequences=args.seq,
            label="baseline (no drift corr)",
        )
        print_results(res_baseline)
        _export_eval_tables(report_dir, "baseline", res_baseline)

        baseline_run_summary = _load_run_summary(baseline_root)
        overview_rows = [
            _method_overview_row(res_baseline, baseline_run_summary),
            _method_overview_row(res, pred_run_summary),
        ]
        _write_csv_rows(report_dir / "method_overview.csv", overview_rows)

        comparison = build_comparison(res_baseline, res)
        print_comparison(comparison)
        _write_csv_rows(report_dir / "comparison_per_sequence.csv", comparison["per_sequence"])
        _write_csv_rows(report_dir / "comparison_per_frame.csv", comparison["per_frame"])
        with open(report_dir / "comparison_summary.json", "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)

        markdown = _build_markdown_report(overview_rows, comparison, pred_run_summary)
        with open(report_dir / "paper_report.md", "w", encoding="utf-8") as f:
            f.write(markdown)

        combined_output = {
            "baseline": res_baseline,
            "with_drift_correction": res,
            "comparison": comparison,
            "method_overview": overview_rows,
        }
    else:
        _write_csv_rows(report_dir / "method_overview.csv", overview_rows)
        markdown = _build_markdown_report(overview_rows, None, pred_run_summary)
        with open(report_dir / "paper_report.md", "w", encoding="utf-8") as f:
            f.write(markdown)

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(combined_output, f, indent=2, ensure_ascii=False)
        log.info("Results saved to %s", args.output_json)

    log.info("Paper-oriented report artifacts saved to %s", report_dir)


if __name__ == "__main__":
    main()

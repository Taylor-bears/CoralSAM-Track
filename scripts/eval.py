#!/usr/bin/env python
"""
eval.py – Evaluate predicted masks against GT on CoralVOS sequences.

Metrics (DAVIS / VOS standard):
  J  – Mean Jaccard (region similarity / IoU) per frame, averaged over the
       sequence, then over the dataset.
  F  – Mean F-measure (contour accuracy) per frame, averaged similarly.
  J&F – Mean of J and F.

Usage
-----
# Evaluate with-drift-correction predictions:
python scripts/eval.py --pred_dir outputs/with_drift_corr/masks

# Evaluate baseline (no drift correction):
python scripts/eval.py --pred_dir outputs/baseline/masks

# Compare both side-by-side:
python scripts/eval.py \
    --pred_dir outputs/with_drift_corr/masks \
    --baseline_dir outputs/baseline/masks

# Evaluate specific sequences only:
python scripts/eval.py --pred_dir outputs/with_drift_corr/masks --seq video102 video75
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, list_sequences, read_mask

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval")


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------

def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Jaccard index (IoU) between two binary masks."""
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = (pred_b & gt_b).sum()
    union = (pred_b | gt_b).sum()
    if union == 0:
        # Both empty → perfect match
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def _boundary_map(mask: np.ndarray, bound_th: float = 0.008) -> np.ndarray:
    """Compute the binary boundary map of a segmentation mask.

    Uses morphological erosion to extract the contour pixels, scaled
    by the ``bound_th`` fraction of the diagonal.
    """
    from scipy.ndimage import binary_erosion

    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)

    H, W = mask.shape
    # Threshold determines contour width relative to image diagonal
    bound_pix = max(1, int(round(bound_th * np.sqrt(H ** 2 + W ** 2))))
    mask_bool = mask.astype(bool)
    eroded = binary_erosion(mask_bool, iterations=bound_pix)
    return mask_bool ^ eroded  # XOR → boundary pixels


def compute_f_measure(
    pred: np.ndarray, gt: np.ndarray, bound_th: float = 0.008
) -> float:
    """Boundary F-measure between two binary masks (DAVIS convention).

    Precision = |{pred_boundary ∩ gt_boundary_dilated}| / |pred_boundary|
    Recall    = |{gt_boundary ∩ pred_boundary_dilated}| / |gt_boundary|
    F         = 2 * P * R / (P + R)
    """
    from scipy.ndimage import binary_dilation

    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)

    if gt_b.sum() == 0 and pred_b.sum() == 0:
        return 1.0
    if gt_b.sum() == 0 or pred_b.sum() == 0:
        return 0.0

    pred_bound = _boundary_map(pred_b, bound_th)
    gt_bound = _boundary_map(gt_b, bound_th)

    H, W = pred.shape
    bound_pix = max(1, int(round(bound_th * np.sqrt(H ** 2 + W ** 2))))

    # Dilate boundaries for tolerance
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


# ---------------------------------------------------------------------------
# Sequence-level evaluation
# ---------------------------------------------------------------------------

def evaluate_sequence(
    pred_dir: Path,
    gt_dir: Path,
    seq_name: str,
) -> Dict[str, float]:
    """Compute per-frame J and F for one sequence, return mean values.

    Iterates over GT annotated frames (standard VOS evaluation convention).
    Frames with GT but no prediction are scored as 0. Frames with prediction
    but no GT (unannotated frames common in CoralVOS) are silently ignored.

    Args:
        pred_dir:  Directory containing <frame_name>.png predicted masks.
        gt_dir:    Directory containing <frame_name>.png GT masks.
        seq_name:  Name (for logging only).

    Returns:
        dict with keys: J_mean, F_mean, JF_mean, n_frames
    """
    gt_files = sorted(gt_dir.glob("*.png"))
    if not gt_files:
        gt_files = sorted(gt_dir.glob("*.jpg"))
    if not gt_files:
        log.warning("No GT masks found in %s", gt_dir)
        return {"J_mean": 0.0, "F_mean": 0.0, "JF_mean": 0.0, "n_frames": 0}

    j_scores: List[float] = []
    f_scores: List[float] = []
    missing_pred = 0

    for gt_path in tqdm(gt_files, desc=seq_name, leave=False):
        frame_name = gt_path.stem
        pred_path = pred_dir / f"{frame_name}.png"
        if not pred_path.exists():
            pred_path = pred_dir / f"{frame_name}.jpg"
        if not pred_path.exists():
            missing_pred += 1
            j_scores.append(0.0)
            f_scores.append(0.0)
            continue

        pred_mask = read_mask(str(pred_path))
        gt_mask = read_mask(str(gt_path))

        # Resize pred to GT size if needed
        if pred_mask.shape != gt_mask.shape:
            import cv2
            pred_mask = cv2.resize(
                pred_mask.astype(np.uint8),
                (gt_mask.shape[1], gt_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        j_scores.append(compute_iou(pred_mask, gt_mask))
        f_scores.append(compute_f_measure(pred_mask, gt_mask))

    if missing_pred:
        log.warning(
            "Sequence '%s': %d GT frames had no prediction (scored as 0).",
            seq_name, missing_pred,
        )

    n = len(j_scores)
    if n == 0:
        return {"J_mean": 0.0, "F_mean": 0.0, "JF_mean": 0.0, "n_frames": 0}

    j_mean = float(np.mean(j_scores))
    f_mean = float(np.mean(f_scores))
    return {
        "J_mean": round(j_mean, 4),
        "F_mean": round(f_mean, 4),
        "JF_mean": round((j_mean + f_mean) / 2, 4),
        "n_frames": n,
    }


# ---------------------------------------------------------------------------
# Dataset-level evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(
    pred_root: Path,
    gt_root: Path,
    sequences: Optional[List[str]] = None,
    label: str = "Method",
) -> Dict:
    """Evaluate all sequences under pred_root / gt_root.

    Returns a dict with per-sequence metrics and overall mean.
    """
    if sequences is None:
        # Auto-discover from pred_root
        sequences = sorted([d.name for d in pred_root.iterdir() if d.is_dir()])

    results = {}
    for seq in sequences:
        pred_dir = pred_root / seq
        gt_dir = gt_root / seq
        if not pred_dir.exists():
            log.warning("Prediction dir not found: %s", pred_dir)
            continue
        if not gt_dir.exists():
            log.warning("GT dir not found: %s — skipping", gt_dir)
            continue
        results[seq] = evaluate_sequence(pred_dir, gt_dir, seq)

    # Aggregate
    valid = [v for v in results.values() if v["n_frames"] > 0]
    if valid:
        overall = {
            "J_mean": round(float(np.mean([v["J_mean"] for v in valid])), 4),
            "F_mean": round(float(np.mean([v["F_mean"] for v in valid])), 4),
            "JF_mean": round(float(np.mean([v["JF_mean"] for v in valid])), 4),
            "n_sequences": len(valid),
        }
    else:
        overall = {"J_mean": 0, "F_mean": 0, "JF_mean": 0, "n_sequences": 0}

    return {"label": label, "per_sequence": results, "overall": overall}


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_results(eval_dict: dict) -> None:
    label = eval_dict["label"]
    overall = eval_dict["overall"]
    per_seq = eval_dict["per_sequence"]

    print(f"\n{'='*65}")
    print(f"  Results for: {label}")
    print(f"{'='*65}")
    print(f"  {'Sequence':<20} {'J':>8} {'F':>8} {'J&F':>8} {'Frames':>8}")
    print(f"  {'-'*52}")
    for seq, m in sorted(per_seq.items()):
        print(
            f"  {seq:<20} {m['J_mean']:>8.4f} {m['F_mean']:>8.4f} "
            f"{m['JF_mean']:>8.4f} {m['n_frames']:>8}"
        )
    print(f"  {'='*52}")
    print(
        f"  {'OVERALL':<20} {overall['J_mean']:>8.4f} {overall['F_mean']:>8.4f} "
        f"{overall['JF_mean']:>8.4f} ({overall['n_sequences']} seqs)"
    )
    print(f"{'='*65}\n")


def print_comparison(res_a: dict, res_b: dict) -> None:
    """Print a side-by-side comparison table of two evaluation results."""
    label_a = res_a["label"]
    label_b = res_b["label"]
    seqs = sorted(set(res_a["per_sequence"]) | set(res_b["per_sequence"]))

    col = 14
    print(f"\n{'='*85}")
    print(f"  Comparison: {label_a}  vs  {label_b}")
    print(f"{'='*85}")
    hdr = (
        f"  {'Sequence':<18}"
        f"  {'J(A)':>{col}} {'J(B)':>{col}} {'ΔJ':>{col}}"
        f"  {'JF(A)':>{col}} {'JF(B)':>{col}} {'ΔJF':>{col}}"
    )
    print(hdr)
    print(f"  {'-'*81}")
    for seq in seqs:
        ma = res_a["per_sequence"].get(seq, {})
        mb = res_b["per_sequence"].get(seq, {})
        ja = ma.get("J_mean", float("nan"))
        jb = mb.get("J_mean", float("nan"))
        jfa = ma.get("JF_mean", float("nan"))
        jfb = mb.get("JF_mean", float("nan"))
        dj = jb - ja
        djf = jfb - jfa
        dj_s = f"{dj:+.4f}"
        djf_s = f"{djf:+.4f}"
        print(
            f"  {seq:<18}"
            f"  {ja:>{col}.4f} {jb:>{col}.4f} {dj_s:>{col}}"
            f"  {jfa:>{col}.4f} {jfb:>{col}.4f} {djf_s:>{col}}"
        )
    print(f"  {'='*81}")
    oa = res_a["overall"]
    ob = res_b["overall"]
    dj = ob["J_mean"] - oa["J_mean"]
    djf = ob["JF_mean"] - oa["JF_mean"]
    print(
        f"  {'OVERALL':<18}"
        f"  {oa['J_mean']:>{col}.4f} {ob['J_mean']:>{col}.4f} {dj:>+{col}.4f}"
        f"  {oa['JF_mean']:>{col}.4f} {ob['JF_mean']:>{col}.4f} {djf:>+{col}.4f}"
    )
    print(f"{'='*85}\n")

    gain = "improved" if djf >= 0 else "decreased"
    print(
        f"  Drift correction {gain} J&F by "
        f"{abs(djf):.4f} ({abs(djf)*100:.2f} pp)"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
    )
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

    # Main evaluation
    log.info("Evaluating predictions in %s", pred_root)
    res = evaluate_dataset(
        pred_root=pred_root,
        gt_root=gt_root,
        sequences=args.seq,
        label="w/ drift correction" if "with_drift" in str(pred_root) else str(pred_root.parent.name),
    )
    print_results(res)

    # Comparison with baseline
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
        print_comparison(res_baseline, res)

        if args.output_json:
            combined = {
                "baseline": res_baseline,
                "with_drift_correction": res,
            }
            Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_json, "w") as f:
                json.dump(combined, f, indent=2)
            log.info("Results saved to %s", args.output_json)
    else:
        if args.output_json:
            Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_json, "w") as f:
                json.dump(res, f, indent=2)
            log.info("Results saved to %s", args.output_json)


if __name__ == "__main__":
    main()

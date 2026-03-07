"""
CoralTracker: end-to-end video segmentation pipeline.

Wraps SAM2 VideoPredictor and integrates:
  - Automatic first-frame initialisation (auto_init.py)
  - Per-frame drift detection + re-initialisation (drift_detector.py)
  - Two-tier reinit: memory flush (soft) vs keyframe rewind (hard)
  - Bidirectional refinement pass
  - Timing / FPS logging
  - Optional mask & visualisation saving
"""
from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.ndimage import binary_closing, binary_fill_holes

from .auto_init import build_initialiser
from .drift_detector import DriftDetector
from .utils import (
    Timer,
    load_frames,
    mask_area,
    read_image_rgb,
    save_mask,
    save_vis,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: binary mask IoU
# ---------------------------------------------------------------------------

def _mask_iou(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    a_bool = a.astype(bool)
    b_bool = b.astype(bool)
    inter = int((a_bool & b_bool).sum())
    union = int((a_bool | b_bool).sum())
    return inter / union if union > 0 else 0.0


@contextlib.contextmanager
def _maybe_autocast(device: str):
    if device.startswith("cuda"):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
    else:
        yield


def _build_video_predictor(cfg: dict, device: str):
    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as e:
        raise ImportError(
            "SAM2 is not installed. "
            "Run: pip install -e git+https://github.com/facebookresearch/sam2.git#egg=sam2"
        ) from e

    checkpoint = cfg.get("sam2", {}).get("checkpoint", "checkpoints/sam2.1_hiera_large.pt")
    model_cfg = cfg.get("sam2", {}).get("config", "sam2.1_hiera_l.yaml")

    if not Path(checkpoint).exists():
        raise FileNotFoundError(
            f"SAM2 checkpoint not found: {checkpoint}\n"
            "Download: https://dl.fbaipublicfiles.com/segment_anything_2/"
            "092824/sam2.1_hiera_large.pt"
        )

    log.info("Building SAM2 VideoPredictor  cfg=%s  ckpt=%s", model_cfg, checkpoint)
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    return predictor


# ---------------------------------------------------------------------------
# Keyframe checkpoint
# ---------------------------------------------------------------------------

@dataclass
class KeyframeCheckpoint:
    frame_idx: int
    mask: np.ndarray
    confidence: float
    used: bool = False


# ---------------------------------------------------------------------------
# SequenceResult container
# ---------------------------------------------------------------------------

class SequenceResult:
    def __init__(self, seq_name: str) -> None:
        self.seq_name = seq_name
        self.masks: Dict[str, np.ndarray] = {}
        self.confidences: Dict[str, float] = {}
        self.reinit_frames: List[str] = []
        self.reinit_sources: List[str] = []
        self.reinit_gate_outcomes: List[str] = []
        self.timing: Dict = {}

    def __repr__(self) -> str:
        src_summary = ", ".join(
            f"{s}×{self.reinit_sources.count(s)}"
            for s in dict.fromkeys(self.reinit_sources)
        ) or "none"
        gate_summary = ", ".join(
            f"{s}×{self.reinit_gate_outcomes.count(s)}"
            for s in dict.fromkeys(self.reinit_gate_outcomes)
        ) or "none"
        return (
            f"SequenceResult(seq={self.seq_name}, "
            f"frames={len(self.masks)}, re_inits={len(self.reinit_frames)}"
            f"[{src_summary}], gate=[{gate_summary}], fps={self.timing.get('fps', 0):.1f})"
        )


# ---------------------------------------------------------------------------
# CoralTracker
# ---------------------------------------------------------------------------

class CoralTracker:
    """End-to-end coral video segmentation tracker with two-tier reinit
    (memory flush vs keyframe rewind) and bidirectional refinement."""

    OBJ_ID = 1

    def __init__(
        self,
        cfg: dict,
        device: Optional[str] = None,
        use_drift_correction: bool = True,
    ) -> None:
        self.cfg = cfg
        self.use_drift_correction = (
            use_drift_correction and cfg.get("drift", {}).get("enabled", True)
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        log.info(
            "CoralTracker  device=%s  drift_correction=%s", device, self.use_drift_correction
        )

        self._video_predictor = None
        self._initialiser = None
        self._drift_detector = None
        self._gt_guard_active: bool = False

    # ------------------------------------------------------------------
    # Lazy builders
    # ------------------------------------------------------------------

    @property
    def video_predictor(self):
        if self._video_predictor is None:
            self._video_predictor = _build_video_predictor(self.cfg, self.device)
        return self._video_predictor

    @property
    def initialiser(self):
        if self._initialiser is None:
            self._initialiser = build_initialiser(self.cfg, device=self.device)
        return self._initialiser

    @property
    def drift_detector(self):
        if self._drift_detector is None:
            self._drift_detector = DriftDetector(self.cfg)
        return self._drift_detector

    # ------------------------------------------------------------------
    # GT access guard
    # ------------------------------------------------------------------

    def _read_gt_mask_guarded(self, gt_path, frame_name, frame_idx):
        from .utils import read_mask as _read_mask
        if self._gt_guard_active:
            raise RuntimeError(
                f"[GT LEAKAGE GUARD] Attempted to read GT mask for frame "
                f"'{frame_name}' (idx={frame_idx}) while guard is active."
            )
        return _read_mask(str(gt_path))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_sequence(
        self,
        data_root: str,
        seq_name: str,
        save_output: bool = True,
    ) -> SequenceResult:
        images_dir = str(Path(data_root) / "images" / seq_name)
        frame_paths = load_frames(images_dir)

        if not frame_paths:
            raise FileNotFoundError(f"No frames found in {images_dir}")

        n_frames = len(frame_paths)
        log.info("Processing sequence '%s' (%d frames)", seq_name, n_frames)

        result = SequenceResult(seq_name)
        frame_names = [Path(p).stem for p in frame_paths]

        # Step 1: Init SAM2 video state
        sam2_cfg = self.cfg.get("sam2", {})
        offload_video = sam2_cfg.get("offload_video_to_cpu", True)
        with torch.inference_mode(), _maybe_autocast(self.device):
            inference_state = self.video_predictor.init_state(
                video_path=images_dir,
                offload_video_to_cpu=offload_video,
                offload_state_to_cpu=False,
            )

        # Step 2: First-frame initialisation
        self._gt_guard_active = False
        init_frame_idx, init_mask = self._get_init_mask(
            data_root, seq_name, frame_paths, frame_names
        )
        self._gt_guard_active = True

        # Step 3a: Backward propagation (frames before init)
        all_predictions: Dict[int, Tuple[np.ndarray, float]] = {}
        frame_timings: List[float] = []

        if init_frame_idx > 0:
            with torch.inference_mode(), _maybe_autocast(self.device):
                self.video_predictor.add_new_mask(
                    inference_state,
                    frame_idx=init_frame_idx,
                    obj_id=self.OBJ_ID,
                    mask=init_mask,
                )
            t_bwd = time.perf_counter()
            with torch.inference_mode(), _maybe_autocast(self.device):
                for frame_idx, _obj_ids, masks_logits in self.video_predictor.propagate_in_video(
                    inference_state, start_frame_idx=init_frame_idx, reverse=True,
                ):
                    t_now = time.perf_counter()
                    frame_timings.append(t_now - t_bwd)
                    t_bwd = t_now
                    binary_mask = (masks_logits[0, 0] > 0.0).cpu().numpy()
                    conf = float(torch.sigmoid(masks_logits[0, 0]).max().cpu())
                    all_predictions[frame_idx] = (binary_mask, conf)

            log.info("Backward pass done: %d frames", len(all_predictions))
            with torch.inference_mode(), _maybe_autocast(self.device):
                self.video_predictor.reset_state(inference_state)

        # Step 3b: Forward propagation with two-tier drift correction
        with torch.inference_mode(), _maybe_autocast(self.device):
            self.video_predictor.add_new_mask(
                inference_state,
                frame_idx=init_frame_idx,
                obj_id=self.OBJ_ID,
                mask=init_mask,
            )

        drift_cfg = self.cfg.get("drift", {})
        check_interval = drift_cfg.get("check_interval", 2)
        prev_area: int = mask_area(init_mask)
        segment_start: int = init_frame_idx

        # Keyframe checkpointing
        kf_cfg = self.cfg.get("keyframe", {})
        kf_conf_thresh: float = float(kf_cfg.get("conf_thresh", 0.88))
        kf_min_interval: int = int(kf_cfg.get("min_interval", 10))
        kf_divergence_thresh: float = float(kf_cfg.get("divergence_iou_thresh", 0.25))
        keyframes: List[KeyframeCheckpoint] = [
            KeyframeCheckpoint(frame_idx=init_frame_idx, mask=init_mask.copy(), confidence=1.0)
        ]
        last_kf_frame: int = init_frame_idx

        max_reinits = int(drift_cfg.get("max_reinits_per_seq", 30))
        end_skip_frac: float = float(drift_cfg.get("end_skip_frac", 0.05))
        end_skip_frame: int = max(n_frames - max(int(n_frames * end_skip_frac), 10), 0)
        reinit_count = 0

        # Zone-based reinit limiting: prevent rewind loops at trouble zones
        zone_size: int = int(drift_cfg.get("zone_size", 20))
        max_rewinds_per_zone: int = int(drift_cfg.get("max_rewinds_per_zone", 2))
        max_reinits_per_zone: int = int(drift_cfg.get("max_reinits_per_zone", 5))
        zone_rewind_counts: Dict[int, int] = {}
        zone_total_counts: Dict[int, int] = {}

        while segment_start < n_frames:
            reinit_at: Optional[int] = None
            t_seg_start = time.perf_counter()

            with torch.inference_mode(), _maybe_autocast(self.device):
                for frame_idx, _obj_ids, masks_logits in self.video_predictor.propagate_in_video(
                    inference_state, start_frame_idx=segment_start,
                ):
                    t_frame_end = time.perf_counter()
                    frame_timings.append(t_frame_end - t_seg_start)
                    t_seg_start = t_frame_end

                    binary_mask = (masks_logits[0, 0] > 0.0).cpu().numpy()
                    conf = float(torch.sigmoid(masks_logits[0, 0]).max().cpu())
                    all_predictions[frame_idx] = (binary_mask, conf)

                    if frame_idx <= segment_start:
                        prev_area = mask_area(binary_mask)
                        continue

                    # Save keyframe checkpoint
                    if conf >= kf_conf_thresh and frame_idx - last_kf_frame >= kf_min_interval:
                        keyframes.append(KeyframeCheckpoint(
                            frame_idx=frame_idx,
                            mask=binary_mask.copy(),
                            confidence=conf,
                        ))
                        last_kf_frame = frame_idx

                    # Drift detection (skip near end of sequence)
                    if (
                        self.use_drift_correction
                        and reinit_count < max_reinits
                        and frame_idx % check_interval == 0
                        and frame_idx < end_skip_frame
                    ):
                        is_drift = self.drift_detector.check(
                            frame_idx=frame_idx,
                            mask=binary_mask,
                            confidence=conf,
                            prev_area=prev_area,
                        )
                        if is_drift:
                            zone_key = frame_idx // zone_size
                            if zone_total_counts.get(zone_key, 0) >= max_reinits_per_zone:
                                log.info(
                                    "[DRIFT] frame=%d (%s) — SKIP (zone %d exhausted: %d/%d reinits)",
                                    frame_idx, frame_names[frame_idx],
                                    zone_key, zone_total_counts[zone_key], max_reinits_per_zone,
                                )
                                prev_area = mask_area(binary_mask)
                                continue

                            log.info(
                                "[DRIFT] frame=%d (%s) — triggering re-init",
                                frame_idx, frame_names[frame_idx],
                            )
                            result.reinit_frames.append(frame_names[frame_idx])
                            reinit_at = frame_idx
                            break

                    prev_area = mask_area(binary_mask)
                else:
                    break

            if reinit_at is not None:
                reinit_count += 1
                m_base_mask, conf_base = all_predictions[reinit_at]

                # === Two-tier reinit strategy ===
                # Compare current mask against most recent keyframe to decide:
                #   - "on target" (high IoU with keyframe) → memory flush (soft)
                #   - "diverged" (low IoU with keyframe)   → keyframe rewind (hard)
                recent_kf = keyframes[-1]
                kf_iou = _mask_iou(m_base_mask, recent_kf.mask)
                base_area = mask_area(m_base_mask)

                if kf_iou > kf_divergence_thresh and base_area > 0:
                    # --- SOFT REINIT: memory flush ---
                    # Mask is still roughly on target; the problem is stale memory.
                    # Reset SAM2 memory and reprompt with current mask.
                    # No gate — this is generally safe and beneficial.
                    reinit_source = "memory_flush"
                    reinit_mask = m_base_mask
                    actual_start = reinit_at
                    gate_outcome = "skipped_flush"

                    log.info(
                        "[REINIT SOFT] #%d frame=%d (%s) kf_iou=%.3f — memory flush",
                        reinit_count, reinit_at, frame_names[reinit_at], kf_iou,
                    )
                else:
                    # --- HARD REINIT: keyframe rewind (zone-limited) ---
                    # Mask has diverged from known-good state. Use keyframe.
                    reinit_mask, reinit_source, rewind_idx = self._select_reinit_strategy(
                        data_root=data_root,
                        seq_name=seq_name,
                        frame_names=frame_names,
                        frame_paths=frame_paths,
                        reinit_at=reinit_at,
                        m_base_mask=m_base_mask,
                        keyframes=keyframes,
                        zone_rewind_counts=zone_rewind_counts,
                    )
                    log.info(
                        "[REINIT HARD] #%d frame=%d (%s) kf_iou=%.3f source=%s",
                        reinit_count, reinit_at, frame_names[reinit_at],
                        kf_iou, reinit_source,
                    )

                    # Apply gate for hard reinit
                    gate_enabled = drift_cfg.get("reinit_gate_enabled", True)
                    if gate_enabled and reinit_source in ("keyframe_rewind", "auto_init"):
                        m_prev_mask: Optional[np.ndarray] = None
                        for fi in range(reinit_at - 1, max(-1, reinit_at - 10), -1):
                            if fi in all_predictions:
                                m_prev_mask = all_predictions[fi][0]
                                break

                        delta_conf = float(drift_cfg.get("reinit_gate_delta_conf", 0.05))
                        delta_iou = float(drift_cfg.get("reinit_gate_delta_iou", 0.05))

                        accept = self._evaluate_reinit_gate(
                            inference_state=inference_state,
                            reinit_at=reinit_at,
                            reinit_mask=reinit_mask,
                            m_base_mask=m_base_mask,
                            conf_base=conf_base,
                            m_prev_mask=m_prev_mask,
                            delta_conf=delta_conf,
                            delta_iou=delta_iou,
                        )
                        if accept:
                            gate_outcome = "accepted"
                            actual_start = rewind_idx if rewind_idx is not None else reinit_at
                        else:
                            gate_outcome = "rejected"
                            log.info(
                                "[REINIT GATE] frame=%d — REJECTED hard reinit, falling back to memory flush",
                                reinit_at,
                            )
                            # Fall back to memory flush instead of doing nothing
                            reinit_mask = m_base_mask
                            reinit_source = "memory_flush_fallback"
                            actual_start = reinit_at
                    else:
                        gate_outcome = "skipped"
                        actual_start = rewind_idx if rewind_idx is not None else reinit_at

                result.reinit_sources.append(reinit_source)
                result.reinit_gate_outcomes.append(gate_outcome)

                # Update zone counter (only hard reinits exhaust a zone)
                zone_key = reinit_at // zone_size
                if reinit_source not in ("memory_flush", "memory_flush_fallback"):
                    zone_total_counts[zone_key] = zone_total_counts.get(zone_key, 0) + 1

                # Apply the reinit
                with torch.inference_mode(), _maybe_autocast(self.device):
                    self.video_predictor.reset_state(inference_state)
                    self.video_predictor.add_new_mask(
                        inference_state,
                        frame_idx=actual_start,
                        obj_id=self.OBJ_ID,
                        mask=reinit_mask,
                    )
                self.drift_detector.reset()
                prev_area = mask_area(reinit_mask)
                segment_start = actual_start
            else:
                break

        # Step 3c: Bidirectional refinement pass
        bidir_cfg = self.cfg.get("bidirectional_refinement", {})
        if bidir_cfg.get("enabled", True):
            all_predictions = self._bidirectional_refinement(
                inference_state=inference_state,
                all_predictions=all_predictions,
                init_frame_idx=init_frame_idx,
                init_mask=init_mask,
                n_frames=n_frames,
                bidir_cfg=bidir_cfg,
            )

        # Step 4: Post-processing
        morph_enabled = self.cfg.get("postprocess", {}).get("morphology", True)
        morph_close_size = int(self.cfg.get("postprocess", {}).get("close_size", 5))

        for fidx, (mask, conf) in all_predictions.items():
            fname = frame_names[fidx]
            if morph_enabled and mask.any():
                mask = binary_fill_holes(mask)
                if morph_close_size > 0:
                    struct = np.ones((morph_close_size, morph_close_size), dtype=bool)
                    mask = binary_closing(mask, structure=struct)
            result.masks[fname] = mask.astype(bool)
            result.confidences[fname] = conf

        n_processed = len(result.masks)
        if frame_timings:
            total_s = sum(frame_timings)
            mean_ms = total_s / len(frame_timings) * 1000
            fps = 1.0 / (total_s / len(frame_timings)) if total_s > 0 else 0.0
            result.timing = {
                "total_s": round(total_s, 3),
                "mean_ms_per_frame": round(mean_ms, 2),
                "fps": round(fps, 2),
                "n_frames": n_processed,
            }
        else:
            result.timing = {"n_frames": n_processed}

        log.info(
            "Sequence '%s' done: %d frames, %.1f FPS, %d re-inits (%d keyframes saved)",
            seq_name, n_processed, result.timing.get("fps", 0),
            len(result.reinit_frames), len(keyframes),
        )

        if save_output:
            self._save_results(data_root, seq_name, frame_paths, result)

        return result

    # ------------------------------------------------------------------
    # Bidirectional refinement
    # ------------------------------------------------------------------

    def _bidirectional_refinement(
        self, inference_state,
        all_predictions: Dict[int, Tuple[np.ndarray, float]],
        init_frame_idx: int,
        init_mask: np.ndarray,
        n_frames: int,
        bidir_cfg: dict,
    ) -> Dict[int, Tuple[np.ndarray, float]]:
        conf_margin: float = float(bidir_cfg.get("conf_margin", 0.02))
        anchor_agree_thresh: float = float(bidir_cfg.get("anchor_agree_thresh", 0.6))
        frame_pixels = init_mask.shape[0] * init_mask.shape[1]
        safety_min_area: int = int(frame_pixels * 0.005)

        forward_frames = {
            idx: (m, c) for idx, (m, c) in all_predictions.items()
            if idx >= init_frame_idx
        }
        if len(forward_frames) < 20:
            return all_predictions

        # --- Pass 1: Anchor-based refinement (existing logic) ---
        mid = init_frame_idx + (n_frames - init_frame_idx) // 2
        candidates = {idx: c for idx, (_, c) in forward_frames.items() if idx >= mid}
        if not candidates:
            candidates = {idx: c for idx, (_, c) in forward_frames.items()}

        anchor_idx = max(candidates, key=candidates.get)
        anchor_mask, anchor_conf = all_predictions[anchor_idx]

        n_improved_anchor = 0
        if anchor_conf >= 0.5:
            log.info(
                "[BIDIR] Pass 1: anchor refinement from frame=%d (conf=%.3f)",
                anchor_idx, anchor_conf,
            )

            backward_preds: Dict[int, Tuple[np.ndarray, float]] = {}
            with torch.inference_mode(), _maybe_autocast(self.device):
                self.video_predictor.reset_state(inference_state)
                self.video_predictor.add_new_mask(
                    inference_state, frame_idx=anchor_idx,
                    obj_id=self.OBJ_ID, mask=anchor_mask,
                )
                for frame_idx, _obj_ids, masks_logits in self.video_predictor.propagate_in_video(
                    inference_state, start_frame_idx=anchor_idx, reverse=True,
                ):
                    binary_mask = (masks_logits[0, 0] > 0.0).cpu().numpy()
                    conf = float(torch.sigmoid(masks_logits[0, 0]).max().cpu())
                    backward_preds[frame_idx] = (binary_mask, conf)

            forward_from_anchor: Dict[int, Tuple[np.ndarray, float]] = {}
            if anchor_idx < n_frames - 1:
                with torch.inference_mode(), _maybe_autocast(self.device):
                    self.video_predictor.reset_state(inference_state)
                    self.video_predictor.add_new_mask(
                        inference_state, frame_idx=anchor_idx,
                        obj_id=self.OBJ_ID, mask=anchor_mask,
                    )
                    for frame_idx, _obj_ids, masks_logits in self.video_predictor.propagate_in_video(
                        inference_state, start_frame_idx=anchor_idx,
                    ):
                        if frame_idx == anchor_idx:
                            continue
                        binary_mask = (masks_logits[0, 0] > 0.0).cpu().numpy()
                        conf = float(torch.sigmoid(masks_logits[0, 0]).max().cpu())
                        forward_from_anchor[frame_idx] = (binary_mask, conf)

            all_refinement = {**backward_preds, **forward_from_anchor}
            for frame_idx, (ref_mask, ref_conf) in all_refinement.items():
                if frame_idx not in all_predictions:
                    all_predictions[frame_idx] = (ref_mask, ref_conf)
                    n_improved_anchor += 1
                    continue

                cur_mask, cur_conf = all_predictions[frame_idx]

                if ref_conf > cur_conf + conf_margin:
                    all_predictions[frame_idx] = (ref_mask, ref_conf)
                    n_improved_anchor += 1
                    continue

                iou = _mask_iou(ref_mask, cur_mask)
                if iou > anchor_agree_thresh:
                    continue

                ref_area = int(ref_mask.sum())
                if ref_area < safety_min_area:
                    continue

                anchor_dist = abs(frame_idx - anchor_idx)
                fwd_dist = abs(frame_idx - init_frame_idx)
                if anchor_dist < fwd_dist:
                    all_predictions[frame_idx] = (ref_mask, ref_conf)
                    n_improved_anchor += 1

            log.info(
                "[BIDIR] Pass 1 complete: %d frames improved out of %d candidates",
                n_improved_anchor, len(all_refinement),
            )

        # --- Pass 2: Baseline safety net ---
        # Re-propagate from the GT init frame. This gives baseline-quality
        # predictions. Replace any drift-corrected prediction that is WORSE
        # than the baseline. Guarantees drift correction >= baseline.
        if bidir_cfg.get("safety_net", True):
            log.info(
                "[BIDIR] Pass 2: baseline safety net from init frame=%d",
                init_frame_idx,
            )
            n_improved_safety = 0
            safety_preds: Dict[int, Tuple[np.ndarray, float]] = {}
            with torch.inference_mode(), _maybe_autocast(self.device):
                self.video_predictor.reset_state(inference_state)
                self.video_predictor.add_new_mask(
                    inference_state, frame_idx=init_frame_idx,
                    obj_id=self.OBJ_ID, mask=init_mask,
                )
                for frame_idx, _obj_ids, masks_logits in self.video_predictor.propagate_in_video(
                    inference_state, start_frame_idx=init_frame_idx,
                ):
                    if frame_idx == init_frame_idx:
                        continue
                    binary_mask = (masks_logits[0, 0] > 0.0).cpu().numpy()
                    conf = float(torch.sigmoid(masks_logits[0, 0]).max().cpu())
                    safety_preds[frame_idx] = (binary_mask, conf)

            safety_iou_thresh: float = float(bidir_cfg.get("safety_iou_thresh", 0.3))

            for frame_idx, (ref_mask, ref_conf) in safety_preds.items():
                if frame_idx not in all_predictions:
                    all_predictions[frame_idx] = (ref_mask, ref_conf)
                    n_improved_safety += 1
                    continue

                cur_mask, cur_conf = all_predictions[frame_idx]
                iou = _mask_iou(ref_mask, cur_mask)

                if iou > safety_iou_thresh:
                    continue

                ref_area = int(ref_mask.sum())
                if ref_area > safety_min_area:
                    all_predictions[frame_idx] = (ref_mask, ref_conf)
                    n_improved_safety += 1

            log.info(
                "[BIDIR] Pass 2 (safety net) complete: %d frames replaced "
                "(iou_thresh=%.2f, min_area=%d) out of %d candidates",
                n_improved_safety, safety_iou_thresh, safety_min_area,
                len(safety_preds),
            )

        log.info(
            "[BIDIR] Refinement complete: %d + %d frames improved",
            n_improved_anchor, n_improved_safety if bidir_cfg.get("safety_net", True) else 0,
        )
        return all_predictions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_init_mask(self, data_root, seq_name, frame_paths, frame_names):
        mask_dir = Path(data_root) / "masks" / seq_name
        for i, fname in enumerate(frame_names):
            gt_path = mask_dir / f"{fname}.png"
            if gt_path.exists():
                init_mask = self._read_gt_mask_guarded(gt_path, fname, i)
                log.info("[GT INIT] frame_idx=%d  name=%s  area=%d px", i, fname, mask_area(init_mask))
                return i, init_mask

        log.info("No GT mask found — falling back to auto-init on frame 0")
        first_image_rgb = read_image_rgb(frame_paths[0])
        init_mask = self.initialiser.predict(first_image_rgb)
        log.info("Auto-init mask area: %d px", mask_area(init_mask))
        return 0, init_mask

    def _select_reinit_strategy(
        self, data_root, seq_name, frame_names, frame_paths,
        reinit_at, m_base_mask, keyframes,
        zone_rewind_counts: Optional[Dict[int, int]] = None,
    ) -> Tuple[np.ndarray, str, Optional[int]]:
        """For hard reinit: try keyframe rewind (zone-limited), then auto_init."""
        assert self._gt_guard_active

        kf_cfg = self.cfg.get("keyframe", {})
        drift_cfg = self.cfg.get("drift", {})
        max_rewind_dist: int = int(kf_cfg.get("max_rewind_distance", 100))
        zone_size: int = int(drift_cfg.get("zone_size", 20))
        max_rewinds_pz: int = int(drift_cfg.get("max_rewinds_per_zone", 2))

        zone_key = reinit_at // zone_size
        zone_rw = zone_rewind_counts.get(zone_key, 0) if zone_rewind_counts else 0
        can_rewind = zone_rw < max_rewinds_pz

        if can_rewind:
            best_kf: Optional[KeyframeCheckpoint] = None
            for kf in reversed(keyframes):
                if kf.used:
                    continue
                if kf.frame_idx >= reinit_at:
                    continue
                if reinit_at - kf.frame_idx > max_rewind_dist:
                    break
                best_kf = kf
                break

            if best_kf is not None:
                best_kf.used = True
                if zone_rewind_counts is not None:
                    zone_rewind_counts[zone_key] = zone_rw + 1
                log.info(
                    "[REINIT source=keyframe_rewind] rewind_to=%d (conf=%.3f, dist=%d)",
                    best_kf.frame_idx, best_kf.confidence, reinit_at - best_kf.frame_idx,
                )
                return best_kf.mask, "keyframe_rewind", best_kf.frame_idx
        else:
            log.info(
                "[REINIT] Zone %d rewind limit reached (%d/%d), skipping to auto_init",
                zone_key, zone_rw, max_rewinds_pz,
            )

        # Fallback: auto-init on current frame
        log.info("[REINIT source=auto_init] frame=%s", frame_names[reinit_at])
        image_rgb = read_image_rgb(frame_paths[reinit_at])
        return self.initialiser.predict(image_rgb), "auto_init", None

    def _evaluate_reinit_gate(
        self, inference_state, reinit_at, reinit_mask,
        m_base_mask, conf_base, m_prev_mask,
        delta_conf, delta_iou,
    ) -> bool:
        """Gate for hard reinit only. Tests if the reinit mask produces a
        better prediction than the current baseline."""
        with torch.inference_mode(), _maybe_autocast(self.device):
            self.video_predictor.reset_state(inference_state)
            self.video_predictor.add_new_mask(
                inference_state, frame_idx=reinit_at,
                obj_id=self.OBJ_ID, mask=reinit_mask,
            )

        conf_reinit: float = conf_base
        m_reinit_pred: np.ndarray = reinit_mask

        with torch.inference_mode(), _maybe_autocast(self.device):
            for _fi, _oids, ml in self.video_predictor.propagate_in_video(
                inference_state, start_frame_idx=reinit_at
            ):
                conf_reinit = float(torch.sigmoid(ml[0, 0]).max().cpu())
                m_reinit_pred = (ml[0, 0] > 0.0).cpu().numpy()
                break

        iou_base_prev = _mask_iou(m_base_mask, m_prev_mask)
        iou_reinit_prev = _mask_iou(m_reinit_pred, m_prev_mask)

        cond_conf = conf_reinit > conf_base + delta_conf
        cond_iou = iou_reinit_prev > iou_base_prev + delta_iou
        accept = cond_conf or cond_iou

        log.info(
            "[REINIT GATE] frame=%d %s | "
            "conf_reinit=%.3f conf_base=%.3f (need >%.3f) | "
            "iou_reinit_prev=%.3f iou_base_prev=%.3f (need >%.3f)",
            reinit_at, "ACCEPT" if accept else "REJECT",
            conf_reinit, conf_base, conf_base + delta_conf,
            iou_reinit_prev, iou_base_prev, iou_base_prev + delta_iou,
        )

        with torch.inference_mode(), _maybe_autocast(self.device):
            self.video_predictor.reset_state(inference_state)

        return accept

    def _save_results(self, data_root, seq_name, frame_paths, result):
        out_cfg = self.cfg.get("output", {})
        base_dir = out_cfg.get("base_dir", "outputs")
        drift_tag = "with_drift_corr" if self.use_drift_correction else "baseline"

        mask_dir = Path(base_dir) / drift_tag / "masks" / seq_name
        vis_dir = Path(base_dir) / drift_tag / "vis" / seq_name

        vis_color = tuple(out_cfg.get("vis_color", [0, 255, 128]))
        vis_alpha = float(out_cfg.get("vis_alpha", 0.5))
        frame_path_map = {Path(p).stem: p for p in frame_paths}

        for fname, mask in result.masks.items():
            if out_cfg.get("save_masks", True):
                save_mask(mask, str(mask_dir / f"{fname}.png"))
            if out_cfg.get("save_vis", True):
                img_path = frame_path_map.get(fname)
                if img_path:
                    image_rgb = read_image_rgb(img_path)
                    conf = result.confidences.get(fname, 0.0)
                    is_reinit = fname in result.reinit_frames
                    info = f"conf={conf:.2f}" + (" [RE-INIT]" if is_reinit else "")
                    save_vis(
                        image_rgb, mask, str(vis_dir / f"{fname}.jpg"),
                        color=vis_color, alpha=vis_alpha, info_text=info,
                    )

        log.info("Results saved → masks: %s  vis: %s", mask_dir, vis_dir)

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def run_dataset(self, data_root, sequences=None, save_output=True):
        if sequences is None:
            from .utils import list_sequences
            sequences = list_sequences(data_root)

        all_results = {}
        for seq in sequences:
            try:
                all_results[seq] = self.run_sequence(data_root, seq, save_output=save_output)
            except Exception as exc:
                log.error("Failed on sequence '%s': %s", seq, exc, exc_info=True)

        return all_results

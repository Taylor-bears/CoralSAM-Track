"""
CoralTracker: end-to-end video segmentation pipeline.

Wraps SAM2 VideoPredictor and integrates:
  - Automatic first-frame initialisation (auto_init.py)
  - Per-frame drift detection + re-initialisation (drift_detector.py)
  - Timing / FPS logging
  - Optional mask & visualisation saving

Usage example
-------------
    from src.tracker import CoralTracker
    from src.utils import load_config

    cfg = load_config("configs/default.yaml")
    tracker = CoralTracker(cfg)
    results = tracker.run_sequence("partial_coralvos/partial", "video102")
    # results.masks    -> {frame_name: np.ndarray bool H×W}
    # results.timing   -> {"fps": ..., "total_s": ..., ...}
"""
from __future__ import annotations

import contextlib
import logging
import time
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
    """Compute binary IoU between two masks. Returns 0.0 if either is None."""
    if a is None or b is None:
        return 0.0
    a_bool = a.astype(bool)
    b_bool = b.astype(bool)
    inter = int((a_bool & b_bool).sum())
    union = int((a_bool | b_bool).sum())
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Helper: autocast context that degrades gracefully on CPU
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _maybe_autocast(device: str):
    """Use bfloat16 autocast on CUDA; no-op on CPU."""
    if device.startswith("cuda"):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
    else:
        yield


# ---------------------------------------------------------------------------
# Helper: build SAM2 VideoPredictor
# ---------------------------------------------------------------------------

def _build_video_predictor(cfg: dict, device: str):
    """Instantiate a SAM2VideoPredictor from config."""
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
# SequenceResult container
# ---------------------------------------------------------------------------

class SequenceResult:
    """Holds all outputs from processing one video sequence."""

    def __init__(self, seq_name: str) -> None:
        self.seq_name = seq_name
        self.masks: Dict[str, np.ndarray] = {}
        self.confidences: Dict[str, float] = {}
        self.reinit_frames: List[str] = []
        # Parallel to reinit_frames: records where each reinit mask came from.
        # Values: "prev_mask" | "auto_init"
        self.reinit_sources: List[str] = []
        # Gate outcome per re-init event: "skipped" | "accepted" | "rejected"
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
    """End-to-end coral video segmentation tracker.

    Parameters
    ----------
    cfg : dict
        Loaded from configs/default.yaml (or equivalent).
    device : str
        "cuda" or "cpu". Auto-detected when None.
    use_drift_correction : bool
        If False, disable drift detection (baseline mode).
    """

    OBJ_ID = 1  # single-object tracking

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

        # GT access guard (Safeguard 1).
        # Set to True after first-frame init; any GT read after that raises.
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
    # GT access guard (Safeguard 1)
    # ------------------------------------------------------------------

    def _read_gt_mask_guarded(
        self,
        gt_path: "Path",
        frame_name: str,
        frame_idx: int,
    ) -> np.ndarray:
        """Central chokepoint for every GT mask read.

        Raises ``RuntimeError`` if the guard is active (i.e. init is already
        done), so that any accidental GT access during inference is caught
        immediately rather than silently inflating metrics.

        All GT reads in this class MUST go through this method.
        """
        from .utils import read_mask as _read_mask

        if self._gt_guard_active:
            raise RuntimeError(
                f"[GT LEAKAGE GUARD] Attempted to read GT mask for frame "
                f"'{frame_name}' (idx={frame_idx}) while the inference guard "
                f"is active. GT is only allowed at first-frame initialisation. "
                f"Reading GT during tracking would constitute data leakage."
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
        """Process one video sequence end-to-end.

        Args:
            data_root:    Path containing images/<seq_name>/ and masks/<seq_name>/.
            seq_name:     Sequence subdirectory name (e.g. "video102").
            save_output:  Whether to save predicted masks and visualisations.

        Returns:
            SequenceResult with all per-frame predictions and timing info.
        """
        images_dir = str(Path(data_root) / "images" / seq_name)
        frame_paths = load_frames(images_dir)

        if not frame_paths:
            raise FileNotFoundError(f"No frames found in {images_dir}")

        n_frames = len(frame_paths)
        log.info("Processing sequence '%s' (%d frames)", seq_name, n_frames)

        result = SequenceResult(seq_name)
        frame_names = [Path(p).stem for p in frame_paths]

        # ------------------------------------------------------------------
        # Step 1: Init SAM2 video state (loads frames into memory)
        # ------------------------------------------------------------------
        sam2_cfg = self.cfg.get("sam2", {})
        offload_video = sam2_cfg.get("offload_video_to_cpu", True)
        with torch.inference_mode(), _maybe_autocast(self.device):
            inference_state = self.video_predictor.init_state(
                video_path=images_dir,
                offload_video_to_cpu=offload_video,
                offload_state_to_cpu=False,
            )

        # ------------------------------------------------------------------
        # Step 2: Find the first available GT frame and load its mask.
        #   Semi-supervised VOS protocol: use the earliest annotated frame.
        #   Falls back to auto-init on frame 0 when no GT is present at all.
        # ------------------------------------------------------------------
        # Reset GT guard so the init read is permitted, then activate it
        # immediately after to block any subsequent GT reads.
        self._gt_guard_active = False
        init_frame_idx, init_mask = self._get_init_mask(
            data_root, seq_name, frame_paths, frame_names
        )
        self._gt_guard_active = True
        log.debug(
            "[GT Guard] Activated after init at frame %d — GT reads are now forbidden",
            init_frame_idx,
        )

        # ------------------------------------------------------------------
        # Step 3a: Backward propagation (only when init frame is not frame 0)
        #   This fills predictions for frames [0 … init_frame_idx-1] that
        #   precede the first GT annotation.
        # ------------------------------------------------------------------
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
                    inference_state,
                    start_frame_idx=init_frame_idx,
                    reverse=True,
                ):
                    t_now = time.perf_counter()
                    frame_timings.append(t_now - t_bwd)
                    t_bwd = t_now
                    binary_mask = (masks_logits[0, 0] > 0.0).cpu().numpy()
                    conf = float(torch.sigmoid(masks_logits[0, 0]).max().cpu())
                    all_predictions[frame_idx] = (binary_mask, conf)

            log.info(
                "Backward pass done: %d frames (idx 0..%d)",
                len(all_predictions),
                init_frame_idx,
            )

            # Reset tracking state (keeps loaded video frames) for forward pass
            with torch.inference_mode(), _maybe_autocast(self.device):
                self.video_predictor.reset_state(inference_state)

        # ------------------------------------------------------------------
        # Step 3b: Forward propagation with optional drift correction
        #   Starts at init_frame_idx and runs to the end of the video.
        # ------------------------------------------------------------------
        with torch.inference_mode(), _maybe_autocast(self.device):
            self.video_predictor.add_new_mask(
                inference_state,
                frame_idx=init_frame_idx,
                obj_id=self.OBJ_ID,
                mask=init_mask,
            )

        drift_cfg = self.cfg.get("drift", {})
        check_interval = drift_cfg.get("check_interval", 10)
        prev_area: int = mask_area(init_mask)
        segment_start: int = init_frame_idx

        while segment_start < n_frames:
            reinit_at: Optional[int] = None
            t_seg_start = time.perf_counter()

            with torch.inference_mode(), _maybe_autocast(self.device):
                for frame_idx, _obj_ids, masks_logits in self.video_predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=segment_start,
                ):
                    t_frame_end = time.perf_counter()
                    frame_timings.append(t_frame_end - t_seg_start)
                    t_seg_start = t_frame_end

                    binary_mask = (masks_logits[0, 0] > 0.0).cpu().numpy()
                    conf = float(torch.sigmoid(masks_logits[0, 0]).max().cpu())
                    all_predictions[frame_idx] = (binary_mask, conf)

                    # Skip checks on the very first frame of a segment
                    if frame_idx <= segment_start:
                        prev_area = mask_area(binary_mask)
                        continue

                    # ---- Drift detection re-init ----
                    if (
                        self.use_drift_correction
                        and frame_idx % check_interval == 0
                    ):
                        is_drift = self.drift_detector.check(
                            frame_idx=frame_idx,
                            mask=binary_mask,
                            confidence=conf,
                            prev_area=prev_area,
                        )
                        if is_drift:
                            log.info(
                                "[DRIFT] frame=%d (%s) — triggering re-init",
                                frame_idx,
                                frame_names[frame_idx],
                            )
                            result.reinit_frames.append(frame_names[frame_idx])
                            reinit_at = frame_idx
                            break

                    prev_area = mask_area(binary_mask)

                else:
                    break

            if reinit_at is not None:
                m_base_mask, conf_base = all_predictions[reinit_at]

                reinit_mask, reinit_source = self._get_reinit_mask(
                    data_root=data_root,
                    seq_name=seq_name,
                    frame_name=frame_names[reinit_at],
                    frame_path=frame_paths[reinit_at],
                    prev_mask=m_base_mask,
                )
                log.info(
                    "[REINIT] frame=%d (%s)  source=%s  mask_area=%d px",
                    reinit_at,
                    frame_names[reinit_at],
                    reinit_source,
                    mask_area(reinit_mask),
                )
                result.reinit_sources.append(reinit_source)

                # ----------------------------------------------------------
                # Accept/Reject gate: only replace M_base when M_reinit is
                # demonstrably better.  Gate is skipped when reinit_mask IS
                # M_base (source="prev_mask") — resetting SAM2 memory with
                # the same mask is always safe and beneficial.
                # ----------------------------------------------------------
                gate_enabled = drift_cfg.get("reinit_gate_enabled", True)
                gate_skipped = reinit_source == "prev_mask"

                if gate_enabled and not gate_skipped:
                    # Find the most recent previous-frame mask for IoU comparison
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
                    gate_outcome = "accepted" if accept else "rejected"
                    if not accept:
                        log.info(
                            "[REINIT GATE] frame=%d — rejected, reverting to M_base prompt",
                            reinit_at,
                        )
                        reinit_mask = m_base_mask
                else:
                    gate_outcome = "skipped"
                    log.debug(
                        "[REINIT GATE] frame=%d skipped (gate_enabled=%s, source=%s)",
                        reinit_at, gate_enabled, reinit_source,
                    )

                result.reinit_gate_outcomes.append(gate_outcome)

                # Apply chosen mask: unconditional clean reset + re-prompt
                with torch.inference_mode(), _maybe_autocast(self.device):
                    self.video_predictor.reset_state(inference_state)
                    self.video_predictor.add_new_mask(
                        inference_state,
                        frame_idx=reinit_at,
                        obj_id=self.OBJ_ID,
                        mask=reinit_mask,
                    )
                self.drift_detector.reset()
                prev_area = mask_area(reinit_mask)
                segment_start = reinit_at
            else:
                break

        # ------------------------------------------------------------------
        # Step 4: Pack results and timing
        #   P4: Apply morphological post-processing to each predicted mask:
        #     - binary_fill_holes: fills interior holes → improves J (region IoU)
        #     - binary_closing: smooths ragged boundaries → improves F (contour)
        # ------------------------------------------------------------------
        morph_enabled = self.cfg.get("postprocess", {}).get("morphology", True)
        morph_close_size = int(self.cfg.get("postprocess", {}).get("close_size", 5))

        for fidx, (mask, conf) in all_predictions.items():
            fname = frame_names[fidx]
            if morph_enabled and mask.any():
                mask = binary_fill_holes(mask)
                if morph_close_size > 0:
                    struct = np.ones(
                        (morph_close_size, morph_close_size), dtype=bool
                    )
                    mask = binary_closing(mask, structure=struct)
            result.masks[fname] = mask.astype(bool)
            result.confidences[fname] = conf

        n_processed = len(result.masks)
        # Build timing summary directly from collected frame times
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
            "Sequence '%s' done: %d frames, %.1f FPS, %d re-inits",
            seq_name,
            n_processed,
            result.timing.get("fps", 0),
            len(result.reinit_frames),
        )

        # ------------------------------------------------------------------
        # Step 5: Optional save
        # ------------------------------------------------------------------
        if save_output:
            self._save_results(data_root, seq_name, frame_paths, result)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_init_mask(
        self,
        data_root: str,
        seq_name: str,
        frame_paths: List[str],
        frame_names: List[str],
    ) -> Tuple[int, np.ndarray]:
        """Find the first frame with a GT mask and return (frame_idx, mask).

        Searches all frames in order; returns the index of the earliest
        annotated frame plus the loaded binary mask.  Falls back to
        auto-init on frame 0 when the sequence has no GT at all.

        All GT reads go through ``_read_gt_mask_guarded`` so that the runtime
        guard catches any accidental call made after init completes.
        """
        mask_dir = Path(data_root) / "masks" / seq_name
        for i, fname in enumerate(frame_names):
            gt_path = mask_dir / f"{fname}.png"
            if gt_path.exists():
                # Guard is not yet active here — this is the one allowed GT read.
                init_mask = self._read_gt_mask_guarded(gt_path, fname, i)
                log.info(
                    "[GT INIT] frame_idx=%d  name=%s  area=%d px",
                    i, fname, mask_area(init_mask),
                )
                return i, init_mask

        log.info("No GT mask found — falling back to auto-init on frame 0")
        first_image_rgb = read_image_rgb(frame_paths[0])
        init_mask = self.initialiser.predict(first_image_rgb)
        log.info("Auto-init mask area: %d px", mask_area(init_mask))
        return 0, init_mask

    def _get_reinit_mask(
        self,
        data_root: str,
        seq_name: str,
        frame_name: str,
        frame_path: str,
        prev_mask: np.ndarray,
    ) -> "Tuple[np.ndarray, str]":
        """Get a mask for re-initialisation. Returns (mask, source_tag).

        source_tag is one of: "prev_mask" | "auto_init"

        GT masks are intentionally NOT used here.  The GT access guard
        (_gt_guard_active) is active at this point and will raise if any
        code path accidentally tries to read GT — providing belt-and-suspenders
        protection on top of the explicit non-use below.

        Priority:
          1. Previous tracked mask when use_prev_mask_for_reinit=True (default),
             BUT only when the previous mask is non-empty (area > 0).
          2. Fresh auto-init via SAM2 image predictor (fallback when prev mask
             is empty/degenerate).
        """
        # Belt-and-suspenders: prove the guard is active (it must be at this
        # point) so that any accidental GT read would raise immediately.
        assert self._gt_guard_active, (
            "_get_reinit_mask called before GT guard was activated — "
            "this indicates a logic error in run_sequence."
        )

        # Informational log: GT file exists but we deliberately do NOT read it.
        gt_path = Path(data_root) / "masks" / seq_name / f"{frame_name}.png"
        if gt_path.exists():
            log.debug(
                "[GT Guard] GT mask exists for reinit frame '%s' — NOT reading it "
                "(guard active; reading would raise RuntimeError).",
                frame_name,
            )

        use_prev = self.cfg.get("drift", {}).get("use_prev_mask_for_reinit", True)
        prev_area = mask_area(prev_mask)
        if use_prev and prev_area > 0:
            log.info(
                "[REINIT source=prev_mask] frame=%s  area=%d px",
                frame_name, prev_area,
            )
            return prev_mask, "prev_mask"

        # Previous mask is empty or use_prev is disabled — run fresh auto-init.
        if prev_area == 0:
            reason = "prev_mask_empty"
        else:
            reason = "use_prev_disabled"
        log.info(
            "[REINIT source=auto_init] frame=%s  reason=%s",
            frame_name, reason,
        )
        image_rgb = read_image_rgb(frame_path)
        return self.initialiser.predict(image_rgb), "auto_init"

    def _evaluate_reinit_gate(
        self,
        inference_state,
        reinit_at: int,
        reinit_mask: np.ndarray,
        m_base_mask: np.ndarray,
        conf_base: float,
        m_prev_mask: Optional[np.ndarray],
        delta_conf: float,
        delta_iou: float,
    ) -> bool:
        """Accept/reject gate for re-initialisation.

        Runs one SAM2 forward step with ``reinit_mask`` to obtain the model's
        actual confidence and predicted mask (``M_reinit_pred``), then accepts
        only when either quality condition is met:

            conf(M_reinit_pred) > conf(M_base) + delta_conf   [conf gate]
            OR
            IoU(M_reinit_pred, M_prev) > IoU(M_base, M_prev) + delta_iou  [IoU gate]

        After the evaluation the inference_state is left in a *reset* state so
        that the caller can unconditionally add the chosen mask and restart
        propagation.

        Returns
        -------
        True  → accept M_reinit (caller should use it as the new prompt)
        False → reject M_reinit (caller should fall back to M_base)
        """
        # ---- One-step test: reset → add M_reinit → propagate one frame ----
        with torch.inference_mode(), _maybe_autocast(self.device):
            self.video_predictor.reset_state(inference_state)
            self.video_predictor.add_new_mask(
                inference_state,
                frame_idx=reinit_at,
                obj_id=self.OBJ_ID,
                mask=reinit_mask,
            )

        conf_reinit: float = conf_base   # safe fallback
        m_reinit_pred: np.ndarray = reinit_mask

        with torch.inference_mode(), _maybe_autocast(self.device):
            for _fi, _oids, ml in self.video_predictor.propagate_in_video(
                inference_state, start_frame_idx=reinit_at
            ):
                conf_reinit = float(torch.sigmoid(ml[0, 0]).max().cpu())
                m_reinit_pred = (ml[0, 0] > 0.0).cpu().numpy()
                break  # only the first (reinit) frame is needed

        # ---- Compute IoU against previous frame ----
        iou_base_prev = _mask_iou(m_base_mask, m_prev_mask)
        iou_reinit_prev = _mask_iou(m_reinit_pred, m_prev_mask)

        # ---- Gate condition ----
        cond_conf = conf_reinit > conf_base + delta_conf
        cond_iou = iou_reinit_prev > iou_base_prev + delta_iou
        accept = cond_conf or cond_iou

        log.info(
            "[REINIT GATE] frame=%d %s | "
            "conf_reinit=%.3f conf_base=%.3f (need >%.3f) | "
            "iou_reinit_prev=%.3f iou_base_prev=%.3f (need >%.3f)",
            reinit_at,
            "ACCEPT" if accept else "REJECT",
            conf_reinit, conf_base, conf_base + delta_conf,
            iou_reinit_prev, iou_base_prev, iou_base_prev + delta_iou,
        )

        # Leave state clean for the caller's unconditional reset+add below.
        with torch.inference_mode(), _maybe_autocast(self.device):
            self.video_predictor.reset_state(inference_state)

        return accept

    def _save_results(
        self,
        data_root: str,
        seq_name: str,
        frame_paths: List[str],
        result: SequenceResult,
    ) -> None:
        """Write predicted masks and coloured visualisations to disk."""
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
                        image_rgb,
                        mask,
                        str(vis_dir / f"{fname}.jpg"),
                        color=vis_color,
                        alpha=vis_alpha,
                        info_text=info,
                    )

        log.info("Results saved → masks: %s  vis: %s", mask_dir, vis_dir)

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def run_dataset(
        self,
        data_root: str,
        sequences: Optional[List[str]] = None,
        save_output: bool = True,
    ) -> Dict[str, SequenceResult]:
        """Run the tracker on multiple sequences.

        Args:
            data_root:   Root with images/ and masks/ subdirectories.
            sequences:   Sequence names; if None, auto-discover all.
            save_output: Passed through to run_sequence.
        """
        if sequences is None:
            from .utils import list_sequences
            sequences = list_sequences(data_root)

        all_results: Dict[str, SequenceResult] = {}
        for seq in sequences:
            try:
                all_results[seq] = self.run_sequence(
                    data_root, seq, save_output=save_output
                )
            except Exception as exc:
                log.error("Failed on sequence '%s': %s", seq, exc, exc_info=True)

        return all_results

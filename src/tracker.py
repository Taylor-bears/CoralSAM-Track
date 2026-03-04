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
        self.timing: Dict = {}

    def __repr__(self) -> str:
        return (
            f"SequenceResult(seq={self.seq_name}, "
            f"frames={len(self.masks)}, re_inits={len(self.reinit_frames)}, "
            f"fps={self.timing.get('fps', 0):.1f})"
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
        with torch.inference_mode(), _maybe_autocast(self.device):
            inference_state = self.video_predictor.init_state(
                video_path=images_dir,
                offload_video_to_cpu=False,
                offload_state_to_cpu=False,
            )

        # ------------------------------------------------------------------
        # Step 2: Auto-init first frame → add mask prompt to SAM2
        # ------------------------------------------------------------------
        log.info("Auto-initialising first frame...")
        first_image_rgb = read_image_rgb(frame_paths[0])
        init_mask = self.initialiser.predict(first_image_rgb)
        log.info("Init mask area: %d px", mask_area(init_mask))

        with torch.inference_mode(), _maybe_autocast(self.device):
            self.video_predictor.add_new_mask(
                inference_state, frame_idx=0, obj_id=self.OBJ_ID, mask=init_mask
            )

        # ------------------------------------------------------------------
        # Step 3: Propagate with optional drift correction
        # ------------------------------------------------------------------
        check_interval = self.cfg.get("drift", {}).get("check_interval", 10)

        # {frame_idx: (binary_mask, confidence)}
        all_predictions: Dict[int, Tuple[np.ndarray, float]] = {}
        frame_timings: List[float] = []  # per-frame wall-clock seconds

        prev_area: int = mask_area(init_mask)
        segment_start: int = 0

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
                    t_seg_start = t_frame_end  # reset for next frame

                    binary_mask = (masks_logits[0, 0] > 0.0).cpu().numpy()
                    # Use max sigmoid of logits as proxy confidence (0→1)
                    conf = float(torch.sigmoid(masks_logits[0, 0]).max().cpu())

                    all_predictions[frame_idx] = (binary_mask, conf)

                    # -- Drift check every `check_interval` frames --
                    if (
                        self.use_drift_correction
                        and frame_idx > segment_start
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
                            break  # stop this propagation segment

                    prev_area = mask_area(binary_mask)

                else:
                    # for-else: loop completed without break → all frames done
                    break

            # -- Reinit: reset SAM2 state, re-prompt, continue from reinit_at --
            if reinit_at is not None:
                log.info("Re-initialising SAM2 from frame %d", reinit_at)
                reinit_mask = self._get_reinit_mask(
                    frame_path=frame_paths[reinit_at],
                    prev_mask=all_predictions[reinit_at][0],
                )
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
                # Don't break – continue outer while loop from reinit_at
            else:
                break  # done with this sequence

        # ------------------------------------------------------------------
        # Step 4: Pack results and timing
        # ------------------------------------------------------------------
        for fidx, (mask, conf) in all_predictions.items():
            fname = frame_names[fidx]
            result.masks[fname] = mask
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

    def _get_reinit_mask(
        self, frame_path: str, prev_mask: np.ndarray
    ) -> np.ndarray:
        """Get a fresh mask for the re-initialisation frame.

        use_prev_mask_for_reinit=True  → reuse the drifted mask (fast).
        use_prev_mask_for_reinit=False → re-run auto-init on this frame (slow,
                                          but recovers from real drift).
        """
        use_prev = self.cfg.get("drift", {}).get("use_prev_mask_for_reinit", False)
        if use_prev:
            return prev_mask
        image_rgb = read_image_rgb(frame_path)
        return self.initialiser.predict(image_rgb)

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

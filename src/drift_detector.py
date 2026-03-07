"""
Drift detection for CoralSAM-Track.

Inspired by "Prompt Self-Correction for SAM2 Zero-Shot Video Object
Segmentation" which observes that SAM2's internal prediction quality can
degrade over long sequences and proposes detecting such degradation and
re-prompting the model.

Three complementary drift signals are monitored:

  1. Confidence (iou_predictions proxy)
     SAM2's mask logit magnitude is used as a surrogate confidence score.
     When max(sigmoid(logits)) < conf_thresh, the mask quality is suspect.

  2. Area consistency
     A sudden jump or collapse in segmented area between consecutive
     observed frames is a reliable indicator of tracking failure.

  3. IoU consistency
     Low mask-to-mask IoU between consecutive check frames.

Additional robustness features:
  - Cooldown period after each re-init to prevent cascade failures.
  - Configurable minimum signal count (require N out of 3 signals to fire).

Usage
-----
    detector = DriftDetector(cfg)
    is_drift = detector.check(frame_idx, mask, confidence, prev_area)
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Optional

import numpy as np

log = logging.getLogger(__name__)


class DriftDetector:
    """Detects tracking drift and decides whether to trigger re-initialisation.

    Parameters
    ----------
    cfg : dict
        Loaded from configs/default.yaml.  Relevant sub-keys under ``drift``:
          - conf_thresh          (float, default 0.50)
          - area_ratio_thresh    (float, default 1.8)
          - iou_thresh           (float, default 0.30)
          - ema_alpha            (float, default 0.4)
          - consecutive_low_conf (int,   default 2)
          - cooldown_frames      (int,   default 10) – skip checks after reinit
          - min_signals          (int,   default 1)  – how many signals must
                                                       agree to fire drift
    """

    def __init__(self, cfg: dict) -> None:
        drift_cfg = cfg.get("drift", {})
        self.conf_thresh: float = float(drift_cfg.get("conf_thresh", 0.50))
        self.area_ratio_thresh: float = float(drift_cfg.get("area_ratio_thresh", 1.8))
        self.iou_thresh: float = float(drift_cfg.get("iou_thresh", 0.30))
        self._ema_alpha: float = float(drift_cfg.get("ema_alpha", 0.4))
        self._consec_required: int = int(drift_cfg.get("consecutive_low_conf", 2))
        self._cooldown_frames: int = int(drift_cfg.get("cooldown_frames", 4))
        self._min_signals: int = int(drift_cfg.get("min_signals", 1))

        # Internal state
        self._ema_conf: Optional[float] = None
        self._low_conf_streak: int = 0
        self._prev_mask: Optional[np.ndarray] = None
        self._history: Deque[dict] = deque(maxlen=50)
        self._cooldown_remaining: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        frame_idx: int,
        mask: np.ndarray,
        confidence: float,
        prev_area: Optional[int],
    ) -> bool:
        """Return True if drift is detected at this frame.

        Respects the cooldown period: returns False unconditionally during
        cooldown, while still updating internal state for accurate EMA.
        """
        current_mask = mask.astype(bool)
        current_area = int(current_mask.sum())

        # ---- 1. Update EMA confidence (always, even during cooldown) ----
        if self._ema_conf is None:
            self._ema_conf = confidence
        else:
            self._ema_conf = (
                self._ema_alpha * confidence + (1 - self._ema_alpha) * self._ema_conf
            )

        # ---- Cooldown guard ----
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            self._prev_mask = current_mask.copy()
            log.debug(
                "[DriftDetector] frame=%d COOLDOWN (%d remaining)",
                frame_idx, self._cooldown_remaining,
            )
            return False

        # ---- 2. Confidence check ----
        conf_low = self._ema_conf < self.conf_thresh
        if conf_low:
            self._low_conf_streak += 1
        else:
            self._low_conf_streak = 0

        conf_drift = self._low_conf_streak >= self._consec_required

        # ---- 3. Area ratio check ----
        area_drift = False
        area_ratio = 1.0
        if prev_area is not None and prev_area > 0 and current_area > 0:
            area_ratio = max(current_area / prev_area, prev_area / current_area)
            area_drift = area_ratio > self.area_ratio_thresh
        elif prev_area is not None and prev_area > 0 and current_area == 0:
            area_drift = True
            area_ratio = float("inf")

        # ---- 4. IoU consistency check ----
        iou_drift = False
        iou_score = 1.0
        if self._prev_mask is not None and self.iou_thresh > 0:
            inter = (current_mask & self._prev_mask).sum()
            union = (current_mask | self._prev_mask).sum()
            iou_score = float(inter) / float(union) if union > 0 else 1.0
            iou_drift = iou_score < self.iou_thresh

        self._prev_mask = current_mask.copy()

        # ---- 5. Record history ----
        self._history.append(
            {
                "frame_idx": frame_idx,
                "conf": confidence,
                "ema_conf": self._ema_conf,
                "area": current_area,
                "area_ratio": area_ratio,
                "iou": iou_score,
                "conf_drift": conf_drift,
                "area_drift": area_drift,
                "iou_drift": iou_drift,
            }
        )

        # ---- 6. Decision: require min_signals to agree ----
        n_active = sum([conf_drift, area_drift, iou_drift])
        is_drift = n_active >= self._min_signals

        if is_drift:
            reasons = []
            if conf_drift:
                reasons.append(
                    f"low_conf (ema={self._ema_conf:.3f} < {self.conf_thresh}, "
                    f"streak={self._low_conf_streak})"
                )
            if area_drift:
                reasons.append(
                    f"area_ratio={area_ratio:.2f} > {self.area_ratio_thresh}"
                )
            if iou_drift:
                reasons.append(
                    f"iou={iou_score:.3f} < {self.iou_thresh}"
                )
            log.debug(
                "[DriftDetector] frame=%d DRIFT (%d/%d signals): %s",
                frame_idx, n_active, 3, "; ".join(reasons),
            )
            self._low_conf_streak = 0
        else:
            log.debug(
                "[DriftDetector] frame=%d OK  ema_conf=%.3f  area_ratio=%.2f  iou=%.3f",
                frame_idx,
                self._ema_conf,
                area_ratio,
                iou_score,
            )

        return is_drift

    def reset(self) -> None:
        """Reset internal state and activate cooldown (call after re-init)."""
        self._ema_conf = None
        self._low_conf_streak = 0
        self._prev_mask = None
        self._cooldown_remaining = self._cooldown_frames

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_history(self) -> list:
        """Return a copy of the recent detection history (for debugging)."""
        return list(self._history)

    def summary(self) -> dict:
        """Return aggregate statistics over the tracked history."""
        if not self._history:
            return {}
        confs = [h["conf"] for h in self._history]
        areas = [h["area"] for h in self._history]
        n_drift = sum(1 for h in self._history if h["conf_drift"] or h["area_drift"])
        return {
            "frames_checked": len(self._history),
            "n_drift_events": n_drift,
            "mean_conf": float(np.mean(confs)),
            "min_conf": float(np.min(confs)),
            "mean_area": float(np.mean(areas)),
        }

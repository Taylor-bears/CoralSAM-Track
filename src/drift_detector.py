"""
Drift detection for CoralSAM-Track.

Inspired by "Prompt Self-Correction for SAM2 Zero-Shot Video Object
Segmentation" which observes that SAM2's internal prediction quality can
degrade over long sequences and proposes detecting such degradation and
re-prompting the model.

Two complementary drift signals are monitored:

  1. Confidence (iou_predictions proxy)
     SAM2's mask logit magnitude is used as a surrogate confidence score.
     When max(sigmoid(logits)) < conf_thresh, the mask quality is suspect.

  2. Area consistency
     A sudden jump or collapse in segmented area between consecutive
     observed frames is a reliable indicator of tracking failure.
     When max(s_t/s_{t-1}, s_{t-1}/s_t) > area_ratio_thresh, trigger.

Both checks are gated by an additional history: the detector maintains a
short EMA of recent confidences to suppress single-frame false positives.

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
          - conf_thresh          (float, default 0.70)
          - area_ratio_thresh    (float, default 3.0)
          - ema_alpha            (float, default 0.3)  – EMA smoothing factor
          - consecutive_low_conf (int,   default 2)    – consecutive low-conf
                                                          frames needed to fire
    """

    def __init__(self, cfg: dict) -> None:
        drift_cfg = cfg.get("drift", {})
        self.conf_thresh: float = float(drift_cfg.get("conf_thresh", 0.70))
        self.area_ratio_thresh: float = float(drift_cfg.get("area_ratio_thresh", 3.0))
        # EMA smoothing coefficient for confidence history
        self._ema_alpha: float = float(drift_cfg.get("ema_alpha", 0.3))
        # Number of consecutive below-threshold frames required to fire
        self._consec_required: int = int(drift_cfg.get("consecutive_low_conf", 2))

        # Internal state
        self._ema_conf: Optional[float] = None
        self._low_conf_streak: int = 0
        self._history: Deque[dict] = deque(maxlen=50)

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

        Args:
            frame_idx:   Current frame index (for logging).
            mask:        Binary predicted mask (H, W bool/uint8).
            confidence:  SAM2 proxy confidence score in [0, 1].
            prev_area:   Pixel area of the mask at the previous check frame.
                         Pass None on the very first call.

        Returns:
            True  → drift detected, caller should trigger re-init.
            False → tracking looks healthy.
        """
        current_area = int(mask.astype(bool).sum())

        # ---- 1. Update EMA confidence ----
        if self._ema_conf is None:
            self._ema_conf = confidence
        else:
            self._ema_conf = (
                self._ema_alpha * confidence + (1 - self._ema_alpha) * self._ema_conf
            )

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
            # Mask completely disappeared
            area_drift = True
            area_ratio = float("inf")

        # ---- 4. Record history ----
        self._history.append(
            {
                "frame_idx": frame_idx,
                "conf": confidence,
                "ema_conf": self._ema_conf,
                "area": current_area,
                "area_ratio": area_ratio,
                "conf_drift": conf_drift,
                "area_drift": area_drift,
            }
        )

        # ---- 5. Decision ----
        is_drift = conf_drift or area_drift

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
            log.debug(
                "[DriftDetector] frame=%d DRIFT: %s", frame_idx, "; ".join(reasons)
            )
            # Reset streak so we don't fire every subsequent frame unnecessarily
            self._low_conf_streak = 0
        else:
            log.debug(
                "[DriftDetector] frame=%d OK  ema_conf=%.3f  area_ratio=%.2f",
                frame_idx,
                self._ema_conf,
                area_ratio,
            )

        return is_drift

    def reset(self) -> None:
        """Reset internal state (call after each re-initialisation)."""
        self._ema_conf = None
        self._low_conf_streak = 0

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

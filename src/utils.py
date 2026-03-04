"""
Utility functions: dataset loading, mask I/O, visualisation, FPS timing.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def list_sequences(data_root: str) -> List[str]:
    """Return sorted list of video sequence names under data_root/images/."""
    images_dir = Path(data_root) / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"images/ not found under {data_root}")
    seqs = sorted([d.name for d in images_dir.iterdir() if d.is_dir()])
    return seqs


def load_frames(seq_dir: str, ext: str = ".jpg") -> List[str]:
    """Return sorted list of absolute frame paths for a sequence directory."""
    paths = sorted(Path(seq_dir).glob(f"*{ext}"))
    if not paths:
        paths = sorted(Path(seq_dir).glob("*.png"))
    return [str(p) for p in paths]


def read_image_rgb(path: str) -> np.ndarray:
    """Read an image as RGB uint8 numpy array (H, W, 3)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Mask I/O
# ---------------------------------------------------------------------------

def read_mask(path: str) -> np.ndarray:
    """Read a GT or predicted mask as bool (H, W)."""
    mask = np.array(Image.open(path).convert("L"))
    return mask > 0


def save_mask(mask: np.ndarray, path: str) -> None:
    """Save a binary bool/uint8 mask as a PNG (0 / 255)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = (mask.astype(np.uint8) * 255)
    Image.fromarray(out, mode="L").save(path)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def overlay_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 128),
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend a binary mask over an RGB image.

    Args:
        image_rgb: H×W×3 uint8.
        mask:      H×W bool or uint8.
        color:     RGB colour for the overlay.
        alpha:     Transparency in [0, 1].

    Returns:
        H×W×3 uint8 blended image.
    """
    overlay = image_rgb.copy()
    overlay[mask.astype(bool)] = (
        (1 - alpha) * overlay[mask.astype(bool)] + alpha * np.array(color)
    ).astype(np.uint8)
    # Draw contour for clarity
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)
    return overlay


def save_vis(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    path: str,
    color: Tuple[int, int, int] = (0, 255, 128),
    alpha: float = 0.5,
    info_text: Optional[str] = None,
) -> None:
    """Save a visualisation image (mask overlay + optional text)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vis = overlay_mask(image_rgb, mask, color=color, alpha=alpha)
    if info_text:
        cv2.putText(
            vis,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    # Save as BGR
    cv2.imwrite(path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# FPS / timing
# ---------------------------------------------------------------------------

class Timer:
    """Simple wall-clock timer for tracking per-frame / per-sequence latency."""

    def __init__(self) -> None:
        self._t0: float = 0.0
        self._times: List[float] = []

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self) -> float:
        elapsed = time.perf_counter() - self._t0
        self._times.append(elapsed)
        return elapsed

    @property
    def frame_times(self) -> List[float]:
        return list(self._times)

    @property
    def mean_fps(self) -> float:
        if not self._times:
            return 0.0
        return 1.0 / (sum(self._times) / len(self._times))

    @property
    def total_seconds(self) -> float:
        return sum(self._times)

    def summary(self, n_frames: int) -> Dict[str, float]:
        total = self.total_seconds
        mean_ms = (total / n_frames * 1000) if n_frames else 0.0
        fps = self.mean_fps
        return {
            "total_s": round(total, 3),
            "mean_ms_per_frame": round(mean_ms, 2),
            "fps": round(fps, 2),
            "n_frames": n_frames,
        }

    def reset(self) -> None:
        self._times.clear()


# ---------------------------------------------------------------------------
# Mask merging helpers
# ---------------------------------------------------------------------------

def merge_masks(masks: List[np.ndarray], min_area: int = 100) -> np.ndarray:
    """Union of a list of binary masks, optionally filtering tiny regions."""
    if not masks:
        return np.zeros((0, 0), dtype=bool)
    merged = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        region_area = int(m.sum())
        if region_area >= min_area:
            merged |= m.astype(bool)
    return merged


def mask_area(mask: np.ndarray) -> int:
    return int(mask.astype(bool).sum())


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load a YAML config file and return a plain dict."""
    import yaml  # lazy import – only needed at runtime

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_nested(cfg: dict, *keys, default=None):
    """Safe nested dict access: get_nested(cfg, 'drift', 'enabled')."""
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

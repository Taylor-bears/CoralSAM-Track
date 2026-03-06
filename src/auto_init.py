"""
Automatic first-frame initialisation for CoralSAM-Track.

Two strategies are supported (selected via cfg['init_method']):

  sam2_auto  (default)
    - Use SAM2's image predictor in "everything" mode: sample an N×N grid of
      candidate foreground points, run the SAM2 image predictor for each, and
      merge all masks that pass score/area thresholds into a single coral mask.
    - Only the SAM2 checkpoint is required.

  coralscop
    - Use CoralSCOP's SamAutomaticMaskGenerator (coral-domain fine-tuned SAM).
    - Requires `segment_anything` to be importable and the CoralSCOP ViT-B
      checkpoint at cfg['coralscop']['checkpoint'].
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from .utils import merge_masks, mask_area, read_image_rgb

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SAM2-Auto initialiser
# ---------------------------------------------------------------------------

class SAM2AutoInitialiser:
    """Grid-based automatic first-frame initialisation using SAM2.

    Workflow
    --------
    1. Load the SAM2 *image* predictor (not the video predictor).
    2. Set the first-frame image.
    3. Sample a uniform N×N grid of (x, y) point prompts.
    4. For each point, predict a mask and keep it when
       iou_predictions >= score_thresh AND area >= min_mask_area.
    5. Union all accepted masks → single binary coral mask.
    """

    def __init__(self, cfg: dict, device: str = "cuda") -> None:
        self.cfg = cfg
        self.device = device
        self.grid_size: int = cfg.get("sam2", {}).get("grid_size", 16)
        self.min_mask_area: int = cfg.get("sam2", {}).get("min_mask_area", 500)
        self.score_thresh: float = cfg.get("sam2", {}).get("score_thresh", 0.70)
        self._predictor = None  # lazy init

    def _build_predictor(self):
        """Lazy-load the SAM2 image predictor."""
        if self._predictor is not None:
            return

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as e:
            raise ImportError(
                "SAM2 is not installed. "
                "Run: pip install -e git+https://github.com/facebookresearch/sam2.git#egg=sam2"
            ) from e

        checkpoint = self.cfg.get("sam2", {}).get("checkpoint", "checkpoints/sam2.1_hiera_large.pt")
        model_cfg = self.cfg.get("sam2", {}).get("config", "sam2.1_hiera_l.yaml")

        if not Path(checkpoint).exists():
            raise FileNotFoundError(
                f"SAM2 checkpoint not found: {checkpoint}\n"
                "Download from https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
            )

        log.info("Loading SAM2 image predictor from %s", checkpoint)
        sam2_model = build_sam2(model_cfg, checkpoint, device=self.device)
        self._predictor = SAM2ImagePredictor(sam2_model)
        log.info("SAM2 image predictor loaded.")

    def predict(self, image_rgb: np.ndarray) -> np.ndarray:
        """Return a binary mask (H, W, bool) for the given RGB image.

        Improvement over naive union: collect (score, mask) pairs, sort by
        score descending, then greedily add masks only while the running union
        area stays below max_area_ratio of the image.  This prevents the
        merged mask from exploding to half the frame when many grid points
        happen to land on background.
        """
        self._build_predictor()

        H, W = image_rgb.shape[:2]
        total_pixels = H * W
        # Cap merged mask at 30% of image area to avoid over-segmentation
        max_area_frac: float = self.cfg.get("sam2", {}).get("max_area_frac", 0.30)

        self._predictor.set_image(image_rgb)

        # Build grid of foreground prompt points
        grid_pts = self._make_grid(W, H, self.grid_size)

        scored: List[tuple] = []  # (score, mask)

        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            for x, y in grid_pts:
                pts = np.array([[x, y]], dtype=np.float32)
                lbls = np.array([1], dtype=np.int32)
                masks, scores, _ = self._predictor.predict(
                    point_coords=pts,
                    point_labels=lbls,
                    multimask_output=True,
                )
                # masks: (3, H, W); dtype may be float under bfloat16 autocast
                best_idx = int(np.argmax(scores))
                best_mask = masks[best_idx].astype(bool)
                best_score = float(scores[best_idx])

                if best_score >= self.score_thresh and mask_area(best_mask) >= self.min_mask_area:
                    scored.append((best_score, best_mask))

        if not scored:
            log.warning(
                "SAM2-Auto init: no mask passed the thresholds. "
                "Falling back to centre-point mask."
            )
            pts = np.array([[W // 2, H // 2]], dtype=np.float32)
            lbls = np.array([1], dtype=np.int32)
            with torch.inference_mode():
                masks, scores, _ = self._predictor.predict(
                    point_coords=pts, point_labels=lbls, multimask_output=True
                )
            return masks[int(np.argmax(scores))].astype(bool)

        # Sort by score descending; greedily merge while area stays bounded
        scored.sort(key=lambda t: t[0], reverse=True)
        running_union = np.zeros((H, W), dtype=bool)
        accepted: List[np.ndarray] = []
        for score, mask in scored:
            candidate = running_union | mask
            if candidate.sum() / total_pixels <= max_area_frac:
                running_union = candidate
                accepted.append(mask)
            # Once area budget exceeded, skip remaining (they have lower scores)

        if not accepted:
            # All masks individually exceed area budget — take the highest-score one
            accepted = [scored[0][1]]
            running_union = accepted[0].copy()

        merged = merge_masks(accepted, min_area=self.min_mask_area)
        log.info(
            "SAM2-Auto init: %d/%d grid prompts accepted → merged area=%d px (%.1f%% of frame)",
            len(accepted),
            len(grid_pts),
            mask_area(merged),
            mask_area(merged) / total_pixels * 100,
        )
        return merged

    @staticmethod
    def _make_grid(W: int, H: int, n: int) -> List[tuple]:
        """Return a list of (x, y) pixel coordinates for an n×n uniform grid."""
        xs = np.linspace(W * 0.1, W * 0.9, n, dtype=int)
        ys = np.linspace(H * 0.1, H * 0.9, n, dtype=int)
        return [(int(x), int(y)) for y in ys for x in xs]


# ---------------------------------------------------------------------------
# CoralSCOP initialiser
# ---------------------------------------------------------------------------

class CoralSCOPInitialiser:
    """First-frame initialisation using CoralSCOP's automatic mask generator.

    CoralSCOP exposes the same interface as SAM's SamAutomaticMaskGenerator
    but uses a coral-domain fine-tuned ViT-B checkpoint.

    Requirements
    ------------
    - ``segment_anything`` package installed (SAM base library).
    - CoralSCOP repo cloned and on sys.path (or installed via pip).
    - Weight file at cfg['coralscop']['checkpoint'].
    """

    def __init__(self, cfg: dict, device: str = "cuda") -> None:
        self.cfg = cfg
        self.device = device
        self._generator = None  # lazy init

    def _build_generator(self):
        if self._generator is not None:
            return

        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        except ImportError as e:
            raise ImportError(
                "segment_anything is not installed. "
                "Run: pip install git+https://github.com/facebookresearch/segment-anything.git"
            ) from e

        cs_cfg = self.cfg.get("coralscop", {})
        checkpoint = cs_cfg.get("checkpoint", "checkpoints/vit_b_coralscop.pth")
        model_type = cs_cfg.get("model_type", "vit_b")

        if not Path(checkpoint).exists():
            raise FileNotFoundError(
                f"CoralSCOP checkpoint not found: {checkpoint}\n"
                "Download from https://www.dropbox.com/scl/fi/pw5jiq9oc8e8kvkx1fdk0/"
                "vit_b_coralscop.pth?rlkey=qczdohnzxwgwoadpzeht0lim2&dl=1"
            )

        log.info("Loading CoralSCOP from %s", checkpoint)
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=self.device)

        self._generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=cs_cfg.get("points_per_side", 32),
            pred_iou_thresh=cs_cfg.get("pred_iou_thresh", 0.72),
            stability_score_thresh=cs_cfg.get("stability_score_thresh", 0.62),
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=cs_cfg.get("min_mask_region_area", 500),
        )
        log.info("CoralSCOP generator loaded.")

    def predict(self, image_rgb: np.ndarray) -> np.ndarray:
        """Return a binary mask (H, W, bool) – union of all CoralSCOP detections."""
        self._build_generator()

        annotations = self._generator.generate(image_rgb)
        if not annotations:
            log.warning("CoralSCOP: no annotations returned. Returning empty mask.")
            H, W = image_rgb.shape[:2]
            return np.zeros((H, W), dtype=bool)

        masks = [ann["segmentation"].astype(bool) for ann in annotations]
        merged = merge_masks(
            masks,
            min_area=self.cfg.get("coralscop", {}).get("min_mask_region_area", 500),
        )
        log.info(
            "CoralSCOP init: %d detections → merged area=%d px",
            len(masks),
            mask_area(merged),
        )
        return merged


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_initialiser(cfg: dict, device: str = "cuda"):
    """Return the appropriate initialiser based on cfg['init_method']."""
    method = cfg.get("init_method", "sam2_auto").lower()
    if method == "sam2_auto":
        return SAM2AutoInitialiser(cfg, device=device)
    elif method == "coralscop":
        return CoralSCOPInitialiser(cfg, device=device)
    else:
        raise ValueError(
            f"Unknown init_method '{method}'. Choose 'sam2_auto' or 'coralscop'."
        )


def auto_init_frame(
    image_path: str,
    cfg: dict,
    device: str = "cuda",
    initialiser=None,
) -> np.ndarray:
    """Convenience wrapper: load image and run auto-initialisation.

    Args:
        image_path:  Path to the first frame image.
        cfg:         Config dict.
        device:      Torch device string.
        initialiser: Optional pre-built initialiser (avoids reloading weights).

    Returns:
        Binary mask as bool numpy array (H, W).
    """
    image_rgb = read_image_rgb(image_path)
    if initialiser is None:
        initialiser = build_initialiser(cfg, device=device)
    return initialiser.predict(image_rgb)

# CoralSAM-Track

**End-to-end coral video segmentation** on the CoralVOS dataset.

Combines:
- **SAM2** ([facebookresearch/sam2](https://github.com/facebookresearch/sam2)) — video object propagation engine.
- **CoralSCOP** ([zhengziqiang/CoralSCOP](https://github.com/zhengziqiang/CoralSCOP), optional) — coral-domain auto-detector for first-frame initialisation.
- **Drift-correction module** — inspired by *"Prompt Self-Correction for SAM2 Zero-Shot Video Object Segmentation"* — automatically detects tracking drift and re-initialises SAM2.

---

## Quick overview

```
Input video sequence (images/)
        │
        ▼
Auto first-frame init  ←── SAM2-Auto grid OR CoralSCOP
        │
        ▼
SAM2 VideoPredictor.propagate_in_video()
        │
        ├── every K frames: drift check (confidence + area ratio)
        │         │
        │   drift detected?──yes──► re-detect current frame → re-prompt SAM2
        │         │
        │        no
        ▼
Predicted masks + visualisations saved
        │
        ▼
eval.py  →  J / F / J&F mean  (w/ vs w/o drift correction)
```

---

## Repository layout

```
CoralSAM-Track/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml          ← all hyper-parameters
├── src/
│   ├── auto_init.py          ← first-frame initialiser (SAM2-Auto & CoralSCOP)
│   ├── tracker.py            ← CoralTracker pipeline
│   ├── drift_detector.py     ← drift detection logic
│   └── utils.py              ← I/O, visualisation, timing
├── scripts/
│   ├── demo.py               ← run one (or all) sequence(s)
│   └── eval.py               ← J&F evaluation + comparison table
├── checkpoints/              ← place model weights here (see below)
├── outputs/                  ← auto-created; saved masks & visualisations
└── partial_coralvos/
    └── partial/
        ├── images/<video_name>/*.jpg
        └── masks/<video_name>/*.png
```

---

## Environment setup

### 1. Python environment

```bash
conda create -n coralsam python=3.10 -y
conda activate coralsam
```

### 2. PyTorch (CUDA 11.8 example — adjust to your CUDA version)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. SAM2

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
# SAM2 installs its own model configs under sam2/configs/
cd ..
```

### 4. Other dependencies

```bash
pip install -r requirements.txt
```

### 5. (Optional) CoralSCOP

Only required when `init_method: coralscop` in the config.

```bash
# Base SAM library
pip install git+https://github.com/facebookresearch/segment-anything.git

# CoralSCOP repo (for SamAutomaticMaskGenerator with coral weights)
git clone https://github.com/zhengziqiang/CoralSCOP.git
# No install needed; the generator is loaded directly via segment_anything
```

---

## Download model weights

Place all checkpoints in the `checkpoints/` directory.

### SAM2 (required)

| Model | Size | Download |
|-------|------|----------|
| `sam2.1_hiera_large.pt` (recommended) | 224 MB | `wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -P checkpoints/` |
| `sam2.1_hiera_base_plus.pt` | 81 MB | `wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt -P checkpoints/` |
| `sam2.1_hiera_small.pt` | 46 MB | `wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt -P checkpoints/` |
| `sam2.1_hiera_tiny.pt` | 39 MB | `wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt -P checkpoints/` |

Or use the SAM2 helper script:

```bash
cd sam2/checkpoints && ./download_ckpts.sh && cd ../..
# Then copy / symlink to CoralSAM-Track/checkpoints/
```

### CoralSCOP (optional)

Download `vit_b_coralscop.pth` from:

> [Dropbox link](https://www.dropbox.com/scl/fi/pw5jiq9oc8e8kvkx1fdk0/vit_b_coralscop.pth?rlkey=qczdohnzxwgwoadpzeht0lim2&st=actcedwy&dl=1)

```bash
# Linux/Mac
wget "https://www.dropbox.com/scl/fi/pw5jiq9oc8e8kvkx1fdk0/vit_b_coralscop.pth?rlkey=qczdohnzxwgwoadpzeht0lim2&dl=1" \
     -O checkpoints/vit_b_coralscop.pth
```

---

## Configuration

All settings live in `configs/default.yaml`.  Key options:

```yaml
init_method: sam2_auto   # "sam2_auto" or "coralscop"

sam2:
  checkpoint: checkpoints/sam2.1_hiera_large.pt
  config: sam2.1_hiera_l.yaml   # resolved from installed sam2 package
  grid_size: 16          # NxN grid for SAM2-Auto init
  score_thresh: 0.70     # min confidence to keep a candidate mask

drift:
  enabled: true
  conf_thresh: 0.70      # EMA confidence below this → suspect drift
  area_ratio_thresh: 3.0 # mask area change ratio above this → suspect drift
  check_interval: 10     # check every N frames
```

To disable drift correction globally: set `drift.enabled: false`.

---

## One-click pipeline script (remote Linux server)

`run_pipeline.sh` automates the full flow: environment check → baseline inference → drift-correction inference → evaluation comparison.

**Quick start on a remote server:**

```bash
# 1. Edit the top-level variables in the script:
#    PYTHON  → your conda env python, e.g. /data1/xxx/envs/coralsam/bin/python
#    CUDA_DEVICE → GPU id from `nvidia-smi`, e.g. 0
#    ALL_SEQS / SEQUENCES → which sequences to process

nano run_pipeline.sh   # or vim

# 2. Make executable and run
chmod +x run_pipeline.sh
./run_pipeline.sh
```

The script will:
1. Verify Python / PyTorch / CUDA / SAM2 / dataset are all reachable
2. Auto-download the SAM2 checkpoint if it is missing
3. Run **baseline** inference (no drift correction) → `outputs/baseline/`
4. Run **drift-corrected** inference → `outputs/with_drift_corr/`
5. Run evaluation and print the J / F / J&F comparison table
6. Save all logs to `logs/` and the metric JSON to `outputs/eval_results.json`

Key switches at the top of the script:

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTHON` | `/data1/…/coralsam/bin/python` | Path to your conda env Python |
| `CUDA_DEVICE` | `0` | GPU id (`nvidia-smi` to check) |
| `ALL_SEQS` | `true` | `true` = all sequences; `false` = use `SEQUENCES` list |
| `SEQUENCES` | all 6 videos | Space-separated sequence names |
| `RUN_BASELINE` | `true` | Whether to run the no-drift-correction baseline |
| `RUN_DRIFT_CORR` | `true` | Whether to run drift-corrected inference |
| `RUN_EVAL` | `true` | Whether to compute J&F evaluation |

---

## Running the demo

### Single sequence

```bash
# With drift correction (default)
python scripts/demo.py --seq video102

# Baseline — no drift correction
python scripts/demo.py --seq video102 --no_drift_correction

# Use CoralSCOP initialiser (requires checkpoint)
python scripts/demo.py --seq video102 --config configs/coralscop.yaml
```

### All sequences

```bash
# Run with drift correction
python scripts/demo.py --all

# Run baseline (needed for comparison eval)
python scripts/demo.py --all --no_drift_correction
```

Output is saved under `outputs/with_drift_corr/` and `outputs/baseline/`:

```
outputs/
├── with_drift_corr/
│   ├── masks/video102/00006.png ...
│   ├── vis/video102/00006.jpg   ...   (coloured overlay)
│   └── timing.json
└── baseline/
    ├── masks/video102/00006.png ...
    └── timing.json
```

---

## Evaluation

```bash
# Evaluate with-drift-correction results
python scripts/eval.py \
    --pred_dir outputs/with_drift_corr/masks

# Compare both methods side-by-side
python scripts/eval.py \
    --pred_dir outputs/with_drift_corr/masks \
    --baseline_dir outputs/baseline/masks \
    --output_json outputs/eval_results.json
```

Example output:

```
=================================================================
  Results for: w/ drift correction
=================================================================
  Sequence              J        F      J&F   Frames
  ------------------------------------------------------------
  video102         0.7234   0.6891   0.7063      423
  video75          0.6812   0.6543   0.6678      389
  ...
  ================================================================
  OVERALL          0.7021   0.6719   0.6870  (6 seqs)


  Comparison: baseline (no drift corr)  vs  w/ drift correction
  =============================================================
  Sequence           J(A)         J(B)           ΔJ   JF(A)  ...
  video102         0.6900       0.7234       +0.0334  ...
  ...
  Drift correction improved J&F by 0.0183 (1.83 pp)
```

---

## Metrics

| Symbol | Full name | Description |
|--------|-----------|-------------|
| **J** | Jaccard / IoU | `|pred ∩ gt| / |pred ∪ gt|` — region overlap |
| **F** | F-measure | Contour accuracy (boundary precision-recall) |
| **J&F** | J&F mean | Standard VOS metric (DAVIS benchmark) |

---

## References

1. Zheng et al., **"CoralSCOP: Segment any COral Image on this Planet"**, CVPR 2024 Highlight.
2. *"Prompt Self-Correction for SAM2 Zero-Shot Video Object Segmentation"*, CVPR Workshop 2024.
3. Ravi et al., **"SAM 2: Segment Anything in Images and Videos"**, arXiv 2024.
4. CoralVOS dataset: *"CoralVOS: Dataset and Benchmark for Coral Video Segmentation"*.

---

## Citation

If you use this code in your research, please cite the relevant papers above.

# MachVis: Synthetic-to-Real Transfer in Neural-Guided RANSAC

MachVis studies whether an NG-RANSAC guidance network trained only on
synthetic correspondences can transfer to real phone-camera image pairs for
robust epipolar geometry estimation.

This repository builds on the [NG-RANSAC](https://arxiv.org/abs/1905.04132)
codebase (Brachmann & Rother, ICCV 2019) with a synthetic data generator,
synthetic end-to-end training, and a real-image transfer evaluation pipeline.

**Authors:** Donovan Holgado (`dmh313`), Tony Lu (`hl990`), Aman Patel (`anp181`)

## What It Does

- Generates synthetic sparse correspondences with controllable geometry, noise,
  outliers, and hand-crafted Lowe ratio side-info
- Trains a PointCN-style guidance network (`CNNet`) end-to-end with a RANSAC
  task loss (inlier count) via the custom C++ NG-RANSAC extension
- Runs OpenCV RANSAC vs NG-guided filtering on real SIFT matches from phone
  image pairs
- Reports inlier counts and median epipolar error, with per-pair visualizations
- Includes a demo that compares OpenCV RANSAC and NG-RANSAC on an image pair

## Requirements

- Python 3.10+ (developed with Python 3.12)
- PyTorch (CUDA recommended for training/demo)
- OpenCV (headers + libs needed to build the C++ extension)
- NumPy, Matplotlib, SciPy

Core work lives under `ngransac/`. From the repo root:

```bash
cd ngransac
```

## Setup

1. Install Python deps (PyTorch, OpenCV, NumPy, Matplotlib, SciPy).
2. Point `ngransac/setup.py` at your local OpenCV include/lib directories if needed.
3. Build the C++ extension:

```bash
cd ngransac
python setup.py install
```

On Windows, some scripts also call `os.add_dll_directory(...)` so OpenCV DLLs
can be found at runtime.

## Quick Start: Demo

Compare OpenCV RANSAC and NG-RANSAC on the bundled demo images (essential
matrix, pretrained or synthetic weights):

```bash
cd ngransac
python ngransac_demo.py -img1 images/demo1.jpg -fl1 900 -img2 images/demo2.jpg -fl2 900
```

Demo with a synthetic-trained checkpoint (2 residual blocks):

```bash
python ngransac_demo.py -img1 images/demo1.jpg -fl1 900 -img2 images/demo2.jpg -fl2 900 -m synthetic_weights_e2e/weights_best.net -rb 2
```

Output is written to `demo.png` (OpenCV inliers on top, NG-RANSAC inliers below).

## Synthetic Data

Generate synthetic correspondence packs used for training:

```bash
python synthetic_data.py
# or the harder / larger generator:
python dataset_synthetic_2.py
```

These write `.npz` samples under `synthetic_train/` or `synthetic_train2/`
(random cameras, 3D points, projection noise, outliers, synthetic Lowe ratios).
Ground-truth pose and intrinsics are stored for each sample.

## Training

End-to-end synthetic training (inlier-count loss, early stopping):

```bash
python ngransac_train_e2e_synthetic.py
```

Default config in that script:

- Data: `synthetic_train2/` (random subset, 90/10 train/val)
- Network: 2 residual blocks
- Loss: `inliers` (pose recovery path is currently disabled / TODO)
- Checkpoints: `synthetic_weights_e2e_2/`
- Early stopping patience: 5 epochs

In the reported run, **epoch 12** was selected as best validation loss
(`-0.073855`). Plot curves with:

```bash
python plot_training_curves.py
```

Upstream paper training/testing on precomputed `traindata/` is still available
via `ngransac_train_init.py`, `ngransac_train_e2e.py`, and `ngransac_test.py`
(see `ngransac/README.md`). Download the official packs if you need those
scripts.

## Real Transfer Evaluation

Real pairs live under `images/real/` (`imgNNa.jpg` / `imgNNb.jpg`, pairs 00–14):

- Pairs 0–4: indoor
- Pairs 5–9: outdoor, darker / neutral snow
- Pairs 10–14: outdoor, brighter snow

Extract SIFT correspondences:

```bash
python generate_real_npz.py
```

This writes `real_eval_npz/`. Then evaluate baseline RANSAC vs NG-guided
filtering (network scores → keep top-k → OpenCV fundamental-matrix RANSAC):

```bash
python eval_transfer.py
```

Results and figures go to `viz/` (`eval_log.txt`, `viz_pair_*.png`). Paths for
weights / folders are set near the top of `eval_transfer.py`.

## Results (Summary)

This project is a small-scale case study of synthetic-to-real transfer, not a
full benchmark. Metrics are inlier count and median epipolar error (not pose
error).

Observed behavior on the 15 phone pairs:

- **Indoor (0–4):** NG-guided usually finds fewer inliers and higher median
  epipolar error than OpenCV RANSAC; pair 04 is especially hard (little overlap /
  large rotation)
- **Outdoor dark / grey (5–9):** NG-guided often matches or beats RANSAC on
  inlier count—closest alignment with simplified synthetic statistics
- **Outdoor bright snow (10–14):** NG-guided consistently underperforms RANSAC

On a qualitative demo pair in the report: OpenCV RANSAC ~196 inliers,
pretrained NG-RANSAC ~19 (over-selective), synthetic-trained NG ~1999
(over-accepting on that pair).

Takeaway: transfer is environment-dependent. The minimal geometric generator
(no texture, lighting, or true descriptor–geometry coupling) creates a domain
gap that hurts complex scenes but can accidentally help low-variance ones.

## Project Layout

```
machine_vision/
├── README.md                 # this file
└── ngransac/                 # NG-RANSAC code + project extensions
    ├── ngransac/             # C++ extension sources + setup.py
    ├── network.py            # CNNet guidance network
    ├── synthetic_data.py     # synthetic correspondence generator
    ├── dataset_synthetic*.py # loaders / harder generator
    ├── ngransac_train_e2e_synthetic.py
    ├── ngransac_demo.py
    ├── generate_real_npz.py
    ├── eval_transfer.py
    ├── images/real/          # 15 phone image pairs
    ├── models/               # upstream pretrained .net weights
    ├── synthetic_train*/     # generated .npz training packs
    ├── synthetic_weights_e2e*/  # local training checkpoints + logs
    └── viz/                  # transfer eval plots + eval_log.txt
```

## Current Limitations

- Transfer eval uses the guidance net as a **correspondence filter**, then
  OpenCV RANSAC—not the full C++ guided sampler used in training/demo
- Pose-loss training / pose-error reporting are not enabled
- Synthetic Lowe ratios are sampled from hand-crafted distributions, not real
  descriptors
- OpenCV paths in `setup.py` (and some DLL hooks) are machine-specific
- CUDA is assumed in several scripts; no formal `requirements.txt` or test suite
- Paper `traindata/` packs are not checked in (gitignored / download separately)

## References

- Brachmann, E., & Rother, C. (2019). Neural-Guided RANSAC: Learning Where to
  Sample Model Hypotheses. ICCV 2019.
- Yi, K. M., et al. (2018). Learning to Find Good Correspondences. CVPR 2018.
- Upstream code and docs: `ngransac/README.md`

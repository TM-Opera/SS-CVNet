# SS-CVNet

SS-CVNet predicts street view indicators — **BVI** (Green View Index), **GVI** (Sky View Index), and **SVF** (Sky View Factor) — directly from satellite imagery. It establishes a geometric correspondence between the satellite view and the ground panorama via physics-based ray casting, and fuses RGB appearance with building/tree height information using depth-aware attention.

## Quick Start

### 1. Prerequisite: generate the mask data

SS-CVNet requires pre-computed geometric masks (visibility mask, semantic mask, and mapping matrix). Generate them first with [generate/generate_masks.py](generate/generate_masks.py):

```bash
python generate/generate_masks.py \
    --config configs/SS-CVNet.yaml \
    --cities all cities
```

### 2. Train

```bash
python train.py --config configs/SS-CVNet.yaml

# or in the background
nohup python train.py --config configs/SS-CVNet.yaml > /dev/null 2>&1 &
```

Outputs (best model, per-epoch metrics) are written to `results/<perspective>/<exp_name>/`.

Useful overrides:

```bash
# Train on specific cities only
python train.py --config configs/SS-CVNet.yaml --cities Adelaide San_Francisco

# Leave-One-City-Out validation
python train.py --config configs/SS-CVNet.yaml --set data.val_cities "['Tokyo']"
```

### 3. Inference

Batch prediction over all cities:

```bash
python generate/Cities_predicate_multi_view.py --config configs/SS-CVNet.yaml --perspective multi_view
```

## Project Structure

```
├── train.py                                # Training entry point
├── configs/
│   └── SS-CVNet.yaml                       # Training configuration (incl. ablation options)
├── models/                                 # Model definitions
│   ├── ss_cvnet.py                         # SS-CVNet full network + factory
│   ├── VDWSA.py                            # Depth-aware window self-attention (VDWSA)
│   ├── modules.py                          # Shared modules: alignment, backbone encoder, heads
│   ├── model.py                            # Multi-architecture model factory
│   ├── loss.py                             # Loss: weighted MSE + physics constraint
│   ├── cross_view_geometric.py             # Cross-view geometric correspondence
│   ├── ray_casting.py                      # Vectorized cross-view ray casting
│   ├── visibility.py                       # Ground visibility analyzer (Bresenham)
│   ├── voxel_grid.py                       # 3D voxel grid builder (CUDA)
│   ├── projection_transformer.py           # Equirectangular → fisheye/perspective projections
│   └── polar_transform_model.py            # Polar transformation baseline
├── utils/                                  # Data loader, logger, seed utilities
├── generate/
│   ├── generate_masks.py                   # Mask generation (prerequisite for training)
│   └── Cities_predicate_multi_view.py      # Batch multi-view inference
└── tools/                                  # Preprocessing and data integrity tests
```

## Ablation Options

All ablations are controlled by `model.ablation` in [configs/SS-CVNet.yaml](configs/SS-CVNet.yaml):

| Module | Option | Config key |
|---|---|---|
| 1. Semantic embedding | disable embedding | `use_semantic_embedding` |
| 2. VDWSA | disable height maps | `use_height_map` |
| 2. VDWSA | center circular mask instead of visibility mask | `use_center_circle_mask` + `circle_mask_radius` |
| 2. VDWSA | SEBlock attention instead of visibility mask | `use_se_attention` |
| 3. Alignment | polar transformation instead of M_geo | `use_polar_transform` |
| 3. Alignment | learnable mapping matrix instead of M_geo | `use_learnable_matrix` |
| 3. Alignment | no alignment (dual-branch gated fusion) | `use_cross_view_alignment` |

Run an ablation via `--set` overrides and give the experiment a distinct name:

```bash
# Center circular mask (radius 50)
python train.py --config configs/SS-CVNet.yaml \
    --set model.ablation.use_center_circle_mask true \
    --set model.ablation.use_visibility_mask false \
    --set exp_name sscvnet_circle50

# Polar transformation alignment
python train.py --config configs/SS-CVNet.yaml \
    --set model.ablation.use_polar_transform true \
    --set exp_name sscvnet_polar
```

More examples and a recommended ablation workflow can be found in the comments of [configs/SS-CVNet.yaml](configs/SS-CVNet.yaml).

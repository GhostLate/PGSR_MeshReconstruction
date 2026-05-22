# Run Guide

End-to-end pipeline for high-fidelity 3D mesh reconstruction with PGSR (Planar-based Gaussian Splatting). The full motivation, method choice, and benchmark results are in [GS_Report.md](GS_Report.md).

---

## 1. Install submodules

Builds the two CUDA submodules pinned to PGSR's planar rasterization formulation.

```bash
pip install -r requirements.txt

export CC=$(which gcc-14)
export CXX=$(which g++-14)
export CUDAHOSTCXX=$(which g++-14)
pip install --no-build-isolation submodules/diff-plane-rasterization
pip install submodules/simple-knn
```

- **GCC 14 pinned** for `CC`, `CXX`, and `CUDAHOSTCXX` because the CUDA extensions fail to compile against newer host toolchains on this machine.
- **`--no-build-isolation`** for `diff-plane-rasterization` lets the build see the active PyTorch install (the extension links against `libtorch`).
- **`diff-plane-rasterization`** is PGSR's modified rasterizer that splats flat planar Gaussians and emits the unbiased depth used by the geometric losses.
- **`simple-knn`** is the small CUDA k-NN helper used during densification.

The conda environment itself (PyTorch + CUDA + Python deps) is created separately following the upstream PGSR README; this script only handles the CUDA submodules that have to be (re)built locally.

---

## 2. Prepare data

Turns raw photos into a COLMAP scene. The commands are listed in the order they were used; not every dataset needs every step. The `data/datasets/<variant>` naming convention matches the variants reported in [GS_Report.md §2](GS_Report.md) (`base_r1`, `base_r2`, `base_r2_crop`, etc.).

### 2.1 Downscale

```bash
python scripts/preprocess/downsize_images.py data/datasets/base_r1 data/datasets/base_r2 --ratio 2
```

Resizes every image under the source folder by `--ratio` (×2 in this case) using LANCZOS resampling and writes JPEGs at quality 100. Used to produce the `*_r2` variants — they train faster than full-resolution and, per the report, reach essentially the same PSNR and geometry.

### 2.2 Object masks

```bash
rembg p -om data/datasets/base_r1/input data/datasets/base_r1/masks -m bria-rmbg
```

Runs `rembg` in batch mode (`p`) with the BRIA RMBG-2.0 model to produce a foreground mask for every image. `-om` writes mask-only PNGs to the `masks/` folder. PGSR consumes these via the alpha channel after cropping (see 2.4) and uses them to zero out background depth during TSDF fusion.

The report also mentions SAM3 as an alternative mask source; either works as long as the output is a single-channel mask sharing the image's filename stem.

### 2.3 Filter images

```bash
python scripts/preprocess/image_filter.py \
      --input data/datasets/base_r2_crop/input \
      --masks data/datasets/base_r2_crop/masks \
      --masks-crop data/datasets/base_r2_crop/masks_crop \
      --output data/datasets/base_r2_f2_crop/input \
      --masks-out data/datasets/base_r2_f2_crop/masks \
      --masks-crop-out data/datasets/base_r2_f2_crop/masks_crop \
      --sharpness-percentile 10.0 \
      --brightness-std 2.0 \
      --score-resize -1 \
      --report data/datasets/base_r2_f2_crop/report.txt
```

Drops blurry frames and exposure outliers. Sharpness is the Laplacian variance and brightness is the mean intensity, both computed inside the supplied mask so a sharp subject on a blurry background still scores well.

- `--sharpness-percentile 10.0` drops the bottom 10% sharpest-rejecting frames (mild). The report's `*_f` variants used a more aggressive cutoff and lost too many frames for COLMAP to register all 87.
- `--brightness-std 2.0` rejects frames whose mean brightness is >2σ from the dataset mean.
- `--score-resize -1` disables the score-time resize (uses the original resolution for scoring).
- Masks are propagated to the output folder but never baked into the images.
- `--report` writes a CSV of per-frame scores for tuning the thresholds.

The naming convention `base_r2_f2_crop` = downscale ×2, filter level 2 (mild), cropped.

### 2.4 Crop + mask (RGBA)

```bash
python scripts/preprocess/crop_and_mask.py \
      --images data/datasets/base_r1/input \
      --masks  data/datasets/base_r1/masks \
      --masks_crop  data/datasets/base_r1/masks_crop \
      --out    data/datasets/base_r1_crop/input \
      --out_masks data/datasets/base_r1_crop/masks \
      --pad 0.025 --feather 5 --dilate 3 #--rgba
```

Computes the union bounding box of all masks, applies it as a global crop (so framing is consistent across views), then writes the cropped images plus their masks. Optionally bakes the (feathered, dilated) mask into the alpha channel when `--rgba` is enabled — PGSR's `scene/cameras.py` reads alpha as `view.mask`, which `render.py` uses to zero out background depth during TSDF fusion.

- `--pad 0.025` adds 2.5% padding around the bbox.
- `--feather 5` softens the mask edge by 5 px.
- `--dilate 3` grows the mask by 3 px before feathering so the silhouette isn't clipped.

Per the report, fixed-bbox cropping concentrates features on the object during SfM but causes a large train/test PSNR gap because held-out views fall partially outside the trained volume.

### 2.5 Convert to COLMAP

```bash
cp -r data/datasets/base_r2 data/colmap/base_r2
python scripts/preprocess/convert_extra.py --data_path data/colmap/base_r2 # --masks --rgba
```

Runs feature extraction → matching → mapping (via PGSR's `convert_extra.py`) on the prepared dataset to produce `sparse/0/{cameras,images,points3D}.bin`, the COLMAP outputs PGSR needs at training time. The `--masks` flag tells COLMAP to use the per-image masks during feature extraction (so background features are excluded); `--rgba` tells it the inputs are RGBA with the alpha channel as the mask.

### 2.6 Inspect the COLMAP result

```bash
colmap model_analyzer --path data/colmap/base_r2/sparse
colmap gui
# File → Import model → scene/sparse/0
```

`model_analyzer` prints the registered-image count, point count, and mean reprojection error — the headline numbers used in [GS_Report.md §2](GS_Report.md) to pick which variants moved on to training. The GUI is for visually checking the sparse point cloud and camera trajectory.

---

## 3. Train & Eval

### 3.1 Train

```bash
python train.py -s ./data/custom_crop_2a/s1 -m ./data/out_crop2a --opacity_cull_threshold 0.05 -r 1 # --max_abs_split_points 0 --white_background
```

Trains PGSR on the COLMAP scene at `-s` and writes the checkpoint + logs to `-m`.

- `-r 1` keeps images at full resolution (1 = no downscaling at load time).
- `--opacity_cull_threshold 0.05` prunes near-transparent Gaussians during densification.
- `--max_abs_split_points 0` (commented) disables the abs-grad split cap — the "ASP" variant from the report. It slightly improves recall at a small precision cost.
- `--white_background` (commented) is for datasets without alpha masks where the background is white.

Each run takes ≈ 3 hours on the reference hardware and trains for 30 000 iterations by default.

### 3.2 Render and fuse a mesh

```bash
python render.py -m ./data/output/base_r2 --max_depth 10.0 --voxel_size 0.002 # --use_depth_filter
```

Renders depth maps from the trained model and runs TSDF fusion to produce a mesh under `-m/mesh/`.

- `--max_depth 10.0` is the TSDF depth-truncation cutoff. The report's `d10_f` configuration combines this with `--use_depth_filter` and produces the best mean Chamfer distance.
- `--voxel_size 0.002` is the TSDF voxel size. Smaller = finer mesh, but RAM-bound — the report notes 0.02 m was the practical floor at 32 GB.
- `--use_depth_filter` (commented) drops grazing-angle depth pixels before integration. Use it together with a larger `--max_depth`.

### 3.3 Visualize

```bash
python scripts/visualize_mesh.py data/other/fusion.ply --wireframe --show-axes --show-bbox
```

Opens the mesh in an Open3D viewer. `--wireframe` makes hole patterns easy to spot; `--show-axes` and `--show-bbox` overlay the world frame and the mesh bounding box for sanity checks during alignment.

### 3.4 Evaluate against ground truth

The predicted mesh must first be rigidly aligned to the GT mesh in **CloudCompare** (the report's protocol — see [GS_Report.md §4.2](GS_Report.md)).

```bash
python scripts/compare_meshes.py \
      --pred data/output/out1/mesh/tsdf_fusion_d7_post_cut.ply \
      --gt   data/other/fusion.ply \
      --out-dir data/eval/results
```

Samples both meshes to point clouds and reports Chamfer (pred→gt, gt→pred, mean), Precision/Recall/F-score at an auto-estimated threshold τ, and Hausdorff distance (max and p95). Writes two error-colored point clouds under `--out-dir`:

```bash
python scripts/visualize_mesh.py data/eval/results/gt_error.ply
python scripts/visualize_mesh.py data/eval/results/pred_error.ply
```

`gt_error.ply` colors each GT point by its distance to the predicted surface (highlights what the prediction is missing — recall failures). `pred_error.ply` colors each predicted point by its distance to GT (highlights spurious geometry — precision failures). Per the report, the dominant error mode is the former: precision is ~0.99 but recall ~0.84, so the GT-side error cloud is where the holes show up.


# Downscale
python scripts/preprocess/downsize_images.py data/datasets/base_r1 data/datasets/base_r2 --ratio 2

# Get masks
rembg p -om data/datasets/base_r1/input data/datasets/base_r1/masks -m bria-rmbg

# Filter Images
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

# Crop | Masking
python scripts/preprocess/crop_and_mask.py \
      --images data/datasets/base_r1/input \
      --masks  data/datasets/base_r1/masks \
      --masks_crop  data/datasets/base_r1/masks_crop \
      --out    data/datasets/base_r1_crop/input \
      --out_masks data/datasets/base_r1_crop/masks \
      --pad 0.025 --feather 5 --dilate 3 #--rgba

# Convert to COLMAP
cp -r data/datasets/base_r2 data/colmap/base_r2
python scripts/preprocess/convert_extra.py --data_path data/colmap/base_r2 # --masks --rgba

# Analyze COLMAP
colmap model_analyzer --path data/colmap/base_r2/sparse
colmap gui
# File → Import model → scene/sparse/0

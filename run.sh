# Train
python train.py -s ./data/custom_crop_2a/s1 -m ./data/out_crop2a  --opacity_cull_threshold 0.05 -r 1 # --max_abs_split_points 0 --white_background

# Render
python render.py -m ./data/output/base_r2 --max_depth 10.0 --voxel_size 0.002 # --use_depth_filter

# Visualize
python scripts/visualize_mesh.py data/other/fusion.ply --wireframe --show-axes --show-bbox

# Eval meshes (after aligning)
python scripts/compare_meshes.py \
      --pred data/output/out1/mesh/tsdf_fusion_d7_post_cut.ply \
      --gt   data/other/fusion.ply \
      --out-dir data/eval/results

python scripts/visualize_mesh.py data/eval/results/gt_error.ply
python scripts/visualize_mesh.py data/eval/results/pred_error.ply

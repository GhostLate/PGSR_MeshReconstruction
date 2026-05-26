# python train.py -s ./data/colmap/base_r1_crop -m ./data/output/base_r1_crop --opacity_cull_threshold 0.05 -r 2 --iterations 7000 --eval --start_checkpoint ./data/output/base_r1_crop/app_model/iteration_7000/app.pth
# [ITER 7000] Evaluating test: L1 0.04660647794265639 PSNR 24.216180888089266 [20/05 20:16:20]
# [ITER 7000] Evaluating train: L1 0.013001064583659173 PSNR 30.942652130126955 [20/05 20:16:27]

# python train.py -s ./data/colmap/base_r1_f2 -m ./data/output/base_r1_f2 --opacity_cull_threshold 0.05 -r 2 --iterations 7000 --eval # --start_checkpoint ./data/output/base_r1_f2/app_model/iteration_7000/app.pth
# [ITER 7000] Evaluating test: L1 0.07058006972074508 PSNR 19.780845642089844 [21/05 01:44:57]
# [ITER 7000] Evaluating train: L1 0.016095251962542533 PSNR 26.458304977416994 [21/05 01:45:08]

# python train.py -s ./data/colmap/base_r1_f2_crop -m ./data/output/base_r1_f2_crop --opacity_cull_threshold 0.05 -r 2 --iterations 7000 --eval --start_checkpoint ./data/output/base_r1_f2_crop/app_model/iteration_7000/app.pth
# [ITER 7000] Evaluating test: L1 0.05347097013145685 PSNR 22.583127784729005 [21/05 02:40:07]
# [ITER 7000] Evaluating train: L1 0.012384273670613767 PSNR 30.854606246948244 [21/05 02:40:13]


# python train.py -s ./data/colmap/base_r1_crop -m ./data/output/base_r1_crop_asp --opacity_cull_threshold 0.05 -r 2 --iterations 7000 --max_abs_split_points 0 --eval --start_checkpoint ./data/output/base_r1_crop_asp/app_model/iteration_7000/app.pth
# [ITER 7000] Evaluating test: L1 0.039274625734172085 PSNR 24.729190653020687 [20/05 21:11:08]
# [ITER 7000] Evaluating train: L1 0.013308095186948777 PSNR 30.705913162231447 [20/05 21:11:15]

# python train.py -s ./data/colmap/base_r1_f2 -m ./data/output/base_r1_f2_asp --opacity_cull_threshold 0.05 -r 2 --iterations 7000 --max_abs_split_points 0 --eval --start_checkpoint ./data/output/base_r1_f2_asp/app_model/iteration_7000/app.pth
# [ITER 7000] Evaluating test: L1 0.06865525618195534 PSNR 19.899058437347414 [21/05 02:06:43]
# [ITER 7000] Evaluating train: L1 0.017129814065992833 PSNR 25.998581314086916 [21/05 02:06:54]

# python train.py -s ./data/colmap/base_r1_f2_crop -m ./data/output/base_r1_f2_crop_asp --opacity_cull_threshold 0.05 -r 2 --iterations 7000 --max_abs_split_points 0 --eval --start_checkpoint ./data/output/base_r1_f2_crop_asp/app_model/iteration_7000/app.pth
# [ITER 7000] Evaluating test: L1 0.06402467153966428 PSNR 21.15404739379883 [21/05 02:52:51]
# [ITER 7000] Evaluating train: L1 0.011502085812389852 PSNR 31.018287658691406 [21/05 02:52:57]


# python train.py -s ./data/colmap/base_r2 -m ./data/output/base_r2_asp --opacity_cull_threshold 0.05 -r 1 --iterations 7000 --max_abs_split_points 0 --eval --start_checkpoint ./data/output/base_r2_asp/app_model/iteration_7000/app.pth
# [ITER 7000] Evaluating test: L1 0.01910792316564105 PSNR 24.519408312710848 [20/05 21:21:43]
# [ITER 7000] Evaluating train: L1 0.015345619246363641 PSNR 24.723571395874025 [20/05 21:21:54]

# python train.py -s ./data/colmap/base_r2_crop -m ./data/output/base_r2_crop_asp --opacity_cull_threshold 0.05 -r 1 --iterations 7000 --max_abs_split_points 0 --eval --start_checkpoint ./data/output/base_r2_crop_asp/app_model/iteration_7000/app.pth
# [ITER 7000] Evaluating test: L1 0.044059348885308616 PSNR 25.512306560169566 [20/05 21:28:11]
# [ITER 7000] Evaluating train: L1 0.011000568978488445 PSNR 31.924663543701172 [20/05 21:28:18]

# python train.py -s ./data/colmap/base_r2_f2 -m ./data/output/base_r2_f2_asp --opacity_cull_threshold 0.05 -r 1 --iterations 7000 --max_abs_split_points 0 --eval --start_checkpoint ./data/output/base_r2_f2_asp/app_model/iteration_7000/app.pth
# [ITER 7000] Evaluating test: L1 0.018945168238133192 PSNR 25.148893356323242 [21/05 02:17:25]
# [ITER 7000] Evaluating train: L1 0.013236897438764572 PSNR 26.995811843872072 [21/05 02:17:35]

# python train.py -s ./data/colmap/base_r2_f2_crop -m ./data/output/base_r2_f2_crop_asp --opacity_cull_threshold 0.05 -r 1 --iterations 7000 --max_abs_split_points 0 --eval --start_checkpoint ./data/output/base_r2_f2_crop_asp/app_model/iteration_7000/app.pth
# [ITER 7000] Evaluating test: L1 0.06454939376562834 PSNR 21.487690734863282 [21/05 02:59:33]
# [ITER 7000] Evaluating train: L1 0.01333622317761183 PSNR 30.719336318969727 [21/05 02:59:39]


# python scripts/compare_meshes.py \
#      --pred data/eval/base_r1/tsdf_fusion_v-0.002_d-6.0_post_cut1.ply \
#      --gt   data/other/fusion_cut.ply \
#      --out-dir data/eval/base_r1/tsdf_fusion_v-0.002_d-6.0_post_cut1
#
#python scripts/compare_meshes.py \
#      --pred data/eval/base_r1/tsdf_fusion_v-0.002_d-10.0_f_post_cut1.ply \
#      --gt   data/other/fusion_cut.ply \
#      --out-dir data/eval/base_r1/tsdf_fusion_v-0.002_d-10.0_f_post_cut1
#
#
#python scripts/compare_meshes.py \
#      --pred data/eval/base_r1_asp/tsdf_fusion_v-0.002_d-6.0_post_cut1.ply \
#      --gt   data/other/fusion_cut.ply \
#      --out-dir data/eval/base_r1_asp/tsdf_fusion_v-0.002_d-6.0_post_cut1
#
#python scripts/compare_meshes.py \
#      --pred data/eval/base_r1_asp/tsdf_fusion_v-0.002_d-10.0_f_post_cut1.ply \
#      --gt   data/other/fusion_cut.ply \
#      --out-dir data/eval/base_r1_asp/tsdf_fusion_v-0.002_d-10.0_f_post_cut1
#
#
#python scripts/compare_meshes.py \
#      --pred data/eval/base_r1_asp/tsdf_fusion_v-0.002_d-6.0_post_cut2.ply \
#      --gt   data/other/fusion_cut.ply \
#      --out-dir data/eval/base_r1_asp/tsdf_fusion_v-0.002_d-6.0_post_cut2
#
#python scripts/compare_meshes.py \
#      --pred data/eval/base_r1_asp/tsdf_fusion_v-0.002_d-10.0_f_post_cut2.ply \
#      --gt   data/other/fusion_cut.ply \
#      --out-dir data/eval/base_r1_asp/tsdf_fusion_v-0.002_d-10.0_f_post_cut2
#
#
#python scripts/compare_meshes.py \
#      --pred data/eval/base_r2/tsdf_fusion_v-0.002_d-6.0_post_cut1.ply \
#      --gt   data/other/fusion_cut.ply \
#      --out-dir data/eval/base_r2/tsdf_fusion_v-0.002_d-6.0_post_cut1
#
#python scripts/compare_meshes.py \
#      --pred data/eval/base_r2/tsdf_fusion_v-0.002_d-10.0_f_post_cut1.ply \
#      --gt   data/other/fusion_cut.ply \
#      --out-dir data/eval/base_r2/tsdf_fusion_v-0.002_d-10.0_f_post_cut1
#
#
#python scripts/compare_meshes.py \
#      --pred data/eval/base_r2_f2/tsdf_fusion_v-0.002_d-6.0_post_cut1.ply \
#      --gt   data/other/fusion_cut.ply \
#      --out-dir data/eval/base_r2_f2/tsdf_fusion_v-0.002_d-6.0_post_cut1
#
#python scripts/compare_meshes.py \
#      --pred data/eval/base_r2_f2/tsdf_fusion_v-0.002_d-10.0_f_post_cut1.ply \
#      --gt   data/other/fusion_cut.ply \
#      --out-dir data/eval/base_r2_f2/tsdf_fusion_v-0.002_d-10.0_f_post_cut1
#
#
#python scripts/compare_meshes.py \
#      --pred data/eval/base_r2_crop/tsdf_fusion_v-0.002_d-10.0_f_post_cut1.ply \
#      --gt   data/other/fusion_cut.ply \
#      --out-dir data/eval/base_r2_crop/tsdf_fusion_v-0.002_d-10.0_f_post_cut1

python scripts/preprocess/convert_extra.py --data_path data/datasets/arctic

python train.py -s ./data/colmap/arctic1 -m ./data/output/arctic --opacity_cull_threshold 0.05 --max_abs_split_points 0 -r 2 --multi_view_max_dis 8.0 --multi_view_max_angle 90 --use_virtul_cam --init_ply dense/fused_photo.ply
python render.py -m ./data/output/arctic --max_depth 10.0 --voxel_size 0.002 # --use_depth_filter
python scripts/visualize_mesh.py data/output/arctic/mesh/tsdf_fusion_v-0.002_d-10.0_post.ply

python train.py -s ./data/colmap/arctic1 -m ./data/output/arctic1 -r 2 \
    --init_ply sparse/points3D.ply \
    --iterations 10000 \
    --multi_view_max_dis 8.0 --multi_view_max_angle 90 \
    --single_view_weight_from_iter 500 --multi_view_weight_from_iter 500 \
    --single_view_weight 0.04 \
    --multi_view_pixel_noise_th 3.0 \
    --opacity_reset_interval 100000 \
    --densify_until_iter 4000 --densify_grad_threshold 0.0005 \
    --opacity_cull_threshold 0.05 --max_abs_split_points 0

colmap model_converter \
    --input_path $SCENE/sparse \
    --output_path $SCENE/sparse/points.ply \
    --output_type PLY

python scripts/visualize_mesh.py data/colmap/arctic/sparse/points3D.ply
python scripts/visualize_mesh.py data/colmap/base_r1/sparse/points3D.ply
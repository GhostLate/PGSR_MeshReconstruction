
import os
import logging
from argparse import ArgumentParser
import shutil

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def prepare_colmap_masks(input_dir, masks_dir, out_dir):
    """COLMAP expects masks at <out_dir>/<image_filename>.png — e.g. for
    input/foo.jpg it reads out_dir/foo.jpg.png. Returns the count copied,
    or None if no masks dir was supplied."""
    if masks_dir is None or not os.path.isdir(masks_dir):
        return None
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        try:
            os.remove(os.path.join(out_dir, f))
        except OSError:
            pass
    n = 0
    for fname in sorted(os.listdir(input_dir)):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in IMG_EXTS:
            continue
        src = os.path.join(masks_dir, stem + ".png")
        if not os.path.exists(src):
            print(f"[warn] no mask for {fname} in {masks_dir}")
            continue
        shutil.copy2(src, os.path.join(out_dir, fname + ".png"))
        n += 1
    return n


def _extract_alpha(input_dir, masks_dir, fname):
    """Single-channel alpha for `fname`: prefer the alpha channel of the
    RGBA input image (preserves feathering), fall back to the binary mask."""
    stem, _ = os.path.splitext(fname)
    img = cv2.imread(os.path.join(input_dir, fname), cv2.IMREAD_UNCHANGED)
    if img is not None and img.ndim == 3 and img.shape[2] == 4:
        return img[..., 3]
    src = os.path.join(masks_dir, stem + ".png")
    if os.path.exists(src):
        return cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    return None


def undistort_masks(colmap_command, scene_path):
    """Undistort per-image alpha masks against the SAME distorted sparse
    model that drives the real image undistortion. Output lands at
    <scene_path>/_undist_masks/ for later alpha merging. MUST be called
    BEFORE image_undistorter on the images (which overwrites the sparse
    model with the PINHOLE version)."""
    input_dir = os.path.join(scene_path, "input")
    masks_dir = os.path.join(scene_path, "masks")
    sparse_dir = os.path.join(scene_path, "sparse")
    tmp_in = os.path.join(scene_path, "_mask_in")
    tmp_out = os.path.join(scene_path, "_mask_out")
    if os.path.isdir(tmp_in):
        shutil.rmtree(tmp_in)
    if os.path.isdir(tmp_out):
        shutil.rmtree(tmp_out)
    os.makedirs(tmp_in, exist_ok=True)
    os.makedirs(tmp_out, exist_ok=True)

    n = 0
    for fname in sorted(os.listdir(input_dir)):
        _, ext = os.path.splitext(fname)
        if ext.lower() not in IMG_EXTS:
            continue
        alpha = _extract_alpha(input_dir, masks_dir, fname)
        if alpha is None:
            continue
        # Stage as 3-channel under the image's original filename so the
        # sparse model's image records resolve. image_undistorter cares
        # about pixel dimensions, not channel semantics.
        alpha_bgr = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(os.path.join(tmp_in, fname), alpha_bgr)
        n += 1
    if n == 0:
        shutil.rmtree(tmp_in, ignore_errors=True)
        shutil.rmtree(tmp_out, ignore_errors=True)
        return False

    cmd = (colmap_command + " image_undistorter "
           "--image_path " + tmp_in + " "
           "--input_path " + sparse_dir + " "
           "--output_path " + tmp_out + " "
           "--output_type COLMAP")
    rc = os.system(cmd)
    shutil.rmtree(tmp_in, ignore_errors=True)
    if rc != 0:
        print(f"[warn] mask undistortion failed (rc={rc}); alpha will not be re-merged")
        shutil.rmtree(tmp_out, ignore_errors=True)
        return False

    final = os.path.join(scene_path, "_undist_masks")
    if os.path.isdir(final):
        shutil.rmtree(final)
    shutil.move(os.path.join(tmp_out, "images"), final)
    shutil.rmtree(tmp_out, ignore_errors=True)
    print(f"[info] undistorted {n} masks at {final}")
    return True


def merge_alpha_from_undistorted_masks(scene_path):
    """Merge <scene>/_undist_masks/* into the alpha channel of
    <scene>/images/*, writing RGBA PNGs. Cleans up the temp dir."""
    udm = os.path.join(scene_path, "_undist_masks")
    images_dir = os.path.join(scene_path, "images")
    if not os.path.isdir(udm) or not os.path.isdir(images_dir):
        return 0
    merged = 0
    for fname in sorted(os.listdir(images_dir)):
        stem, _ = os.path.splitext(fname)
        mpath = os.path.join(udm, fname)
        if not os.path.exists(mpath):
            continue
        img_path = os.path.join(images_dir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        if img is None or m is None:
            continue
        if m.shape != img.shape[:2]:
            m = cv2.resize(m, (img.shape[1], img.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
        rgba = np.dstack([img, m])
        out_path = os.path.join(images_dir, stem + ".png")
        cv2.imwrite(out_path, rgba)
        if out_path != img_path:
            os.remove(img_path)
        merged += 1
    shutil.rmtree(udm, ignore_errors=True)
    print(f"[info] merged alpha into {merged} images at {images_dir}")
    return merged


# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
def init_colmap(args):
    colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
    magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
    use_gpu = 1 if not args.no_gpu else 0

    scene_path = args.data_path
    os.makedirs(scene_path + "/sparse", exist_ok=True)

    os.system(f"rm -rf {scene_path}/images/*")
    os.system(f"rm -rf {scene_path}/sparse/*")
    os.system(f"rm -f {scene_path}/database.db")

    # Build COLMAP-formatted mask dir (<filename>.png) so feature_extractor
    # ignores background features during SfM.
    colmap_masks_dir = os.path.join(scene_path, "colmap_masks")
    if args.masks:
        n_masks = prepare_colmap_masks(
            os.path.join(scene_path, "input"),
            os.path.join(scene_path, "masks"),
            colmap_masks_dir,
        )
        if n_masks:
            print(f"[info] prepared {n_masks} masks for COLMAP at {colmap_masks_dir}")
        else:
            print(f"[info] no masks found at {scene_path}/masks; SfM will use full images")
            colmap_masks_dir = None

    ## Feature extraction
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + scene_path + "/database.db \
        --image_path " + scene_path + "/input \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + args.camera + " \
        --FeatureExtraction.use_gpu " + str(use_gpu)
    if colmap_masks_dir is not None and args.masks:
        feat_extracton_cmd += " --ImageReader.mask_path " + colmap_masks_dir
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching
    feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + scene_path + "/database.db \
        --FeatureMatching.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + scene_path + "/database.db \
        --image_path "  + scene_path + "/input \
        --output_path "  + scene_path + "/sparse \
        --Mapper.ba_global_function_tolerance=0.000001")
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    files = os.listdir(scene_path + "/sparse/0")
    # Copy each file from the source directory to the destination directory
    for file in files:
        destination_file = os.path.join(scene_path, "sparse", file)
        source_file = os.path.join(scene_path, "sparse", "0", file)
        shutil.move(source_file, destination_file)

    ### Mask undistortion — runs BEFORE image_undistorter so it uses the
    ### still-distorted sparse model. Output staged at <scene>/_undist_masks.
    if colmap_masks_dir is not None and args.masks:
        undistort_masks(colmap_command, scene_path)

    ### Image undistortion
    ## We need to undistort our images into ideal pinhole intrinsics.
    img_undist_cmd = (colmap_command + " image_undistorter \
        --image_path " + scene_path + "/input \
        --input_path " + scene_path + "/sparse \
        --output_path " + scene_path + "\
        --output_type COLMAP")
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Re-merge undistorted masks as alpha into <scene>/images/*.png
    ### so PGSR's loader (scene/cameras.py:34-36) picks them up.
    if colmap_masks_dir is not None and args.rgba:
        merge_alpha_from_undistorted_masks(scene_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = ArgumentParser("Colmap converter")
    parser.add_argument('--data_path', type=str, default=None, help='Path to dataset')
    parser.add_argument("--no_gpu", action='store_true')
    parser.add_argument("--rgba", action='store_true')
    parser.add_argument("--masks", action='store_true')
    parser.add_argument("--camera", default="OPENCV", type=str)
    parser.add_argument("--colmap_executable", default="", type=str)
    parser.add_argument("--magick_executable", default="", type=str)
    args = parser.parse_args()

    init_colmap(args)

    print("Done.")

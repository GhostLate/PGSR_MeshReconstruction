"""
Crop images to a global object bounding box (union of all masks) and write
RGBA PNGs where the alpha channel carries the (feathered) object mask. PGSR's
scene/cameras.py reads the alpha channel as view.mask, which render.py uses
to zero out background depth during TSDF fusion.

Usage:
    python scripts/preprocess/crop_and_mask.py \
        --images data/custom/input \
        --masks  data/custom/masks \
        --out    data/custom/input_processed \
        --pad 0.10 --feather 5 --dilate 3
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(path)
    if m.ndim == 3:
        m = m[..., -1] if m.shape[2] == 4 else cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    return (m > 127).astype(np.uint8)


def pair_files(images_dir: Path, masks_dir: Path):
    images = {p.stem: p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
    masks = {p.stem: p for p in masks_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
    common = sorted(set(images) & set(masks))
    missing_masks = sorted(set(images) - set(masks))
    if missing_masks:
        print(f"[warn] {len(missing_masks)} images have no mask, skipping (e.g. {missing_masks[:3]})")
    if not common:
        raise SystemExit("No matching image/mask pairs found (matched by filename stem).")
    return [(images[k], masks[k]) for k in common]


def global_bbox(pairs, pad_frac: float):
    boxes = []
    H, W = None, None
    for _, mask_path in tqdm(pairs, desc="bbox"):
        m = load_mask(mask_path)
        if H is None:
            H, W = m.shape
        ys, xs = np.where(m)
        if xs.size == 0:
            continue
        boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
    if not boxes:
        raise SystemExit("All masks are empty.")
    boxes = np.array(boxes)
    x0, y0 = boxes[:, 0].min(), boxes[:, 1].min()
    x1, y1 = boxes[:, 2].max(), boxes[:, 3].max()
    pad_x = int(pad_frac * (x1 - x0))
    pad_y = int(pad_frac * (y1 - y0))
    x0 = max(0, int(x0) - pad_x)
    y0 = max(0, int(y0) - pad_y)
    x1 = min(W, int(x1) + pad_x + 1)
    y1 = min(H, int(y1) + pad_y + 1)
    return x0, y0, x1, y1, (H, W)


def feather(mask: np.ndarray, dilate_px: int, feather_px: int) -> np.ndarray:
    m = mask.astype(np.uint8) * 255
    if dilate_px > 0:
        k = np.ones((dilate_px, dilate_px), np.uint8)
        m = cv2.dilate(m, k)
    if feather_px > 0:
        k = feather_px | 1  # ensure odd
        m = cv2.GaussianBlur(m, (k, k), 0)
    return m.astype(np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--masks", required=True, type=Path)
    ap.add_argument("--masks_crop", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path,
                    help="Output dir for cropped+composited RGB images.")
    ap.add_argument("--out_masks", type=Path, default=None,
                    help="Optional: also save cropped masks here (for COLMAP --ImageReader.mask_path).")
    ap.add_argument("--pad", type=float, default=0.10, help="Bbox padding fraction.")
    ap.add_argument("--feather", type=int, default=5, help="Gaussian blur kernel (px) for soft alpha edges. 0 disables.")
    ap.add_argument("--dilate", type=int, default=3, help="Mask dilation kernel (px) before feather. 0 disables.")
    ap.add_argument("--rgba", action="store_true", help="Fuse RGB and Alpha channel.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.out_masks:
        args.out_masks.mkdir(parents=True, exist_ok=True)

    pairs = pair_files(args.images, args.masks_crop)
    x0, y0, x1, y1, (H, W) = global_bbox(pairs, args.pad)
    print(f"Global bbox: x=[{x0},{x1}) y=[{y0},{y1}) from {W}x{H} -> {x1-x0}x{y1-y0}")

    pairs = pair_files(args.images, args.masks)
    for img_path, mask_path in tqdm(pairs, desc="crop+rgba"):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)  # BGR
        if img is None:
            print(f"[skip] could not read {img_path}")
            continue
        if img.shape[:2] != (H, W):
            print(f"[skip] {img_path.name} has size {img.shape[1]}x{img.shape[0]}, expected {W}x{H}")
            continue

        mask = load_mask(mask_path)
        mask_soft = feather(mask, args.dilate, args.feather)  # HxW float [0,1]

        bgr_c = img[y0:y1, x0:x1]

        if args.rgba:
            alpha_c = (mask_soft[y0:y1, x0:x1] * 255).clip(0, 255).astype(np.uint8)
            bgr_c = np.dstack([bgr_c, alpha_c])  # BGRA for cv2.imwrite

        out_path = args.out / (img_path.stem + ".png")
        cv2.imwrite(str(out_path), bgr_c)

        if args.out_masks:
            m_bin = (mask[y0:y1, x0:x1] * 255).astype(np.uint8)
            cv2.imwrite(str(args.out_masks / (out_path.stem + ".png")), m_bin)

    print(f"Wrote {len(pairs)} images to {args.out}")
    if args.out_masks:
        print(f"Wrote cropped masks to {args.out_masks}")


if __name__ == "__main__":
    main()
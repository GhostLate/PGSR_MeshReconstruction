#!/usr/bin/env python3
"""
filter_frames.py — Prepare images for COLMAP / 3DGS by removing
blurry frames, exposure outliers, and near-duplicates.
 
When masks are supplied (COLMAP convention: <image_filename>.png), sharpness
and brightness are computed only inside the masked region — so a sharp
subject on a blurry background still scores well, and vice versa. Masks
are NEVER baked into the output images; they're only used to inform the
filtering decision (and optionally copied to --masks-out for COLMAP).
 
Usage examples:
    # Filter a folder of images, keep ~200 best frames
    python filter_frames.py -i raw/ -o filtered/ --target 200
 
    # Extract frames from video at 2 fps, then filter
    python filter_frames.py -i clip.mp4 -o filtered/ --fps 2 --target 200
 
    # Mask-aware scoring + sync masks to output folder
    python filter_frames.py -i raw/ -o filtered/ \\
        --masks masks/ --masks-out filtered_masks/ --target 200
 
    # Sync masks to output but score on the full image
    python filter_frames.py -i raw/ -o filtered/ \\
        --masks masks/ --masks-out filtered_masks/ --no-mask-scoring
 
    # Dry run with a per-frame CSV report — useful for tuning thresholds
    python filter_frames.py -i raw/ -o filtered/ --report scores.csv --dry-run -v
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path
 
import cv2
import numpy as np
 
 
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
 
 
def extract_frames_from_video(video_path: Path, out_dir: Path, fps: float) -> Path:
    """Extract frames from a video at approximately the given fps."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"Could not open video: {video_path}")
 
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(video_fps / fps))
 
    print(f"Video: {video_fps:.1f} fps, {total} frames; "
          f"keeping every {step}th frame (~{video_fps / step:.1f} fps)")
 
    saved = 0
    for i in range(total):
        if not cap.grab():
            break
        if i % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                cv2.imwrite(str(out_dir / f"frame_{saved:05d}.jpg"), frame)
                saved += 1
    cap.release()
    print(f"Extracted {saved} frames to {out_dir}")
    return out_dir
 
 
def score_image(path: Path, mask_path: Path | None = None,
                resize_long: int = 1024, min_mask_pixels: int = 500):
    """Return (sharpness, brightness, used_mask) for an image.
 
    If a mask is provided and has enough foreground pixels, sharpness and
    brightness are computed only over the masked-in region (with a small
    erosion to exclude Laplacian artifacts along the mask boundary).
    Otherwise the full image is used.
 
    Downscaling first makes scoring much faster on high-res inputs without
    materially changing the relative ranking.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
 
    mask = None
    if mask_path is not None and mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None and mask.shape != img.shape:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
 
    # Match the same downscale to both image and mask.
    h, w = img.shape
    long_edge = max(h, w)
    if long_edge > resize_long > 0:
        scale = resize_long / long_edge
        new_size = (int(w * scale), int(h * scale))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        if mask is not None:
            mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)
 
    lap = cv2.Laplacian(img, cv2.CV_64F)
 
    used_mask = False
    if mask is not None:
        # Erode so the Laplacian's response along the mask boundary
        # (a hard edge it would otherwise see as huge variance) is excluded.
        binary = (mask > 0).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        eroded = cv2.erode(binary, kernel, iterations=1)
        if int(eroded.sum()) >= min_mask_pixels:
            sel = eroded > 0
            sharpness = float(lap[sel].var())
            brightness = float(img[sel].mean())
            used_mask = True
        else:
            # Mask too small / empty after erosion — fall back to full image.
            sharpness = float(lap.var())
            brightness = float(img.mean())
    else:
        sharpness = float(lap.var())
        brightness = float(img.mean())
 
    return sharpness, brightness, used_mask
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Filter images for COLMAP / 3DGS reconstruction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", "-i", required=True, type=Path,
                        help="Input directory of images OR a video file")
    parser.add_argument("--output", "-o", required=True, type=Path,
                        help="Output directory for filtered images")
    parser.add_argument("--masks", type=Path, default=None,
                        help="Optional dir of masks (COLMAP naming: <image_filename>.png)")
    parser.add_argument("--masks-crop", type=Path, default=None,
                        help="Optional dir of masks (COLMAP naming: <image_filename>.png)")
    parser.add_argument("--masks-out", type=Path, default=None,
                        help="Output dir for synced masks (default: <output>_masks)")
    parser.add_argument("--masks-crop-out", type=Path, default=None,
                        help="Output dir for synced masks (default: <output>_masks)")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="If input is a video, extract at this fps (default: 2)")
    parser.add_argument("--sharpness-percentile", type=float, default=25.0,
                        help="Drop frames below this sharpness percentile; 0 disables (default: 25)")
    parser.add_argument("--brightness-std", type=float, default=2.0,
                        help="Drop frames > N std devs from median brightness; 0 disables (default: 2)")
    parser.add_argument("--target", type=int, default=0,
                        help="Subsample uniformly to this many frames; 0 = no subsample")
    parser.add_argument("--score-resize", type=int, default=1024,
                        help="Downscale long edge to this size before scoring (default: 1024)")
    parser.add_argument("--report", type=Path, default=None,
                        help="Optional CSV with per-frame scores and decisions")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print decisions without copying files")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print per-frame info")
    parser.add_argument("--no-mask-scoring", action="store_true",
                        help="Don't use masks when computing sharpness/brightness "
                             "(masks still get synced to --masks-out if provided)")
    args = parser.parse_args()
 
    # ---- Resolve input ----------------------------------------------------
    if args.input.is_file() and args.input.suffix.lower() in VIDEO_EXTS:
        tmp_dir = args.output.parent / f"_frames_{args.input.stem}"
        image_dir = extract_frames_from_video(args.input, tmp_dir, args.fps)
    elif args.input.is_dir():
        image_dir = args.input
    else:
        sys.exit(f"Input must be a video file or directory: {args.input}")
 
    images = sorted(p for p in image_dir.iterdir()
                    if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        sys.exit(f"No images found in {image_dir}")
    print(f"Scanning {len(images)} images...")
    if args.masks and not args.no_mask_scoring:
        print(f"Using masks from {args.masks} to restrict scoring region")
 
    # ---- Score every image -----------------------------------------------
    data = []
    mask_used_count = 0
    mask_missing_count = 0
    for p in images:
        mask_path = None
        if args.masks and not args.no_mask_scoring:
            candidate = args.masks / (p.stem + ".png")
            if candidate.exists():
                mask_path = candidate
            else:
                mask_missing_count += 1
        r = score_image(p, mask_path=mask_path, resize_long=args.score_resize)
        if r is None:
            print(f"  skip unreadable: {p.name}")
            continue
        sharp, bright, used_mask = r
        if used_mask:
            mask_used_count += 1
        data.append({"path": p, "sharp": sharp, "bright": bright,
                     "keep": True, "reason": ""})
 
    if not data:
        sys.exit("No readable images.")
 
    if args.masks and not args.no_mask_scoring:
        print(f"Mask-restricted scoring: {mask_used_count} / {len(data)} images"
              + (f"  ({mask_missing_count} missing masks)" if mask_missing_count else "")
              + (f"  ({len(data) - mask_used_count - mask_missing_count} masks too small, "
                 f"fell back to full image)"
                 if (len(data) - mask_used_count - mask_missing_count) > 0 else ""))
 
    sharps = np.array([d["sharp"] for d in data])
    brights = np.array([d["bright"] for d in data])
    print(f"Sharpness: min={sharps.min():.1f}  median={np.median(sharps):.1f}  "
          f"max={sharps.max():.1f}")
    print(f"Brightness: min={brights.min():.1f}  median={np.median(brights):.1f}  "
          f"max={brights.max():.1f}")
 
    # ---- Stage 1: sharpness ----------------------------------------------
    if args.sharpness_percentile > 0:
        thresh = float(np.percentile(sharps, args.sharpness_percentile))
        for d in data:
            if d["sharp"] < thresh:
                d["keep"] = False
                d["reason"] = f"blur(s={d['sharp']:.1f}<{thresh:.1f})"
        print(f"After sharpness >= {thresh:.1f} "
              f"(p{args.sharpness_percentile:g}): "
              f"{sum(d['keep'] for d in data)} / {len(data)}")
 
    # ---- Stage 2: brightness outliers ------------------------------------
    if args.brightness_std > 0:
        med = float(np.median(brights))
        std = float(np.std(brights))
        lo, hi = med - args.brightness_std * std, med + args.brightness_std * std
        for d in data:
            if d["keep"] and not (lo <= d["bright"] <= hi):
                d["keep"] = False
                d["reason"] = f"exposure(b={d['bright']:.1f} outside [{lo:.1f},{hi:.1f}])"
        print(f"After brightness within {args.brightness_std}σ of median: "
              f"{sum(d['keep'] for d in data)} / {len(data)}")
 
    # ---- Stage 3: uniform subsample --------------------------------------
    if args.target > 0:
        survivors = [d for d in data if d["keep"]]
        if len(survivors) > args.target:
            idx = set(np.linspace(0, len(survivors) - 1,
                                  args.target, dtype=int).tolist())
            for i, d in enumerate(survivors):
                if i not in idx:
                    d["keep"] = False
                    d["reason"] = "subsample"
            print(f"After subsample to {args.target}: "
                  f"{sum(d['keep'] for d in data)} / {len(data)}")
        else:
            print(f"Subsample target ({args.target}) >= survivors "
                  f"({len(survivors)}); no subsampling needed")
 
    # ---- Reporting --------------------------------------------------------
    args.output.mkdir(parents=True, exist_ok=True)
    if args.verbose:
        for d in data:
            status = "KEEP" if d["keep"] else f"DROP ({d['reason']})"
            print(f"  {d['path'].name}: sharp={d['sharp']:.1f} "
                  f"bright={d['bright']:.1f} -> {status}")
 
    if args.report:
        with open(args.report, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filename", "sharpness", "brightness", "kept", "reason"])
            for d in data:
                w.writerow([d["path"].name, f"{d['sharp']:.2f}",
                            f"{d['bright']:.2f}", int(d["keep"]), d["reason"]])
        print(f"Wrote report to {args.report}")
 
    keepers = [d for d in data if d["keep"]]
    print(f"\nFinal: {len(keepers)} / {len(data)} images kept")
 
    if len(keepers) < 30:
        print("WARNING: very few frames kept — COLMAP may fail. "
              "Try loosening --sharpness-percentile or --brightness-std.")
 
    if args.dry_run:
        print("(dry run; no files copied)")
        return
 
    # ---- Copy outputs -----------------------------------------------------
    for d in keepers:
        shutil.copy2(d["path"], args.output / d["path"].name)
    print(f"Copied {len(keepers)} images to {args.output}")
 
    if args.masks:
        masks_out = args.masks_out or args.output.parent / f"{args.output.name}_masks"
        masks_out.mkdir(parents=True, exist_ok=True)
        copied = missing = 0
        for d in keepers:
            # COLMAP convention: mask file is named <image_filename>.png
            mask_name = d["path"].stem + ".png"
            src_mask = args.masks / mask_name
            if src_mask.exists():
                shutil.copy2(src_mask, masks_out / mask_name)
                copied += 1
            else:
                missing += 1
        print(f"Copied {copied} masks to {masks_out}"
              + (f"  ({missing} missing!)" if missing else ""))

    if args.masks_crop:
        masks_out = args.masks_crop_out or args.output.parent / f"{args.output.name}_masks"
        masks_out.mkdir(parents=True, exist_ok=True)
        copied = missing = 0
        for d in keepers:
            # COLMAP convention: mask file is named <image_filename>.png
            mask_name = d["path"].stem + ".jpg"
            src_mask = args.masks_crop / mask_name
            if src_mask.exists():
                shutil.copy2(src_mask, masks_out / mask_name)
                copied += 1
            else:
                missing += 1
        print(f"Copied {copied} masks to {masks_out}"
              + (f"  ({missing} missing!)" if missing else ""))
 
if __name__ == "__main__":
    main()
 
"""
Compare a predicted mesh against a ground-truth mesh (already aligned).

Samples both meshes to point clouds, then computes:
  - Chamfer distance (mean pred->gt, mean gt->pred, symmetric mean)
  - Precision / Recall / F-score at a distance threshold tau
  - Hausdorff distance (and a robust 95th-percentile variant)

Either input may also be a .ply point cloud — auto-detected by trying to
read triangles; if none, treated as a point cloud.
"""
import argparse
import json
import os

import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors


def load_points(path, n_samples, voxel_size=None):
    geom = o3d.io.read_triangle_mesh(path)
    if len(geom.triangles) > 0:
        geom.remove_unreferenced_vertices()
        pcd = geom.sample_points_uniformly(number_of_points=n_samples)
    else:
        pcd = o3d.io.read_point_cloud(path)
    if voxel_size is not None and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    return np.asarray(pcd.points), pcd


def nn_distances(src, dst):
    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree", n_jobs=-1).fit(dst)
    d, _ = nn.kneighbors(src, return_distance=True)
    return d[:, 0]


def estimate_tau(gt_pts, frac=0.005):
    """Auto-estimate tau as a fraction of the GT bounding-box diagonal,
    floored by the GT point spacing so we don't go below sensor noise."""
    bbox_min = gt_pts.min(axis=0)
    bbox_max = gt_pts.max(axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    tau_bbox = frac * diag

    # Estimate median GT point spacing (nearest-neighbor distance to *another* point)
    n_probe = min(20000, len(gt_pts))
    idx = np.random.default_rng(0).choice(len(gt_pts), n_probe, replace=False)
    nn = NearestNeighbors(n_neighbors=2, algorithm="kd_tree", n_jobs=-1).fit(gt_pts)
    d, _ = nn.kneighbors(gt_pts[idx], return_distance=True)
    spacing = float(np.median(d[:, 1]))
    tau_floor = 2.0 * spacing

    tau = max(tau_bbox, tau_floor)
    return tau, diag, spacing, tau_bbox, tau_floor


def write_error_pcd(path, points, dists, vis_max):
    alpha = np.clip(dists / vis_max, 0.0, 1.0)[:, None]
    red = np.array([[1.0, 0.0, 0.0]])
    white = np.array([[1.0, 1.0, 1.0]])
    colors = red * alpha + white * (1.0 - alpha)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path, pcd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True, help="path to predicted mesh / pcd")
    parser.add_argument("--gt", required=True, help="path to ground-truth mesh / pcd")
    parser.add_argument("--tau", type=float, default=None,
                        help="distance threshold for precision/recall/F-score; if omitted, auto-estimated from GT")
    parser.add_argument("--tau-frac", type=float, default=0.005,
                        help="fraction of GT bbox diagonal used when auto-estimating tau (default 0.5%%)")
    parser.add_argument("--tau-sweep", action="store_true",
                        help="also report P/R/F at 0.5x, 1x, 2x, 5x the chosen tau")
    parser.add_argument("--n-samples", type=int, default=1_000_000,
                        help="points to sample from each mesh")
    parser.add_argument("--voxel-size", type=float, default=0.0,
                        help="optional voxel downsample size; 0 disables")
    parser.add_argument("--out-dir", default="", help="if set, writes results.json and error PLYs")
    parser.add_argument("--no-vis", action="store_true", help="skip writing colored error point clouds")
    args = parser.parse_args()

    print(f"Loading pred: {args.pred}")
    pred_pts, _ = load_points(args.pred, args.n_samples, args.voxel_size or None)
    print(f"  {len(pred_pts):,} points")

    print(f"Loading gt:   {args.gt}")
    gt_pts, _ = load_points(args.gt, args.n_samples, args.voxel_size or None)
    print(f"  {len(gt_pts):,} points")

    if args.tau is None:
        tau, diag, spacing, tau_bbox, tau_floor = estimate_tau(gt_pts, args.tau_frac)
        print(f"Auto-tau: bbox diag={diag:.6f}, gt point spacing={spacing:.6f}")
        print(f"          tau_bbox ({args.tau_frac*100:.2f}% of diag) = {tau_bbox:.6f}")
        print(f"          tau_floor (2x spacing)                     = {tau_floor:.6f}")
        print(f"          -> tau = max(...) = {tau:.6f}")
    else:
        tau = args.tau
        diag = spacing = tau_bbox = tau_floor = None

    print("Computing pred -> gt distances")
    d_p2g = nn_distances(pred_pts, gt_pts)
    print("Computing gt -> pred distances")
    d_g2p = nn_distances(gt_pts, pred_pts)

    mean_p2g = float(d_p2g.mean())
    mean_g2p = float(d_g2p.mean())
    chamfer = 0.5 * (mean_p2g + mean_g2p)

    def prf(t):
        p = float((d_p2g < t).mean())
        r = float((d_g2p < t).mean())
        f = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
        return p, r, f

    precision, recall, fscore = prf(tau)

    hausdorff = float(max(d_p2g.max(), d_g2p.max()))
    hausdorff95 = float(max(np.percentile(d_p2g, 95), np.percentile(d_g2p, 95)))

    results = {
        "tau": tau,
        "tau_auto": args.tau is None,
        "tau_frac": args.tau_frac if args.tau is None else None,
        "gt_bbox_diag": diag,
        "gt_point_spacing": spacing,
        "n_pred": int(len(pred_pts)),
        "n_gt": int(len(gt_pts)),
        "chamfer_pred_to_gt": mean_p2g,
        "chamfer_gt_to_pred": mean_g2p,
        "chamfer_mean": chamfer,
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
        "hausdorff": hausdorff,
        "hausdorff_p95": hausdorff95,
    }

    if args.tau_sweep:
        sweep = {}
        for mult in (0.5, 1.0, 2.0, 5.0):
            p, r, f = prf(tau * mult)
            sweep[f"{mult:g}x"] = {"tau": tau * mult, "precision": p, "recall": r, "fscore": f}
        results["sweep"] = sweep

    print("=" * 50)
    print(f"tau                : {tau:.6f}{'  (auto)' if args.tau is None else ''}")
    print(f"chamfer pred->gt   : {mean_p2g:.6f}")
    print(f"chamfer gt->pred   : {mean_g2p:.6f}")
    print(f"chamfer mean       : {chamfer:.6f}")
    print(f"precision (<tau)   : {precision:.4f}")
    print(f"recall    (<tau)   : {recall:.4f}")
    print(f"f-score            : {fscore:.4f}")
    print(f"hausdorff          : {hausdorff:.6f}")
    print(f"hausdorff p95      : {hausdorff95:.6f}")
    if args.tau_sweep:
        print("-- sweep --")
        print(f"{'mult':>6} {'tau':>12} {'P':>8} {'R':>8} {'F':>8}")
        for k, v in results["sweep"].items():
            print(f"{k:>6} {v['tau']:>12.6f} {v['precision']:>8.4f} {v['recall']:>8.4f} {v['fscore']:>8.4f}")
    print("=" * 50)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        if not args.no_vis:
            vis_max = 2.0 * tau
            write_error_pcd(os.path.join(args.out_dir, "pred_error.ply"), pred_pts, d_p2g, vis_max)
            write_error_pcd(os.path.join(args.out_dir, "gt_error.ply"), gt_pts, d_g2p, vis_max)
        print(f"Wrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
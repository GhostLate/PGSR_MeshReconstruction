import argparse
import os
import sys

import numpy as np
import open3d as o3d


def parse_args():
    parser = argparse.ArgumentParser(
        description="Keep only the largest connected cluster of a triangle mesh."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to input mesh (.ply, .obj, .off, .stl, .gltf).",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Path to write the cleaned mesh.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Mutate the loaded mesh instead of copying before filtering.",
    )
    return parser.parse_args()


def keep_largest_cluster(
    mesh: o3d.geometry.TriangleMesh, in_place: bool = False
) -> o3d.geometry.TriangleMesh:
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    if cluster_n_triangles.size == 0:
        return mesh

    largest_cluster_idx = cluster_n_triangles.argmax()
    triangles_to_remove = triangle_clusters != largest_cluster_idx

    if not in_place:
        mesh = o3d.geometry.TriangleMesh(mesh)
    mesh.remove_triangles_by_mask(triangles_to_remove)
    mesh.remove_unreferenced_vertices()
    return mesh


def main():
    args = parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"[clean_mesh] file not found: {args.input}")

    mesh = o3d.io.read_triangle_mesh(args.input)
    if len(mesh.triangles) == 0:
        sys.exit(f"[clean_mesh] no triangles found in {args.input}")

    n_tris_before = len(mesh.triangles)
    n_verts_before = len(mesh.vertices)

    mesh = keep_largest_cluster(mesh, in_place=args.in_place)

    print(
        f"[clean_mesh] {n_verts_before} -> {len(mesh.vertices)} verts, "
        f"{n_tris_before} -> {len(mesh.triangles)} tris"
    )

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    o3d.io.write_triangle_mesh(args.output, mesh)
    print(f"[clean_mesh] wrote {args.output}")


if __name__ == "__main__":
    main()
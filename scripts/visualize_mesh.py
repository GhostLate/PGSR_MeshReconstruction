import argparse
import os
import sys

import numpy as np
import open3d as o3d


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize a mesh (or point cloud) with Open3D."
    )
    parser.add_argument(
        "mesh_path",
        type=str,
        help="Path to a mesh file (.ply, .obj, .off, .stl, .gltf) or point cloud.",
    )
    parser.add_argument(
        "--point-cloud",
        action="store_true",
        help="Load the file as a point cloud instead of a triangle mesh.",
    )
    parser.add_argument(
        "--wireframe",
        action="store_true",
        help="Render the mesh as wireframe.",
    )
    parser.add_argument(
        "--no-shading",
        action="store_true",
        help="Disable lighting / shading (flat color).",
    )
    parser.add_argument(
        "--no-normals",
        action="store_true",
        help="Do not compute vertex normals (faster, but flatter look).",
    )
    parser.add_argument(
        "--color",
        type=float,
        nargs=3,
        metavar=("R", "G", "B"),
        default=None,
        help="Paint the mesh with a uniform RGB color in [0,1].",
    )
    parser.add_argument(
        "--background",
        type=float,
        nargs=3,
        metavar=("R", "G", "B"),
        default=[0.1, 0.1, 0.1],
        help="Window background color in [0,1] (default: dark gray).",
    )
    parser.add_argument(
        "--show-axes",
        action="store_true",
        help="Show a coordinate frame at the origin.",
    )
    parser.add_argument(
        "--axis-size",
        type=float,
        default=None,
        help="Size of the coordinate frame (default: 10%% of mesh bbox diagonal).",
    )
    parser.add_argument(
        "--show-bbox",
        action="store_true",
        help="Draw the axis-aligned bounding box.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=2.0,
        help="Point size used when rendering point clouds (default: 2.0).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Window width in pixels (default: 1280).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Window height in pixels (default: 720).",
    )
    parser.add_argument(
        "--screenshot",
        type=str,
        default=None,
        help="If set, render once and save a PNG to this path instead of opening a window.",
    )
    return parser.parse_args()


def load_geometry(path: str, as_point_cloud: bool):
    if not os.path.isfile(path):
        sys.exit(f"[visualize_mesh] file not found: {path}")

    if as_point_cloud:
        pcd = o3d.io.read_point_cloud(path)
        if len(pcd.points) == 0:
            sys.exit(f"[visualize_mesh] empty point cloud: {path}")
        return pcd, "pointcloud"

    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.triangles) == 0:
        print(
            f"[visualize_mesh] no triangles found in {path}, falling back to point cloud.",
            file=sys.stderr,
        )
        pcd = o3d.io.read_point_cloud(path)
        if len(pcd.points) == 0:
            sys.exit(f"[visualize_mesh] file has neither triangles nor points: {path}")
        return pcd, "pointcloud"
    return mesh, "mesh"


def prepare_mesh(mesh, args):
    if not args.no_normals:
        mesh.compute_vertex_normals()

    if args.color is not None:
        mesh.paint_uniform_color(np.clip(args.color, 0.0, 1.0))
    elif not mesh.has_vertex_colors():
        mesh.paint_uniform_color([0.75, 0.75, 0.75])
    return mesh


def build_extras(geometry, args):
    extras = []
    bbox = geometry.get_axis_aligned_bounding_box()
    diag = float(np.linalg.norm(bbox.get_extent()))

    if args.show_axes:
        size = args.axis_size if args.axis_size is not None else max(diag * 0.1, 1e-3)
        extras.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=size))

    if args.show_bbox:
        bbox.color = (1.0, 0.0, 0.0)
        extras.append(bbox)
    return extras


def run_visualizer(geometry, kind, extras, args):
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=os.path.basename(args.mesh_path),
        width=args.width,
        height=args.height,
        visible=args.screenshot is None,
    )

    vis.add_geometry(geometry)
    for extra in extras:
        vis.add_geometry(extra)

    opt = vis.get_render_option()
    opt.background_color = np.asarray(args.background, dtype=np.float64)
    opt.mesh_show_wireframe = args.wireframe
    opt.light_on = not args.no_shading
    opt.point_size = args.point_size
    if kind == "mesh":
        opt.mesh_show_back_face = True

    vis.poll_events()
    vis.update_renderer()

    if args.screenshot is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.screenshot)), exist_ok=True)
        vis.capture_screen_image(args.screenshot, do_render=True)
        print(f"[visualize_mesh] saved screenshot to {args.screenshot}")
        vis.destroy_window()
    else:
        vis.run()
        vis.destroy_window()


def main():
    args = parse_args()
    geometry, kind = load_geometry(args.mesh_path, args.point_cloud)
    if kind == "mesh":
        geometry = prepare_mesh(geometry, args)
        print(
            f"[visualize_mesh] loaded mesh: "
            f"{len(geometry.vertices)} verts, {len(geometry.triangles)} tris"
        )
    else:
        if args.color is not None:
            geometry.paint_uniform_color(np.clip(args.color, 0.0, 1.0))
        print(f"[visualize_mesh] loaded point cloud: {len(geometry.points)} points")

    extras = build_extras(geometry, args)
    run_visualizer(geometry, kind, extras, args)


if __name__ == "__main__":
    main()
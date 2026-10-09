import argparse
import os
from time import perf_counter

import numpy as np

from cam_calibration import intrinsics_for_image, list_images, load_or_calibrate, scale_intrinsics
from features import extract_features
from pose import camera_center
from sfm import SfMOptions, run_sfm


def parse_args():
    parser = argparse.ArgumentParser(description="Incremental Structure from Motion from calibrated images")
    parser.add_argument("images", nargs="*", help="input images (or use --images-dir)")
    parser.add_argument("--images-dir", help="folder containing the input images")
    parser.add_argument("--calib-file", default="calibrated_data/calibrated_data.npz",
                        help="calibration file, created from --calib-dir if missing")
    parser.add_argument("--calib-dir", default="Calibration2", help="chessboard pictures")
    parser.add_argument("--board", type=int, nargs=2, default=(8, 6), metavar=("COLS", "ROWS"),
                        help="inner corners of the chessboard")
    parser.add_argument("--square-size", type=float, default=24.0, help="chessboard square size (mm)")
    parser.add_argument("--recalibrate", action="store_true", help="ignore an existing calibration file")
    parser.add_argument("--max-size", type=int, default=2000,
                        help="downscale images so that max(width, height) <= this (0: full resolution)")
    parser.add_argument("--max-features", type=int, default=10000)
    parser.add_argument("--sift-contrast", type=float, default=0.04,
                        help="SIFT contrast threshold, lower (0.01) for weakly textured objects")
    parser.add_argument("--min-pair-inliers", type=int, default=30,
                        help="verified matches needed to link two images")
    parser.add_argument("--min-pnp-inliers", type=int, default=30,
                        help="2D-3D inliers needed to register an image")
    parser.add_argument("--ba-every", type=int, default=3, help="global bundle adjustment frequency")
    parser.add_argument("--min-angle", type=float, default=1.5,
                        help="minimum triangulation angle in degrees (lower keeps noisier, distant points)")
    parser.add_argument("--out", default="output", help="output folder")
    parser.add_argument("--no-clean", action="store_true", help="skip statistical outlier removal")
    parser.add_argument("--mesh", action="store_true", help="also build a Poisson mesh")
    parser.add_argument("--colmap", action="store_true", help="export a COLMAP text model for MVS")
    parser.add_argument("--show", action="store_true", help="open an Open3D viewer")
    args = parser.parse_args()

    images = list(args.images)
    if args.images_dir:
        images += list_images(args.images_dir)
    if len(images) < 2:
        parser.error("need at least 2 images")
    return args, images


def main():
    args, image_paths = parse_args()
    if args.recalibrate and os.path.exists(args.calib_file):
        os.remove(args.calib_file)
    calib = load_or_calibrate(args.calib_file, args.calib_dir, chessboard_dim=tuple(args.board),
                              square_size=args.square_size)

    start = perf_counter()
    views = [extract_features(p, args.max_size, args.max_features, args.sift_contrast) for p in image_paths]
    print(f"SIFT on {len(views)} images: {sum(len(v.keypoints) for v in views)} keypoints "
          f"in {perf_counter() - start:.1f}s")

    original_sizes = {tuple(round(s / v.scale) for s in v.image_size) for v in views}
    if len(original_sizes) != 1 or len({v.image_size for v in views}) != 1:
        raise ValueError(f"All images must have the same size, got {sorted(original_sizes)}")
    original_size = original_sizes.pop()
    K_original = intrinsics_for_image(calib, original_size)
    K = scale_intrinsics(K_original, views[0].scale)

    options = SfMOptions(ba_every=args.ba_every, min_triangulation_angle=args.min_angle,
                         min_pair_inliers=args.min_pair_inliers, min_pnp_inliers=args.min_pnp_inliers,
                         init_min_median_angle=max(args.min_angle, 3.0))
    recon = run_sfm(views, K, calib.dist, options)
    print(f"Final reprojection RMS: {recon.reprojection_rms():.2f} px "
          f"(at {views[0].image_size[0]}x{views[0].image_size[1]})")

    from pointcloud import clean_pointcloud, create_pointcloud, save_geometry, show

    points, colors = recon.points_and_colors()
    view_points = np.array([camera_center(*recon.poses[next(iter(t.obs))]) for t in recon.tracks.values()])
    pcd = create_pointcloud(points, colors)
    if not args.no_clean:
        pcd, kept = clean_pointcloud(pcd)
        view_points = view_points[kept]
        print(f"Outlier removal kept {len(kept)}/{len(points)} points")
    save_geometry(pcd, os.path.join(args.out, "sparse.ply"))

    cameras = []
    if args.show:
        import open3d as o3d
        scale = np.linalg.norm(np.ptp(points, axis=0)) * 0.03
        for R, t in recon.poses.values():
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
            frame.rotate(R.T, center=(0, 0, 0))
            frame.translate(camera_center(R, t))
            cameras.append(frame)

    if args.colmap:
        from export import export_colmap
        export_colmap(recon, K_original, calib.dist, original_size, os.path.join(args.out, "colmap"))

    if args.mesh:
        from mesh import poisson_mesh
        mesh = poisson_mesh(pcd, view_points)
        save_geometry(mesh, os.path.join(args.out, "mesh.ply"))
        if args.show:
            mesh.paint_uniform_color([1, 0.706, 0])
            show(mesh, *cameras)
    elif args.show:
        show(pcd, *cameras)


if __name__ == "__main__":
    main()

"""Run the SfM pipeline on a Middlebury multi-view dataset and compare with the ground truth.

    python benchmarks/middlebury.py templeRing
    python benchmarks/middlebury.py dinoRing --sift-contrast 0.01 --min-pnp-inliers 15

Datasets: https://vision.middlebury.edu/mview/data/ (images + calibrated cameras, no distortion).
"""
import argparse
import glob
import os
import sys
import urllib.request
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import compare_poses  # noqa: E402
from features import extract_features  # noqa: E402
from sfm import SfMOptions, run_sfm  # noqa: E402

URL = "https://vision.middlebury.edu/mview/data/data/{}.zip"


def download(name, data_dir):
    folder = os.path.join(data_dir, name)
    if not os.path.isdir(folder):
        os.makedirs(data_dir, exist_ok=True)
        archive = os.path.join(data_dir, f"{name}.zip")
        print(f"Downloading {URL.format(name)}")
        urllib.request.urlretrieve(URL.format(name), archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(data_dir)
    return folder


def read_cameras(folder):
    """Middlebury *_par.txt: name k11..k33 r11..r33 t1 t2 t3 with x = K (R X + t)."""
    par = glob.glob(os.path.join(folder, "*_par.txt"))[0]
    cams = {}
    for line in open(par).read().splitlines()[1:]:
        f = line.split()
        if len(f) == 22:
            v = np.array(f[1:], dtype=float)
            cams[f[0]] = (v[:9].reshape(3, 3), v[9:18].reshape(3, 3), v[18:21].reshape(3, 1))
    return cams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="templeRing",
                        help="templeRing, dinoRing, templeSparseRing, dinoSparseRing, ...")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="output")
    parser.add_argument("--sift-contrast", type=float, default=0.04)
    parser.add_argument("--min-pnp-inliers", type=int, default=30)
    args = parser.parse_args()

    folder = download(args.dataset, args.data_dir)
    gt = read_cameras(folder)
    names = sorted(gt)
    K = next(iter(gt.values()))[0]

    views = [extract_features(os.path.join(folder, n), 0, 10000, args.sift_contrast) for n in names]
    recon = run_sfm(views, K, None, SfMOptions(min_pnp_inliers=args.min_pnp_inliers))

    est = {recon.views[v].name: pose for v, pose in recon.poses.items()}
    center_err, rot_err, (s, R_align, t_align) = compare_poses(est, {n: (R, t) for n, (_, R, t) in gt.items()})
    centers = np.array([(-R.T @ t).ravel() for _, R, t in gt.values()])
    radius = np.mean(np.linalg.norm(centers - centers.mean(0), axis=1))

    print(f"\n{args.dataset}: {len(recon.poses)}/{len(views)} images registered, {len(recon.tracks)} points, "
          f"reprojection RMS {recon.reprojection_rms():.2f} px")
    print(f"camera centers: mean error {1000 * center_err.mean():.2f} mm, max {1000 * center_err.max():.2f} mm "
          f"({100 * center_err.max() / radius:.2f}% of the {radius:.3f} m ring radius)")
    print(f"camera rotations: mean error {rot_err.mean():.3f} deg, max {rot_err.max():.3f} deg")

    from pointcloud import clean_pointcloud, create_pointcloud, save_geometry

    points, colors = recon.points_and_colors()
    points = (s * (R_align @ points.T)).T + t_align
    pcd, _ = clean_pointcloud(create_pointcloud(points, colors))
    save_geometry(pcd, os.path.join(args.out, f"{args.dataset}_sparse.ply"))


if __name__ == "__main__":
    main()

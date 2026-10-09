"""Export a reconstruction in COLMAP text format (cameras.txt, images.txt, points3D.txt).

The exported model refers to the original images at full resolution, so it can be densified
with COLMAP (image_undistorter + patch_match_stereo + stereo_fusion) or OpenMVS.
"""
import os

import numpy as np
from scipy.spatial.transform import Rotation

from triangulation import reprojection_errors


def _to_original_pixels(points, scale):
    return (points + 0.5) / scale - 0.5


def export_colmap(recon, K, dist_coeffs, image_size, out_dir):
    """K, dist_coeffs and image_size (width, height) describe the original images."""
    os.makedirs(out_dir, exist_ok=True)
    width, height = image_size
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    dist = np.zeros(8)
    coeffs = np.asarray(dist_coeffs, dtype=np.float64).ravel()[:8]
    dist[:len(coeffs)] = coeffs
    k1, k2, p1, p2, k3, k4, k5, k6 = dist

    with open(os.path.join(out_dir, "cameras.txt"), "w") as f:
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 FULL_OPENCV {width} {height} {fx} {fy} {cx} {cy} "
                f"{k1} {k2} {p1} {p2} {k3} {k4} {k5} {k6}\n")

    track_ids = {tid: n + 1 for n, tid in enumerate(recon.tracks)}
    with open(os.path.join(out_dir, "images.txt"), "w") as f:
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for v in sorted(recon.poses):
            R, t = recon.poses[v]
            qx, qy, qz, qw = Rotation.from_matrix(R).as_quat()
            view = recon.views[v]
            f.write(f"{v + 1} {qw} {qx} {qy} {qz} {t[0, 0]} {t[1, 0]} {t[2, 0]} 1 {view.name}\n")
            pts = _to_original_pixels(view.keypoints, view.scale)
            ids = [track_ids.get(tid, -1) for tid in recon.kp_to_track[v]]
            f.write(" ".join(f"{x:.2f} {y:.2f} {pid}" for (x, y), pid in zip(pts, ids)) + "\n")

    with open(os.path.join(out_dir, "points3D.txt"), "w") as f:
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for tid, track in recon.tracks.items():
            errors = [reprojection_errors(track.xyz[None], recon.points2d[v][k][None], recon.K,
                                          *recon.poses[v])[0] / recon.views[v].scale
                      for v, k in track.obs.items()]
            x, y, z = track.xyz
            r, g, b = (int(c) for c in track.color)
            obs = " ".join(f"{v + 1} {k}" for v, k in track.obs.items())
            f.write(f"{track_ids[tid]} {x} {y} {z} {r} {g} {b} {np.mean(errors):.4f} {obs}\n")
    print(f"COLMAP model exported to {out_dir}")

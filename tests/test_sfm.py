import numpy as np

from sfm import SfMOptions, run_sfm
from synthetic import DIST, K, circle_cameras, make_views, rotation_angle_deg, sphere_points, umeyama


def _gt_and_estimated_centers(recon, poses):
    views = sorted(recon.poses)
    est = np.array([(-R.T @ t).ravel() for R, t in (recon.poses[v] for v in views)])
    gt = np.array([(-R.T @ t).ravel() for R, t in (poses[v] for v in views)])
    return views, est, gt


def test_incremental_sfm_recovers_cameras_and_structure():
    rng = np.random.default_rng(1)
    points, normals = sphere_points(1500, rng)
    poses = circle_cameras(np.arange(0, 120, 15))
    views, gt_ids = make_views(points, normals, poses, rng)

    recon = run_sfm(views, K, DIST, SfMOptions(verbose=False))

    assert len(recon.poses) == len(views)
    assert recon.reprojection_rms() < 1.0
    assert recon.mean_track_length() > 2.5
    assert len(recon.tracks) > 0.6 * len(np.unique(np.concatenate(gt_ids)))

    views_idx, est, gt = _gt_and_estimated_centers(recon, poses)
    s, R_align, t_align = umeyama(est, gt)
    aligned = (s * (R_align @ est.T)).T + t_align
    assert np.max(np.linalg.norm(aligned - gt, axis=1)) < 0.02 * 3.0  # 2% of the orbit radius

    for v in views_idx:
        R_est = recon.poses[v][0] @ R_align.T
        assert rotation_angle_deg(R_est @ poses[v][0].T) < 0.5

    # Tracks only contain keypoints of the same 3D point.
    wrong = sum(len({gt_ids[v][k] for v, k in t.obs.items()}) > 1 for t in recon.tracks.values())
    assert wrong < 0.01 * len(recon.tracks)

    X, _ = recon.points_and_colors()
    X_aligned = (s * (R_align @ X.T)).T + t_align
    radii = np.linalg.norm(X_aligned, axis=1)
    assert np.mean((radii > 0.85) & (radii < 1.15)) > 0.98


def test_initial_pair_avoids_small_baseline():
    rng = np.random.default_rng(2)
    points, normals = sphere_points(1200, rng)
    # Views 0 and 1 are almost identical: most matches, but no parallax.
    poses = circle_cameras([0.0, 0.3, 20.0, 35.0])
    views, _ = make_views(points, normals, poses, rng)
    recon = run_sfm(views, K, DIST, SfMOptions(verbose=False))
    assert set(recon.initial_pair) != {0, 1}
    assert len(recon.poses) == 4


def test_colmap_export_reprojects_in_original_images(tmp_path):
    import cv2
    from scipy.spatial.transform import Rotation

    from cam_calibration import scale_intrinsics
    from export import export_colmap
    from features import ViewFeatures

    rng = np.random.default_rng(4)
    points, normals = sphere_points(800, rng)
    poses = circle_cameras([0, 15, 30])
    views, _ = make_views(points, normals, poses, rng, noise_px=0.3)
    scale = 0.5
    small = [ViewFeatures(name=v.name, keypoints=(v.keypoints + 0.5) * scale - 0.5,
                          descriptors=v.descriptors, colors=v.colors,
                          image_size=(v.image_size[0] // 2, v.image_size[1] // 2), scale=scale)
             for v in views]
    recon = run_sfm(small, scale_intrinsics(K, scale), DIST, SfMOptions(verbose=False))
    export_colmap(recon, K, DIST, views[0].image_size, str(tmp_path))

    xyz = {}
    for line in open(tmp_path / "points3D.txt"):
        if not line.startswith("#"):
            f = line.split()
            xyz[int(f[0])] = np.array(f[1:4], dtype=float)
    lines = [l for l in open(tmp_path / "images.txt") if not l.startswith("#")]
    errors = []
    for header, obs in zip(lines[0::2], lines[1::2]):
        f = header.split()
        qw, qx, qy, qz = map(float, f[1:5])
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        t = np.array(f[5:8], dtype=float)
        o = np.array(obs.split(), dtype=float).reshape(-1, 3)
        o = o[o[:, 2] > 0]
        X = np.array([xyz[int(i)] for i in o[:, 2]])
        proj, _ = cv2.projectPoints(X, cv2.Rodrigues(R)[0], t, K, DIST)
        errors.append(np.linalg.norm(proj.reshape(-1, 2) - o[:, :2], axis=1))
    assert np.median(np.concatenate(errors)) < 1.0


def _alignment_errors(recon, poses):
    views, est, gt = _gt_and_estimated_centers(recon, poses)
    s, R_align, t_align = umeyama(est, gt)
    center_err = np.max(np.linalg.norm((s * (R_align @ est.T)).T + t_align - gt, axis=1))
    X, _ = recon.points_and_colors()
    radii = np.linalg.norm((s * (R_align @ X.T)).T + t_align, axis=1)
    return center_err, np.mean(np.abs(radii - 1.0) > 0.1)


def test_ignoring_distortion_degrades_reconstruction():
    rng = np.random.default_rng(3)
    points, normals = sphere_points(1500, rng)
    poses = circle_cameras(np.arange(0, 75, 15))
    dist = np.array([-0.25, 0.08, 0, 0, 0])
    views, _ = make_views(points, normals, poses, rng, dist=dist)
    center_ok, off_surface_ok = _alignment_errors(run_sfm(views, K, dist, SfMOptions(verbose=False)), poses)
    center_bad, off_surface_bad = _alignment_errors(run_sfm(views, K, None, SfMOptions(verbose=False)), poses)
    assert center_ok < 0.2 * center_bad
    assert off_surface_ok < 0.05 < off_surface_bad

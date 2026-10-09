import numpy as np
import pytest

from bundle_adjustment import bundle_adjustment, pose_to_params, params_to_pose
from matching import match_descriptors
from pose import camera_center, estimate_pose, undistort_keypoints
from synthetic import DIST, K, circle_cameras, make_views, rotation_angle_deg, sphere_points
from triangulation import project_points, triangulate_points, valid_triangulations


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _two_view_correspondences(rng, outlier_ratio=0.2, angle=20.0):
    points, normals = sphere_points(800, rng)
    (R1, t1), (R2, t2) = circle_cameras([0.0, angle])
    views, ids = make_views(points, normals, [(R1, t1), (R2, t2)], rng, n_clutter=0)
    common = np.intersect1d(ids[0], ids[1])
    idx1 = np.array([np.flatnonzero(ids[0] == i)[0] for i in common])
    idx2 = np.array([np.flatnonzero(ids[1] == i)[0] for i in common])
    p1 = undistort_keypoints(views[0].keypoints[idx1], K, DIST)
    p2 = undistort_keypoints(views[1].keypoints[idx2], K, DIST)
    n_out = int(outlier_ratio * len(p2))
    p2[:n_out] = rng.uniform([0, 0], [1000, 750], size=(n_out, 2))
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1
    return p1, p2, R_rel, t_rel, n_out


def test_undistortion_matches_pinhole_projection(rng):
    points, normals = sphere_points(300, rng)
    poses = circle_cameras([0.0])
    views, ids = make_views(points, normals, poses, rng, noise_px=0.0, n_clutter=0)
    R, t = poses[0]
    ideal = project_points(points[ids[0]], K, R, t)
    assert np.abs(views[0].keypoints - ideal).max() > 1.0  # distortion is significant
    np.testing.assert_allclose(undistort_keypoints(views[0].keypoints, K, DIST), ideal, atol=1e-3)


def test_estimate_pose_with_outliers(rng):
    p1, p2, R_rel, t_rel, n_out = _two_view_correspondences(rng)
    R, t, mask = estimate_pose(p1, p2, K, threshold=2.0)
    assert rotation_angle_deg(R @ R_rel.T) < 1.0
    cos = float(t.ravel() @ t_rel.ravel() / np.linalg.norm(t_rel))
    assert np.degrees(np.arccos(min(1.0, cos))) < 2.0
    # Outliers are rejected, inliers kept.
    assert mask[:n_out].mean() < 0.05
    assert mask[n_out:].mean() > 0.9


def test_triangulation_filters(rng):
    p1, p2, R_rel, t_rel, n_out = _two_view_correspondences(rng)
    R0, t0 = np.eye(3), np.zeros((3, 1))
    X = triangulate_points(p1, p2, K, R0, t0, R_rel, t_rel)
    ok = valid_triangulations(X, p1, p2, K, R0, t0, R_rel, t_rel)
    assert ok[:n_out].mean() < 0.05
    assert ok[n_out:].mean() > 0.9


def test_bundle_adjustment_converges_and_keeps_gauge(rng):
    points, normals = sphere_points(400, rng)
    poses = circle_cameras([0, 15, 30, 45])
    views, ids = make_views(points, normals, poses, rng, noise_px=0.5, n_clutter=0)
    cam_idx, pt_idx, uv = [], [], []
    for c, (view, vid) in enumerate(zip(views, ids)):
        cam_idx += [c] * len(vid)
        pt_idx += list(vid)
        uv.append(undistort_keypoints(view.keypoints, K, DIST))
    cam_idx, pt_idx, uv = np.array(cam_idx), np.array(pt_idx), np.vstack(uv)

    gt_cams = np.array([pose_to_params(R, t) for R, t in poses])
    cams0 = gt_cams.copy()
    cams0[1:, :3] += rng.normal(scale=0.01, size=(3, 3))
    cams0[1:, 3:] += rng.normal(scale=0.05, size=(3, 3))
    pts0 = points + rng.normal(scale=0.03, size=points.shape)

    cams, pts, info = bundle_adjustment(cams0, pts0, cam_idx, pt_idx, uv, K, fixed_cams=(0,),
                                        scale_pair=(0, 1))
    assert info["initial_rms"] > 5.0
    assert info["final_rms"] < 0.8  # noise is 0.5 px per axis
    np.testing.assert_array_equal(cams[0], cams0[0])
    d0 = np.linalg.norm(camera_center(*params_to_pose(cams0[0])) - camera_center(*params_to_pose(cams0[1])))
    d1 = np.linalg.norm(camera_center(*params_to_pose(cams[0])) - camera_center(*params_to_pose(cams[1])))
    assert abs(d1 - d0) / d0 < 1e-2
    for c in range(1, 4):
        R_est, _ = params_to_pose(cams[c])
        assert rotation_angle_deg(R_est @ poses[c][0].T) < 0.5


def test_mutual_matching_rejects_ambiguous(rng):
    desc1 = rng.uniform(size=(50, 128)).astype(np.float32)
    desc2 = desc1[::-1] + 0.01 * rng.normal(size=(50, 128)).astype(np.float32)
    desc2 = np.vstack((desc2, desc2[:5]))  # duplicated descriptors are ambiguous
    m = match_descriptors(desc1, desc2.astype(np.float32))
    assert len(m) == 45
    assert np.all(m[:, 1] == 49 - m[:, 0])

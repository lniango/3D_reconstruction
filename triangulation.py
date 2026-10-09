import cv2 as cv
import numpy as np

from pose import camera_center


def projection_matrix(K, R, t):
    return K @ np.hstack((R, np.asarray(t, dtype=np.float64).reshape(3, 1)))


def triangulate_points(pts1, pts2, K, R1, t1, R2, t2):
    """Linear triangulation of undistorted correspondences. Returns (N, 3) world points."""
    if len(pts1) == 0:
        return np.zeros((0, 3))
    points_4d = cv.triangulatePoints(projection_matrix(K, R1, t1), projection_matrix(K, R2, t2),
                                     np.asarray(pts1, np.float64).T, np.asarray(pts2, np.float64).T)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (points_4d[:3] / points_4d[3]).T


def depths(points, R, t):
    return (points @ R.T + np.asarray(t).reshape(1, 3))[:, 2]


def project_points(points, K, R, t):
    cam = points @ R.T + np.asarray(t).reshape(1, 3)
    proj = cam @ K.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return proj[:, :2] / proj[:, 2:3]


def reprojection_errors(points, pts2d, K, R, t):
    err = np.linalg.norm(project_points(points, K, R, t) - pts2d, axis=1)
    return np.where(np.isfinite(err), err, np.inf)


def triangulation_angles(points, center1, center2):
    """Angle in degrees between the two viewing rays of each point."""
    ray1 = points - center1
    ray2 = points - center2
    cos = np.sum(ray1 * ray2, axis=1) / (np.linalg.norm(ray1, axis=1) * np.linalg.norm(ray2, axis=1))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def valid_triangulations(points, pts1, pts2, K, R1, t1, R2, t2, max_error=4.0, min_angle=1.5):
    """Points in front of both cameras, with small reprojection error and enough parallax."""
    finite = np.all(np.isfinite(points), axis=1)
    safe = np.where(finite[:, None], points, 0.0)
    ok = finite
    ok &= depths(safe, R1, t1) > 0
    ok &= depths(safe, R2, t2) > 0
    ok &= reprojection_errors(safe, pts1, K, R1, t1) < max_error
    ok &= reprojection_errors(safe, pts2, K, R2, t2) < max_error
    ok &= triangulation_angles(safe, camera_center(R1, t1), camera_center(R2, t2)) >= min_angle
    return ok

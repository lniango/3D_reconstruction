import cv2 as cv
import numpy as np


def undistort_keypoints(points, K, dist):
    """Map distorted pixel coordinates to the ideal pinhole camera with the same K."""
    if len(points) == 0 or dist is None or not np.any(dist):
        return np.asarray(points, dtype=np.float64).reshape(-1, 2).copy()
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv.undistortPoints(pts, K, dist, P=K).reshape(-1, 2)


def camera_center(R, t):
    return (-R.T @ np.asarray(t).reshape(3, 1)).ravel()


def _recover_pose(E, pts1, pts2, K, mask):
    n_good, R, t, mask_pose = cv.recoverPose(E, pts1, pts2, K, mask=mask.astype(np.uint8).copy())
    return n_good, R, t, mask_pose.ravel() > 0


def estimate_pose(pts1, pts2, K, threshold=1.0, prob=0.999):
    """Relative pose of camera 2 w.r.t. camera 1 from undistorted correspondences.

    Returns R, t (|t| = 1) and a boolean mask of matches that are RANSAC inliers of the
    essential matrix and lie in front of both cameras. Returns (None, None, mask) on failure.
    """
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)
    if len(pts1) < 5:
        return None, None, np.zeros(len(pts1), dtype=bool)

    E, mask = cv.findEssentialMat(pts1, pts2, K, method=cv.RANSAC, prob=prob, threshold=threshold)
    if E is None or mask is None:
        return None, None, np.zeros(len(pts1), dtype=bool)
    mask = mask.ravel() > 0

    # With degenerate data findEssentialMat may return several stacked 3x3 solutions.
    best = None
    for k in range(E.shape[0] // 3):
        n_good, R, t, good = _recover_pose(E[3 * k:3 * k + 3], pts1, pts2, K, mask)
        if best is None or n_good > best[0]:
            best = (n_good, R, t, good & mask)
    _, R, t, good = best
    return R, t, good


def essential_inliers(pts1, pts2, K, threshold=1.0, prob=0.999):
    """Boolean mask of correspondences consistent with an essential matrix (geometric verification)."""
    if len(pts1) < 5:
        return np.zeros(len(pts1), dtype=bool)
    E, mask = cv.findEssentialMat(np.asarray(pts1, np.float64), np.asarray(pts2, np.float64), K,
                                  method=cv.RANSAC, prob=prob, threshold=threshold)
    if E is None or mask is None:
        return np.zeros(len(pts1), dtype=bool)
    return mask.ravel() > 0


def homography_inlier_count(pts1, pts2, threshold=4.0):
    """Many inliers to a homography means pure rotation or a planar scene: bad initial pair."""
    if len(pts1) < 4:
        return 0
    H, mask = cv.findHomography(np.asarray(pts1, np.float64), np.asarray(pts2, np.float64),
                                cv.RANSAC, threshold)
    return 0 if mask is None else int(mask.sum())

import cv2 as cv
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def rotate(points, rot_vecs):
    """Rodrigues rotation of each point by its own rotation vector (vectorized)."""
    theta = np.linalg.norm(rot_vecs, axis=1)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.nan_to_num(rot_vecs / theta)
    dot = np.sum(points * v, axis=1)[:, None]
    cos, sin = np.cos(theta), np.sin(theta)
    return cos * points + sin * np.cross(v, points) + dot * (1 - cos) * v


def project(points, cam_params, K):
    """cam_params rows are [rvec (3), t (3)] world-to-camera."""
    p = rotate(points, cam_params[:, :3]) + cam_params[:, 3:6]
    p = p @ K.T
    return p[:, :2] / p[:, 2:3]


def pose_to_params(R, t):
    rvec, _ = cv.Rodrigues(R)
    return np.hstack((rvec.ravel(), np.asarray(t, dtype=np.float64).ravel()))


def params_to_pose(params):
    R, _ = cv.Rodrigues(params[:3].reshape(3, 1))
    return R, params[3:6].reshape(3, 1)


def _center(params):
    R, t = params_to_pose(params)
    return (-R.T @ t).ravel()


def bundle_adjustment(cam_params, points, cam_idx, pt_idx, uv, K, fixed_cams=(0,), scale_pair=None,
                      loss="soft_l1", f_scale=2.0, max_nfev=100, verbose=0):
    """Jointly refine camera poses and 3D points by minimizing the reprojection error.

    cam_params: (C, 6) [rvec, t]; points: (P, 3); observation k sees point pt_idx[k] in camera
    cam_idx[k] at pixel uv[k] (undistorted). Cameras in fixed_cams are held constant, which
    removes the rotation/translation gauge freedom. The remaining scale freedom is removed by
    softly constraining the distance between the centers of the two cameras in scale_pair.

    scipy's "huber" loss has zero curvature beyond f_scale, which stalls the solver when the
    starting errors are large; "soft_l1" is robust to outliers and still converges.

    Returns refined (cam_params, points) and a dict with the initial and final RMS errors.
    """
    cam_params = np.asarray(cam_params, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    cam_idx = np.asarray(cam_idx, dtype=int)
    pt_idx = np.asarray(pt_idx, dtype=int)
    uv = np.asarray(uv, dtype=np.float64)

    n_cams, n_points, n_obs = len(cam_params), len(points), len(uv)
    fixed = np.zeros(n_cams, dtype=bool)
    fixed[list(fixed_cams)] = True
    free_cams = np.flatnonzero(~fixed)
    free_pos = -np.ones(n_cams, dtype=int)
    free_pos[free_cams] = np.arange(len(free_cams))
    n_cam_vars = 6 * len(free_cams)

    if scale_pair is not None:
        a, b = scale_pair
        target_dist = np.linalg.norm(_center(cam_params[a]) - _center(cam_params[b]))
        # Large weight: a 1% scale drift costs as much as ~10 px of reprojection error.
        scale_weight = 1000.0 / max(target_dist, 1e-12)

    def unpack(x):
        cams = cam_params.copy()
        cams[free_cams] = x[:n_cam_vars].reshape(-1, 6)
        return cams, x[n_cam_vars:].reshape(n_points, 3)

    def residuals(x):
        cams, pts = unpack(x)
        res = (project(pts[pt_idx], cams[cam_idx], K) - uv).ravel()
        if scale_pair is None:
            return res
        dist = np.linalg.norm(_center(cams[a]) - _center(cams[b]))
        return np.append(res, scale_weight * (dist - target_dist))

    n_res = 2 * n_obs + (scale_pair is not None)
    sparsity = lil_matrix((n_res, n_cam_vars + 3 * n_points), dtype=int)
    rows = np.arange(n_obs)
    obs_free = free_pos[cam_idx] >= 0
    for s in range(6):
        cols = 6 * free_pos[cam_idx[obs_free]] + s
        sparsity[2 * rows[obs_free], cols] = 1
        sparsity[2 * rows[obs_free] + 1, cols] = 1
    for s in range(3):
        cols = n_cam_vars + 3 * pt_idx + s
        sparsity[2 * rows, cols] = 1
        sparsity[2 * rows + 1, cols] = 1
    if scale_pair is not None:
        for c in scale_pair:
            if free_pos[c] >= 0:
                sparsity[n_res - 1, 6 * free_pos[c]:6 * free_pos[c] + 6] = 1

    x0 = np.hstack((cam_params[free_cams].ravel(), points.ravel()))
    r0 = residuals(x0)[:2 * n_obs]
    result = least_squares(residuals, x0, jac_sparsity=sparsity, method="trf", x_scale="jac",
                           loss=loss, f_scale=f_scale, max_nfev=max_nfev, ftol=1e-6, xtol=1e-10,
                           verbose=verbose)
    cams, pts = unpack(result.x)
    r1 = result.fun[:2 * n_obs]
    info = {
        "initial_rms": float(np.sqrt(np.mean(r0 ** 2))),
        "final_rms": float(np.sqrt(np.mean(r1 ** 2))),
        "nfev": int(result.nfev),
    }
    return cams, pts, info

import numpy as np


def umeyama(src, dst):
    """Similarity (s, R, t) minimizing |dst - (s R src + t)| for (N, 3) point sets."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / np.mean(np.sum(xs ** 2, axis=1))
    return s, R, mu_d - s * R @ mu_s


def rotation_angle_deg(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def compare_poses(est_poses, gt_poses):
    """Align estimated world-to-camera poses to ground truth with a similarity transform.

    est_poses, gt_poses: dicts key -> (R, t). Returns per-camera center errors (in ground-truth
    units), rotation errors (degrees) and the similarity (s, R, t) mapping estimate -> truth.
    """
    keys = [k for k in est_poses if k in gt_poses]
    est_c = np.array([(-R.T @ np.reshape(t, (3, 1))).ravel() for R, t in (est_poses[k] for k in keys)])
    gt_c = np.array([(-R.T @ np.reshape(t, (3, 1))).ravel() for R, t in (gt_poses[k] for k in keys)])
    s, R_align, t_align = umeyama(est_c, gt_c)
    aligned = (s * (R_align @ est_c.T)).T + t_align
    center_err = np.linalg.norm(aligned - gt_c, axis=1)
    rot_err = np.array([rotation_angle_deg(est_poses[k][0] @ R_align.T @ gt_poses[k][0].T) for k in keys])
    return center_err, rot_err, (s, R_align, t_align)

"""Synthetic scenes with known geometry, used to test the pipeline without photographs."""
import cv2 as cv
import numpy as np

from features import ViewFeatures

IMAGE_SIZE = (1000, 750)
K = np.array([[800.0, 0.0, 499.5], [0.0, 800.0, 374.5], [0.0, 0.0, 1.0]])
DIST = np.array([-0.08, 0.02, 0.0, 0.0, 0.0])


def look_at(center, target=np.zeros(3), down=np.array([0.0, 1.0, 0.0])):
    z = target - center
    z /= np.linalg.norm(z)
    x = np.cross(down, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack((x, y, z))
    return R, (-R @ center).reshape(3, 1)


def circle_cameras(angles_deg, radius=3.0, height=-0.5):
    poses = []
    for a in np.radians(angles_deg):
        center = np.array([radius * np.sin(a), height, -radius * np.cos(a)])
        poses.append(look_at(center))
    return poses


def sphere_points(n, rng, radius=1.0):
    """Points on a bumpy sphere with outward normals (for self-occlusion)."""
    normals = rng.normal(size=(n, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    r = radius * (1.0 + 0.1 * rng.uniform(-1, 1, size=(n, 1)))
    return normals * r, normals


def make_views(points, normals, poses, rng, noise_px=0.5, n_clutter=200, dist=DIST,
               descriptor_noise=0.05):
    """Project points into each camera and build fake SIFT-like features.

    Each 3D point owns a random descriptor; every observation gets a slightly perturbed copy.
    Returns the views and, for each view, the 3D point index of every keypoint (-1 for clutter).
    """
    base_desc = rng.uniform(0, 1, size=(len(points), 128)).astype(np.float32)
    base_desc *= 512 / np.linalg.norm(base_desc, axis=1, keepdims=True)
    w, h = IMAGE_SIZE
    views, gt_ids = [], []
    for v, (R, t) in enumerate(poses):
        center = (-R.T @ t).ravel()
        cam = points @ R.T + t.ravel()
        facing = np.sum(normals * (center - points), axis=1) > 0
        rvec, _ = cv.Rodrigues(R)
        uv, _ = cv.projectPoints(points, rvec, t, K, dist)
        uv = uv.reshape(-1, 2) + rng.normal(scale=noise_px, size=(len(points), 2))
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        visible = np.flatnonzero(facing & (cam[:, 2] > 0) & inside)

        desc = base_desc[visible] * (1 + descriptor_noise * rng.normal(size=(len(visible), 128)))
        clutter_uv = rng.uniform([0, 0], [w, h], size=(n_clutter, 2))
        clutter_desc = rng.uniform(0, 1, size=(n_clutter, 128))
        clutter_desc *= 512 / np.linalg.norm(clutter_desc, axis=1, keepdims=True)

        kps = np.vstack((uv[visible], clutter_uv))
        ids = np.concatenate((visible, -np.ones(n_clutter, dtype=int)))
        order = rng.permutation(len(kps))
        colors = np.tile(np.array([[200, 120, 40]], dtype=np.uint8), (len(kps), 1))
        views.append(ViewFeatures(
            name=f"view{v}", keypoints=kps[order],
            descriptors=np.vstack((desc, clutter_desc)).astype(np.float32)[order],
            colors=colors, image_size=IMAGE_SIZE))
        gt_ids.append(ids[order])
    return views, gt_ids


def umeyama(src, dst):
    """Similarity (s, R, t) minimizing |dst - (s R src + t)|."""
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

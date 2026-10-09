import os
from dataclasses import dataclass

import cv2 as cv
import numpy as np

from cam_calibration import read_image


@dataclass
class ViewFeatures:
    name: str
    keypoints: np.ndarray    # (N, 2) pixel coordinates in the processing resolution, still distorted
    descriptors: np.ndarray  # (N, D) float32
    colors: np.ndarray       # (N, 3) uint8 RGB sampled at each keypoint
    image_size: tuple        # (width, height) of the processing resolution
    scale: float = 1.0       # processing resolution / original resolution
    path: str = None


def detect_sift(gray, max_features=0, contrast_threshold=0.04):
    sift = cv.SIFT_create(nfeatures=max_features, contrastThreshold=contrast_threshold)
    kp, desc = sift.detectAndCompute(gray, None)
    return kp, desc


def detect_orb(gray):
    orb = cv.ORB_create()
    kp, desc = orb.detectAndCompute(gray, None)
    return kp, desc


def sample_colors(image_bgr, points):
    h, w = image_bgr.shape[:2]
    x = np.clip(np.round(points[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.round(points[:, 1]).astype(int), 0, h - 1)
    return image_bgr[y, x][:, ::-1].copy()


def extract_features(image_path, max_size=2000, max_features=10000, contrast_threshold=0.04):
    """SIFT features of one image, optionally downscaled so that max(width, height) <= max_size.

    Lower contrast_threshold (e.g. 0.01) yields more keypoints on weakly textured objects.
    """
    image = read_image(image_path)
    h, w = image.shape[:2]
    scale = 1.0
    if max_size and max(h, w) > max_size:
        scale = max_size / max(h, w)
        image = cv.resize(image, (round(w * scale), round(h * scale)), interpolation=cv.INTER_AREA)

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    kp, desc = detect_sift(gray, max_features, contrast_threshold)
    if desc is None:
        kp, desc = [], np.zeros((0, 128), np.float32)
    points = np.array([k.pt for k in kp], dtype=np.float64).reshape(-1, 2)

    return ViewFeatures(
        name=os.path.basename(image_path),
        keypoints=points,
        descriptors=desc.astype(np.float32),
        colors=sample_colors(image, points),
        image_size=(image.shape[1], image.shape[0]),
        scale=image.shape[1] / w,
        path=image_path,
    )

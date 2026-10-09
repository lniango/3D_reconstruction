import glob
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

IMAGE_EXTENSIONS = ("*.JPG", "*.jpg", "*.JPEG", "*.jpeg", "*.png", "*.PNG")


def read_image(path, flags=cv2.IMREAD_COLOR):
    """Read an image in raw sensor orientation.

    Phone pictures carry an EXIF rotation flag that cv2.imread applies by default, so a
    portrait photo comes back with swapped width/height. Intrinsics are tied to the sensor,
    so calibration and reconstruction images must be read without that rotation.
    """
    img = cv2.imread(path, flags | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def list_images(folder_path):
    files = set()
    for ext in IMAGE_EXTENSIONS:
        files.update(glob.glob(os.path.join(folder_path, ext)))
    return sorted(files)


@dataclass
class Calibration:
    K: np.ndarray
    dist: np.ndarray
    image_size: tuple  # (width, height)
    rms: float
    per_view_errors: dict = field(default_factory=dict)


def _per_view_errors(obj_points, img_points, K, dist, rvecs, tvecs):
    errors = []
    for obj, img, rvec, tvec in zip(obj_points, img_points, rvecs, tvecs):
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        diff = proj.reshape(-1, 2) - img.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))))
    return np.array(errors)


def _find_corners(gray, chessboard_dim, criteria, detection_size=1000):
    """Detect on a downscaled copy (full-resolution detection is slow), refine at full resolution."""
    scale = min(1.0, detection_size / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    found, corners = cv2.findChessboardCorners(
        small, chessboard_dim, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        return False, None
    corners = (corners.reshape(-1, 1, 2) / scale).astype(np.float32)
    win = max(5, int(round(5 / scale)))
    corners = cv2.cornerSubPix(gray, corners, (win, win), (-1, -1), criteria)
    return True, corners


def calibrate(folder_path, chessboard_dim=(8, 6), square_size=24.0, flags=cv2.CALIB_FIX_K3,
              max_view_error=1.0, show=False):
    """Calibrate a camera from chessboard pictures.

    chessboard_dim: inner corners (cols, rows). square_size: in millimeters.
    Views whose RMS reprojection error exceeds max(max_view_error, 2 * median) are
    discarded and the calibration is run again.
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    obj_3d = np.zeros((chessboard_dim[0] * chessboard_dim[1], 3), np.float32)
    obj_3d[:, :2] = np.mgrid[0:chessboard_dim[0], 0:chessboard_dim[1]].T.reshape(-1, 2)
    obj_3d *= square_size

    image_files = list_images(folder_path)
    print(f"Calibration images found: {len(image_files)}")

    obj_points, img_points, names = [], [], []
    image_size = None
    for path in image_files:
        gray = read_image(path, cv2.IMREAD_GRAYSCALE)
        size = gray.shape[::-1]
        if image_size is None:
            image_size = size
        elif size != image_size:
            raise ValueError(f"{path} has size {size}, expected {image_size}: all calibration "
                             "images must share the same resolution and orientation")

        found, corners = _find_corners(gray, chessboard_dim, criteria)
        if not found:
            print(f"  chessboard not found: {os.path.basename(path)}")
            continue
        obj_points.append(obj_3d)
        img_points.append(corners)
        names.append(os.path.basename(path))

        if show:
            vis = cv2.drawChessboardCorners(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
                                            chessboard_dim, corners, found)
            cv2.imshow("Chessboard", cv2.resize(vis, None, fx=0.25, fy=0.25))
            cv2.waitKey(50)
    if show:
        cv2.destroyAllWindows()

    if len(obj_points) < 3:
        raise ValueError(f"Only {len(obj_points)} chessboards detected in {folder_path}, need >= 3")

    keep = np.ones(len(obj_points), dtype=bool)
    for rejection_round in range(4):
        kept_idx = np.flatnonzero(keep)
        obj = [obj_points[i] for i in kept_idx]
        img = [img_points[i] for i in kept_idx]
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, image_size, None, None, flags=flags)
        errors = _per_view_errors(obj, img, K, dist, rvecs, tvecs)
        bad = errors > max(max_view_error, 2.0 * np.median(errors))
        if rejection_round == 3 or not bad.any() or len(kept_idx) - bad.sum() < 3:
            break
        for j in np.flatnonzero(bad):
            print(f"  rejecting {names[kept_idx[j]]} (error {errors[j]:.2f} px)")
        keep[kept_idx[bad]] = False

    kept_names = [n for n, k in zip(names, keep) if k]
    print(f"Calibrated on {len(kept_names)} views, RMS reprojection error = {rms:.3f} px")
    return Calibration(K=K, dist=dist.ravel(), image_size=tuple(int(v) for v in image_size),
                       rms=float(rms), per_view_errors=dict(zip(kept_names, errors.tolist())))


def save_calibration(calib, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, camMatrix=calib.K, distCoeff=calib.dist, imageSize=np.array(calib.image_size),
             repError=calib.rms)


def load_calibration(path):
    """Return a Calibration, or None if the file is missing or predates the image size field."""
    if not os.path.exists(path):
        return None
    data = np.load(path)
    if "imageSize" not in data:
        return None
    return Calibration(K=data["camMatrix"], dist=data["distCoeff"].ravel(),
                       image_size=tuple(int(v) for v in data["imageSize"]),
                       rms=float(data["repError"]))


def load_or_calibrate(calib_file, folder_path=None, **kwargs):
    calib = load_calibration(calib_file)
    if calib is not None:
        print(f"Loaded calibration from {calib_file} (RMS {calib.rms:.3f} px, size {calib.image_size})")
        return calib
    if folder_path is None:
        raise ValueError(f"No usable calibration in {calib_file} and no chessboard folder given")
    calib = calibrate(folder_path, **kwargs)
    save_calibration(calib, calib_file)
    print(f"Calibration saved to {calib_file}")
    return calib


def scale_intrinsics(K, scale):
    """Intrinsics of the same camera after resizing the image by `scale`."""
    S = np.diag([scale, scale, 1.0])
    K_scaled = S @ K
    # Pixel centers: x' = (x + 0.5) * s - 0.5
    K_scaled[0, 2] = (K[0, 2] + 0.5) * scale - 0.5
    K_scaled[1, 2] = (K[1, 2] + 0.5) * scale - 0.5
    return K_scaled


def intrinsics_for_image(calib, image_size):
    """K for an image of `image_size` (width, height) taken with the calibrated camera."""
    cw, ch = calib.image_size
    w, h = image_size
    if (w, h) == (cw, ch):
        return calib.K.copy()
    if (w, h) == (ch, cw):
        raise ValueError(f"Image size {image_size} is the calibration size {calib.image_size} rotated "
                         "by 90 degrees; read images with read_image() to ignore EXIF rotation")
    sx, sy = w / cw, h / ch
    if abs(sx - sy) > 1e-3:
        raise ValueError(f"Image size {image_size} does not have the calibration aspect ratio "
                         f"{calib.image_size}: images were cropped, intrinsics are unknown")
    return scale_intrinsics(calib.K, sx)

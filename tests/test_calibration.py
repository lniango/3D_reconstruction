import cv2
import numpy as np
import pytest

from cam_calibration import Calibration, calibrate, intrinsics_for_image, load_calibration, save_calibration

K_TRUE = np.array([[1500.0, 0.0, 799.5], [0.0, 1500.0, 599.5], [0.0, 0.0, 1.0]])
SIZE = (1600, 1200)


def _board_texture(cols=9, rows=7, px=100, margin=100):
    board = np.full((rows * px + 2 * margin, cols * px + 2 * margin), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                y, x = margin + r * px, margin + c * px
                board[y:y + px, x:x + px] = 0
    return board, px, margin


def _render_boards(folder, n=12, square_size=24.0):
    rng = np.random.default_rng(0)
    board, px, margin = _board_texture()
    mm_per_px = square_size / px
    # Texture pixel -> board plane (mm); the first inner corner sits at the origin.
    T = np.array([[mm_per_px, 0, -(margin + px) * mm_per_px],
                  [0, mm_per_px, -(margin + px) * mm_per_px],
                  [0, 0, 1]])
    for i in range(n):
        rvec = rng.uniform(-0.4, 0.4, size=3)
        R, _ = cv2.Rodrigues(rvec)
        t = np.array([-80.0, -60.0, 450.0]) + rng.uniform(-40, 40, size=3)
        H = K_TRUE @ np.column_stack((R[:, 0], R[:, 1], t)) @ T
        img = cv2.warpPerspective(board, H, SIZE, borderValue=128)
        cv2.imwrite(str(folder / f"board_{i:02d}.png"), img)


def test_calibration_recovers_intrinsics(tmp_path):
    _render_boards(tmp_path)
    calib = calibrate(str(tmp_path), chessboard_dim=(8, 6), square_size=24.0)
    assert calib.image_size == SIZE
    assert calib.rms < 0.5
    np.testing.assert_allclose(calib.K[0, 0], 1500.0, rtol=0.01)
    np.testing.assert_allclose(calib.K[:2, 2], [799.5, 599.5], atol=10)

    path = tmp_path / "calib.npz"
    save_calibration(calib, str(path))
    loaded = load_calibration(str(path))
    np.testing.assert_allclose(loaded.K, calib.K)
    assert loaded.image_size == SIZE


def test_intrinsics_for_image():
    calib = Calibration(K=K_TRUE, dist=np.zeros(5), image_size=SIZE, rms=0.1)
    np.testing.assert_allclose(intrinsics_for_image(calib, SIZE), K_TRUE)
    K_half = intrinsics_for_image(calib, (800, 600))
    np.testing.assert_allclose(K_half[0, 0], 750.0)
    np.testing.assert_allclose(K_half[:2, 2], [399.5, 299.5])
    with pytest.raises(ValueError, match="rotated"):
        intrinsics_for_image(calib, (1200, 1600))
    with pytest.raises(ValueError, match="aspect ratio"):
        intrinsics_for_image(calib, (1600, 900))

# 3D Structure from Motion (SfM)

A Python pipeline for incremental multi-view 3D reconstruction from calibrated images, using
classical computer vision (OpenCV, SciPy, Open3D).

---

## Pipeline

```
Chessboard images ─► Calibration (K, distortion)
Input images ─► SIFT ─► Undistorted keypoints
            ─► Pairwise matching (FLANN, ratio test, mutual check)
            ─► Geometric verification (essential matrix + RANSAC)
            ─► Initial pair (inliers, homography ratio, triangulation angle)
            ─► Two-view triangulation + bundle adjustment
            ─► loop: register next view (PnP RANSAC + LM refinement)
                     extend tracks, triangulate new points
                     global bundle adjustment every N views, outlier filtering
            ─► Sparse colored point cloud (.ply) ─► optional Poisson mesh / COLMAP export
```

Key points:

- **Images are read in sensor orientation** (`IMREAD_IGNORE_ORIENTATION`). Phone photos carry an
  EXIF rotation flag; applying it swaps width/height and makes the calibrated `K` wrong.
- **Lens distortion is removed from keypoints** (`cv.undistortPoints`), so all geometry uses an
  ideal pinhole camera with the calibrated `K`.
- **Tracks**: each 3D point stores its observations `{image: keypoint}`, and each image maps its
  keypoints to 3D points. This drives 2D-3D correspondences for PnP and the bundle adjustment.
- **Initial pair**: chosen among pairs with many verified matches, rejecting near-pure rotations
  and planar scenes (homography inlier ratio > 0.8) and preferring a median triangulation angle >= 3°.
- **Point filtering**: positive depth in every observing camera, reprojection error < 4 px,
  triangulation angle >= 1.5°.
- **Bundle adjustment**: all registered cameras and points, sparse Jacobian
  (`scipy.optimize.least_squares` with `jac_sparsity`), robust `soft_l1` loss. The first camera is
  fixed and the distance between the two initial cameras is constrained, which removes the
  7-DoF similarity gauge freedom.

---

## Installation

```bash
conda env create -f environment.yml
conda activate 3d_geometry
# or: pip install -r requirements.txt
```

On headless Linux Open3D also needs `libegl1 libgl1 libgomp1`.

---

## Usage

```bash
# Reconstruct from a folder of images
python main.py --images-dir for_reconstruction3 --mesh --show

# Or from explicit images
python main.py b1.JPG b2.JPG
```

The calibration in `calibrated_data/calibrated_data.npz` is reused when present; otherwise the
chessboard pictures of `--calib-dir` (default `Calibration2`) are used and the result is saved.

| Option | Default | Description |
|---|---|---|
| `--images-dir` | | Folder of input images (all with the same resolution) |
| `--calib-file` | `calibrated_data/calibrated_data.npz` | Calibration to load or create |
| `--calib-dir` | `Calibration2` | Chessboard pictures |
| `--board COLS ROWS` | `8 6` | Inner chessboard corners |
| `--square-size` | `24` | Square size (mm) |
| `--recalibrate` | | Ignore the existing calibration file |
| `--max-size` | `2000` | Downscale images for processing (`0` = full resolution) |
| `--min-angle` | `1.5` | Minimum triangulation angle (degrees) |
| `--ba-every` | `3` | Global bundle adjustment every N registered images |
| `--out` | `output` | Output folder |
| `--mesh` | | Build a Poisson mesh (`mesh.ply`) |
| `--colmap` | | Export a COLMAP text model (`colmap/`) |
| `--show` | | Open the Open3D viewer with the cameras |

Outputs: `output/sparse.ply` (colored sparse point cloud), `output/mesh.ply`, `output/colmap/`.

### Capturing images

- Lock focus and exposure (AE/AF lock): autofocus changes the focal length between shots.
- Move the camera around the object (translation, not just rotation), ~10-20° between
  consecutive views, with large overlap. Pure rotations or distant scenes give tiny
  triangulation angles: the depth of such points is undetermined and they are filtered out.
- Use the same resolution and camera as for the calibration.

The reconstruction datasets (`for_reconstruction*/`) are not versioned because of their size.

### Dense reconstruction

The sparse SfM cloud is too sparse for a detailed mesh. Export the model with `--colmap` and
densify it, e.g. with COLMAP:

```bash
colmap image_undistorter --image_path for_reconstruction3 --input_path output/colmap \
    --output_path output/dense
colmap patch_match_stereo --workspace_path output/dense
colmap stereo_fusion --workspace_path output/dense --output_path output/dense/fused.ply
```

COLMAP reads text models from a folder containing `cameras.txt`, `images.txt` and `points3D.txt`.

---

## Calibration (`cam_calibration.py`)

- Chessboard corners are detected on a downscaled copy and refined at full resolution.
- `square_size` scales the board, all images must share one size and orientation.
- Views whose error exceeds `max(1 px, 2 × median)` are discarded.
- Default model fixes `k3` (`CALIB_FIX_K3`): with `Calibration2` the 5-coefficient model gives
  `k3 ≈ 1.1`, which over-fits and moves image corners by ~150 px compared to the 4-coefficient model.

Current `Calibration2` result: RMS ≈ 1.96 px over 30 views, uniform across views, which points to
autofocus changes or a non-flat board rather than a few bad pictures.

---

## Project structure

```
.
├── main.py               # Command line entry point
├── sfm.py                # Incremental SfM (tracks, initialization, registration, filtering)
├── bundle_adjustment.py  # Sparse multi-camera bundle adjustment
├── cam_calibration.py    # Chessboard calibration, intrinsics handling, image reading
├── features.py           # SIFT extraction (ViewFeatures)
├── matching.py           # FLANN matching, ratio test, mutual check
├── pose.py               # Undistortion, essential matrix, relative pose
├── triangulation.py      # Triangulation, reprojection, parallax and cheirality filters
├── pointcloud.py         # Open3D point cloud helpers
├── mesh.py               # Poisson meshing
├── export.py             # COLMAP text export
├── help.py               # Grid drawing helper (undistortion check)
└── tests/                # Synthetic tests (known cameras and scene)
```

## Tests

```bash
python -m pytest tests
```

The tests build synthetic scenes (known cameras with lens distortion, noisy keypoints, clutter,
SIFT-like descriptors) and check calibration, relative pose, triangulation filters, bundle
adjustment convergence and gauge, the full incremental pipeline (camera poses within 2% of the
orbit radius after similarity alignment), the choice of the initial pair, the effect of
distortion, and the COLMAP export.

---

## References

- [SfM tutorial — CMSC426](https://cmsc426.github.io/sfm/)
- [Schönberger & Frahm, Structure-from-Motion Revisited (COLMAP), CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/papers/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.pdf)
- [Large-scale bundle adjustment in SciPy](https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html)
- [Camera calibration — OpenCV docs](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)

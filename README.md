# 3D Structure from Motion (SfM)

A Python pipeline for 3D reconstruction from 2D images using classical computer vision techniques. The pipeline covers camera calibration, keypoint detection and matching, essential matrix estimation, pose recovery, triangulation, and 3D point cloud visualization.

---

## Pipeline Overview

```
Images → Camera Calibration → Undistortion → Keypoint Matching → 
Essential Matrix → Pose Recovery → Triangulation → 3D Point Cloud
```

---

## Features

- **Camera calibration** using a chessboard pattern (OpenCV)
- **Distortion correction** with optional optimal camera matrix
- **Keypoint detection** with ORB and SIFT descriptors
- **Feature matching** using Brute-Force matcher with Lowe's ratio test
- **Essential matrix** estimation with RANSAC
- **Camera pose recovery** (R, t) from matched points
- **3D triangulation** and point cloud generation
- **Point cloud visualization** and export (`.ply`) with Open3D

---

## Project Structure

```
.
├── main.py               # Main pipeline
├── cam_calibration.py    # Camera calibration module
├── help.py               # Utility functions (grid overlay)
├── environment.yml       # Conda environment
└── README.md
```

---

## Installation

```bash
conda env create -f environment.yml
conda activate 3d_geometry
```

---

## Usage

### Calibrate camera only
```bash
python main.py -c
```

### Calibrate + apply optimal undistortion matrix
```bash
python main.py -c -o
```

### Run without calibration (uses approximate focal length)
```bash
python main.py
```

---

## Configuration

Before running, update the hardcoded paths in `main.py` and `cam_calibration.py` to match your local setup:

```python
# In main.py
path1 = "/path/to/your/image1.JPG"
path2 = "/path/to/your/image2.JPG"

# In cam_calibration.py
image_files = glob.glob("/path/to/calibration/images/*.JPG")
param_path  = "/path/to/save/calibrated_data.npz"
```

### Chessboard settings (`cam_calibration.py`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `Ch_Dim`  | (8, 6)  | Inner corners (cols, rows) |
| `Sq_size` | 24 mm   | Physical square size |

---

## Output

| File | Description |
|------|-------------|
| `calibrated_data.npz` | Camera matrix, distortion coefficients, R/T vectors |
| `undistort_img1.jpg`  | Undistorted image |
| `Original_grid.jpg`   | Original image with grid overlay |
| `Undistorted_grid.jpg`| Undistorted image with grid overlay |
| `output.ply`          | 3D point cloud (Open3D / MeshLab compatible) |

---

## Dependencies

Key libraries (see `environment.yml` for full list):

- [OpenCV](https://opencv.org/) — image processing, calibration, feature detection
- [Open3D](http://www.open3d.org/) — 3D point cloud processing and visualization
- [NumPy](https://numpy.org/) — matrix operations

---

## References

- [SfM tutorial — CMSC426](https://cmsc426.github.io/sfm/)
- [Camera calibration — OpenCV docs](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [SIFT descriptor — GeeksforGeeks](https://www.geeksforgeeks.org/sift-interest-point-detector-using-python-opencv/)
- [ORB feature descriptor](https://github.com/ImranNawar/orb_feature_descriptor)
- [3D SfM — ekrrems](https://github.com/ekrrems/3D-Structure-from-Motion)

import os

import numpy as np


def create_pointcloud(points, colors=None, normals=None):
    """Open3D point cloud from (N, 3) points and optional (N, 3) uint8 RGB colors."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None and len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64) / 255.0)
    if normals is not None and len(normals) == len(points):
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=np.float64))
    return pcd


def clean_pointcloud(pcd, nb_neighbors=20, std_ratio=2.0):
    """Statistical outlier removal. Returns the filtered cloud and the kept indices; colors and
    normals stay aligned because Open3D filters them together with the points."""
    if len(pcd.points) <= nb_neighbors:
        return pcd, np.arange(len(pcd.points))
    filtered, kept = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return filtered, np.asarray(kept)


def save_geometry(geometry, path):
    import open3d as o3d

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if isinstance(geometry, o3d.geometry.TriangleMesh):
        o3d.io.write_triangle_mesh(path, geometry)
    else:
        o3d.io.write_point_cloud(path, geometry)
    print(f"Saved {path}")


def show(*geometries):
    import open3d as o3d

    o3d.visualization.draw_geometries(list(geometries))

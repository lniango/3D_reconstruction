import open3d as o3d
import numpy as np


def create_pointcloud(points_3d):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d.T)

    return pcd


def clean_pointcloud(pcd, pts1, pts2, img):
    pcd, ind = pcd.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=2.0
    )
    
    #Color
    colors = []
    for pt in pts1:
        x, y = int(pt[0]), int(pt[1])
        colors.append(img[y, x] / 255.0)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    #Save sparse Point cloud
    o3d.io.write_point_cloud("point_cloud/output.ply", pcd)

    o3d.visualization.draw_geometries([pcd])

    return pcd
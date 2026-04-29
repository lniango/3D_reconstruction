import open3d as o3d
import numpy as np


def create_pointcloud(points_3d, save_path="point_cloud/output_pc.ply"):
    pcd = o3d.geometry.PointCloud()
    
    if points_3d is None or len(points_3d) == 0:
        print("Warning: No points to create pointcloud")
        return pcd
    
    # Convert to numpy array 
    points_3d = np.asarray(points_3d)
    
    # Verify the shape
    print(f"Shape before fixing: {points_3d.shape}")
    
    # Case 1:  3xN, transpose to Nx3
    if points_3d.shape[0] == 3 and len(points_3d.shape) == 2:
        points_3d = points_3d.T
    
    # Case 2:  1xNx3 or another strange format
    if len(points_3d.shape) == 3:
        points_3d = points_3d.reshape(-1, 3)
    
    # Case 3: A 1D vector
    if len(points_3d.shape) == 1:
        points_3d = points_3d.reshape(-1, 3)
    
    # Case 4: float64
    points_3d = points_3d.astype(np.float64)
    
    
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    
    # Display point cloud
    o3d.io.write_point_cloud(save_path, pcd)

    o3d.visualization.draw_geometries([pcd])

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
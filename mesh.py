import numpy as np


def poisson_mesh(pcd, view_points=None, depth=8, density_quantile=0.05):
    """Poisson surface from a point cloud.

    Reconstruction scale is arbitrary, so the normal estimation radius is derived from the
    point spacing. If view_points (N, 3) is given (e.g. the center of a camera observing each
    point), normals are oriented towards it; otherwise a consistent tangent-plane orientation
    is used. Low-density vertices, i.e. surface extrapolated far from any point, are removed.

    A sparse SfM cloud gives a coarse surface; densify it with an MVS tool for a detailed mesh.
    """
    import open3d as o3d

    spacing = float(np.mean(pcd.compute_nearest_neighbor_distance()))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5 * spacing, max_nn=30))
    if view_points is not None:
        normals = np.asarray(pcd.normals)
        to_view = np.asarray(view_points) - np.asarray(pcd.points)
        flip = np.sum(normals * to_view, axis=1) < 0
        normals[flip] *= -1
        pcd.normals = o3d.utility.Vector3dVector(normals)
    else:
        pcd.orient_normals_consistent_tangent_plane(15)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    densities = np.asarray(densities)
    mesh.remove_vertices_by_mask(densities < np.quantile(densities, density_quantile))
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox = bbox.scale(1.1, bbox.get_center())
    mesh = mesh.crop(bbox)
    mesh.compute_vertex_normals()
    return mesh

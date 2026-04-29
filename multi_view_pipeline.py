import cv2 as cv
import open3d as o3d
import numpy as np
from PIL import Image

from cam_calibration import calib
from features import detect_sift
from matching import match_sift, extract_points
from pose import estimate_pose
from triangulation import triangulate_points
from pointcloud import create_pointcloud, clean_pointcloud
from mesh import poisson_mesh

# Data structure for view
class View:
    def __init__(self, image_path, K):
        self.image_path = image_path
        self.image = cv.imread(image_path)
        self.gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        self.kp, self.desc = detect_sift(self.gray)
        self.R = None
        self.t = None
        self.K = K
        self.points3D = []  # Indexes of visible 3D points 
        self.point_indices = []  # Correspondance avec points 3D
        

def sfm(image_paths, K):
    # Initialization with best views
    views = [View((path, K)) for path in image_paths]
    
    # Step 1: Find first pair with high number of correspondence
    best_pair = None
    best_inliers = 0
    
    for i in range(len(views)):
        for j in range(i+1, len(views)):
            matches = match_sift(views[i].desc, views[j].desc)
            pts1, pts2 = extract_points(matches, views[i].kp, views[j].kp)
            
            # Estimate epipolar geometry between the 2 cameras
            E, mask = cv.findEssentialMat(pts1, pts2, K, cv.RANSAC, prob=0.999, threshold=1.0)
            R, t, _, mask_pose = cv.recoverPose(E, pts1, pts2, K)
            
            # Count the number of 3D points we can reconstruct
            inliers = np.sum(mask_pose)
            if inliers > best_inliers:
                best_inliers = inliers
                best_pair = (i, j, R, t, mask_pose, pts1, pts2)
                
            # Step 2 : Initialize reconstruction with best pairs
            i, j, R_init, t_init, mask, pts1, pts2 = best_pair
            
            # Update poses
            views[i].R = np.eye(3)
            views[i].t = np.zeros((3, 1))
            views[j].R = R_init
            views[j].t = t_init
            
            # triangulate initial points
            points3D = triangulate_points(pts1, pts2, K, R_init, t_init)
            
            # Step 3 : Add iteratively views left
            registered = {i, j}
            unregistered = set(range(len(views))) - registered
            
            while unregistered:
                # Find view with more correspondances with registered 
                best_view = None
                best_reconstruction = None
                best_points3D = None
        
                for idx in unregistered:
                    # Count correspondances 2D-3D
                    correspondances = []
                    for reg_idx in registered:
                        # Check similarities between registered img and new img
                        matches = match_sift(views[idx].desc, views[reg_idx].desc)
                        pts2D, pts3D = [], []
                
                        for match in matches[:100]:  # Limit for performance
                            pt2D = views[idx].kp[match.trainIdx].pt
                            
                            # Verify if 3D point exists
                            if match.queryIdx < len(views[reg_idx].point_indices):
                                pt3D_idx = views[reg_idx].point_indices[match.queryIdx]
                                if pt3D_idx < len(points3D):
                                    pts2D.append(pt2D)
                                    pts3D.append(points3D[pt3D_idx])
                        
                        # Keep image pairs with at least 20 matching 2D-3D
                        if len(pt2D) > 20:
                            correspondances.append((reg_idx, np.array(pts2D), np.array(pts3D)))
                
                if correspondances:
                    # Resolve PnP to estimate the pose
                    _, rvec, tvec, inliers = cv.solvePnPRansac(
                        correspondances[0][2], 
                        correspondances[0][1], 
                        K, None, 
                        iterationsCount=100
                        )
                
                    if inliers is not None and len(inliers) > 30:
                        best_view = idx
                        R, _ = cv.Rodrigues(rvec)
                        views[idx].R = R
                        views[idx].t = tvec
                        break
                
                if best_view is None:
                    break
                    
                # Triangulate new points with added view
                for reg_idx in registered:
                    matches = match_sift(views[best_view].desc, views[reg_idx].desc)
                    pts_new, pts_reg = extract_points(matches[:100], 
                                             views[best_view].kp, 
                                             views[reg_idx].kp)
            
                    # Triangulation
                    P1 = K @ np.hstack((views[best_view].R, views[best_view].t))
                    P2 = K @ np.hstack((views[reg_idx].R, views[reg_idx].t))
            
                    new_points_3d = cv.triangulatePoints(P1, P2, pts_new.T, pts_reg.T)
                    new_points_3d = (new_points_3d[:3] / new_points_3d[3]).T
            
                    # Filter points in front of cameras
                    depths1 = (views[best_view].R @ new_points_3d.T + views[best_view].t)[2]
                    depths2 = (views[reg_idx].R @ new_points_3d.T + views[reg_idx].t)[2]
                    valid = (depths1 > 0) & (depths2 > 0)
            
                    new_points_3d = new_points_3d[valid]
                
                    # Add to global 3D points
                    start_idx = len(points3D)
                    points3D = np.vstack([points3D, new_points_3d]) if len(points3D) > 0 else new_points_3d
            
                    # Update correspondences
                    for k, match in enumerate(np.array(matches)[:100][valid]):
                        views[best_view].point_indices.append(start_idx + k)
                        views[reg_idx].point_indices.append(start_idx + k)
        
                registered.add(best_view)
                unregistered.remove(best_view)
        
                # Bundle adjustment optionnel
                #if len(registered) % 3 == 0:
                #    points3D = bundle_adjustment(views, points3D, registered)
    
    return create_pointcloud(points3D)
            
            
def bundle_adjustment(views, points3D, registered_indices):
    """Simplified Implementation of bundle adjustment"""
    # use scipy.optimize.least_squares
    # or g2o (https://github.com/RainerKuemmerle/g2o)
    from scipy.optimize import least_squares
    
    def residuals(params):
        # À implémenter : paramètres = poses + points 3D
        # Retourner les erreurs de reprojection
        pass
    
    # Configuration des paramètres
    # ...
    
    return points3D
    

#Sources: https://github.com/Abhishek-Aditya-bs/MultiView-3D-Reconstruction/blob/main/Reports/Final%20Paper.pdf
'''def multi_view(images, K):
    all_points = [] 
    all_pcd = o3d.geometry.PointCloud()

    # Pose du référentiel monde : caméra 0 = identité
    R_world = np.eye(3)
    t_world = np.zeros((3, 1))
    
    for i in range(len(images) - 1):
        if i + 1 < len(images):
            #Convert images
            img1 = np.array(Image.open(images[i]))
            img2 = np.array(Image.open(images[i+1]))
            gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
            gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
            
            #Detect keypoints sift
            kp1, desc1 = detect_sift(gray1)
            kp2, desc2 = detect_sift(gray2)
            
            #match keypoints
            good_matches = match_sift(desc1, desc2)
            
            #extract points
            pts1, pts2 = extract_points(good_matches, kp1, kp2)
            
            #Estimate pose: local camera i -> i+1
            R, t, pts1, pts2 = estimate_pose(pts1, pts2, K) #K after calibration and optimalcammatx
            
            #Triangulation: 3D point in local space of (img i, img i+1)
            point3D = triangulate_points(pts1, pts2, K, R, t)
            
            #https://perso.univ-lemans.fr/~fvaret/opi/cours_maj/OPI_fr_M04_C01_web_gen_auroraW/co/Contenu03.html
            #Transformation towards world coordinates
            #print(f"R_world : {R_world.shape} - point3D : {point3D.shape} - t_word : {t_world.shape}")
            pts_world = (R_world @ point3D).T + t_world.T #182x1
            
            #Create point cloud and stack them in real world space
            pcd = create_pointcloud(pts_world.T)
            #Clean point cloud
            pcd = clean_pointcloud(pcd, pts1, pts2, img1)
            
            pts = np.asarray(pcd.points)
            all_points.append(pts)
            
            #update world pose for the next match (next 2 images)
            # t_world = R_world · t  +  t_world
            t_world = R_world @ t.reshape(3, 1) + t_world
            # R_world = R_world · R
            R_world = R_world @ R
    
    # final merge
    all_points = np.vstack(all_points)
    
    all_pcd = o3d.geometry.PointCloud()
    all_pcd.points = o3d.utility.Vector3dVector(all_points)
     
    o3d.visualization.draw_geometries([all_pcd])
    return all_pcd
'''

def multi_view(images, K):
    # Select initial pair 
    
    gray_images = [cv.cvtColor(cv.imread(img), cv.COLOR_BGR2GRAY) for img in images]
    kp_list = []
    desc_list = []
    
    for gray in gray_images:
        kp, desc = detect_sift(gray)
        kp_list.append(kp)
        desc_list.append(desc)
    
    # Trouver la meilleure paire initiale
    best_score = 0
    best_pair = None
    best_R, best_t = None, None
    best_pts1, best_pts2 = None, None
    
    for i in range(len(images)):
        for j in range(i+1, len(images)):
            matches = match_sift(desc_list[i], desc_list[j])
            if len(matches) < 50:
                continue
                
            pts1, pts2 = extract_points(matches, kp_list[i], kp_list[j])
            
            # Epipolar geometry
            E, mask = cv.findEssentialMat(pts1, pts2, K, cv.RANSAC, 0.999, 1.0)
            _, R, t, mask_pose = cv.recoverPose(E, pts1, pts2, K)
            
            score = np.sum(mask_pose)
            if score > best_score:
                best_score = score
                best_pair = (i, j)
                best_R, best_t = R, t
                best_pts1, best_pts2 = pts1[mask_pose.ravel() == 255], pts2[mask_pose.ravel() == 255]
    
    if best_pair is None:
        raise ValueError("Aucune paire d'images valide trouvée")
    
    # Initial Triangulation 
    points3D = triangulate_points(best_pts1, best_pts2, K, best_R, best_t)
    
    # Transformation to world frame
    pts_world = points3D.T  # 3D Points camera i frame
    
    pcd = create_pointcloud(pts_world)
    # Clean point cloud
    #pcd = clean_pointcloud(pcd, pts1, pts2, gray_images[0])
    return pcd
            
if __name__ == "__main__":
    #Choose camera to calibrate among all cameras
    '''images = ["for_reconstruction/IMG_4148.JPG", "for_reconstruction/IMG_4149.JPG",
              "for_reconstruction/IMG_4150.JPG", "for_reconstruction/IMG_4151.JPG",
              "for_reconstruction/IMG_4152.JPG", "for_reconstruction/IMG_4155.JPG",
              "for_reconstruction/IMG_4156.JPG", "for_reconstruction/IMG_4157.JPG",
              "for_reconstruction/IMG_4158.JPG"]'''
        
    images = ["for_reconstruction2/IMG_4159.JPG", "for_reconstruction2/IMG_4160.JPG",
                  "for_reconstruction2/IMG_4161.JPG", "for_reconstruction2/IMG_4162.JPG",
                  "for_reconstruction2/IMG_4163.JPG", "for_reconstruction2/IMG_4164.JPG",
                  "for_reconstruction2/IMG_4165.JPG", "for_reconstruction2/IMG_4166.JPG",
                  "for_reconstruction2/IMG_4167.JPG", "for_reconstruction2/IMG_4168.JPG",
                  "for_reconstruction2/IMG_4169.JPG", "for_reconstruction2/IMG_4170.JPG",
                  "for_reconstruction2/IMG_4171.JPG"]
    
    #Calibration: Choose a camera for calibration
    ret, K, dist = calib(folder_path="Calibration2")
    
    all_pcd = multi_view(images, K)
    mesh = poisson_mesh(all_pcd)
    
    mesh.paint_uniform_color([1, 0.706, 0]) #0.7, 0.7, 0.7
    mesh.compute_vertex_normals()

    o3d.visualization.draw_geometries([mesh]
                                  ) #mesh_show_back_face=True
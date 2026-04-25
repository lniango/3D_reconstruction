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

#Sources: https://github.com/Abhishek-Aditya-bs/MultiView-3D-Reconstruction/blob/main/Reports/Final%20Paper.pdf
def multi_view(images, K):
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
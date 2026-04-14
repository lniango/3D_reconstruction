#Ressources: https://cmsc426.github.io/sfm/
# https://learnopencv.com/mast3r-sfm-grounding-image-matching-3d/
# https://learnopencv.com/mast3r-sfm-grounding-image-matching-3d/
# https://github.com/zhangshi0512/cs5330_project06_landmark/
# https://cmsc426.github.io/math-tutorial/#svd
# https://github.com/topics/real-time-computer-vision
# https://github.com/ekrrems/3D-Structure-from-Motion
# https://github.com/topics/structure-from-motion?o=asc&s=forks
# https://imkaywu.github.io/tutorials/sfm/

import sys
import cv2 as cv
import numpy as np
import open3d as o3d
from cam_calibration import *
from help import draw_lines

'''3D reconstruction from 2 images'''
#Load images
img1 = cv.imread("/Users/kyo/Documents/projects/CVision/SfM/nz1.JPG")
img2 = cv.imread("/Users/kyo/Documents/projects/CVision/SfM/nz2.JPG")
#cv.imshow("Image 1", img1)
#cv.waitKey(0)
#cv.destroyAllWindows()

#Reshape images
'''h1, w1, c1 = img1.shape
h2, w2, c2 = img2.shape
h = min(h1, h2)
w = min(h1, h2)
c = min(c1, c2)
img1 = cv.resize(img1, (w, h), interpolation=cv.INTER_CUBIC)
img2 = cv.resize(img2, (w, h), interpolation=cv.INTER_CUBIC)'''

#gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
#gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

'''cv.imshow("Image 1", gray1)
cv.imshow("Image 2", gray2)
cv.waitKey(0)
cv.destroyAllWindows() '''

if sys.argv[1] == "calibrate" and sys.argv[2] == "opti_matrix":
    # Camera calibration using opencv
    # https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-3-1-homogeneous-transformation-matrices/
    # let's find K matrix
    ret, K, dist_coeff, R_vecs, T_vecs = calib(showPix=False) #mtx = K
    #verify quality
    print("Erreur reprojection :", ret) # retg -> 0

    #distorsion correction
    # https://csundergrad.science.uoit.ca/courses/cv-notes/notebooks/02-camera-calibration.html
    newimg = cv.imread("/Users/kyo/Documents/projects/CVision/SfM/nz1.JPG")
    h, w = newimg.shape[:2]
    
    ######################################################
    ##VERIFY UNDISTORTION##
    # Original Image 
    img_orig1 = img1.copy()
    ######################################################
    #With alpha = 1, trying to keep the whole content in the image
    #-> alpha = 0; zoom + crop to keep only valid pixels  
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(K, dist_coeff, (w, h), 0, (w, h))
    #using new camera matrix to undistort image
    img1_undist = cv.undistort(img1, K, dist_coeff, None, newcameramtx)
    img2_undist = cv.undistort(img2, K, dist_coeff, None, newcameramtx)
    
    # crop the image
    x, y, w, h = roi
    dst = dist_coeff[y:y+h, x:x+w]
    cv.imwrite("/Users/kyo/Documents/projects/CVision/SfM/calibrated_data/undistort_img1.jpg", img1_undist)
    cv.imwrite("/Users/kyo/Documents/projects/CVision/SfM/calibrated_data/undistort_img2.jpg", img2_undist)
    
    gray1_undist = cv.cvtColor(img1_undist, cv.COLOR_BGR2GRAY)
    gray2_undist = cv.cvtColor(img2_undist, cv.COLOR_BGR2GRAY)
    #using newcameramtx instead of K
    K = newcameramtx
    
    ######################################################
    ##VERIFY UNDISTORTION##
    # Dessiner les grilles
    img_orig_lines = draw_lines(img_orig1)
    img_undist_lines = draw_lines(img1_undist)
    
    cv.imwrite("/Users/kyo/Documents/projects/CVision/SfM/calibrated_data/Original_grid.jpg", img_orig_lines)
    cv.imwrite("/Users/kyo/Documents/projects/CVision/SfM/calibrated_data/Undistorted_grid.jpg", img_undist_lines)
    ######################################################


    
elif sys.argv[1] == "calibrate":
    # Camera calibration using opencv
    # https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-3-1-homogeneous-transformation-matrices/
    # let's find K matrix
    ret, K, dist_coeff, R_vecs, T_vecs = calib(showPix=False) #mtx = K
    #verify quality
    print("Erreur reprojection :", ret) # retg -> 0

    #distorsion correction
    # https://csundergrad.science.uoit.ca/courses/cv-notes/notebooks/02-camera-calibration.html
    img1_undist = cv.undistort(img1, K, dist_coeff)
    cv.imwrite("/Users/kyo/Documents/projects/CVision/SfM/calibrated_data/undistort_img.jpg", img1_undist)
    img2_undist = cv.undistort(img2, K, dist_coeff)

    gray1_undist = cv.cvtColor(img1_undist, cv.COLOR_BGR2GRAY)
    gray2_undist = cv.cvtColor(img2_undist, cv.COLOR_BGR2GRAY)
else:
    gray1_undist = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2_undist = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    
    #Essential matrix
    # Simple approximation
    f = 800  # focale approximation 
    cx, cy = img1.shape[1] / 2, img1.shape[0] / 2

    K = np.array([[f, 0, cx],
              [0, f, cy],
              [0, 0, 1]])



#Keypoints detection using ORB
'''How does it work: https://github.com/ImranNawar/orb_feature_descriptor
practical: https://github.com/ImranNawar/orb_feature_descriptor/blob/main/code.ipynb
ORB uses the FAST algorithm
for keypoint detection. FAST identifies keypoints by comparing pixel 
intensities in a circular pattern around each pixel.

For a pixel 𝑝 with intensity 𝐼(𝑝), if there exist N contiguous pixels 
in a circle around 𝑝 that are all brighter or darker than 𝐼(𝑝) by a certain 
threshold 𝑡, it's marked as a keypoint.
'''

orb = cv.ORB_create()
kp1, dsc1 = orb.detectAndCompute(gray1_undist, None)
kp2, dsc2 = orb.detectAndCompute(gray2_undist, None)

#Match the keypoints 
matcher = cv.BFMatcher()
matches = matcher.match(dsc1, dsc2)
print(f"Matches points OBR: {len(matches)}")
#Draw matches and display 
img_matches = cv.drawMatches(gray1_undist, kp1, gray2_undist, kp2, 
                             matches[:], None, matchColor=(0, 0, 255),)
img_matches = cv.resize(img_matches, (1920, 1920))

#cv.imshow("Matches image ORB", img_matches)
#cv.waitKey(0)
#cv.destroyAllWindows()
'''Analyze: OBR detector is too noisy'''

#Keypoints detection using SIFT descriptor
'''How does it work: 
https://www.geeksforgeeks.org/machine-learning/sift-interest-point-detector-using-python-opencv/
'''
sift = cv.SIFT_create()
kp1_sift, desc1_sift = sift.detectAndCompute(gray1_undist, None)
kp2_sift, desc2_sift = sift.detectAndCompute(gray2_undist, None)

#Match the keypoints 
matcher_sift = cv.BFMatcher()
#matches_sift = matcher_sift.match(desc1_sift, desc2_sift)
matcher_sift_knn = matcher_sift.knnMatch(desc1_sift, desc2_sift, k=2)
good_matches = []
for m, n in matcher_sift_knn:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print(len(good_matches))
#Draw matches and display 
img_matches_sift = cv.drawMatches(gray1_undist, kp1_sift, gray2_undist, kp2_sift, 
                             good_matches[:200], None, matchColor=(0, 0, 255),)
img_matches_sift = cv.resize(img_matches_sift, (1920, 1920))
print(f"Matches points SIFT: {len(matcher_sift_knn)}")

cv.imshow("image matching using SIFT", img_matches_sift)
cv.waitKey(0)
cv.destroyAllWindows()
'''Analyse: SIFT descriptor finds more keypoints'''

#Extract matches points from OBR
best1 = np.float32([kp1[m.queryIdx].pt for m in matches])
best2 = np.float32([kp2[m.trainIdx].pt for m in matches])

#Extract matches points from SIFT
#best1 = np.float32([kp1_sift[m.queryIdx].pt for m in matches_sift])
#best2 = np.float32([kp2_sift[m.trainIdx].pt for m in matches_sift])
#print(f"The points: {best1} // {best2}")

'''
#Essential matrix
# Simple approximation
f = 800  # focale approximation 
cx, cy = img1.shape[1] / 2, img1.shape[0] / 2

K = np.array([[f, 0, cx],
              [0, f, cy],
              [0, 0, 1]])

E, mask = cv.findEssentialMat(best1, best2, K, method=cv.RANSAC)

#Recover camero pose
_, R, t, mask = cv.recoverPose(E, best1, best2, K)'''
 #Essential matrix
E, mask = cv.findEssentialMat(best1, best2, K, method=cv.RANSAC)

#Recover camero pose
_, R, t, mask = cv.recoverPose(E, best1, best2, K)  #pose of cam1 wrt cam2

#Triangulation
# https://staff.fnwi.uva.nl/r.vandenboomgaard/ComputerVision/LectureNotes/CV/StereoVision/triangulation.html
# https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://arxiv.org/abs/2501.14277&ved=2ahUKEwiGmuCNo-KTAxUXTqQEHWbeG6kQFnoECBYQAQ&usg=AOvVaw3wAOMjtZDSFAamycdN-1dr
proj1 = np.hstack((np.eye(3), np.zeros((3,1))))
proj2 = np.hstack((R, t))

proj1 = K @ proj1 # proj1 = K @ [I | 0]
proj2 = K @ proj2 # proj1 = K @ [I | 0]

points_4d = cv.triangulatePoints(proj1, proj2, best1.T, best2.T)
points_3d = points_4d[:3] / points_4d[3]

#print(points_4d)
#3D visualization using Open3D 
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_3d.T)

o3d.visualization.draw_geometries([pcd])


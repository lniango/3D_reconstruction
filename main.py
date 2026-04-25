import cv2 as cv
import open3d as o3d

from cam_calibration import calib
from features import detect_sift
from matching import match_sift, extract_points
from pose import estimate_pose
from triangulation import triangulate_points
from pointcloud import create_pointcloud, clean_pointcloud
from mesh import poisson_mesh


img1 = cv.imread("ball1.JPG")
img2 = cv.imread("ball2.JPG")

gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

ret, K, dist = calib(folder_path="Calibration2")

#distorsion correction
# https://csundergrad.science.uoit.ca/courses/cv-notes/notebooks/02-camera-calibration.html
#newimg = cv.imread(path1)
h, w = img1.shape[:2]
    
newK, roi = cv.getOptimalNewCameraMatrix(
    K,
    dist,
    (w,h),
    0,
    (w,h)
)
#Undistort image
img_undist1 = cv.undistort(img1, K, dist, None, newK)
img_undist2 = cv.undistort(img2, K, dist, None, newK)

# crop the image
x, y, w, h = roi
dst = dist[y:y+h, x:x+w]
cv.imwrite("calibrated_data/undistort_img1.jpg", img_undist1)
cv.imwrite("calibrated_data/undistort_img2.jpg", img_undist2)
    
gray1_undist = cv.cvtColor(img_undist1, cv.COLOR_BGR2GRAY)
gray2_undist = cv.cvtColor(img_undist2, cv.COLOR_BGR2GRAY)
#using newcameramtx instead of K

kp1, desc1 = detect_sift(gray1_undist)
kp2, desc2 = detect_sift(gray2_undist)

matches = match_sift(desc1, desc2)

pts1, pts2 = extract_points(matches, kp1, kp2)

R, t, pts1, pts2 = estimate_pose(pts1, pts2, K)

points_3d = triangulate_points(pts1, pts2, K, R, t)

pcd = create_pointcloud(points_3d)
pcd = clean_pointcloud(pcd, pts1, pts2, img1)

mesh = poisson_mesh(pcd)

o3d.visualization.draw_geometries([mesh])
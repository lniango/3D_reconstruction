import cv2 as cv
import numpy as np

#Triangulation
# https://staff.fnwi.uva.nl/r.vandenboomgaard/ComputerVision/LectureNotes/CV/StereoVision/triangulation.html
# https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://arxiv.org/abs/2501.14277&ved=2ahUKEwiGmuCNo-KTAxUXTqQEHWbeG6kQFnoECBYQAQ&usg=AOvVaw3wAOMjtZDSFAamycdN-1dr
def triangulate_points(pts1, pts2, K, R, t):
    proj1 = np.hstack((np.eye(3), np.zeros((3,1))))
    proj2 = np.hstack((R, t))

    proj1 = K @ proj1
    proj2 = K @ proj2

    points_4d = cv.triangulatePoints(
        proj1,
        proj2,
        pts1.T,
        pts2.T
    )

    points_3d = points_4d[:3] / points_4d[3]

    return points_3d
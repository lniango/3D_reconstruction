import cv2 as cv
import numpy as np


def match_sift(desc1, desc2, ratio=0.75):
    matcher = cv.BFMatcher()
    matches_knn = matcher.knnMatch(desc1, desc2, k=2)
    good_matches = []

    for m, n in matches_knn:
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    return good_matches


def extract_points(matches, kp1, kp2):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    return pts1, pts2
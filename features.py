import cv2 as cv

'''Detect keypoints'''

def detect_sift(gray):
    sift = cv.SIFT_create()
    kp, desc = sift.detectAndCompute(gray, None)
    return kp, desc


def detect_orb(gray):
    orb = cv.ORB_create()
    kp, desc = orb.detectAndCompute(gray, None)
    return kp, desc
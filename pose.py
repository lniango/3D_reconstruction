import cv2 as cv


def estimate_pose(pts1, pts2, K):
     #Essential matrix
    E, mask = cv.findEssentialMat(
        pts1,
        pts2,
        K,
        method=cv.RANSAC
    )
    #Clean point cloud
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    _, R, t, mask_pose = cv.recoverPose(E, pts1, pts2, K)

    return R, t, pts1, pts2
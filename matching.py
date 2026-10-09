import cv2 as cv
import numpy as np

_FLANN_INDEX_KDTREE = 1


def _knn(desc_query, desc_train, k=2):
    if len(desc_query) < k or len(desc_train) < k:
        return []
    matcher = cv.FlannBasedMatcher(dict(algorithm=_FLANN_INDEX_KDTREE, trees=5), dict(checks=64))
    return matcher.knnMatch(desc_query, desc_train, k=k)


def match_descriptors(desc1, desc2, ratio=0.8, mutual=True):
    """Lowe ratio test, optionally keeping only mutual nearest neighbours.

    Returns an (M, 2) int array of (index in desc1, index in desc2).
    """
    forward = {}
    for pair in _knn(desc1, desc2):
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            forward[pair[0].queryIdx] = pair[0].trainIdx
    if not forward:
        return np.zeros((0, 2), dtype=int)

    if mutual:
        backward = {pair[0].queryIdx: pair[0].trainIdx for pair in _knn(desc2, desc1) if pair}
        forward = {i: j for i, j in forward.items() if backward.get(j) == i}

    return np.array(sorted(forward.items()), dtype=int).reshape(-1, 2)

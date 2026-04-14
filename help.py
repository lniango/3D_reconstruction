import cv2 as cv
import numpy as np

def draw_lines(img, step=50):
    img_lines = img.copy()
    h, w = img.shape[:2]

    # lignes verticales
    for x in range(0, w, step):
        cv.line(img_lines, (x, 0), (x, h), (0, 255, 0), 1)

    # lignes horizontales
    for y in range(0, h, step):
        cv.line(img_lines, (0, y), (w, y), (0, 255, 0), 1)

    return img_lines
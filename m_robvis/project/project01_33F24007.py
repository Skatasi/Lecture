'''
Student Number: 33F24007
Student Name: Seinosuke Sakata
'''
import numpy as np
import cv2 as cv

img = cv.imread('./image00.png', 0)    # Read out as a gray scale image
img = img.astype(float)
# f + f * g
g = np.array([[0, -1, 0],
              [-1, 5, -1],
              [0, -1, 0]], float)

h = cv.filter2D(img, -1, g)
h = (h - np.min(h)) / (np.max(h) - np.min(h)) * 255.0
cv.imshow('Input', img.astype(np.uint8))
cv.imshow('Filtered', h.astype(np.uint8))
cv.waitKey()
cv.destroyAllWindows()
'''
Student Number: 33F24007
Student Name: Seinosuke Sakata
'''
import numpy as np
import cv2 as cv

img = cv.imread('./sample_salt_and_pepper.jpg', 0)    # Read out as a gray scale image
img = img.astype(float)
#median filter
h = np.zeros_like(img)
for i in range(1, img.shape[0]-1):
    for j in range(1, img.shape[1]-1):
        h[i, j] = np.median(img[i-1:i+2, j-1:j+2])
h = (h - np.min(h)) / (np.max(h) - np.min(h)) * 255.0
cv.imshow('Input', img.astype(np.uint8))
cv.imshow('Filtered', h.astype(np.uint8))
cv.waitKey()
cv.destroyAllWindows()
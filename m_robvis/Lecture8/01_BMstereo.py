import cv2

# Read out images
img_l = cv2.imread("view0_small.png", 0)  # left image
img_r = cv2.imread("view1_small.png", 0)  # right image

# Use OpenCV Block Matching (BM) stereo method (SAD)
stereo = cv2.StereoBM_create(numDisparities=32, blockSize=15)
disparity = stereo.compute(img_l, img_r)  # Compute disparity
disparity = cv2.normalize(src=disparity, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

cv2.imshow("disparity", disparity)
cv2.waitKey(0)
cv2.destroyAllWindows()

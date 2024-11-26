'''
33F24007
Seinosuke Sakata
'''
import numpy as np
import cv2

def warpTwoImages(img1, img2, H):
    '''
    This function will be used for Project 2.
    Warps img2 to img1 coordinates with a 3x3 homograph matrix H.
    '''
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    pts1 = np.array([[[0, 0], [0, h1], [w1, h1], [w1, 0]]], dtype=np.float32)
    pts2 = np.array([[[0, 0], [0, h2], [w2, h2], [w2, 0]]], dtype=np.float32)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1[0, :, :], pts2_[0, :, :]), axis=0)
    [xmin, ymin] = np.array(pts.min(axis=0).ravel() - 0.5, dtype=np.float32)
    [xmax, ymax] = np.array(pts.max(axis=0).ravel() + 0.5, dtype=np.float32)
    t = np.array([-xmin, -ymin], dtype=np.int32)
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]], dtype=np.float32)  # translate

    result = cv2.warpPerspective(img2, Ht.dot(H), (int(xmax - xmin), int(ymax - ymin)))
    result[t[1]:h1 + t[1], t[0]:w1 + t[0]] = img1
    return result

def find_overlap():
    '''
    Find overlap area between two images
    '''
    overlap = []
    return overlap

def find_baundary():
    '''
    Find baundary via graph cut
    '''
    baundary = []
    return baundary

def multiband_blending():
    '''
    maltiband blending with baundary
    '''
    result = []
    return result

if __name__=='__main__':
    img1 = cv2.imread('kyoto01.jpg')  # Image 1
    img2 = cv2.imread('kyoto02.jpg')  # Image 2
    img3 = cv2.imread('kyoto03.jpg', 0)  # Image 3
    # Initiate AKAZE detector
    akaze = cv2.AKAZE_create()
    # find the keypoints and descriptors with AKAZE
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)
    # Compute matches
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)  # sort by good matches
    #get Homography matrix
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) 
    
    # warp two images
    out = warpTwoImages(img2, img1, H)

    # kp2, des2 = akaze.detectAndCompute(out, None)
    # kp3, des3 = akaze.detectAndCompute(img3, None)

    # matches = matcher.match(des2, des3)
    # matches = sorted(matches, key=lambda x: x.distance)  # sort by good matches
    # #get Homography matrix
    # src_pts = np.float32([kp2[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    # dst_pts = np.float32([kp3[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    # H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    # out = warpTwoImages(out, img3, np.linalg.inv(H))
    images = [img1, img2, img3]
    stitcher = cv2.Stitcher_create()
    _, out = stitcher.stitch(images)
    cv2.imshow('Stitched image', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


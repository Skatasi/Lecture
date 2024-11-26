'''
33F24007
Seinosuke Sakata

issue1:
when we use old function 1 by 1, all imgaes are warped to middle 'kyoto03.jpg' and there are some black area around baundary.

solution1:
I know 'kyoto02.jpg' is middle image.
So, I warp 'kyoto01.jpg' to 'kyoto02.jpg' and 'kyoto03.jpg' to 'kyoto02.jpg'.

issue2:
Baundaries of each images is clear due to difference of color value.

sollution2:
I use multiband blending to blur boundary.
but I can't blur baoundary due to resize error.
'''

import numpy as np
import cv2

images = ['kyoto01.jpg', 'kyoto02.jpg', 'kyoto03.jpg']

def getHomographyMatrix(img1, img2):
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
    return H

def get_transformed_coordinate(img1, img2, H):
    '''
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

    return pts

def warp_images(img1, img2, H, t):
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]], dtype=np.float32)  # translate
    return cv2.warpPerspective(img2, Ht.dot(H), (img1.shape[1], img1.shape[0]))

def get_intersection(corners):
    '''
    get intersection of two quadrilaterals
    '''
    x1, y1 = corners[0]
    x2, y2 = corners[1]
    x3, y3 = corners[2]
    x4, y4 = corners[3]

    # 2つの線分の方向ベクトルの計算
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # 平行な場合（交点がない場合）
    if denom == 0:
        return None

    # パラメータ t, u の計算
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denom

    # t, u が [0, 1] の範囲内にあるか確認（線分が交差するかどうか）
    if 0 <= t <= 1 and 0 <= u <= 1:
        # 交点の計算
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        return np.array([px, py])
    else:
        # 交差していない場合
        return None

def make_mask(blank, corners):
    mask = np.zeros_like(blank)
    cv2.fillPoly(mask, [corners], (255, 255, 255))
    return mask

def gausian_pyramid(image, level = 2):
    '''
    gausian pyramid
    '''
    G = image.copy()
    gp = [G]
    for i in range(level):
        G = cv2.pyrDown(G)
        gp.append(G)
    return gp

def laplacian_pyramid(gp, level = 2):
    '''
    laplacian pyramid
    '''
    lp = [gp[level-1]]
    for i in range(level-1,0,-1):
        GE = cv2.pyrUp(gp[i])
        if GE.shape != gp[i - 1].shape:
            GE = cv2.resize(GE, (gp[i - 1].shape[1], gp[i - 1].shape[0]))
        L = cv2.subtract(gp[i-1],GE)
        lp.append(L)
    return lp

def multiband_blending(lap_ps, masks, level = 2):
    '''
    maltiband blending
    '''
    result = np.zeros(masks[0].shape, np.uint8)
    for i in range(len(lap_ps)):
        ls_ = lap_ps[i][0]
        for j in range(1,level):
            ls_ = cv2.pyrUp(ls_)
            if ls_.shape != lap_ps[i][j].shape:
                ls_ = cv2.resize(ls_, (lap_ps[i][j].shape[1], lap_ps[i][j].shape[0]))
            ls_ = cv2.add(ls_, lap_ps[i][j])
        # cv2.imshow('Stitched image', ls_)
        # cv2.waitKey(0)
        result += cv2.bitwise_and(ls_, masks[i])
    return result

if __name__=='__main__':
    imgs = [cv2.imread(image) for image in images]

    H1 = getHomographyMatrix(imgs[0], imgs[1])
    H2 = getHomographyMatrix(imgs[2], imgs[1])

    pts1 = get_transformed_coordinate(imgs[1], imgs[0], H1)
    pts2 = get_transformed_coordinate(imgs[1], imgs[2], H2)
    pts = np.concatenate((pts1, pts2), axis=0)
    pts = pts - pts.min(axis=0)
    [w, h] = np.array(pts.max(axis=0).ravel(), dtype=np.float32)

    blank = np.zeros((int(h), int(w), 3), np.uint8)
    warped_img1 = warp_images(blank, imgs[0], H1, pts[0])
    warped_img2 = warp_images(blank, imgs[1], np.eye(3), pts[0])
    warped_img3 = warp_images(blank, imgs[2], H2, pts[0])

    #make mask1
    intersection1 = get_intersection([pts[0],pts[3],pts[4],pts[7]])
    intersection2 = get_intersection([pts[1],pts[2],pts[5],pts[6]])
    intersection3 = get_intersection([pts[0],pts[3],pts[12],pts[15]])
    intersection4 = get_intersection([pts[1],pts[2],pts[13],pts[14]])
    mask1 = make_mask(blank, np.array([pts[4],pts[5],intersection4,intersection1], dtype=np.int32))
    mask2 = make_mask(blank, np.array([intersection1,intersection4,intersection2,intersection3], dtype=np.int32))
    mask3 = make_mask(blank, np.array([intersection3,intersection2,pts[14],pts[15]], dtype=np.int32))

    gs = [gausian_pyramid(warped_img1), gausian_pyramid(warped_img2), gausian_pyramid(warped_img3)]
    lp = [laplacian_pyramid(gs[0]), laplacian_pyramid(gs[1]), laplacian_pyramid(gs[2])]
    masks = [mask1, mask2, mask3]

    blended = multiband_blending(lp, masks)

    cv2.imshow('Stitched image', blended)
    # cv2.imshow('Stitched image', warped_img1)
    cv2.waitKey(0)

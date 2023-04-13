import numpy as np
import cv2


def get_features(img):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors

def match_keypoints(desc1, desc2, ratio=0.75):
    m = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = []

    matcher = m.knnMatch(desc1, desc2, k=2)

    for m,n in matcher:
        if m.distance < ratio*n.distance:
            matches.append(m)

    return matches

# do ransac to filter matches
# find homography
#get pose relative to robot position

def triangulation(kp1, kp2, T_1w, T_2w):
    """Triangulation to get 3D points
    Args:
        kp1 (Nx2): keypoint in view 1 (normalized)
        kp2 (Nx2): keypoints in view 2 (normalized)
        T_1w (4x4): pose of view 1 w.r.t  i.e. T_1w (from w to 1)
        T_2w (4x4): pose of view 2 w.r.t world, i.e. T_2w (from w to 2)
    Returns:
        X (3xN): 3D coordinates of the keypoints w.r.t world coordinate
        X1 (3xN): 3D coordinates of the keypoints w.r.t view1 coordinate
        X2 (3xN): 3D coordinates of the keypoints w.r.t view2 coordinate
    """
    kp1_3D = np.ones((3, kp1.shape[0]))
    kp2_3D = np.ones((3, kp2.shape[0]))
    kp1_3D[0], kp1_3D[1] = kp1[:, 0].copy(), kp1[:, 1].copy()
    kp2_3D[0], kp2_3D[1] = kp2[:, 0].copy(), kp2[:, 1].copy()
    X = cv2.triangulatePoints(T_1w[:3], T_2w[:3], kp1_3D[:2], kp2_3D[:2])
    X /= X[3]
    X1 = T_1w[:3] @ X
    X2 = T_2w[:3] @ X
    return X[:3], X1, X2 



if __name__ == '__main__':
    imgL = cv2.imread("imgl.jpg")
    keyL = None
    descL = None

    imgR = cv2.imread("imgr.jpg")
    keyR = None
    descR = None

    
    keyL, descL = get_features(imgL)
    keyR, descR = get_features(imgR)
    matches = match_keypoints(descL, descR)
    
    match_plot = cv2.drawMatches(imgL, keyL, imgR, keyR, matches[:1], None, flags=2)
    cv2.imshow("frame", match_plot)

    poseL = np.eye(4)
    poseR = np.eye(4)
    poseL[0,3] = 2

    kpL = []
    kpR = []
    for mat in matches[0:1]:
        kpL.append(keyL[mat.queryIdx].pt)
        kpR.append(keyR[mat.trainIdx].pt)


    wCoords, v1Coords, v2Coords = triangulation(np.array(kpL), np.array(kpR), poseL, poseR)
    #print(wCoords.T)
    print(v1Coords.T)
    #print(v2Coords)

    while(True):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
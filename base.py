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


if __name__ == '__main__':
    vid = cv2.VideoCapture(0)

    prevFrame = None
    prev_key = None
    prev_desc = None
    c = 0
    while True:
        ret, frame = vid.read()

        if c != 0:
            key1, desc1 = get_features(prevFrame)
            key2, desc2 = get_features(frame)
            matches = match_keypoints(desc1, desc2)
            
            match_plot = cv2.drawMatches(prevFrame, key1, frame, key2, matches[:20], None, flags=2)
            cv2.imshow("frame", match_plot)

        prevFrame =frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        c += 1

    vid.release()
    cv2.destroyAllWindows()
import struct
import cv2
import numpy as np


def voc_load(voc_file, max_words=1000):
    v_file = open(voc_file, 'r')
    header = v_file.readline().split(' ')
    header = [val for val in header if val]
    header = [int(val) for val in header]

    descriptors = np.zeros((max_words, 32))
    #scores = np.zeros((max_words, 1))
    for i in range(max_words):
        line = v_file.readline()
        line = line.strip(' ').split(' ')
        line = [int(val) for val in line[:32]]
        
        #score = float(line[32])
        #scores[i, 0] = score

        descriptors[i] = line
                
    words = np.arange(max_words)
    return words, descriptors#, scores

def match_ORB_to_vocab(descriptor, words, descriptors):
    # compute Euclidean distance between descriptor and all visual words
    dists = np.linalg.norm(descriptors - descriptor, axis=1)
    # find the index of the closest visual word
    idx = np.argmin(dists)
    # return the ID of the closest visual word
    return words[idx]

def filter_descriptors(keypoints, descriptors, thresh=20):
    filtered_kp = []
    filtered_desc = []

    for i in range(len(keypoints)):
        far = True
        for j in range(len(keypoints)):
            if i != j:
                pt1 = keypoints[i].pt
                pt2 = keypoints[j].pt
                dist = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
                if dist < thresh:
                    far = False
                    break
        if far:
            filtered_kp.append(keypoints[i])
            filtered_desc.append(descriptors[i])


    
    return filtered_kp, filtered_desc

if __name__ == '__main__':
    words, voc_descriptors = voc_load("underwater_orb_vocab.txt", max_words=83468)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (0, 255, 0)
    thickness = 1

    video = cv2.VideoCapture('underwater.mp4')
    nfeat = 50
    orb = cv2.ORB_create(nfeatures=nfeat, scaleFactor=2, edgeThreshold=50)#cv2.ORB_create(nfeatures=nfeat, scaleFactor=2, nlevels=6, edgeThreshold=100, firstLevel=2, WTA_K=4, scoreType=cv2.ORB_FAST_SCORE, patchSize=128, fastThreshold=60)#250, 30
    #orb = cv2.ORB_create(nfeatures=nfeat, scaleFactor=2, nlevels=6, edgeThreshold=100, firstLevel=2, WTA_K=4, scoreType=cv2.ORB_FAST_SCORE, patchSize=128, fastThreshold=60)#250, 30
    width  = video.get(3)   # float `width`
    height = video.get(4) 
    print(width, height)

    while True:
        ret, img = video.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if not ret:
            break
        #temp_img = cv2.resize(img, dsize=(int(width/2), int(height/2)), fx=0.5, fy=0.5)
        #img = temp_img
        #temp_img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
        #img = temp_img
        #minmax_img = cv2.normalize(img, None, 25, 255, cv2.NORM_MINMAX)
        #img = minmax_img


        keypoints, descriptors = orb.detectAndCompute(img, None)
        #keypoints, descriptors = filter_descriptors(keypoints, descriptors)
        #print(len(keypoints))

        matched_words = np.zeros(len(keypoints))
        for i in range(len(keypoints)):
            w = match_ORB_to_vocab(descriptors[i], words, voc_descriptors)
            matched_words[i] = w
            pos = (int(keypoints[i].pt[0]), int(keypoints[i].pt[1]))
            img2 = cv2.drawKeypoints(img, keypoints, None, flags=0)
            img2 = cv2.putText(img, str(w), pos, font, fontScale, color, thickness, cv2.LINE_AA)

        #print(np.unique(matched_words).shape[0] == len(keypoints))
        #print(matched_words)
        img2 = cv2.resize(img, dsize=(int(width/2), int(height/2)), fx=0.5, fy=0.5)
        cv2.imshow("frame", img2)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # word_id = match_ORB_to_vocab(descriptors[0], words, descriptors)
    # print(word_id)
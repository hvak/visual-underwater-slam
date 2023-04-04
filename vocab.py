import struct
import cv2
import numpy as np


def voc_load(voc_file, max_words=1000):
    v_file = open(voc_file, 'r')
    header = v_file.readline().split(' ')
    header = [val for val in header if val]
    header = [int(val) for val in header]

    descriptors = np.zeros((max_words, 32))
    for i in range(max_words):
        line = v_file.readline()
        line = line.strip(' ').split(' ')[:32]
        line = [int(val) for val in line]

        descriptors[i] = line
                
    words = np.arange(max_words)
    return words, descriptors

def match_ORB_to_vocab(descriptor, words, descriptors):
    # compute Euclidean distance between descriptor and all visual words
    dists = np.linalg.norm(descriptors - descriptor, axis=1)
    # find the index of the closest visual word
    idx = np.argmin(dists)
    # return the ID of the closest visual word
    return words[idx]


if __name__ == '__main__':
    words, voc_descriptors = voc_load("underwater_orb_vocab.txt", max_words=83468)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (0, 255, 0)
    thickness = 1

    video = cv2.VideoCapture('rock1_camera_image_raw_compressed.mp4')
    nfeat = 3
    orb = cv2.ORB_create(edgeThreshold=10, nfeatures=nfeat)#250, 30
    width  = video.get(3)   # float `width`
    height = video.get(4) 

    while True:
        ret, img = video.read()
        if not ret:
            break
        img2 = cv2.resize(img, dsize=(int(width/2), int(height/2)), fx=0.5, fy=0.5)

        keypoints, descriptors = orb.detectAndCompute(img2, None)
        #print(len(keypoints))

        matched_words = np.zeros(len(keypoints))
        for i in range(len(keypoints)):
            w = match_ORB_to_vocab(descriptors[i], words, voc_descriptors)
            matched_words[i] = w
            pos = (int(keypoints[i].pt[0]), int(keypoints[i].pt[1]))
            img2 = cv2.putText(img2, str(w), pos, font, fontScale, color, thickness, cv2.LINE_AA)

        print(np.unique(matched_words).shape[0] == len(keypoints))
        print(matched_words)
        cv2.imshow("frame", img2)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # word_id = match_ORB_to_vocab(descriptors[0], words, descriptors)
    # print(word_id)
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseStamped
import tf

def construct_pose(x, y, z, roll , pitch, yaw, frame):
        quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = frame
        pose.pose.orientation.x = quaternion[0]
        pose.pose.orientation.y = quaternion[1]
        pose.pose.orientation.z = quaternion[2]
        pose.pose.orientation.w = quaternion[3]
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        return pose

class FeatureTracker:
    def __init__(self, node_handle, img_topic, depth_topic, vocab_file):

        print("Loading Vocabulary...")
        self.load_vocabulary(vocab_file)

        self.img = None
        self.depth = None
        self.br = CvBridge()
        self.orb = cv2.ORB_create(edgeThreshold=25, nfeatures=5)#250, 30

        rospy.Subscriber(img_topic, self.img_callback)
        rospy.Subscriber(depth_topic, self.depth_callback)
        self.tf_listener = tf.TransformListener()

        #intrinsics
        self.K_inv = None

    def img_callback(self, msg):
        self.img = self.br.imgmsg_to_cv2(msg)

    def depth_callback(self, msg):
        self.depth = self.br.imgmsg_to_cv2(msg)

    def load_vocabulary(self, vocab_file, max_words = 80000):
        v_file = open(vocab_file, 'r')
        header = v_file.readline().split(' ')
        header = [val for val in header if val]
        header = [int(val) for val in header]
        self.voc_n_cluster = header[0]
        self.voc_depth = header[1]

        self.descriptors = np.zeros((max_words, 32))
        for i in range(max_words):
            line = v_file.readline()
            line = line.strip(' ').split(' ')[:32]
            line = [int(val) for val in line]
            self.descriptors[i] = line
                
        self.words = np.arange(max_words)
        
    def match_ORB_to_vocab(self, descriptor):
        # compute Euclidean distance between descriptor and all visual words
        dists = np.linalg.norm(self.descriptors - descriptor, axis=1)
        # find the index of the closest visual word
        idx = np.argmin(dists)
        # return the ID of the closest visual word
        return self.words[idx]
    
    def pixel_to_world(self, u, v):
        d = self.depth[u,v]
        cam_frame = d * np.matmul(self.K_inv, np.array([[u], [v], [1]]))
        cam_pose = construct_pose(cam_frame[0,0], cam_frame[1,0], cam_frame[2,0], 0, 0, 0, "image_frame")

        self.tf_listener.waitForTransform("image_frame", "map", rospy.Time.now(), rospy.Duration(1.0))
        world_pose = self.tf_listener.transformPose("world", cam_pose)
        return [world_pose.pose.position.x, world_pose.pose.position.y, world_pose.pose.position.z]


    def get_landmarks(self):
        kp, desc = self.orb.detectAndCompute(self.img, None)
        landmarks = []
        for i in range(len(kp)):
            id = self.match_ORB_to_vocab(desc[i])
            pt = kp[i].pt
            world_pt = self.pixel_to_world(pt[0], pt[1])
            
            landmark = (id, world_pt[0], world_pt[1], world_pt[2])
            landmarks.append(landmark)

        return landmarks
            

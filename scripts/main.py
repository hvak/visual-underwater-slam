import rospy
from uslam.isam import AUV_ISAM
from sensor_msgs.msg import Imu
from waterlinked_a50_ros_driver.msg import DVL
import sys
import tf2_ros

auv_slam = AUV_ISAM()

def callback_imu(data):
    auv_slam.update_imu(data) # whatever this is

def callback_odom(data):
    auv_slam.update_odom(data)
    print(data)

def callback_dvl(data):
    auv_slam.update_dvl(data)
    print(data)

## subscribe to bagfile topics

if __name__=='__main__':
    rospy.init_node('data_listener', anonymous=True)

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    rospy.Subscriber('/mavros/imu/data_raw', Imu, callback_imu)
    #rospy.Subscriber('/mavros/imu/data_raw', sensor_msgs.msg.Imu, callback_odom)
    rospy.Subscriber('/dev/data', DVL, callback_dvl)

    while not rospy.is_shutdown():
        print('here')
        #auv_slam.g_transform = tfBuffer.lookup_transform('world', 'base_link', rospy.Time().now(), rospy.Duration(3.0))
        auv_slam.update_isam()
        rospy.sleep(0.5)

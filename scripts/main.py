import rospy
import isam
from sensor_msgs.msg import Imu
import sys
sys.path.append('~/catkin_ws/src/dvl-a50-ros-driver/scripts')
from waterlinked_a50_ros_driver import DVL
import tf2_ros

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

    auv_slam = isam.AUV_SLAM()

    while not rospy.is_shutdown():
        auv_slam.g_transform = tfBuffer.lookup_transform('world', 'base_link', rospy.Time())
        auv_slam.update_isam()
        rospy.sleep(100)
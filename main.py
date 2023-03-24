import rospy
import isam
from sensor_msgs.msg import Imu
from waterlinked_a50_ros_driver import DVL

def callback_imu(data):
    auv_slam.update_imu(data.) # whatever this is

def callback_odom(data):
    print(data)

def callback_dvl(data):
    print(data)

def callback_transform(data):
    print(data)

## subscribe to bagfile topics

if __name__=='__main__':
    rospy.init_node('data_listener', anonymous=True)

    rospy.Subscriber('/mavros/imu/data_raw', Imu, callback_imu)
    #rospy.Subscriber('/mavros/imu/data_raw', sensor_msgs.msg.Imu, callback_odom)
    rospy.Subscriber('/dev/data', DVL, callback_dvl)

    auv_slam = isam.AUV_SLAM()

    rospy.spin()
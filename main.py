import rospy
import std_msgs
import nav_msgs
import sensor_msgs

def callback_imu(data):
    print(data)

## subscribe to bagfile topics
## put data in ISAM

if __name__=='__main__':
    rospy.init_node('data_listener', anonymous=True)

    rospy.Subscriber('/mavros/imu/data_raw', sensor_msgs.Imu, callback_imu)

    rospy.spin()
import rospy
import std_msgs
import nav_msgs
import sensor_msgs

def callback_imu(data):
    print(data)

def callback_odom(data):
    print(data)

def callback_dvl(data):
    print(data)

## subscribe to bagfile topics

if __name__=='__main__':
    rospy.init_node('data_listener', anonymous=True)

    rospy.Subscriber('/mavros/imu/data_raw', sensor_msgs.msg.Imu, callback_imu)
    rospy.Subscriber('/mavros/imu/data_raw', sensor_msgs.msg.Imu, callback_odom)
    rospy.Subscriber('/mavros/imu/data_raw', sensor_msgs.msg.Imu, callback_dvl)

    rospy.spin()
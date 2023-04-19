import rospy
import tf2_ros
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistStamped, TransformStamped

odom_pose = None
def callback_odom(data):
    global odom_pose
    odom_pose = data


"""
This script fixes the TF tree for our bagfiles
It publishes a transform from world -> baselink such that
the zed odometry is no longer used.
"""
if __name__ == '__main__':
    rospy.init_node('tf_fix', anonymous=True)
    
    rospy.Subscriber("/dvl/local_position", PoseWithCovarianceStamped, callback_odom)
    br = tf2_ros.TransformBroadcaster()

    rate = rospy.Rate(10)


    while not rospy.is_shutdown():

        if odom_pose != None:
            tfs = TransformStamped()
            tfs.header.stamp = rospy.Time.now()
            tfs.header.frame_id = "world"
            tfs.child_frame_id = "base_link"
            tfs.transform.translation.x = odom_pose.pose.pose.position.x
            tfs.transform.translation.y = odom_pose.pose.pose.position.y
            tfs.transform.translation.z = odom_pose.pose.pose.position.z
            tfs.transform.rotation.x = odom_pose.pose.pose.orientation.x
            tfs.transform.rotation.y = odom_pose.pose.pose.orientation.y
            tfs.transform.rotation.z = odom_pose.pose.pose.orientation.z
            tfs.transform.rotation.w = odom_pose.pose.pose.orientation.w
            print(tfs)
            br.sendTransform(tfs)

        rate.sleep()
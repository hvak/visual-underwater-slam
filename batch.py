# pylint: disable=invalid-name, E1101

from __future__ import print_function


import rospy
import rosnode
#from uslam.isam import AUV_ISAM
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistStamped
# from waterlinked_a50_ros_driver.msg import DVL
from typing import Optional, List
import sys
import tf2_ros
from functools import partial


import math

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=W0611

import gtsam
from gtsam import (ISAM2, BetweenFactorConstantBias, Cal3_S2,
                   ConstantTwistScenario, ImuFactor, NonlinearFactorGraph,
                   PinholeCameraCal3_S2, Point3, Pose3,
                   PriorFactorConstantBias, PriorFactorPose3,
                   PriorFactorVector, Rot3, Values)
from gtsam.symbol_shorthand import B, V, X
from gtsam.utils import plot

###############################
#
#   Callbacks, Global
#
################################

def callback_imu(data):
    auv_isam.update_imu(data) # whatever this is
    # auv_isam.imu_transforms.append(auv_isam.last_imu_transform)

    # print("update imu")

def callback_odom(data):
    auv_isam.update_odom(data)
    # print("update odom")

def callback_mavros_vel(data):
    auv_isam.update_mavros_vel(data)
    #print(data)

def callback_dvl(data):
    auv_isam.update_dvl(data)
    #print(data)

def callback_imu_transform(transform):
    
    auv_isam.last_imu_transform = transform
    if auv_isam.do_accum == True:
        auv_isam.odom_accum.append(auv_isam.odom)

        measAcc = np.array([auv_isam.imu_data.linear_acceleration.x, 
                            auv_isam.imu_data.linear_acceleration.y, 
                            auv_isam.imu_data.linear_acceleration.z]) - np.dot(auv_isam.last_imu_transform, auv_isam.g)
        #print("final accel with gravity removed", measAcc)
        measOmega = np.array([auv_isam.imu_data.angular_velocity.x, auv_isam.imu_data.angular_velocity.y, auv_isam.imu_data.angular_velocity.z])
        #print('here', measAcc)
        auv_isam.imu_accum.append(np.array([measAcc, measOmega]))


        auv_isam.dvl_accum.append(auv_isam.dvl)
        print("appending")
    
    # auv_isam.imu_transforms.append(transform)

def callback_dvl_transform(transform):
    
    auv_isam.last_dvl_transform = transform

def vector3(x, y, z):
    """Create 3d double numpy array."""
    return np.array([x, y, z], dtype=float)


g = 9.81
n_gravity = vector3(0, 0, -g)

###############################
#
#   ISAM Class
#
################################

class AUV_ISAM:
    def __init__(self):
        self.PARAMS, self.BIAS_COVARIANCE, self.DELTA = self.preintegration_parameters()
        self.radius = 30
        self.pose_0 = gtsam.Pose3(gtsam.Rot3([[0, 0, -1], [1, 0, 0], [0, -1, 0]]), [0, 0, 0])
        self.delta_t = 1.0/18  # makes for 10 degrees per step
        self.angular_velocity = math.radians(180)  # rad/sec
        self.graph = NonlinearFactorGraph()
        self.isam = ISAM2()
        self.initialEstimate = Values()
        self.timestamp = 0

        # Add a prior on pose x0. This indirectly specifies where the origin is.
        # 30cm std on x,y,z 0.1 rad on roll,pitch,yaw
        self.noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3]))
        # self.graph.push_back(PriorFactorPose3(X(0), self.pose_0, self.noise))

        # Add imu priors
        self.biasKey = B(0)
        self.biasnoise = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
        # self.biasprior = PriorFactorConstantBias(self.biasKey, gtsam.imuBias.ConstantBias(),self. biasnoise)
        # self.graph.push_back(self.biasprior)
        # self.initialEstimate.insert(self.biasKey, gtsam.imuBias.ConstantBias())
        self.velnoise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)

        # Calculate with correct initial velocity
        self.n_velocity = vector3(0, 0, 0)
        # self.velprior = PriorFactorVector(V(0), self.n_velocity, self.velnoise)
        # self.graph.push_back(self.velprior)
        self.initialEstimate.insert(V(0), self.n_velocity)

        self.accum = gtsam.PreintegratedImuMeasurements(self.PARAMS)
        accBias = np.array([-0.3, 0.1, 0.2])
        gyroBias = np.array([0.1, 0.3, -0.1])
        self.bias = gtsam.imuBias.ConstantBias(accBias, gyroBias)


        ### DATA TYPES
        self.mav_vel = None
        self.odom = None
        self.dvl = None
        self.imu_data = None

        ### DATA NOISE
        self.dvl_model = gtsam.noiseModel.Isotropic.Sigma(3, 0.1) ## Bagoren et al


        self.last_imu_transform = np.eye(3)
        self.last_dvl_transform = np.eye(3)

        self.g_transform = np.eye(3)

        self.imu_transforms = []

        self.grav = 9.81
        self.g = vector3(0, 0, -self.grav)

        # batch
        self.do_accum = True
        self.odom_accum = []
        self.imu_accum = []
        self.dvl_accum = []
        self.batch_graph = gtsam.NonlinearFactorGraph()
        self.batch_initial = gtsam.Values()


    def preintegration_parameters(self):
        # IMU preintegration parameters
        # Default Params for a Z-up navigation frame, such as ENU: gravity points along negative Z-axis
        PARAMS = gtsam.PreintegrationParams.MakeSharedU(g)
        I = np.eye(3)
        PARAMS.setAccelerometerCovariance(I * 0.1)
        PARAMS.setGyroscopeCovariance(I * 0.1)
        PARAMS.setIntegrationCovariance(I * 0.1)
        PARAMS.setUse2ndOrderCoriolis(False)
        PARAMS.setOmegaCoriolis(vector3(0, 0, 0))

        BIAS_COVARIANCE = gtsam.noiseModel.Isotropic.Variance(6, 0.1)
        DELTA = Pose3(Rot3.Rodrigues(0, 0, 0),
                    Point3(0.05, -0.10, 0.20))

        return PARAMS, BIAS_COVARIANCE, DELTA
    
    ###############################
    #
    #   Update Data
    #
    ################################
    
    def update_imu(self, data):

        self.imu_data = data
        # measAcc = np.array([data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z]) - np.dot(self.last_imu_transform, self.g)
        # #print("final accel with gravity removed", measAcc)
        # measOmega = np.array([data.angular_velocity.x, data.angular_velocity.y, data.angular_velocity.z])
        # #print('here', measAcc)
        # self.imu = np.array([measAcc, measOmega])

        ## above code moved to transform update
        # print("UPDATE IMU")

        return

    def update_odom(self, data):
        self.odom = {"x": data.pose.pose.position.x, 
                     "y": data.pose.pose.position.y,
                     "z": data.pose.pose.position.z,
                     "i": data.pose.pose.orientation.x,
                     "j": data.pose.pose.orientation.y,
                     "k": data.pose.pose.orientation.z, 
                     "q": data.pose.pose.orientation.w}

        return
    
    def update_dvl(self, data):
        self.dvl = np.array([ data.twist.linear.x, data.twist.linear.y, data.twist.linear.z])
        return
    

    ###############################
    #
    #   Unary Factor Errors
    #
    ################################

    def velocity_error(
        self,
        measurement: np.ndarray,
        this: gtsam.CustomFactor,
        values: gtsam.Values,
        jacobians: Optional[List[np.ndarray]],
    ) -> float:
        """
        Calculate the error betwen the velocity prediction and the velocity measurement
        ana notes: retaining bagoren et al's jacobian and typing (list, optional keywords)
        """
        
        key = this.keys()[0] # key = timestamp
        vel_estimate = values.atPoint3(V(key)) # get previous estimates from graph
        pose_estimate = values.atPose3(X(key))

        rot_mat = pose_estimate.rotation().matrix()
        vx = vel_estimate[0]
        vy = vel_estimate[1]
        vz = vel_estimate[2]
        v = np.array([vx, vy, vz]).reshape((3, 1))

        meas_t = measurement.T
        meas_world = rot_mat @ meas_t # convert dvl vel to world frame using the estimate's transform

        error = np.array(
            [
                meas_world[0, 0] - v[0, 0],
                meas_world[1, 0] - v[1, 0],
                meas_world[2, 0] - v[2, 0],
            ]
        )
        # print(error)
        if jacobians is not None:
            jacobians[0] = rot_mat
        return error


    def create_imu_factor_batch(self, index):
        delta_t = .25
        self.accum.integrateMeasurement(self.imu_accum[index][0], self.imu_accum[index][1], delta_t)
        ## TODO maybe add multiple imu and reset
        imuFactor = ImuFactor(X(index - 1), V(index - 1), X(index), V(index), self.biasKey, self.accum)
        self.accum.resetIntegration()
        return imuFactor
    
    def create_dvl_factor_batch(self, index):
        dvl_factor = gtsam.CustomFactor(
                        self.dvl_model,
                        [index], 
                        partial(self.velocity_error, np.array([self.dvl_accum[index]])),
                    )
        return dvl_factor

    def createBatch(self):
        #poses = self.ODOMDATA
        # self.update_imu(None)
        self.batch_initial.insert(self.biasKey, self.bias)
        velocity = vector3(0,0,0)
        for i in range(len(self.odom_accum)):
            #Prior Estimate
            currPose = self.odom_accum[i]
            #print(each)
            rot = gtsam.Rot3.Quaternion(currPose['q'], currPose['i'], currPose['j'], currPose['k'])
            t = gtsam.Point3(currPose['x'], currPose['y'], currPose['z'])
            #print(rot)
            pose = gtsam.Pose3(rot, t)
            if i == 0:
                PRIOR_NOISE = gtsam.noiseModel.Isotropic.Sigma(6, 0.25)
                self.batch_graph.add(gtsam.PriorFactorPose3(X(0), pose, PRIOR_NOISE))
                self.batch_initial.insert(X(i), pose)
                self.batch_initial.insert(V(i), velocity)
            else:
                self.batch_initial.insert(X(i), pose)
                self.batch_initial.insert(V(i), velocity)
                imuFactor = self.create_imu_factor_batch(i)
                dvlFactor = self.create_dvl_factor_batch(i)
                self.batch_graph.push_back(imuFactor)
                # self.batch_graph.push_back(dvlFactor)
                # print(num_factors)
                
        return

    


if __name__ == '__main__':
    rospy.init_node('data_listener', anonymous=True)

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    rospy.Subscriber('/zedm/zed_node/imu/data', Imu, callback_imu)
    #rospy.Subscriber('/mavros/imu/data', Imu, callback_imu)
    # rospy.Subscriber('/zedm/zed_node/odom', Odometry, callback_odom)
    rospy.Subscriber('/dvl/local_position', PoseWithCovarianceStamped, callback_odom)
    # rospy.Subscriber('/mavros/local_position/velocity_local', TwistStamped, callback_mavros_vel)
    rospy.Subscriber('/dvl/twist', TwistStamped, callback_dvl)

    auv_isam = AUV_ISAM()

    while not rospy.is_shutdown():

        try:
            transform = tfBuffer.lookup_transform('map', 'base_link', rospy.Time(0))
            transform_mat = gtsam.Rot3.Quaternion(transform.transform.rotation.w, 
                                            transform.transform.rotation.x, 
                                            transform.transform.rotation.y, 
                                            transform.transform.rotation.z).matrix()
            callback_imu_transform(transform_mat)
            print('got transform')
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print("exception in imu transform lookup loop, using last transform")

        # try:
        #     dvl_transform = tfBuffer.lookup_transform('map', 'dvl_link', rospy.Time(0))
        #     dvl_transform_mat = gtsam.Rot3.Quaternion(dvl_transform.transform.rotation.w, 
        #                                                  dvl_transform.transform.rotation.x, 
        #                                                  dvl_transform.transform.rotation.y, 
        #                                                  dvl_transform.transform.rotation.z).matrix()
            
        #     callback_dvl_transform(dvl_transform_mat)
        #     print('got dvl transform')
        # except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        #     print("exception in DVL transform lookup loop")


        if 'play' not in '\t'.join(rosnode.get_node_names()):

            auv_isam.do_accum = True
            print("num transforms: ", len(auv_isam.imu_transforms))
            print("odom lentgth:", len(auv_isam.odom_accum))
            print("imu lentgth:", len(auv_isam.imu_accum))

            auv_isam.createBatch()
            results = gtsam.LevenbergMarquardtOptimizer(auv_isam.batch_graph, auv_isam.batch_initial, gtsam.LevenbergMarquardtParams()).optimize()

            auv_isam.timestamp += 1

            break
        
        rospy.sleep(0.25)
    plot.plot_trajectory(1, results)
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x = [auv_isam.odom_accum[i]['x'] for i in range(len(auv_isam.odom_accum))]
    y = [auv_isam.odom_accum[i]['y'] for i in range(len(auv_isam.odom_accum))]
    z = [auv_isam.odom_accum[i]['z'] for i in range(len(auv_isam.odom_accum))]
    ax.plot3D(x, y, z)
    plt.show()

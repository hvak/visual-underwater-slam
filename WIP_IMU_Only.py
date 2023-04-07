from __future__ import print_function

#WIP2 IMU Factor Only
import rospy
#from uslam.isam import AUV_ISAM
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistStamped
from waterlinked_a50_ros_driver.msg import DVL
from typing import Optional, List
import sys
import tf2_ros
from functools import partial


import math

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=W0611

import pandas as pd

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
    #print(data)

def callback_odom(data):
    auv_isam.update_odom(data)
    #print(data.pose.pose.position)

def vector3(x, y, z):
    """Create 3d double numpy array."""
    return np.array([x, y, z], dtype=float)

def importCSV(FILEPATH, type):
    csv = pd.read_csv(FILEPATH).to_numpy()
    if ('IMU' == type):
        data = csv[:, 5:9]
        data = np.append(data, csv[:, 10:13], axis=1)
    elif('ODOM' == type):
        data = csv[:, 5:12]
    print(data[0])
    return data

###############################
#
#   ISAM Class
#
################################
grav = 9.81
class AUV_ISAM:
    def __init__(self):
        #Import CSV Data
        self.IMU = importCSV('23_compressed_merged-mavros-imu-data.csv', 'IMU')
        self.ODOM = importCSV('23_compressed_merged-dvl-local_position.csv', 'ODOM')

        self.PARAMS, self.BIAS_COVARIANCE, self.DELTA = self.preintegration_parameters()
        self.pose_0 = gtsam.Pose3(gtsam.Rot3([[0, 0, -1], [1, 0, 0], [0, -1, 0]]), [0, 0, 0])
        self.graph = NonlinearFactorGraph()
        self.isam = ISAM2()
        self.initialEstimate = Values()
        self.timestamp = 0

        # Add a prior on pose x0. This indirectly specifies where the origin is.
        # 30cm std on x,y,z 0.1 rad on roll,pitch,yaw
        self.noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3]))
        self.graph.push_back(PriorFactorPose3(X(0), self.pose_0, self.noise))

        # Add imu priors
        self.biasKey = B(0)
        self.biasnoise = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
        self.biasprior = PriorFactorConstantBias(self.biasKey, gtsam.imuBias.ConstantBias(),self. biasnoise)
        self.graph.push_back(self.biasprior)
        self.initialEstimate.insert(self.biasKey, gtsam.imuBias.ConstantBias())
        self.velnoise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)

        self.accum = gtsam.PreintegratedImuMeasurements(self.PARAMS)

        # Calculate with correct initial velocity
        self.n_velocity = vector3(0, 0, 0)
        self.velprior = PriorFactorVector(V(0), self.n_velocity, self.velnoise)
        self.graph.push_back(self.velprior)
        self.initialEstimate.insert(V(0), self.n_velocity)

        ### DATA TYPES
        self.mav_vel = None
        self.odom = None

        self.g_transform = None
        self.g = vector3(0, 0, -grav)


    def preintegration_parameters(self):
        # IMU preintegration parameters
        # Default Params for a Z-up navigation frame, such as ENU: gravity points along negative Z-axis
        PARAMS = gtsam.PreintegrationParams.MakeSharedU(grav)
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
        print("IMU Update")
        #print("linear accel raw", np.array([data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z]))
        #print("transform: ", self.g_transform)
        #print("transformed gravity ", np.dot(self.g_transform, self.g))
        if (np.array([data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z]) is None):
            print(data)
        measAcc = np.array([data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z]) - np.flip(np.dot(self.g_transform, self.g))
        print("final accel with gravity removed", measAcc)
        measOmega = np.array([data.angular_velocity.x, data.angular_velocity.y, data.angular_velocity.z])
        #print('here', measAcc)
        self.imu = np.array([measAcc, measOmega])
        return
    
    def update_odom(self, data):
        #print("Odom Update")
        self.odom = {"x": data.pose.pose.position.x, 
                     "y": data.pose.pose.position.y,
                     "z": data.pose.pose.position.z,
                     "i": data.pose.pose.orientation.x,
                     "j": data.pose.pose.orientation.y,
                     "k": data.pose.pose.orientation.z, 
                     "q": data.pose.pose.orientation.w}
        return
    
    def update_mavros_vel(self, data):
        # print("mavros vel Update")
        self.mav_vel = {"x" : data.twist.linear.x,
                "y" : data.twist.linear.y,
                "z" : data.twist.linear.z}
        return
    
    ###############################
    #
    #   Create Factors
    #
    ################################
    
    def create_imu_factor(self):
        delta_t = .01
        self.accum.integrateMeasurement(self.imu[0], self.imu[1], delta_t)
        imuFactor = ImuFactor(X(self.timestamp - 1), V(self.timestamp - 1), X(self.timestamp), V(self.timestamp), self.biasKey, self.accum)
        return imuFactor
    
    def get_factors(self):
        imuFactor = self.create_imu_factor()
        #dvlFactor = self.create_dvl_factor()
        # depthFactor = self.create_depth_factor()
        # orbFactor = self.create_orb_factor()
        factors = [imuFactor]#, dvlFactor]
        return factors
    
    ################################
    #
    #   ISAM Update Loop
    #
    ################################

    def update(self):
        # Simulate poses and imu measurements, adding them to the factor graph
            if self.timestamp == 0:  # First time add two poses
                self.initialEstimate.insert(X(0), gtsam.Pose3(gtsam.Rot3([[0, 0, -1], [1, 0, 0], [0, -1, 0]]), [self.odom['x'], self.odom['y'], self.odom['z']]))
                self.initialEstimate.insert(X(1), gtsam.Pose3(gtsam.Rot3([[0, 0, -1], [1, 0, 0], [0, -1, 0]]), [self.odom['x'], self.odom['y'], self.odom['z']]))
                               
            elif self.timestamp >= 2:  # Add more poses as necessary
                self.initialEstimate.insert(X(self.timestamp), gtsam.Pose3(gtsam.Rot3.Quaternion(self.odom['q'], self.odom['i'], self.odom['j'], self.odom['k']), [self.odom['x'], self.odom['y'], self.odom['z']]))

            if self.timestamp > 0:
                # Add Bias variables periodically
                if self.timestamp % 5 == 0:
                    self.biasKey += 1
                    factor = BetweenFactorConstantBias(
                        self.biasKey - 1, self.biasKey, gtsam.imuBias.ConstantBias(), self.BIAS_COVARIANCE)
                    self.graph.add(factor)
                    self.initialEstimate.insert(self.biasKey, gtsam.imuBias.ConstantBias())

                # Add Factors
                for factor in self.get_factors():
                    self.graph.add(factor)

                # insert new velocity, which is wrong
                if self.mav_vel is not None:
                    #print("adding vel ", self.mav_vel)
                    # rotate?
                    # try removing this and letting isam fill in
                    self.initialEstimate.insert(V(self.timestamp), vector3(self.mav_vel['x'], self.mav_vel['y'], self.mav_vel['z']))
                else:
                    #print("not adding vel", self.mav_vel)
                    self.initialEstimate.insert(V(self.timestamp), vector3(0,0,0))
                self.accum.resetIntegration()

            # Incremental solution
            self.isam.update(self.graph, self.initialEstimate)
            result = self.isam.calculateEstimate()
            plot.plot_incremental_trajectory(0, result,
                                            start=self.timestamp, scale=3, time_interval=0.01)
            plot.plot_pose3(fignum=0, pose=gtsam.Pose3(gtsam.Rot3([[0, 0, -1], [1, 0, 0], [0, -1, 0]]), [self.odom['x'], self.odom['y'], self.odom['z']]), axis_length=0.5)

            # reset
            self.graph = NonlinearFactorGraph()
            self.initialEstimate.clear()
            self.timestamp += 1

if __name__ == '__main__':
    rospy.init_node('data_listener', anonymous=True)

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    rospy.Subscriber('/mavros/imu/data', Imu, callback_imu)
    rospy.Subscriber('/dvl/local_position', PoseWithCovarianceStamped, callback_odom)
    #rospy.Subscriber('/mavros/local_position/velocity_local', TwistStamped, callback_mavros_vel)
    #rospy.Subscriber('/dvl/twist', TwistStamped, callback_dvl)

    auv_isam = AUV_ISAM()

    while not rospy.is_shutdown():
        #auv_isam.g_transform = tfBuffer.lookup_transform('map', 'base_link', rospy.Time().now(), rospy.Duration(3.0))
        got_transform = False    
        while not got_transform:
            try:
                ## todo make sure transform time matches pose time?
                transform = tfBuffer.lookup_transform('base_link', 'map', rospy.Time(0))
                auv_isam.g_transform = gtsam.Rot3.Quaternion(transform.transform.rotation.w, 
                                                             transform.transform.rotation.x, 
                                                             transform.transform.rotation.y, 
                                                             transform.transform.rotation.z).matrix()

                got_transform = True
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                #print("exception in transform lookup loop")
                continue

        if auv_isam.odom is not None:
            auv_isam.update()

            #plt.show()
        rospy.sleep(0.5)
    plt.show()

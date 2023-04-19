##ISAM Implementation

#Imports
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

from scipy.signal import butter,filtfilt


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
###############################

def callback_imu(data):
    auv_isam.update_imu(data) # whatever this is
    #auv_isam.imu_transforms.append(auv_isam.last_imu_transform)

def callback_odom(data):
    auv_isam.update_odom(data)
    #print("update odom")

def callback_dvl(data):
    auv_isam.update_dvl(data)
    #print(data)

def callback_mavros_vel(data):
    auv_isam.update_mavros_vel(data)
    #print(data)

###############################
#
#   Utiliy Functions and Values
#
###############################

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
        #IMU preintegration parameters
        self.PARAMS, self.BIAS_COVARIANCE, self.DELTA = self.preintegration_parameters()

        #Create prior pose
        self.pose_0 = gtsam.Pose3(gtsam.Rot3([[0, 0, -1], [1, 0, 0], [0, -1, 0]]), [0, 0, 0])
        self.delta_t = 1.0/18  # makes for 10 degrees per step

        #Create Prior Non-linear factor graph and ISAM solver
        self.graph = NonlinearFactorGraph()
        self.isam = ISAM2()
        self.initialEstimate = Values()
        self.timestamp = 0

        #Define noise model for prior pose, velocity, and bias
        self.posenoise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05]))
        self.biasnoise = gtsam.noiseModel.Isotropic.Sigma(6, 0.5)
        self.velnoise = gtsam.noiseModel.Isotropic.Sigma(3, 0.5)

        #Create integrator to accumulate the dx and dv values
        self.accum = gtsam.PreintegratedImuMeasurements(self.PARAMS)
        self.total_accum = gtsam.PreintegratedImuMeasurements(self.PARAMS) #Is not reset, represents total change

        #Bias key
        self.biasKey = B(0)

        #Create a constant acceleration and gyro bias for IMU
        accBias = np.array([-0.3, 0.1, 0.2])
        gyroBias = np.array([0.1, 0.3, -0.1])
        self.bias = gtsam.imuBias.ConstantBias(accBias, gyroBias)

        #Insert Bias
        self.initialEstimate.insert(self.biasKey, self.bias)

        #Accum plotting variables
        self.accum_values = []
        self.total_accum_values = []


        ### DATA TYPES
        self.mav_vel = None
        self.odom = None
        self.dvl = None
        self.imu_data = None
        self.prev_imu_data = None

        ### DATA NOISE
        self.dvl_model = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)

        #Transforms for the dvl, IMU, and rotation from map->robot.  Initialized to identity until we receive their true value
        self.last_imu_transform = np.eye(3)
        self.last_dvl_transform = np.eye(3)
        self.g_transform = np.eye(3)

    #Define integration parameters for the IMU
    def preintegration_parameters(self):
        # IMU preintegration parameters
        # Default Params for a Z-up navigation frame, such as ENU: gravity points along negative Z-axis
        PARAMS = gtsam.PreintegrationParams.MakeSharedU(g)
        I = np.eye(3)
        PARAMS.setAccelerometerCovariance(I * 0.1)#8.999999999999999e-08 ) 
        PARAMS.setGyroscopeCovariance(I * 0.1)#1.2184696791468346e-07) 
        PARAMS.setIntegrationCovariance(I * 1e-1) 
        PARAMS.setUse2ndOrderCoriolis(False)
        PARAMS.setOmegaCoriolis(vector3(0, 0, 0))

        BIAS_COVARIANCE = gtsam.noiseModel.Isotropic.Variance(6, 0.1)
        DELTA = Pose3(Rot3.Rodrigues(0, 0, 0),
                    Point3(0.05, -0.10, 0.20))

        return PARAMS, BIAS_COVARIANCE, DELTA
    
    ################################
    #
    #   Update Data
    #
    ################################
    
    def update_imu(self, data):
        #print("IMU updated")
        #Store the previous value and update self.imu
        self.prev_imu_data = self.imu_data
        self.imu_data = data
        
        #Get the map->robot rotation matrix from teh imu data
        imu_transform = gtsam.Rot3.Quaternion(self.imu_data.orientation.w, 
                                          self.imu_data.orientation.x, 
                                          self.imu_data.orientation.y, 
                                          self.imu_data.orientation.z).matrix()
        transformed_grav = (np.dot(imu_transform, n_gravity))

        #Read IMU acceleration and remove the acceleration due to gravity
        measAcc = np.array([self.imu_data.linear_acceleration.x, 
                            self.imu_data.linear_acceleration.y, 
                            self.imu_data.linear_acceleration.z]) + transformed_grav
        #Prints for debugging
        #print("raw linear accel", self.imu_data.linear_acceleration)
        #print("gravity in bot frame", transformed_grav)
        #print("final accel with gravity removed", measAcc)

        #Read IMU angular velocity
        measOmega = np.array([self.imu_data.angular_velocity.x, self.imu_data.angular_velocity.y, self.imu_data.angular_velocity.z])

        if (self.prev_imu_data is not None):
            #Calculate the chagne in time between the two IMU updates
            current_time_s = self.imu_data.header.stamp.secs
            current_time_ns = self.imu_data.header.stamp.nsecs
            prev_time_s = self.prev_imu_data.header.stamp.secs
            prev_time_ns = self.prev_imu_data.header.stamp.nsecs
            dt = (current_time_s - prev_time_s) + 1e-9*(current_time_ns - prev_time_ns)
        else:
            dt = 0.1

        #Integrate the IMU measurements to give us change in position and velocity
        self.accum.integrateMeasurement(measAcc, measOmega, dt)
        self.total_accum.integrateMeasurement(measAcc, measOmega, dt)

        #Append the accum values into the printing arrays for later figure generation
        self.accum_values.append(self.accum)
        #self.accum.print()
        self.total_accum_values.append(self.total_accum)
        #self.total_accum.print()

        return
    
    #Get Odom data and update the dictionary values
    def update_odom(self, data):
        self.odom = {"x": data.pose.pose.position.x, 
                     "y": data.pose.pose.position.y,
                     "z": data.pose.pose.position.z,
                     "i": data.pose.pose.orientation.x,
                     "j": data.pose.pose.orientation.y,
                     "k": data.pose.pose.orientation.z, 
                     "q": data.pose.pose.orientation.w}
        if ((self.imu_data is not None) and (self.mav_vel is not None)):
            auv_isam.update()
        return
    
    #Get the DVL data and update the array values
    def update_dvl(self, data):
        self.dvl = np.array([ data.twist.linear.x, data.twist.linear.y, data.twist.linear.z])
        return
    
    def update_mavros_vel(self, data):
        # print("mavros vel Update")
        self.mav_vel = np.array([data.twist.linear.x,
                                 data.twist.linear.y,
                                 data.twist.linear.z])
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
    
    #Filtering of IMU data
    def smooth_imu(self, data):
        # low pass filter to smooth jittery IMU
        # Filter requirements.
        # T = 5.0         # Sample Period
        fs = 30       # sample rate, Hz
        cutoff = 0.5      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
        nyq = 0.5 * fs  # Nyquist Frequency
        order = 3       # sin wave can be approx represented as quadratic
        # n = int(T * fs) # total number of samples

        normal_cutoff = cutoff / nyq
        # Get the filter coefficients 

        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data.T)

        return y
    def create_imu_factor(self):
        print("Created IMU Factor")
        imuFactor = ImuFactor(X(self.timestamp), V(self.timestamp), X(self.timestamp + 1), V(self.timestamp + 1), auv_isam.biasKey, auv_isam.accum)
        auv_isam.accum.resetIntegration()
        return imuFactor
    
    def create_dvl_factor(self):
        print("Created DVL Factor")
        dvl_factor = gtsam.CustomFactor(
                        self.dvl_model,
                        V(self.timestamp), 
                        partial(self.velocity_error, np.array([self.dvl])))
        
        return dvl_factor
    
    def update(self):
        result = None
        print("Time: ", self.timestamp)
        velocity = vector3(self.mav_vel[0], self.mav_vel[1], self.mav_vel[2])
        #Get current pose from odometry
        pose = gtsam.Pose3(gtsam.Rot3.Quaternion(self.odom['q'], 
                                                 self.odom['i'], 
                                                 self.odom['j'], 
                                                 self.odom['k']), 
                                                 [self.odom['x'], 
                                                  self.odom['y'], 
                                                  self.odom['z']])
        
        if self.timestamp == 0:
            #Add prior into graph on the first iteration
            self.graph.add(gtsam.PriorFactorPose3(X(0), pose, self.posenoise))
            self.graph.add(gtsam.PriorFactorVector(V(0), velocity, self.velnoise))

            #Add pose and velocity to nodes
            self.initialEstimate.insert(X(self.timestamp), pose)
            self.initialEstimate.insert(V(self.timestamp), velocity)
        else:
            #Add pose and velocity to nodes
            self.initialEstimate.insert(X(self.timestamp), pose)
            #velocity = np.array([pose.x(), pose.y(), pose.z()]) - np.array([self.pose_0.x(), self.pose_0.y(), self.pose_0.z()])
            self.initialEstimate.insert(V(self.timestamp), velocity)

            #Create the IMU factor
            imuFactor = self.create_imu_factor()
            #print(imuFactor)
            dvlFactor = self.create_dvl_factor()

            #Add factors to graph
            self.graph.add(imuFactor)
            self.graph.add(dvlFactor)
        self.graph.print()
        if (self.timestamp > -1):
            #Add graph factors and nodes to the ISAM graph
            self.isam.update(self.graph, self.initialEstimate)
            result = self.isam.calculateEstimate()
            plot.plot_incremental_trajectory(0, 
                                            result, 
                                            start=self.timestamp, 
                                            scale=1, 
                                            time_interval=0.01)
            
            self.graph = gtsam.NonlinearFactorGraph()
            self.initialEstimate = gtsam.Values()

        self.timestamp += 1
        return result
    
if __name__ == '__main__':
    #Create rospy listeners for the ros data
    rospy.init_node('data_listener', anonymous=True)

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    rospy.Subscriber('/zedm/zed_node/imu/data', Imu, callback_imu)
    rospy.Subscriber('/dvl/local_position', PoseWithCovarianceStamped, callback_odom)
    rospy.Subscriber('/dvl/twist', TwistStamped, callback_dvl)
    rospy.Subscriber('/mavros/local_position/velocity_local', TwistStamped, callback_mavros_vel)

    auv_isam = AUV_ISAM()
    while not rospy.is_shutdown():
        continue

# pylint: disable=invalid-name, E1101

from __future__ import print_function


import rospy
import rosnode
#from uslam.isam import AUV_ISAM
from std_msgs.msg import String
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistStamped
# from waterlinked_a50_ros_driver.msg import DVL
from typing import Optional, List
import sys
import tf2_ros
from functools import partial
import json

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
################################

def callback_imu(data):
    auv_isam.update_imu(data)

def callback_odom(data):
    auv_isam.update_odom(data)

def callback_mavros_vel(data):
    auv_isam.update_mavros_vel(data)

def callback_dvl(data):
    auv_isam.update_dvl(data)

def callback_pressure(data):
    auv_isam.update_pressure(data)

def callback_imu_transform(transform):
    
    auv_isam.last_imu_transform = transform
    if auv_isam.do_accum == True:
        auv_isam.odom_accum.append(auv_isam.odom)

        print(auv_isam.last_imu_transform)

        transformed_grav = np.flip(np.dot(auv_isam.last_imu_transform.T, auv_isam.g))

        measAcc = np.array([auv_isam.imu_data.linear_acceleration.x, 
                            auv_isam.imu_data.linear_acceleration.y, 
                            auv_isam.imu_data.linear_acceleration.z]) - transformed_grav
        print("raw linear accel", auv_isam.imu_data.linear_acceleration)
        print("gravity in bot frame", transformed_grav)
        print("final accel with gravity removed", measAcc)
        measOmega = np.array([auv_isam.imu_data.angular_velocity.x, auv_isam.imu_data.angular_velocity.y, auv_isam.imu_data.angular_velocity.z])
        auv_isam.imu_accum.append(np.array([measAcc, measOmega]))


        auv_isam.dvl_accum.append(auv_isam.dvl)

## ## Adds all most recent values to arrays 
def record_all_data():
    auv_isam.odom_accum.append(auv_isam.odom)
    auv_isam.dvl_accum.append(auv_isam.dvl)
    auv_isam.depth_accum.append(auv_isam.depth)

    index = int(len(auv_isam.imu_factors))

    if index == 0:
        ## THIS WILL NEVER GET ADDED 
        imuFactor = ImuFactor(X(0), V(0), X(0), V(0), auv_isam.biasKey, auv_isam.accum)
    else:
        imuFactor = ImuFactor(X(index - 1), V(index - 1), X(index), V(index), auv_isam.biasKey, auv_isam.accum)
    auv_isam.imu_factors.append(imuFactor)
    auv_isam.accum.resetIntegration()

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
        self.noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05]))
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
        self.prev_imu_data = None

        ### DATA NOISE
        self.dvl_model = gtsam.noiseModel.Isotropic.Sigma(3, 0.1) ## Bagoren et al


        self.last_imu_transform = np.eye(3)
        self.last_dvl_transform = np.eye(3)

        self.g_transform = np.eye(3)


        self.grav = 9.81
        self.g = vector3(0, 0, -self.grav)

        # batch
        self.do_accum = True
        self.odom_accum = []
        self.imu_accum = []
        self.depth_accum = []
        self.imu_accum_all = np.empty((0, 3))
        self.imu_accum_ang_all = np.empty((0, 3))
        self.imu_accum_indices = []
        self.imu_factors = []
        self.dvl_accum = []
        self.batch_graph = gtsam.NonlinearFactorGraph()
        self.batch_initial = gtsam.Values()


    def preintegration_parameters(self):
        # IMU preintegration parameters
        # Default Params for a Z-up navigation frame, such as ENU: gravity points along negative Z-axis
        PARAMS = gtsam.PreintegrationParams.MakeSharedU(g)
        I = np.eye(3)
        PARAMS.setAccelerometerCovariance(I * 1e-1 ) 
        PARAMS.setGyroscopeCovariance(I * 1e-1 ) 
        PARAMS.setIntegrationCovariance(I * 1e-1 ) 
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

        self.prev_imu_data = self.imu_data
        self.imu_data = data

        if self.imu_data.angular_velocity.x > 0.3: self.imu_data.angular_velocity.x = 0.3
        if self.imu_data.angular_velocity.y > 0.3: self.imu_data.angular_velocity.y = 0.3
        if self.imu_data.angular_velocity.z > 0.3: self.imu_data.angular_velocity.z = 0.3

        if self.imu_data.angular_velocity.x < -0.3: self.imu_data.angular_velocity.x = -0.3
        if self.imu_data.angular_velocity.y < -0.3: self.imu_data.angular_velocity.y = -0.3
        if self.imu_data.angular_velocity.z < -0.3: self.imu_data.angular_velocity.z = -0.3
        
        imu_transform = gtsam.Rot3.Quaternion(self.imu_data.orientation.w, 
                                          self.imu_data.orientation.x, 
                                          self.imu_data.orientation.y, 
                                          self.imu_data.orientation.z).matrix()

        transformed_grav = (np.dot(imu_transform, self.g))

        measAcc = np.array([self.imu_data.linear_acceleration.x, 
                            self.imu_data.linear_acceleration.y, 
                            self.imu_data.linear_acceleration.z]) + transformed_grav
        print("raw linear accel", self.imu_data.linear_acceleration)
        print("gravity in bot frame", transformed_grav)
        print("final accel with gravity removed", measAcc)
        measOmega = np.array([self.imu_data.angular_velocity.x, self.imu_data.angular_velocity.y, self.imu_data.angular_velocity.z])


        self.imu_accum_all = np.append(self.imu_accum_all, [measAcc], axis=0)
        self.imu_accum_ang_all = np.append(self.imu_accum_ang_all, [measOmega], axis=0)

        current_time_s = self.imu_data.header.stamp.secs
        current_time_ns = self.imu_data.header.stamp.nsecs
        prev_time_s = self.prev_imu_data.header.stamp.secs
        prev_time_ns = self.prev_imu_data.header.stamp.nsecs

        dt = (current_time_s - prev_time_s) + 1e-9*(current_time_ns - prev_time_ns)
        self.accum.integrateMeasurement(measAcc, measOmega, dt)

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
    
    def update_pressure(self, data):
        dict = json.loads(data.data)
        measured_pressure = dict['press_abs'] * 100
        pressure_diff = measured_pressure - 98250.0
        depth = pressure_diff / (997 * 9.81)
        self.depth = depth
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
        # delta_t = .25
        # self.accum.integrateMeasurement(self.smoothed_imu[:,self.imu_accum_indices[index]].T, self.smoothed_imu_ang[:,self.imu_accum_indices[index]].T, delta_t)
        ## TODO maybe add multiple imu and reset
        imuFactor = self.imu_factors[index]
        return imuFactor
    
    def create_dvl_factor_batch(self, index):
        dvl_factor = gtsam.CustomFactor(
                        self.dvl_model,
                        [index], 
                        partial(self.velocity_error, np.array([self.dvl_accum[index]])),
                    )
        return dvl_factor
    
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
            t = gtsam.Point3(currPose['x'], currPose['y'], self.depth_accum[i])
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
                
        return

    
def constr3DPoints(values):
    i = 0
    points = np.empty((1,3))
    while values.exists(X(i)):
        pose_i = values.atPose3(X(i))
        point = np.array([pose_i.x(), pose_i.y(), pose_i.z()])
        points = np.append(points, [point], axis=0)
        i += 1

    print(values.exists(i))

    return points

if __name__ == '__main__':
    rospy.init_node('data_listener', anonymous=True)

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    rospy.Subscriber('/zedm/zed_node/imu/data', Imu, callback_imu)
    # rospy.Subscriber('/mavros/imu/data', Imu, callback_imu)
    # rospy.Subscriber('/zedm/zed_node/odom', Odometry, callback_odom)
    rospy.Subscriber('/dvl/local_position', PoseWithCovarianceStamped, callback_odom)
    # rospy.Subscriber('/mavros/local_position/velocity_local', TwistStamped, callback_mavros_vel)
    rospy.Subscriber('/dvl/twist', TwistStamped, callback_dvl)

    rospy.Subscriber('/BlueROV/pressure2', String, callback_pressure)

    auv_isam = AUV_ISAM()

    while not rospy.is_shutdown():

    

        if auv_isam.prev_imu_data is not None:
            record_all_data()


        if 'play' not in '\t'.join(rosnode.get_node_names()):

            # # plot imu linear
            # fig, axs = plt.subplots(3)
            # axs[0].plot(list(range(len(auv_isam.imu_accum_all[:,0]))), auv_isam.imu_accum_all[:,0])
            # axs[1].plot(list(range(len(auv_isam.imu_accum_all[:,1]))), auv_isam.imu_accum_all[:,1])
            # axs[2].plot(list(range(len(auv_isam.imu_accum_all[:,2]))), auv_isam.imu_accum_all[:,2])
            # plt.show()

            # auv_isam.smoothed_imu = np.array(auv_isam.smooth_imu(auv_isam.imu_accum_all))

            # fig, axs = plt.subplots(3)
            # axs[0].plot(list(range(len(auv_isam.smoothed_imu[0]))), auv_isam.smoothed_imu[0])
            # axs[1].plot(list(range(len(auv_isam.smoothed_imu[1]))), auv_isam.smoothed_imu[1])
            # axs[2].plot(list(range(len(auv_isam.smoothed_imu[2]))), auv_isam.smoothed_imu[2])
            # plt.show()

            # # plot imu angular
            # fig, axs = plt.subplots(3)
            # axs[0].plot(list(range(len(auv_isam.imu_accum_ang_all[:,0]))), auv_isam.imu_accum_ang_all[:,0])
            # axs[1].plot(list(range(len(auv_isam.imu_accum_ang_all[:,1]))), auv_isam.imu_accum_ang_all[:,1])
            # axs[2].plot(list(range(len(auv_isam.imu_accum_ang_all[:,2]))), auv_isam.imu_accum_ang_all[:,2])
            # plt.show()

            # auv_isam.smoothed_imu_ang = np.array(auv_isam.smooth_imu(auv_isam.imu_accum_ang_all))

            # fig, axs = plt.subplots(3)
            # axs[0].plot(list(range(len(auv_isam.smoothed_imu_ang[0]))), auv_isam.smoothed_imu_ang[0])
            # axs[1].plot(list(range(len(auv_isam.smoothed_imu_ang[1]))), auv_isam.smoothed_imu_ang[1])
            # axs[2].plot(list(range(len(auv_isam.smoothed_imu_ang[2]))), auv_isam.smoothed_imu_ang[2])
            # plt.show()



            auv_isam.do_accum = True
            print("odom lentgth:", len(auv_isam.odom_accum))
            print("imu lentgth:", len(auv_isam.imu_accum))

            auv_isam.createBatch()
            results = gtsam.LevenbergMarquardtOptimizer(auv_isam.batch_graph, auv_isam.batch_initial, gtsam.LevenbergMarquardtParams()).optimize()

            # marginals = gtsam.Marginals(auv_isam.batch_graph, results)
            # print("Covariance on bias:\n", marginals.marginalCovariance(auv_isam.biasKey))
            # for i in range(auv_isam.timestamp - 1):
            #     print("Covariance on pose {}:\n{}\n".format(
            #         i, marginals.marginalCovariance(X(i))))
            #     print("Covariance on vel {}:\n{}\n".format(
            #         i, marginals.marginalCovariance(V(i))))


            auv_isam.timestamp += 1

            break
        
        rospy.sleep(0.25)

    points = constr3DPoints(results)


    # print(results)
    #plot.plot_trajectory(1, results)
    #plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x = [auv_isam.odom_accum[i]['x'] for i in range(len(auv_isam.odom_accum))]
    y = [auv_isam.odom_accum[i]['y'] for i in range(len(auv_isam.odom_accum))]
    z = [auv_isam.odom_accum[i]['z'] for i in range(len(auv_isam.odom_accum))]
    ax.plot3D(x, y, z, color='orange', linewidth=2)
    ax.plot3D(points[1:,0], points[1:,1], points[1:,2], color='blue')

    plt.show()

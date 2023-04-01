"""
ImuFactor example with iSAM2.
Authors: Robert Truax (C++), Frank Dellaert, Varun Agrawal (Python)
"""
# pylint: disable=invalid-name, E1101

from __future__ import print_function


import rospy
from uslam.isam import AUV_ISAM
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistStamped
# from waterlinked_a50_ros_driver.msg import DVL
import sys
import tf2_ros


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


def callback_imu(data):
    auv_isam.update_imu(data) # whatever this is
    #print(data)

def callback_odom(data):
    auv_isam.update_odom(data)
    #print(data.pose.pose.position)

def callback_mavros_vel(data):
    auv_isam.update_mavros_vel(data)
    print(data)


def vector3(x, y, z):
    """Create 3d double numpy array."""
    return np.array([x, y, z], dtype=float)


g = 9.81
n_gravity = vector3(0, 0, -g)

#x = [4.820593232460892, 4.876136622758349, 4.91834351289019, 4.934849086081899, 4.959993648261925, 4.9872025547738765, 5.0079819483582355]
#y = [0.38682050539411933, 0.3971632041846082, 0.401421910811749, 0.4156703408906957, 0.4260122680672318, 0.43461185988587325, 0.43943945117809496]
#z = [0.04080228845815051, 0.030272680948478296, 0.03397887301502824, 0.03459286951347812, 0.03237211325028328, 0.034703925922842674, 0.049627694983907435]

class AUV_ISAM:
    def __init__(self):
        self.PARAMS, self.BIAS_COVARIANCE, self.DELTA = self.preintegration_parameters()
        self.radius = 30
        self.camera = self.get_camera(self.radius)
        self.pose_0 = gtsam.Pose3(gtsam.Rot3([[0, 0, -1], [1, 0, 0], [0, -1, 0]]), [0, 0, 0])
        self.delta_t = 1.0/18  # makes for 10 degrees per step
        self.angular_velocity = math.radians(180)  # rad/sec
        self.scenario = self.get_scenario(self.radius, self.pose_0, self.angular_velocity)
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

        # Calculate with correct initial velocity
        self.n_velocity = vector3(0, 0, 0)
        self.velprior = PriorFactorVector(V(0), self.n_velocity, self.velnoise)
        self.graph.push_back(self.velprior)
        self.initialEstimate.insert(V(0), self.n_velocity)

        self.accum = gtsam.PreintegratedImuMeasurements(self.PARAMS)

        self.mav_vel = None
        self.odom = None

        self.g_transform = None
        self.grav = 9.81
        self.g = vector3(0, 0, -self.grav)


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


    def get_camera(self, radius):
        up = Point3(0, 0, 1)
        target = Point3(0, 0, 0)
        position = Point3(radius, 0, 0)
        camera = PinholeCameraCal3_S2.Lookat(position, target, up, Cal3_S2())
        return camera


    def get_scenario(self, radius, pose_0, angular_velocity):
        """Create the set of ground-truth landmarks and poses"""
        angular_velocity_vector = vector3(0, -angular_velocity, 0)
        linear_velocity_vector = vector3(radius * angular_velocity, 0, 0)
        scenario = ConstantTwistScenario(
            angular_velocity_vector, linear_velocity_vector, pose_0)

        return scenario
    
    def update_imu(self, data):
        #print("IMU Update")
        measAcc = np.array([data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z]) + np.dot(self.g_transform, self.g)
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
    
    def create_imu_factor(self):
        delta_t = .01
        self.accum.integrateMeasurement(self.imu[0], self.imu[1], delta_t)
        imuFactor = ImuFactor(X(self.timestamp - 1), V(self.timestamp - 1), X(self.timestamp), V(self.timestamp), self.biasKey, self.accum)
        return imuFactor
    
    def create_mavros_vel_factor(self):
        # 
        return
    
    def get_factors(self):
        imuFactor = self.create_imu_factor()
        # dvlFactor = self.create_dvl_factor()
        # depthFactor = self.create_depth_factor()
        # orbFactor = self.create_orb_factor()
        factors = np.array([imuFactor])
        return factors


# def IMU_example_setup():
#     """Run iSAM 2 example with IMU factor."""

#     # Start with a camera on x-axis looking at origin
#     radius = 30
#     #camera = get_camera(radius)
#     #pose_0 = camera.pose()
#     pose_0 = gtsam.Pose3(gtsam.Rot3([[0, 0, -1], [1, 0, 0], [0, -1, 0]]), [x[0], y[0], z[0]])

#     delta_t = 1.0/18  # makes for 10 degrees per step
#     angular_velocity = math.radians(180)  # rad/sec
#     scenario = get_scenario(radius, pose_0, angular_velocity, delta_t)

#     PARAMS, BIAS_COVARIANCE, DELTA = preintegration_parameters()
    
#     # Create a factor graph
#     graph = NonlinearFactorGraph()

#     # Create (incremental) ISAM2 solver
#     isam = ISAM2()

#     # Create the initial estimate to the solution
#     # Intentionally initialize the variables off from the ground truth
#     initialEstimate = Values()

#     # Add a prior on pose x0. This indirectly specifies where the origin is.
#     # 30cm std on x,y,z 0.1 rad on roll,pitch,yaw
#     noise = gtsam.noiseModel.Diagonal.Sigmas(
#         np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3]))
#     graph.push_back(PriorFactorPose3(X(0), pose_0, noise))

#     # Add imu priors
#     biasKey = B(0)
#     biasnoise = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
#     biasprior = PriorFactorConstantBias(biasKey, gtsam.imuBias.ConstantBias(),
#                                         biasnoise)
#     graph.push_back(biasprior)
#     initialEstimate.insert(biasKey, gtsam.imuBias.ConstantBias())
#     velnoise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)

#     # Calculate with correct initial velocity
#     n_velocity = vector3(0, angular_velocity * radius, 0)
#     velprior = PriorFactorVector(V(0), n_velocity, velnoise)
#     graph.push_back(velprior)
#     initialEstimate.insert(V(0), n_velocity)

#     accum = gtsam.PreintegratedImuMeasurements(PARAMS)

    def update(self):

        # Simulate poses and imu measurements, adding them to the factor graph
            t = self.timestamp * self.delta_t  # simulation time
            if self.timestamp == 0:  # First time add two poses
                #pose_1 = self.scenario.pose(self.delta_t)
                self.initialEstimate.insert(X(0), gtsam.Pose3(gtsam.Rot3([[0, 0, -1], [1, 0, 0], [0, -1, 0]]), [self.odom['x'], self.odom['y'], self.odom['z']]))
                self.initialEstimate.insert(X(1), gtsam.Pose3(gtsam.Rot3([[0, 0, -1], [1, 0, 0], [0, -1, 0]]), [self.odom['x'], self.odom['y'], self.odom['z']]))
            elif self.timestamp >= 2:  # Add more poses as necessary
                #pose_i = self.scenario.pose(t)
                self.initialEstimate.insert(X(self.timestamp), gtsam.Pose3(gtsam.Rot3([[0, 0, -1], [1, 0, 0], [0, -1, 0]]), [self.odom['x'], self.odom['y'], self.odom['z']]))

            if self.timestamp > 0:
                # Add Bias variables periodically
                if self.timestamp % 5 == 0:
                    self.biasKey += 1
                    factor = BetweenFactorConstantBias(
                        self.biasKey - 1, self.biasKey, gtsam.imuBias.ConstantBias(), self.BIAS_COVARIANCE)
                    self.graph.add(factor)
                    self.initialEstimate.insert(self.biasKey, gtsam.imuBias.ConstantBias())

                # Predict acceleration and gyro measurements in (actual) body frame
                # nRb = self.scenario.rotation(t).matrix() #Gets rotation matrix from body to navigational frame
                # bRn = np.transpose(nRb) #Converts nRb to be from navigational to body
                # measuredAcc = self.scenario.acceleration_b(t) - np.dot(bRn, n_gravity) #removes measured gravitational accel
                # measuredOmega = self.scenario.omega_b(t) #Gets gryo measurement
                # self.accum.integrateMeasurement(measuredAcc, measuredOmega, self.delta_t)

                # Add Imu Factor
                #imufac = ImuFactor(X(self.timestamp - 1), V(self.timestamp - 1), X(self.timestamp), V(self.timestamp), self.biasKey, self.accum)
                imufac = self.get_factors()[0]
                self.graph.add(imufac)

                # insert new velocity, which is wrong
                if self.mav_vel is not None:
                    self.initialEstimate.insert(V(self.timestamp), vector3(self.mav_vel['x'], self.mav_vel['y'], self.mav_vel['z']))
                else:
                    self.initialEstimate.insert(V(self.timestamp), vector3(0,0,0))
                self.accum.resetIntegration()

            # Incremental solution
            self.isam.update(self.graph, self.initialEstimate)
            result = self.isam.calculateEstimate()
            plot.plot_incremental_trajectory(0, result,
                                            start=self.timestamp, scale=3, time_interval=0.01)

            # reset
            self.graph = NonlinearFactorGraph()
            self.initialEstimate.clear()
            self.timestamp += 1

    


if __name__ == '__main__':
    rospy.init_node('data_listener', anonymous=True)

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    rospy.Subscriber('/mavros/imu/data_raw', Imu, callback_imu)
    rospy.Subscriber('/dvl/local_position', PoseWithCovarianceStamped, callback_odom)
    rospy.Subscriber('/mavros/velocity_local', TwistStamped, callback_mavros_vel)
    
    # rospy.Subscriber('/dev/data', DVL, callback_dvl)

    auv_isam = AUV_ISAM()

    while not rospy.is_shutdown():
        #auv_isam.g_transform = tfBuffer.lookup_transform('map', 'base_link', rospy.Time().now(), rospy.Duration(3.0))
        got_transform = False
        while not got_transform:
            try:
                transform = tfBuffer.lookup_transform('map', 'base_link', rospy.Time(0))
                got_transform = True
                auv_isam.g_transform = gtsam.Rot3.Quaternion(transform.transform.rotation.w, 
                                                             transform.transform.rotation.x, 
                                                             transform.transform.rotation.y, 
                                                             transform.transform.rotation.x).matrix()
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                print("exception in transform lookup loop")
                continue

        if auv_isam.odom is not None:
            auv_isam.update()

            #plt.show()
        rospy.sleep(0.5)

    plt.show()
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
import tf
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
                   PriorFactorVector, Rot3, Values, StereoPoint2, GenericStereoFactor3D, Cal3_S2Stereo)
from gtsam.symbol_shorthand import B, V, X, L
from gtsam.utils import plot
import message_filters
from gtsam_vio.msg import CameraMeasurement
###############################
#
#   Callbacks, Global
#
################################


def ts_callback(imu, odom, dvl, landmarks):
    auv_isam.update_imu(imu)
    auv_isam.update_odom(odom)
    auv_isam.update_dvl(dvl)
    auv_isam.update_landmarks(landmarks)

"""
def callback_imu(data):
    auv_isam.update_imu(data) # whatever this is
    # auv_isam.imu_transforms.append(auv_isam.last_imu_transform)

    #print(data)

def callback_odom(data):
    auv_isam.update_odom(data)
    #print(data.pose.pose.position)

def callback_mavros_vel(data):
    auv_isam.update_mavros_vel(data)
    #print(data)

def callback_dvl(data):
    auv_isam.update_dvl(data)
    #print(data)
"""

def callback_imu_transform(transform):
    
    auv_isam.last_imu_transform = transform
    # auv_isam.imu_transforms.append(transform)

def callback_dvl_transform(transform):
    
    auv_isam.last_dvl_transform = transform

def vector3(x, y, z):
    """Create 3d double numpy array."""
    return np.array([x, y, z], dtype=float)


g = 9.81
n_gravity = vector3(0, 0, -g)

#x = [4.820593232460892, 4.876136622758349, 4.91834351289019, 4.934849086081899, 4.959993648261925, 4.9872025547738765, 5.0079819483582355]
#y = [0.38682050539411933, 0.3971632041846082, 0.401421910811749, 0.4156703408906957, 0.4260122680672318, 0.43461185988587325, 0.43943945117809496]
#z = [0.04080228845815051, 0.030272680948478296, 0.03397887301502824, 0.03459286951347812, 0.03237211325028328, 0.034703925922842674, 0.049627694983907435]

###############################
#
#   ISAM Class
#
################################

class AUV_ISAM:
    def __init__(self):
        self.PARAMS, self.BIAS_COVARIANCE, self.DELTA = self.preintegration_parameters()
        self.radius = 30
        self.camera = self.get_camera(self.radius)
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

        ### DATA NOISE
        self.dvl_model = gtsam.noiseModel.Isotropic.Sigma(3, 0.1) ## Bagoren et al


        self.last_imu_transform = np.eye(3)
        self.last_dvl_transform = np.eye(3)
        # self.g_transform = np.eye(3)

        self.grav = 9.81
        self.g = vector3(0, 0, -self.grav)

        # batch
        self.do_accum = True
        self.odom_accum = []
        self.imu_accum = []
        self.landmark_accum = []
        self.dvl_accum = []
        self.batch_graph = gtsam.NonlinearFactorGraph()
        self.batch_initial = gtsam.Values()

        #landmark stuff
        self.cam_map_transform = None
        self.intrinsic = [669.2387507702717, 669.6062139634853, 647.4136147885813, 348.40757297218505]
        self.f = (self.intrinsic[0] + self.intrinsic[1]) / 2.0
        self.cx = self.intrinsic[2]
        self.cy = self.intrinsic[3]
        self.Tx = -0.001344472879883825
        self.K = Cal3_S2Stereo(self.intrinsic[0], self.intrinsic[1], 0.0, self.cx, self.cy, self.Tx)
        self.resolution_x = 1920
        self.resolution_y = 1080
        self.landmark_noise = gtsam.noiseModel.Isotropic.Sigma(3, 10)


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
    
    ###############################
    #
    #   Update Data
    #
    ################################
    
    def update_imu(self, data):
        # print("IMU Update")
        #print("linear accel raw", np.array([data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z]))
        #print("transform: ", self.g_transform)
        #print("transformed gravity ", np.dot(self.g_transform, self.g))
        if (np.array([data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z]) is None):
            print(data)
        measAcc = np.array([data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z]) +  np.dot(self.last_imu_transform, self.g)
        #print("final accel with gravity removed", measAcc)
        measOmega = np.array([data.angular_velocity.x, data.angular_velocity.y, data.angular_velocity.z])
        #print('here', measAcc)
        self.imu = np.array([measAcc, measOmega])

        if self.do_accum == True:
            self.imu_accum.append(self.imu)
            print(len(self.imu_accum))
        return

    def update_odom(self, data):
        self.odom = {"x": data.pose.pose.position.x, 
                     "y": data.pose.pose.position.y,
                     "z": data.pose.pose.position.z,
                     "i": data.pose.pose.orientation.x,
                     "j": data.pose.pose.orientation.y,
                     "k": data.pose.pose.orientation.z, 
                     "q": data.pose.pose.orientation.w}
        if self.do_accum == True:
            self.odom_accum.append(self.odom)
            print(len(self.odom_accum))

        return

    def update_mavros_vel(self, data):
        # print("mavros vel Update")
        self.mav_vel = {"x" : data.twist.linear.x,
                    "y" : data.twist.linear.y,
                    "z" : data.twist.linear.z}
        return
    
    def update_dvl(self, data):
        # print("dvlUpdate")
        # stays in dvl link frame, but error function changes it to world
        self.dvl_accum.append([ data.twist.linear.x, data.twist.linear.y, data.twist.linear.z])
        return
    
    def update_landmarks(self, data):

        #should give a list of features
        landmarks = []
        if self.cam_map_transform != None:
            for feature in data.features:
                id = feature.id

                uL = (feature.u0 + 1) * 0.5 * self.resolution_x;
                uR = (feature.u1 + 1) * 0.5 * self.resolution_x
                v = ((feature.v0 + feature.v1) / 2.0 + 1) * 0.5 * self.resolution_y

                d = uR - uL
                x = uL
                y = v
                W = d / self.Tx
                x_cam = (x - self.cx) / W
                y_cam = (y - self.cy) / W
                z_cam = self.f / W

                cam_point = np.array([[x_cam], [y_cam], [z_cam]])
                #print(self.cam_map_transform)
                world_point = self.cam_map_transform[0].matrix() @ cam_point + self.cam_map_transform[1].reshape(3,1)
                #print(world_point)
                landmarks.append({
                    'id': id,
                    'pose': world_point.reshape((3,)),
                    'uL': uL,
                    'uR': uR,
                    'v': v
                })
        
        self.landmark_accum.append(landmarks)
            
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
        if jacobians is not None:
            jacobians[0] = rot_mat
        return error


    def update(self):

        # Simulate poses and imu measurements, adding them to the factor graph
            t = self.timestamp * self.delta_t  # simulation time
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
                    self.initialEstimate.insert(V(self.timestamp), vector3(0,0,0))
                else:
                    #print("not adding vel", self.mav_vel)
                    self.initialEstimate.insert(V(self.timestamp), vector3(0,0,0))
                self.accum.resetIntegration()

            # Incremental solution
            self.isam.update(self.graph, self.initialEstimate)
            result = self.isam.calculateEstimate()
            #plot.plot_incremental_trajectory(0, result, start=self.timestamp, scale=3, time_interval=0.01)
            #plot.plot_pose3(fignum=0, pose=gtsam.Pose3(gtsam.Rot3([[0, 0, -1], [1, 0, 0], [0, -1, 0]]), [self.odom['x'], self.odom['y'], self.odom['z']]), axis_length=0.5)

            # reset
            self.graph = NonlinearFactorGraph()
            self.initialEstimate.clear()
            # self.timestamp += 1

    def create_imu_factor_batch(self, index):
        delta_t = .2
        self.accum.integrateMeasurement(self.imu_accum[index][0], self.imu_accum[index][1], delta_t)
        imuFactor = ImuFactor(X(index - 1), V(index - 1), X(index), V(index), self.biasKey, self.accum)
        return imuFactor
    
    def create_dvl_factor_batch(self, index):

        dvlFactor = gtsam.CustomFactor(
                        self.dvl_model,
                        [index], # TODO should this be X or also V?
                        partial(self.velocity_error, np.array([self.dvl_accum[index]])),
                    )
        

        return dvlFactor

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
                self.batch_graph.push_back(dvlFactor)

                for landmark in self.landmark_accum[i]:
                    if not self.batch_initial.exists(L(landmark['id'])):
                        self.batch_initial.insert(L(landmark['id']), landmark['pose'])
                    
                    stereo_factor = GenericStereoFactor3D(
                        StereoPoint2(landmark['uL'], landmark['uR'], landmark['v']),
                        self.landmark_noise,
                        X(i), L(landmark['id']), self.K
                    )
                    self.batch_graph.push_back(stereo_factor)
                
            self.batch_graph.saveGraph("test.dot")

                
        return

    


if __name__ == '__main__':
    rospy.init_node('data_listener', anonymous=True)

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    tf_listener = tf.TransformListener()


    auv_isam = AUV_ISAM()

    imu_sub = message_filters.Subscriber('/zedm/zed_node/imu/data', Imu)
    #rospy.Subscriber('/mavros/imu/data', Imu, callback_imu)
    # rospy.Subscriber('/zedm/zed_node/odom', Odometry, callback_odom)
    odom_sub = message_filters.Subscriber('/dvl/local_position', PoseWithCovarianceStamped)
    # rospy.Subscriber('/mavros/local_position/velocity_local', TwistStamped, callback_mavros_vel)
    dvl_sub = message_filters.Subscriber('/dvl/twist', TwistStamped)

    landmark_sub = message_filters.Subscriber('/auv/image_processor/features', CameraMeasurement)

    ts = message_filters.ApproximateTimeSynchronizer([imu_sub, odom_sub, dvl_sub, landmark_sub], 10, 0.2, allow_headerless=True)
    ts.registerCallback(ts_callback)


    while not rospy.is_shutdown():
        

        try:
            transform = tfBuffer.lookup_transform('map', 'base_link', rospy.Time(0))
            transform_mat = gtsam.Rot3.Quaternion(transform.transform.rotation.w, 
                                            transform.transform.rotation.x, 
                                            transform.transform.rotation.y, 
                                            transform.transform.rotation.z).matrix()
            callback_imu_transform(transform_mat)
            print('got imu transform')
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print("exception in imu transform lookup loop, using last transform")

        try:
            dvl_transform = tfBuffer.lookup_transform('map', 'dvl_link', rospy.Time(0))
            dvl_transform_mat = gtsam.Rot3.Quaternion(dvl_transform.transform.rotation.w, 
                                                         dvl_transform.transform.rotation.x, 
                                                         dvl_transform.transform.rotation.y, 
                                                         dvl_transform.transform.rotation.z).matrix()
            
            callback_dvl_transform(dvl_transform_mat)
            print('got dvl transform')
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print("exception in DVL transform lookup loop")

        if 'play' not in '\t'.join(rosnode.get_node_names()):

            auv_isam.do_accum = True
            # print("num transforms: ", len(auv_isam.imu_transforms))
            print("odom lentgth:", len(auv_isam.odom_accum))
            print("imu lentgth:", len(auv_isam.imu_accum))

            auv_isam.createBatch()
            auv_isam.batch_graph.saveGraph('graph.dot')
            results = gtsam.LevenbergMarquardtOptimizer(auv_isam.batch_graph, auv_isam.batch_initial, gtsam.LevenbergMarquardtParams()).optimize()

            auv_isam.timestamp += 1

            break

        try:
            ## todo make sure transform time matches pose time?
            transform = tfBuffer.lookup_transform('zedm_left_camera_optical_frame', 'map', rospy.Time(0))
            trans = Point3(transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z)
            rot = Rot3.Quaternion(transform.transform.rotation.w, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z)
            auv_isam.cam_map_transform = (rot, trans)
            
            # auv_isam.cam_transform = gtsam.Rot3.Quaternion(transform.transform.rotation.w, 
            #                                              transform.transform.rotation.x, 
            #                                              transform.transform.rotation.y, 
            #                                              transform.transform.rotation.z).matrix()
            
            got_transform = True
        
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            continue
        
        rospy.sleep(0.5)
    plot.plot_trajectory(1, results)
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x = [auv_isam.odom_accum[i]['x'] for i in range(len(auv_isam.odom_accum))]
    y = [auv_isam.odom_accum[i]['y'] for i in range(len(auv_isam.odom_accum))]
    z = [auv_isam.odom_accum[i]['z'] for i in range(len(auv_isam.odom_accum))]
    ax.plot3D(x, y, z)
    plt.show()
    
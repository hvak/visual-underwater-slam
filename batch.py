import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistStamped
from waterlinked_a50_ros_driver.msg import DVL

import rosnode
import tf2_ros
from functools import partial
from typing import Optional, List
import json


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=W0611

import gtsam
from gtsam import (ISAM2, BetweenFactorConstantBias, Cal3_S2,
                   ConstantTwistScenario, ImuFactor, NonlinearFactorGraph,
                   PinholeCameraCal3_S2, Point3, Pose3,
                   PriorFactorConstantBias, PriorFactorPose3,
                   PriorFactorVector, Rot3, Values, PriorFactorPoint3, NavState, 
                   Cal3_S2Stereo, StereoPoint2, GenericStereoFactor3D)
from gtsam.symbol_shorthand import B, V, X, L
from gtsam.utils import plot
import message_filters
from gtsam_vio.msg import CameraMeasurement


def imu_callback(imu):
    global old_time
    slam.update_imu(imu, 0.005)
    old_time = imu.header.stamp

def pressure_callback(pressure):
    slam.process_depth(pressure)

def ts_callback(odom, dvl, landmark):

    try:
        ## todo make sure transform time matches pose time?
        #transform = tfBuffer.lookup_transform('zedm_left_camera_optical_frame', 'world', rospy.Time(0))
        transform = tf_buffer.lookup_transform('zedm_left_camera_optical_frame', 'world', rospy.Time(0))
        trans = Point3(transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z)
        rot = Rot3.Quaternion(transform.transform.rotation.w, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z)
        slam.zed_world_transform = (rot, trans)
        
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        rospy.logerr("Error getting zed to world transform!")
        pass

    if (slam.depth != None):
        slam.batch_update(odom, dvl, slam.depth, landmark)

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

def vector3(x, y, z):
    """Create 3d double numpy array."""
    return np.array([x, y, z], dtype=float)

class AUV_ISAM():

    def __init__(self) -> None:

        #ISAM stuff
        self.isam = ISAM2()
        self.graph = NonlinearFactorGraph()
        self.initial_estimate = Values()
        self.result = Values()
        self.prev_pose = Pose3()
        self.prev_vel = Point3()
        self.timestep = 0

        #IMU
        self.grav = 9.81
        self.g = np.array([0, 0, -self.grav])
        self.PARAMS, self.BIAS_COVARIANCE, self.DELTA = self.preintegration_parameters()
        self.imu_preintegrated = gtsam.PreintegratedImuMeasurements(self.PARAMS)
        self.prev_bias = gtsam.imuBias.ConstantBias()

        # Noise models
        self.pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3]))
        self.vel_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        self.bias_noise = self.BIAS_COVARIANCE
        self.dvl_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)

        self.imu_data = []
        self.odom_accum = []
        self.odom_compare = []
        self.dvl_accum = []
        self.imu_accum = []
        self.depth = None
    
        self.landmark_accum = []
        self.landmarks = []
        self.zed_world_transform = None
        self.baseline = 0.063
        self.intrinsic = [1827.0, 1827.5999755859375, 968.9000244140625, 561.4000244140625]
        self.f = (self.intrinsic[0] + self.intrinsic[1]) / 2.0
        self.cx = self.intrinsic[2]
        self.cy = self.intrinsic[3]
        self.K = Cal3_S2Stereo(self.intrinsic[0], self.intrinsic[1], 0.0, self.cx, self.cy, self.baseline)
        self.resolution_x = 1920
        self.resolution_y = 1080
        self.landmark_noise = gtsam.noiseModel.Isotropic.Sigma(3, 10)


        
    def process_depth(self, data):
        dict = json.loads(data.data)
        measured_pressure = dict['press_abs'] * 100
        pressure_diff = measured_pressure - 98250.0
        self.depth = pressure_diff / (997 * 9.81)
    
    def process_odom(self, odom, depth):
        pos = odom.pose.pose.position
        quat = odom.pose.pose.orientation
        rot = Rot3.Quaternion(quat.w, quat.x, quat.y, quat.z)
        point = Point3(pos.x, pos.y, pos.z)
        point_adjust = Point3(pos.x, pos.y, depth)
        pose_adjust = Pose3(rot, point_adjust)
        pose_norm = Pose3(rot, point)
        return pose_adjust, pose_norm

    def update_imu(self, imu, dt):
        acc = np.array([imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z])
        gyro = np.array([imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z])
        self.imu_data.append(np.hstack((acc, gyro)))


    def get_landmarks(self, data):

        #should give a list of features
        landmarks = []
        if self.zed_world_transform != None:
            for feature in data.features:
                id = feature.id

                uL = (feature.u0 + 1) * 0.5 * self.resolution_x;
                uR = (feature.u1 + 1) * 0.5 * self.resolution_x
                v = ((feature.v0 + feature.v1) / 2.0 + 1) * 0.5 * self.resolution_y

                d = uR - uL
                x = uL
                y = v
                W = d / self.baseline
                x_cam = (x - self.cx) / W
                y_cam = (y - self.cy) / W
                z_cam = self.f / W

                cam_point = np.array([[x_cam], [y_cam], [z_cam]])
                #print(self.cam_map_transform)
                world_point = self.zed_world_transform[0].matrix() @ cam_point + self.zed_world_transform[1].reshape(3,1)
                #print(world_point)
                landmarks.append({
                    'id': id,
                    'pose': world_point.reshape((3,)),
                    'uL': uL,
                    'uR': uR,
                    'v': v
                })

        return landmarks                    

    def preintegration_parameters(self):
        # IMU preintegration parameters
        # Default Params for a Z-up navigation frame, such as ENU: gravity points along negative Z-axis
        PARAMS = gtsam.PreintegrationParams.MakeSharedU(self.grav)
        I = np.eye(3)
        PARAMS.setAccelerometerCovariance(I * 8.999999999999999e-08)
        PARAMS.setGyroscopeCovariance(I * 1.2184696791468346e-07)
        PARAMS.setIntegrationCovariance(I * 1e-07)
        PARAMS.setUse2ndOrderCoriolis(False)
        PARAMS.setOmegaCoriolis(np.array([0,0,0], dtype=float))

        BIAS_COVARIANCE = gtsam.noiseModel.Isotropic.Variance(6, 0.1)
        DELTA = Pose3(Rot3.Rodrigues(0, 0, 0),
                    Point3(0.05, -0.10, 0.20))

        return PARAMS, BIAS_COVARIANCE, DELTA

    
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
        
        key1 = this.keys()[0] # key = timestamp
        key2 = this.keys()[1] # key = timestamp
        vel_estimate = values.atVector(key1) # get previous estimates from graph
        pose_estimate = values.atPose3(key2)

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
            jacobians[1] = rot_mat
        return error

    

    def create_imu_factor(self):
        imu_factor = ImuFactor(X(self.timestep-1), V(self.timestep-1), X(self.timestep), V(self.timestep), B(0), self.imu_preintegrated)
        return imu_factor

    def create_dvl_factor(self, dvl):

        measurement = np.array([[dvl.twist.linear.x, dvl.twist.linear.y, dvl.twist.linear.z]])

        dvl_factor = gtsam.CustomFactor(
                        self.dvl_noise,
                        [V(self.timestep), X(self.timestep)], # TODO not necessarily the same as the data received time?
                        partial(self.velocity_error, measurement),
                    )
        return dvl_factor
    

    def batch_update(self, odom, dvl, depth, landmarks):

        pose_adjust, pose_norm = self.process_odom(odom, depth)
        self.odom_accum.append(pose_adjust)
        self.odom_compare.append(pose_norm)

        self.dvl_accum.append(dvl)

        self.imu_accum.append(self.imu_data)
        self.imu_data = []


        lm = self.get_landmarks(landmarks)
        self.landmark_accum.append(lm)



    def batch_create(self, with_landmark=True):
        self.initial_estimate = Values()
        self.graph = NonlinearFactorGraph()

        self.initial_estimate.insert(B(0), self.prev_bias)
        for i in range(len(self.odom_accum)):
            self.timestep = i
            #Prior Estimate
            pose = self.odom_accum[i]
            velocity = vector3(0,0,0)
            if i == 0:   
                self.graph.add(gtsam.PriorFactorPose3(X(0), pose, self.pose_noise))
                self.graph.add(gtsam.PriorFactorVector(V(0), velocity, self.vel_noise))
                self.initial_estimate.insert(X(i), pose)
                self.initial_estimate.insert(V(i), velocity)
            else:
                
                self.initial_estimate.insert(X(i), pose)
                self.initial_estimate.insert(V(i), velocity)
                for imu in self.imu_accum[i]:
                    self.imu_preintegrated.integrateMeasurement(imu[:3], imu[3:], 0.005)
                self.graph.push_back(self.create_imu_factor())
                self.graph.push_back(self.create_dvl_factor(self.dvl_accum[i]))
                self.imu_preintegrated.resetIntegration()

                if with_landmark:
                    for landmark in self.landmark_accum[i]:
                        if not self.initial_estimate.exists(L(landmark['id'])):
                            self.initial_estimate.insert(L(landmark['id']), landmark['pose'])

                        stereo_factor = GenericStereoFactor3D(
                            StereoPoint2(landmark['uL'], landmark['uR'], landmark['v']),
                            self.landmark_noise,
                            X(i), L(landmark['id']), self.K
                        )
                        self.graph.push_back(stereo_factor)

        

if __name__ == '__main__':
    rospy.init_node("auv_slam", anonymous=True)


    slam = AUV_ISAM()

    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)

    old_time = rospy.Time.now()
    rospy.Subscriber('/zedm/zed_node/imu/data', Imu, imu_callback)
    rospy.Subscriber('/BlueROV/pressure2', String, pressure_callback)
    odom_sub = message_filters.Subscriber('/dvl/local_position', PoseWithCovarianceStamped)
    dvl_sub = message_filters.Subscriber('/dvl/twist', TwistStamped)
    landmark_sub = message_filters.Subscriber('/auv/image_processor/features', CameraMeasurement)

    ts = message_filters.ApproximateTimeSynchronizer([odom_sub, dvl_sub, landmark_sub], 10, 0.2, allow_headerless=True)
    ts.registerCallback(ts_callback)

    rate = rospy.Rate(5)# Hz

    old_time = rospy.Time.now()
    while not rospy.is_shutdown():

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



            #auv_isam.createBatch()
            slam.batch_create(with_landmark=True)
            results = gtsam.LevenbergMarquardtOptimizer(slam.graph, slam.initial_estimate, gtsam.LevenbergMarquardtParams()).optimize()            
            slam.graph.saveGraph("graph.dot")

            break


        rate.sleep()

    points = constr3DPoints(results)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x = np.array([slam.odom_compare[i].x() for i in range(len(slam.odom_compare))])
    y = np.array([slam.odom_compare[i].y() for i in range(len(slam.odom_compare))])
    z = np.array([slam.odom_compare[i].z() for i in range(len(slam.odom_compare))])
    ax.plot3D(x, y, z, color='orange', linewidth=2, label="Odometry")
    ax.plot3D(points[1:,0], points[1:,1], points[1:,2], color='blue', label="Ours (SLAM + Landmarks)")
    #ax.plot3D(points2[1:,0], points2[1:,1], points2[1:,2], color='green')
    plt.legend()
    plt.title("Underwater Visual SLAM Trajectory")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    plt.show()

    odom = np.hstack((x.reshape(-1, 1), y.reshape(-1,1), z.reshape(-1,1)))
    odom -= np.array([0,0,0.7433])

    squared_diff = np.square(odom - points[1:, :])
    mse = np.mean(squared_diff)
    print("MSE=", mse)



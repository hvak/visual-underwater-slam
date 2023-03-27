import rospy
import gtsam.gtsam 
import numpy as np
from matplotlib import pyplot as plt
from gtsam.symbol_shorthand import B, V, X

import gtsam
from gtsam import (ISAM2, BetweenFactorConstantBias, Cal3_S2,
                   ConstantTwistScenario, ImuFactor, NonlinearFactorGraph,
                   PinholeCameraCal3_S2, Point3, Pose3,
                   PriorFactorConstantBias, PriorFactorPose3,
                   PriorFactorVector, Rot3, Values)

BIAS_KEY = B(0)

def vector3(x, y, z):
    """Create 3d double numpy array."""
    return np.array([x, y, z], dtype=float)


class AUV_ISAM:
    ##Iniotialize parameters for the IMU
    def preintegration_parameters(self):
        # IMU preintegration parameters
        # Default Params for a Z-up navigation frame, such as ENU: gravity points along negative Z-axis
        PARAMS = gtsam.PreintegrationParams.MakeSharedU(g)
        I = np.eye(3)
        # later TODO: Set covariance to be actual IMU covariance
        PARAMS.setAccelerometerCovariance(I * 0.1)
        PARAMS.setGyroscopeCovariance(I * 0.1)
        PARAMS.setIntegrationCovariance(I * 0.1)
        PARAMS.setUse2ndOrderCoriolis(False)
        PARAMS.setOmegaCoriolis(vector3(0, 0, 0))

        BIAS_COVARIANCE = gtsam.noiseModel.Isotropic.Variance(6, 0.1)

        return PARAMS, BIAS_COVARIANCE


    ##Retrieve and update ROS data as quickly as it is received.  Store it in various data structures
    def update_imu(self, data):
        measAcc = np.array([data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z]) - np.dot([self.g_transform, self.g])
        measOmega = np.array([data.angular_velocity.x, data.angular_velocity.y, data.angular_velocity.z])
        self.imu = np.array([measAcc, measOmega])
        return

    def update_odom(self, data):
        self.odom = {"x": data.pose.pose.position.x, 
                     "y": data.pose.pose.position.y,
                     "z": data.pose.pose.position.z,
                     "i" : data.pose.pose.orientation.x,
                     "j" : data.pose.pose.orientation.y,
                     "k" : data.pose.pose.orientation.z, 
                     "q" : data.pose.pose.orientation.w}
        return

    def update_dvl(self, data):
        self.dvl = {"x" : data.twist.linear.x,
                    "y" : data.twist.linear.y,
                    "z" : data.twist.linear.z}
        return

    def update_orb(self, data):
        #self.orb = np.array([data.landmarkSeen, data.landMarkId])

        return

    ##Create factors for the graph based on ROS datab
    def create_imu_factor(self):
        delta_t = .01
        self.accum.integrateMeasurement(self.imu[0], self.imu[1], delta_t)
        imuFactor = ImuFactor(X(self.timestep - 1), V(self.timestep - 1), X(self.timestep), V(self.timestep), BIAS_KEY, self.accum)
        return imuFactor
    
    def create_dvl_factor(self):
        dvlFactor = None
        return dvlFactor
    
    def create_depth_factor(self):
        depthFactor = None
        return depthFactor

    def create_orb_factor(self):
        orbFactor = None
        return orbFactor
    
    # returns odometry as gtsam pose3 object
    def get_odom_as_pose3(self):
        t = gtsam.Point3( self.odom['x'], 
                        self.odom['y'], 
                        self.odom['y'])
        rot = gtsam.Rot3(self.odom['q'], self.odom['i'], self.odom['j'], self.odom['k'])
        odom = gtsam.Pose3(rot, t)
        return odom
    
    def __init__(self):
        ## TODO Define Prior Noise
        self.priorPose =  gtsam.pose3(np.eye(3), np.zeros(3))
        self.priorNoise = None
        self.priorVel = gtsam.pose3(np.eye(3), np.zeros(3))
        self.priorVelNoise = None

        ## stores most recent data from rosbag
        self.imu = None
        self.odom = None
        self.dvl = None
        self.orb = None
        self.g = vector3(0, 0, -9.81)
        self.isam = gtsam.gtsam.ISAM2()
        
        self.timestep = 0
        PARAMS, BIAS_COV = self.preintegration_parameters()
        self.accum = gtsam.PreintegratedImuMeasurements(PARAMS)

        self.initial_estimate = Values()
        self.g_transform = 0

     
    def update_isam(self, vertices, edges):
        
        graph = gtsam.gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()
        if self.timestep == 0:
            graph.push_back(gtsam.gtsam.PriorFactorPose3(X(0), self.priorPose, self.priorNoise))
            graph.push_back(gtsam.PriorFactorPoint3(X(0), self.priorVel, self.priorVelNoise ))

        else: 
            currentOdomEst = self.get_odom_as_pose3()
            currentVelEst = vector3(0, 0, 0) ## okay for now, supposed to be bad. Later TODO: replace with DVL estimate?

            ## add the current odom as a variable in estimate, to be updated later
            initial_estimate.insert(X(self.timestep), currentOdomEst)
            initial_estimate.insert(V(self.timestep), currentVelEst)
            
            ## get factors to use in factor update
            factors = self.create_factors_from_recent() 

            ## add most recent measurements as edges in graph
            for factor in factors:
                graph.add(factor)

        self.timestep += 1
        self.isam.update(graph, initial_estimate)                
        result = self.isam.calculateEstimate()
        ## TODO do something with resulting estimates

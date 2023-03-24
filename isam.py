import gtsam.gtsam 
import numpy as np
from matplotlib import pyplot as plt
from gtsam.symbol_shorthand import B, V, X

BIAS_KEY = B(0)



''' 
copied from slam hw needs editing lol
TODO:
    inputs, noise, 

    Parameters: factors, edges
    
'''
class AUV_ISAM:

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

    def update_imu(self, data):
        measAcc = np.array([data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z])
        measOmega = np.array([data.angular_velocity])
        return

    def update_odom(self, data):
        return

    def update_dvl(self, data):
        return

    def update_orb(self, data):
        return


    def __init__(self):
        ## TODO PRIORS
        self.priorPose =  gtsam.pose3(np.eye(3), np.zeros(3))
        self.priorNoise = None
        self.priorVel = None
        self.priorVelNoise = None

        ## TODO Sensor inits
        ## most recent sensor data
        self.imu = None
        self.odom = None
        self.dvl = None
        self.orb = None
        self.g = vector3(0, 0, -g)

        self.optimized_poses = None ## result from isam optimization  :) ## same as isam.estimate() ## dont need
        self.isam = gtsam.gtsam.ISAM2()

    '''
    TODO: this function should be scheduled to run periodically and 
    convert all most recent data variables (self.dvl, self.imu, etc)
    to factors and add them to factor 
    '''
    def create_factors_from_recent():
        pass


    def update_isam(self, factors, edges):

        ## get factors to use in factor update
        edges = self.create_factors_from_recent()

        node_idx = 0
        state_idx = 0
        for factor in factors:
            graph = gtsam.gtsam.NonlinearFactorGraph()
            initial_estimate = gtsam.Values()
            init = False
            if state_idx == 0:

                ## TODO: graph.push_back or graph.add?
                graph.add(gtsam.gtsam.PriorFactorPose3(X(0), self.priorPose, self.priorNoise))

                ## TODO add velocity? why Point3?
                graph.add(gtsam.PriorFactorPoint3(X(0), self.priorVel, self.priorVelNoise ))
                initial_estimate.insert(X(0), self.priorPose)
                initial_estimate.insert(V(0), self.priorVel)

                ## TODO update state idx?
            else: ## TODO this whole else statement is wrong indices

                ## adding all previous poses to initial estimate
                prevPose = self.optimized_poses.atPose3(vertex.i - 1)
                initial_estimate.insert(vertex.i, prevPose)

                ## only add current edges (measurements)
                for edge in edges:
                    ## TODO for all sensor measurements imu, dvl, etc
                    odom = gtsam.gtsam.Pose2(edge.x, edge.y, edge.theta)
                    newFactor = gtsam.gtsam.BetweenFactorPose2(edge.i, edge.j, odom, 
                        gtsam.noiseModel.Gaussian.Covariance(NOISEFROMEDGE))
                    graph.add(newFactor)
            self.isam.update(graph, initial_estimate)                
            self.optimized_poses = self.isam.calculateEstimate()
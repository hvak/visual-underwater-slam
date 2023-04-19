# Underwater Visual-Inertial SLAM

Visual-inertial pose-graph slam using GTSAM

## Build

Required Packages
- TODO dependencies

```bash
$ cd ~/catkin_ws/src
$ git clone https://github.com/hvak/visual-underwater-slam.git
$ git clone https://github.com/hvak/gtsam_vio_orb
$ git clone https://github.com/waterlinked/dvl-a50-ros-driver.git
```

## Run
In separate terminals run the following commands:

```bash
$ roslaunch uslam stereo.launch
$ python3 tf_fix.py
$ rosbag play <bagfile>
$ python3 batch.py
```

When the bag is done playing, the odometry and slam solution will be plotted.

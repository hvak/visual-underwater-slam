# Underwater Visual-Inertial SLAM

This project was done as the final project ROB 530: Mobile Robotics.

Group Members:
- [Anja Sheppard](https://github.com/amstrudy)
- [Ana Warner](https://github.com/aswarner)
- [Ryan Donoghue](https://github.com/rdonoghu-umich)
- [Hersh Vakharia](https://github.com/hvak)
- [Menelik Weatherspoon](https://github.com/menelik002)

Here is our video presentation: [https://drive.google.com/file/d/1fWqroe5BD5YkaPLsU8kTBoGOHVovdXrJ/view?usp=share_link](https://drive.google.com/file/d/1fWqroe5BD5YkaPLsU8kTBoGOHVovdXrJ/view?usp=share_link)

## Build
This system was built using ROS Noetic in Ubuntu 20.04.

Dependencies
- Python packages: numpy, opencv-python, gtsam
- DVL A50 ROS Driver - [Repo](https://github.com/waterlinked/dvl-a50-ros-driver.git) (clone to ```catkin_ws```)
- Our fork of gtsam_vio - [Repo](https://github.com/hvak/gtsam_vio_orb) (clone to ```catkin_ws```)
    - This requires GTSAM C++ and SuiteSparse

Commands:
```bash
$ cd ~/catkin_ws/src
$ git clone https://github.com/hvak/visual-underwater-slam.git
$ git clone https://github.com/hvak/gtsam_vio_orb
$ git clone https://github.com/waterlinked/dvl-a50-ros-driver.git
$ cd ..
$ catkin build
```

## Structure
### ```batch.py```
This contains our batch solution implementation.

### ```tf_fix.py```
This script reorganizes the TF tree for our bag file to ignore ZED odometry, as it is very inaccurate underwater.

### ```isam.py```
This is our experimental implementation of an incremental SLAM solution. It doesn not currently work.

## Run
In separate terminals run the following commands:

```bash
$ roslaunch uslam stereo.launch
$ python3 tf_fix.py
$ rosbag play <bagfile>
$ python3 batch.py
```
When the bag is done playing, the odometry and slam solution will be plotted. The bag can also be stopped early for a partial batch solution. Unfortunately, we are unable to make our bagfile publicly available at this time.

## Acknowledgements
This builds off of work by Onur Bagoren *et al*: [https://github.com/onurbagoren/Underwater_SLAM](https://github.com/onurbagoren/Underwater_SLAM)

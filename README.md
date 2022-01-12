# LiDAR Point Cloud Augmentation for Mobile Robot Safe Navigation in Indoor Environment
The mobile robot is surrounded by various structures. Glass is widely used for constructing walls and doors, and it is difficult for mobile robots to detect glass. Related studies consider the optical properties of glass when reconstructing the position of glass using LiDAR data. However, most of them depend on SLAM (Simultaneous Localization And Mapping) or Localization algorithm and prior information about the environment. This repository proposes a point cloud augmentation algorithm to predict glass panel position. The point cloud outlier is removed with the proposed local standard score, and the boundary of free space is augmented by linear interpolation. It is modularized as a post-processing of LiDAR sensor data that does not require any prior information about the environment. This algorithm was applied to an actual mobile robot. Glass recognition performance was evaluated by SLAM while driving in an office environment.

![figure](https://user-images.githubusercontent.com/16618451/149051384-883fc50d-90f4-449d-b2c4-84825df73845.png)

Proposed point cloud augmentation process. There are four glass panels around the robot(scene 1), but the robot acquires a point cloud containing only one panel(scene 2). The proposed method removes outliers(scene 3, red points) with local standard score, and predict the position of the remaining three panels(scene 4, blue points) using interpolation.


## Simple Usage
Ubuntu 18.04, ROS melodic, VLP-16

``` bash
roslaunch lidar wakeup.launch
rosrun lidar augmentation.py
roslaunch lidar getup.launch
roslaunch lidar show.launch
```

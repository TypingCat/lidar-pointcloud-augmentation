#! /usr/bin/env python

import rospy
import struct
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

import numpy as np
import math
from collections import defaultdict


class Augmentation:
    '''LiDAR point cloud augmentation'''
    
    point_cloud_format = '=ffffHf'
    ring_target = [6]    
    window_size = 10    
    lidar_height = 0.35
    dphi_min = 0.05
    ground_min = -0.05
    ground_max = 0.2
    i_min = 30
    i_accept = 50
    d_max = 1.0
    dn_dphi = 5/(math.pi/180.)

    def __init__(self):
        self.signal_loss_range_publisher = rospy.Publisher("signal_loss_range", Marker, queue_size=10)
        self.point_cloud_publisher = rospy.Publisher("augmented_points", PointCloud2, queue_size=10)
        self.point_cloud_subscriber = rospy.Subscriber("velodyne_points", PointCloud2, self.point_cloud_callback)
    
    def point_cloud_callback(self, data):
        # Get points from LiDAR
        points, frame = defaultdict(list), []
        for step in range(0, data.row_step, data.point_step):
            x, y, z, i, Ring, t = struct.unpack_from(self.point_cloud_format, data.data, step)
            if Ring not in self.ring_target: continue   # limit input ring 
            
            # Get frame points
            h = z + self.lidar_height
            if h < self.ground_min: continue            # Remove ground reflection
            if h < self.ground_max and i > self.i_min:
                frame.append((x, y, z, i, Ring, t))
            
            # Get normal points
            phi = math.atan2(y, x)
            rho = math.sqrt(math.pow(x, 2) + math.pow(y, 2))
            points[Ring].append((x, y, z, i, Ring, t, phi, rho))
            
        # Algorithm 2. Remove outlier
        for Ring in points.keys():
            points[Ring].sort(  # Sort points by phi
                key=lambda point: point[6], reverse=True)
            
            # Remove outliers using standard score
            rho_mean_left = \
                [0 for _ in range(self.window_size)] + \
                [np.mean([p[7] for p in points[Ring][idx-self.window_size:idx]])
                    for idx in range(self.window_size, len(points[Ring]))]
            rho_std_left = \
                [0 for _ in range(self.window_size)] + \
                [np.std([p[7] for p in points[Ring][idx-self.window_size:idx]])
                    for idx in range(self.window_size, len(points[Ring]))]
            rho_mean_right = \
                [np.mean([p[7] for p in points[Ring][idx+1:idx+1+self.window_size]])
                    for idx in range(len(points[Ring]) - self.window_size)] + \
                [0 for _ in range(self.window_size)]
            rho_std_right = \
                [np.std([p[7] for p in points[Ring][idx+1:idx+1+self.window_size]])
                    for idx in range(len(points[Ring]) - self.window_size)] + \
                [0 for _ in range(self.window_size)]
                
            points[Ring] = [p for idx, p in enumerate(points[Ring])
                if abs(p[7] - rho_mean_left[idx]) < 3*rho_std_left[idx] or
                   abs(p[7] - rho_mean_right[idx]) < 3*rho_std_right[idx]]
        
        # Algorithm 3. Search signal loss range
        signal_loss_range = []
        for Ring in points.keys():
            for idx in range(len(points[Ring])-1):
                p0, p1 = points[Ring][idx], points[Ring][idx+1]
                
                # Remove unnecessary interpolation range
                if p0[6] - p1[6] < self.dphi_min: continue
                # if p0[2] < self.ground_max or p1[2] < self.ground_max: continue
                if math.sqrt(math.pow(p0[0] - p1[0], 2) + math.pow(p0[1] - p1[1], 2)) > self.d_max: continue
                
                # Update endpoints with acceptable intensity points
                for i in range(idx, max(idx-self.window_size, 0), -1):
                    if points[Ring][i][3] > self.i_accept:
                        p0 = points[Ring][i]
                        break
                for i in range(idx+1, min(idx+1+self.window_size, len(points[Ring]))):
                    if points[Ring][i][3] > self.i_accept:
                        p1 = points[Ring][i]
                        break
                    
                signal_loss_range.append((p0, p1))

        # Add frame points to the result
        for p in frame:
            data.data += struct.pack(self.point_cloud_format,
                    p[0], p[1], 0., p[3], 16, p[5])
        data.width += len(frame)
        
        # Publish result
        for p in signal_loss_range:
            n = int((p[0][6] - p[1][6]) * self.dn_dphi)
            for rate in [float(r)/n for r in range(1, n)]:
                data.data += struct.pack(self.point_cloud_format,
                    p[0][0] + rate*(p[1][0] - p[0][0]),
                    p[0][1] + rate*(p[1][1] - p[0][1]),
                    p[0][2] + rate*(p[1][2] - p[0][2]),
                    100, #p[0][3] + rate*(p[1][3] - p[0][3]), augmented point height
                    p[0][4],
                    p[0][5] + rate*(p[1][5] - p[0][5]))
            data.width += n - 1
            
        data.row_step = data.width * data.point_step
        self.point_cloud_publisher.publish(data)
        
        # Publish marker
        marker = Marker()
        marker.header = data.header
        marker.ns = "signal_loss_range"
        marker.type = Marker.LINE_LIST
        marker.action = Marker.MODIFY
        marker.pose.orientation.w = 1
        marker.scale.x = 0.01
        marker.color.r = 0
        marker.color.g = 0
        marker.color.b = 1
        marker.color.a = 0.5
        marker.points = []
        [marker.points.extend([Point(p[0][0], p[0][1], p[0][2]), Point(p[1][0], p[1][1], p[1][2])]) for p in signal_loss_range]
        
        self.signal_loss_range_publisher.publish(marker)


if __name__ == '__main__':
    rospy.init_node('lidar_augmentation')
    a = Augmentation()
    rospy.spin()

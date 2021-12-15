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
    window_size = 10
    lidar_height = 0.35

    def __init__(self):
        self.normal_points_publisher = rospy.Publisher("analysis/normal_points", PointCloud2, queue_size=10)
        self.outlier_points_publisher = rospy.Publisher("analysis/outlier_points", PointCloud2, queue_size=10)
        self.augmented_points_publisher = rospy.Publisher("analysis/augmented_points", PointCloud2, queue_size=10)
        self.augmented_pointcloud_publisher = rospy.Publisher("augmented_pointcloud", PointCloud2, queue_size=10)
        self.signal_loss_range_publisher = rospy.Publisher("signal_loss_range", Marker, queue_size=10)

        self.point_cloud_subscriber = rospy.Subscriber("velodyne_points", PointCloud2, self.point_cloud_callback)

    def point_cloud_callback(self, data):
        pointcloud, bump_points = self.get_pointcloud(data)
        
        # Algorithm 2
        pointcloud, normal_points, outlier_points = self.remove_outlier(pointcloud)
        
        # Algorithm 3
        signal_loss_range = self.search_signal_loss_range(pointcloud)
        
        # Augment points to the result
        augmented_points = self.interpolate_range(signal_loss_range)
        data = self.add_points(data, augmented_points + bump_points)
        
        # Publish results
        self.normal_points_publisher.publish(self.create_point_cloud(normal_points, data))
        self.outlier_points_publisher.publish(self.create_point_cloud(outlier_points, data))
        self.augmented_points_publisher.publish(self.create_point_cloud(augmented_points, data))
        self.augmented_pointcloud_publisher.publish(data)
        self.signal_loss_range_publisher.publish(self.create_range_marker(signal_loss_range, data.header))
        print("Normal points: ", len(normal_points), "Outlier points: ", len(outlier_points), "Augmented points: ", len(augmented_points))
        
    def get_pointcloud(self, data, target_ring=[], ground_min=-0.05, ground_max=0.2, i_min=30):
        pointcloud, bump_points = defaultdict(list), []
        for step in range(0, data.row_step, data.point_step):
            x, y, z, i, ring, t = struct.unpack_from(self.point_cloud_format, data.data, step)
            if len(target_ring)!=0 and ring not in target_ring: continue
            
            # Get bump points
            h = z + self.lidar_height
            if h < ground_min: continue     # Remove ground reflection
            if h < ground_max and i > i_min:
                bump_points.append((x, y, 0., i, 16, t))
            
            # Get normal points
            phi = math.atan2(y, x)
            rho = math.sqrt(math.pow(x, 2) + math.pow(y, 2))
            pointcloud[ring].append((x, y, z, i, ring, t, phi, rho))

        return pointcloud, bump_points

    def remove_outlier(self, pointcloud, standard_score_threshold=3):
        normal_points, outlier_points = [], []
        for ring in pointcloud.keys():
            # Sort points by phi
            pointcloud[ring].sort(key=lambda point: point[6], reverse=True)
            
            # Calculate mean and std. from left and right populations
            rho_mean_left = \
                [0 for _ in range(self.window_size)] + \
                [np.mean([p[7] for p in pointcloud[ring][idx-self.window_size:idx]])
                    for idx in range(self.window_size, len(pointcloud[ring]))]
            rho_std_left = \
                [0 for _ in range(self.window_size)] + \
                [np.std([p[7] for p in pointcloud[ring][idx-self.window_size:idx]])
                    for idx in range(self.window_size, len(pointcloud[ring]))]
            rho_mean_right = \
                [np.mean([p[7] for p in pointcloud[ring][idx+1:idx+1+self.window_size]])
                    for idx in range(len(pointcloud[ring]) - self.window_size)] + \
                [0 for _ in range(self.window_size)]
            rho_std_right = \
                [np.std([p[7] for p in pointcloud[ring][idx+1:idx+1+self.window_size]])
                    for idx in range(len(pointcloud[ring]) - self.window_size)] + \
                [0 for _ in range(self.window_size)]

            # Remove outliers using standard score
            for idx, p in enumerate(pointcloud[ring]):
                if abs(p[7] - rho_mean_left[idx]) < standard_score_threshold*rho_std_left[idx] or abs(p[7] - rho_mean_right[idx]) < standard_score_threshold*rho_std_right[idx]:
                    normal_points.append(p)
                else:
                    outlier_points.append(p)
            pointcloud[ring] = [p for idx, p in enumerate(pointcloud[ring])
                if abs(p[7] - rho_mean_left[idx]) < standard_score_threshold*rho_std_left[idx] or abs(p[7] - rho_mean_right[idx]) < standard_score_threshold*rho_std_right[idx]]

        return pointcloud, normal_points, outlier_points

    def search_signal_loss_range(self, pointcloud, dphi_min=0.05, d_max=1.0, i_accept=50):
        signal_loss_range = []
        for ring in pointcloud.keys():
            for idx in range(len(pointcloud[ring])-1):
                p0, p1 = pointcloud[ring][idx], pointcloud[ring][idx+1]
                
                # Check interpolation requirements
                if p0[6] - p1[6] < dphi_min: continue
                if math.sqrt(math.pow(p0[0] - p1[0], 2) + math.pow(p0[1] - p1[1], 2)) > d_max: continue
                
                # Update endpoints with acceptable intensity points
                for i in range(idx, max(idx-self.window_size, 0), -1):
                    if pointcloud[ring][i][3] > i_accept:
                        p0 = pointcloud[ring][i]
                        break
                for i in range(idx+1, min(idx+1+self.window_size, len(pointcloud[ring]))):
                    if pointcloud[ring][i][3] > i_accept:
                        p1 = pointcloud[ring][i]
                        break

                # Add signal loss range
                signal_loss_range.append((p0, p1))

        return signal_loss_range
    
    def add_points(self, data, points):
        for p in points:
            data.data += struct.pack(self.point_cloud_format,
                    p[0], p[1], p[2], p[3], p[4], p[5])

        data.width += len(points)
        return data

    def interpolate_range(self, signal_loss_range, dn_dphi=5/(math.pi/180.)):
        points = []
        for p in signal_loss_range:
            n = int((p[0][6] - p[1][6]) * dn_dphi)
            for rate in [float(r)/n for r in range(1, n)]:
                points.append((
                    p[0][0] + rate*(p[1][0] - p[0][0]),
                    p[0][1] + rate*(p[1][1] - p[0][1]),
                    p[0][2] + rate*(p[1][2] - p[0][2]),
                    p[0][3] + rate*(p[1][3] - p[0][3]),
                    p[0][4],
                    p[0][5] + rate*(p[1][5] - p[0][5])))

        return points

    def create_point_cloud(self, points, data):
        pc = PointCloud2()
        pc.header, pc.fields = data.header, data.fields
        pc.height, pc.point_step = data.height, data.point_step
        pc.is_bigendian, pc.is_dense = data.is_bigendian, data.is_dense

        pc.data, pc.width = '', 0
        for p in points:
            pc.data += struct.pack(self.point_cloud_format,
                    p[0], p[1], p[2], p[3], p[4], p[5])
        pc.width = len(points)
        pc.row_step = pc.width * pc.point_step

        return pc
    
    def create_range_marker(self, signal_loss_range, header):
        marker = Marker()
        marker.header = header
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

        return marker


if __name__ == '__main__':
    rospy.init_node('lidar_augmentation')
    a = Augmentation()
    rospy.spin()

#!/usr/bin/env python

import rospy
import struct

from sensor_msgs.msg import PointCloud2

class DecomposeRing:
    def __init__(self):
        self.format = '=ffffHf'
        self.target = 0

        self.pub = rospy.Publisher("lidar_points", PointCloud2, queue_size=10)
        self.sub = rospy.Subscriber("velodyne_points", PointCloud2, self.pointcloud2_callback)

    def pointcloud2_callback(self, data):
        '''Republish lidar topic'''
        pc = PointCloud2()
        pc.header, pc.fields = data.header, data.fields
        pc.height, pc.point_step = data.height, data.point_step
        pc.is_bigendian, pc.is_dense = data.is_bigendian, data.is_dense

        # Copy target ring points only
        pc.data, width = '', 0
        for i in range(0, data.row_step, data.point_step):
            x, y, z, intensity, ring, time = struct.unpack_from(self.format, data.data, i)

            if ring != self.target: continue
            pc.data += struct.pack(self.format, x, y, z, intensity, ring, time)
            width += 1
        pc.width, pc.row_step = width, width * pc.point_step

        self.pub.publish(pc)


if __name__ == '__main__':
    rospy.init_node('lidar_decompose_ring')
    DecomposeRing()
    rospy.spin()
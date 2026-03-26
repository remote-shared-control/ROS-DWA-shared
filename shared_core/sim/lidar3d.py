#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import tf2_ros
import tf2_geometry_msgs
import sensor_msgs.point_cloud2 as pc2
import matplotlib.pyplot as plt
import threading

from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PointStamped
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


class Mid360Processor:
    def __init__(self):
        

        self.input_topic = rospy.get_param("~input_topic", "/livox/lidar")
        self.target_frame = rospy.get_param("~target_frame", "odom")

        self.z_min = rospy.get_param("~z_min", -0.1)
        self.z_max = rospy.get_param("~z_max", 1.0)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.xy_set = set()
        self.lock = threading.Lock()
        
        rospy.Subscriber(self.input_topic, PointCloud2, self.callback, queue_size=1)

        # ===== Matplotlib 初始化（主线程）=====
        self.enable_plot = False
        if self.enable_plot:
            plt.ion()
            self.fig, self.ax = plt.subplots()

        rospy.loginfo("Mid360 Processor Started (Fixed Plot Thread)")

    def callback(self, msg):
        grid_size = 0.005   # ⭐ 可调

        occupied_voxels = set()   # 只记录格子是否被占
        new_set = set()

        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                msg.header.frame_id,
                rospy.Time(0),
                rospy.Duration(0.1)
            )
        except (LookupException, ConnectivityException, ExtrapolationException):
            return

        for i, point in enumerate(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)):
            if i % 4 != 0:   # ⭐ 每两个点取一个
                continue
            x, y, z = point

            # ===== 基础过滤 =====
            if z < self.z_min or z > self.z_max:
                continue
            if (x**2 + y**2) < (0.22**2) and z < 0.05:
                continue
            if abs(x) > 2 or abs(y) > 2:
                continue

            # ===== TF 变换 =====
            pt = PointStamped()
            pt.header = msg.header
            pt.point.x = x
            pt.point.y = y
            pt.point.z = z

            try:
                pt_transformed = tf2_geometry_msgs.do_transform_point(pt, transform)
            except:
                continue

            tx = pt_transformed.point.x
            ty = pt_transformed.point.y

            # ===== 栅格索引 =====
            gx = int(tx / grid_size)
            gy = int(ty / grid_size)

            # ===== 如果格子已经有点 → 直接跳过（关键优化）=====
            if (gx, gy) in occupied_voxels:
                continue

            # 标记格子已占用
            occupied_voxels.add((gx, gy))

            # ===== 使用格子中心 =====
            cx = (gx + 0.5) * grid_size
            cy = (gy + 0.5) * grid_size

            new_set.add((round(cx, 2), round(cy, 2)))

        # ===== 更新 =====
        with self.lock:
            self.xy_set = new_set
            

    def plot_loop(self):
        rate = rospy.Rate(5)  # 10Hz 刷新

        while not rospy.is_shutdown():
            with self.lock:
                data = list(self.xy_set)

            if len(data) > 0:
                xs, ys = zip(*data)

                self.ax.clear()
                self.ax.scatter(xs, ys)

                self.ax.set_title("Mid360 XY (Odom)")
                self.ax.set_xlabel("X")
                self.ax.set_ylabel("Y")
                self.ax.set_aspect('equal')

                self.ax.set_xlim(-10, 10)
                self.ax.set_ylim(-10, 10)

                plt.draw()
                plt.pause(0.001)

            rate.sleep()


if __name__ == "__main__":
    node = Mid360Processor()
    rospy.init_node("mid360_processor")
    if node.enable_plot:
        node.plot_loop()
    else:
        rospy.spin()
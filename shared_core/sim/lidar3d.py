#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import tf2_ros
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import threading
import matplotlib.pyplot as plt
from typing import Optional, Tuple

from sensor_msgs.msg import PointCloud2
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


class Mid360Processor:
    def __init__(self,robot_radius=0.2):
        self.robot_radius = robot_radius
        self.input_topic  = rospy.get_param("~input_topic",  "/livox/lidar")
        self.target_frame = rospy.get_param("~target_frame", "odom")

        self.z_min = rospy.get_param("~z_min", -0.1)
        self.z_max = rospy.get_param("~z_max",  0.0)

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # 世界坐标系下的有效点：两条对齐的 float32 数组，接口与原 xy_set 兼容
        # xs, ys = self.xy_set
        self.xy_set: Tuple[np.ndarray, np.ndarray] = (
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )
        self.lock = threading.Lock()

        rospy.Subscriber(self.input_topic, PointCloud2, self.callback, queue_size=1)

        # ===== 绘图开关（主线程初始化）=====
        self.enable_plot = False # 设置为 True 可视化点云（仅供调试，生产环境建议关闭）
        if self.enable_plot:
            plt.ion()
            self.fig, self.ax = plt.subplots()

        rospy.loginfo("Mid360 Processor Started")

    # ------------------------------------------------------------------
    def _get_transform_matrix(self, msg) -> Optional[np.ndarray]:
        """从 tf_buffer 提取 4×4 齐次变换矩阵（float32）。"""
        try:
            tf_stamped = self.tf_buffer.lookup_transform(
                self.target_frame,
                msg.header.frame_id,
                msg.header.stamp,
                rospy.Duration(0.1)
            )
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

        t = tf_stamped.transform.translation
        q = tf_stamped.transform.rotation

        # 四元数 → 旋转矩阵
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        R = np.array([
            [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
            [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
        ], dtype=np.float32)

        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[0, 3] = t.x
        T[1, 3] = t.y
        T[2, 3] = t.z
        return T

    # ------------------------------------------------------------------
    def callback(self, msg: PointCloud2):
        GRID      = np.float32(0.01)   # 体素/栅格边长（米）
        STRIDE    = 7                   # 每 STRIDE 个点取一个（粗采样）
        XY_LIMIT  = np.float32(2.5)
        R_SQRD    = np.float32(self.robot_radius ** 2)
        Z_THRESH  = np.float32(0.05)

        # ── 1. 读取点云 → numpy（传感器坐标系） ──────────────────────────
        raw = np.array(
            list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)),
            dtype=np.float32
        )
        if raw.size == 0:
            return

        # 粗步长采样（等价于原来的 i%4!=0 continue）
        raw = raw[::STRIDE]             # shape (N, 3)

        # ── 2. 传感器坐标系过滤（减少后续运算量） ────────────────────────
        x, y, z = raw[:, 0], raw[:, 1], raw[:, 2]

        mask = (
            (z >= self.z_min) & (z <= self.z_max) &
            ~((x**2 + y**2 < R_SQRD) & (z < Z_THRESH)) &
            (np.abs(x) <= XY_LIMIT) & (np.abs(y) <= XY_LIMIT)
        )
        pts = raw[mask]                 # shape (M, 3)
        if pts.shape[0] == 0:
            return

        # ── 3. 传感器坐标系 Voxel 下采样 ─────────────────────────────────
        # 每个体素只保留一个代表点，大幅减少后续 TF 变换次数
        vox_idx = (pts / GRID).astype(np.int32)          # (M, 3)
        # 用结构化方式对 (vx, vy, vz) 三元组去重
        vox_keys = vox_idx[:, 0].astype(np.int64) * 1_000_003 + \
                   vox_idx[:, 1].astype(np.int64) * 1_009 + \
                   vox_idx[:, 2].astype(np.int64)
        _, first = np.unique(vox_keys, return_index=True)
        pts_ds = pts[first]             # 下采样后的点，shape (K, 3)

        # ── 4. 获取 TF 变换矩阵 ──────────────────────────────────────────
        T = self._get_transform_matrix(msg)
        if T is None:
            return

        # ── 5. 批量 TF 变换（矩阵乘法，全程 numpy） ──────────────────────
        ones = np.ones((pts_ds.shape[0], 1), dtype=np.float32)
        pts_h = np.hstack([pts_ds, ones])       # (K, 4)
        world = (T @ pts_h.T).T                 # (K, 4)
        wx, wy = world[:, 0], world[:, 1]

        # ── 6. 世界坐标系栅格索引（整数，天然去重键） ────────────────────
        gx = (wx / GRID).astype(np.int32)
        gy = (wy / GRID).astype(np.int32)

        # 用格子中心作为代表坐标
        cx = (gx.astype(np.float32) + 0.5) * GRID
        cy = (gy.astype(np.float32) + 0.5) * GRID

        # ── 7. 世界坐标系栅格去重（int64 位拼接，全程 numpy，无 Python 循环）──
        # 将两个 int32 拼成一个 int64：高32位=gx，低32位=gy
        # 单整数哈希比 tuple 哈希快 3-5 倍，且这里用 np.unique 根本不进 Python dict
        keys = (gx.astype(np.int64) << 32) | (gy.astype(np.int64) & 0xFFFF_FFFF)
        _, first_idx = np.unique(keys, return_index=True)

        # 格子中心坐标（float32 数组，直接供下游使用）
        cx_out = cx[first_idx]
        cy_out = cy[first_idx]

        # ── 8. 原子替换（两条轻量 float32 数组）────────────────────────────
        with self.lock:
            self.xy_set = (cx_out, cy_out)

    # ------------------------------------------------------------------
    def plot_loop(self):
        rate = rospy.Rate(5)  # 5Hz 刷新

        while not rospy.is_shutdown():
            with self.lock:
                xs, ys = self.xy_set

            if xs.size > 0:
                self.ax.clear()
                self.ax.scatter(xs, ys, s=1)

                self.ax.set_title("Mid360 XY (Odom)")
                self.ax.set_xlabel("X")
                self.ax.set_ylabel("Y")
                self.ax.set_aspect('equal')
                self.ax.set_xlim(-10, 10)
                self.ax.set_ylim(-10, 10)

                plt.draw()
                plt.pause(0.001)

            rate.sleep()


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rospy.init_node("mid360_processor")
    node = Mid360Processor()
    if node.enable_plot:
        node.plot_loop()
    else:
        rospy.spin()
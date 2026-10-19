#!/usr/bin/env python
import rospy
import numpy as np
from gazebo_msgs.msg import ModelStates

class Config:
    def __init__(self):
        self.dt = 0.1
        self.predict_time = 3.0
        self.obs_radius =0.15


class GazeboObsReader:
    def __init__(self, config):
        self.config = config

        # 只读取 human 模型
        self.target_prefix = "obstacle"

        self.obs_info = np.zeros((0, 6))  # x, y, vx, vy, radius, id
        self.moving_future = None        # 保存 num_obs × num_steps × 2

        rospy.Subscriber("/dynamic_obstacle_states", ModelStates, self.callback)


    def filter_models(self, names, poses, twists):
        """只保留 human 模型并组装obs_info"""

        obs_list = []

        id_counter = 0
        for i, name in enumerate(names):

            # 只读取包含 human 的模型
            if self.target_prefix not in name:
                continue

            x = poses[i].position.x
            y = poses[i].position.y
            # vx = twists[i].linear.x
            # vy = twists[i].linear.y
            vx = round(twists[i].linear.x, 3)
            vy = round(twists[i].linear.y, 3)
            # vx = -1.0
            # vy = 0.0

            obs_list.append([x, y, vx, vy, self.config.obs_radius, id_counter])
            id_counter += 1

        if len(obs_list) == 0:
            return np.zeros((0, 6))

        return np.array(obs_list)


    def predict_future(self, obs_info):
        """
        obs_info: (num_obs, 6)
        Return: moving_future = (num_obs, num_steps, 2)
        """

        num_obs = obs_info.shape[0]
        num_steps = int(self.config.predict_time / self.config.dt) + 1

        # 初始化未来轨迹
        future = np.zeros((num_obs, num_steps, 2))  # x,y

        # 逐障碍物预测（numpy向量化）
        for i in range(num_obs):
            x0 = obs_info[i, 0]
            y0 = obs_info[i, 1]
            vx = obs_info[i, 2]
            vy = obs_info[i, 3]

            # 时间序列
            t = np.arange(num_steps) * self.config.dt

            future[i, :, 0] = x0 + vx * t
            future[i, :, 1] = y0 + vy * t

        return future


    def callback(self, data):
        names = data.name
        poses = data.pose
        twists = data.twist

        # 1. 转换为 numpy 的障碍物信息
        self.obs_info = self.filter_models(names, poses, twists)

        # 无障碍物则不继续
        if len(self.obs_info) == 0:
            self.moving_future = np.zeros((0, 1, 2))
            return

        # 2. 计算未来轨迹（numpy）
        self.moving_future = self.predict_future(self.obs_info)

        # 输出测试
        # rospy.loginfo_throttle(1.0,
        #     "\nObs info:\n{}\n\nMoving future shape: {}\n".format(
        #         self.obs_info,
        #         self.moving_future.shape
        #     )
        # )


if __name__ == "__main__":
    rospy.init_node("gazebo_obs_numpy_reader")
    cfg = Config()

    reader = GazeboObsReader(cfg)

    rospy.loginfo("Running Gazebo Obstacle Reader (NumPy version)...")
    rospy.spin()

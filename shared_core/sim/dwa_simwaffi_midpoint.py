#!/usr/bin/env python

# Author: Connor McGuile
# Feel free to use in any way.

# A custom Dynamic Window Approach implementation for use with Turtlebot.
# Obstacles are registered by a front-mounted laser and stored in a set.
# If, for testing purposes or otherwise, you do not want the laser to be used,
# disable the laserscan subscriber and create your own obstacle set in main(),
# before beginning the loop. If you do not want obstacles, create an empty set.
# Implentation based off Fox et al.'s paper, The Dynamic Window Approach to 
# Collision Avoidance (1997).
import rospy
import math
from bisect import bisect_left
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker
from detectmove import GazeboObsReader
import numpy as np
import signal
import sys
from pynput import keyboard  # for keyboard input detect
import simpleaudio as sa
import threading
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
import json
from distancetime import *
from lidar3d import *
# import argparse
# parser = argparse.ArgumentParser(description="设置参数")

# 添加参数
# parser.add_argument("--param", type=int, required=True, help="参数值")

# # 解析参数
# args = parser.parse_args()
import csv


RECORDER = True  # Enable rosbag recording.
RECORDER_CSV = False  # Set to True to enable CSV logging, False to disable.

if RECORDER_CSV:
    log_dir = "/home/frienkie/rosbagpaper"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "vw_cost_log.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "TIME", "v", "w",
        "to_human_cost", "speed_cost",
        "ob_cost", "final_cost"
    ])
    print("✅ CSV logging enabled")
else:
    log_writer = None
    log_file = None
    print("❌ CSV logging disabled")


class RuntimeState:
    """Keep the simulation's shared runtime state in one place."""

    def __init__(self):
        self.control_mode = 0  # 0: human control, 1: shared control
        self.save_requested = False
        self.rosbag_id = 0
        self.result_pending = True  # Handle the result only once.


runtime_state = RuntimeState()


class Config():
    # simulation parameters

    def __init__(self):
        # robot parameter
        #NOTE good params:
        #NOTE 0.55,0.1,1.0,1.6,3.2,0.15,0.05,0.1,1.7,2.4,0.1,3.2,0.18
        self.max_speed = 1.0  # [m/s]
        self.min_speed = -1.0  # [m/s]
        # self.max_yawrate = 1.2  # [rad/s]
        self.max_yawrate = 1.0  # [rad/s]
        self.max_accel = 2.5  # [m/ss]
        self.max_dyawrate = 3.2  # [rad/ss]
        ##################################################33
        self.v_reso = 0.2  # [m/s]
        # self.yawrate_reso = 0.1  # [rad/s]
        self.yawrate_reso = 0.1  # [rad/s]
        #######################################################
        self.dt = 0.4  # [s]
        self.dtsys = 0.05
        self.predict_time = 2.4  # [s]
        # Fixed sampling intervals are prepared only once per configuration.
        times = np.append(
            np.arange(0.0, self.predict_time, self.dt),
            self.predict_time
        )
        self.trajectory_times = tuple(times.tolist())
        self.trajectory_steps = tuple(
            (t1, t1 - t0, t0 + 0.5 * (t1 - t0), True)
            for t0, t1 in zip(
                self.trajectory_times, self.trajectory_times[1:]
            )
        )
        self.goal_radius=0.4
        #########################################
        # self.speed_cost_gain = 1.5 
        # self.obs_cost_gain = 1.0
        # self.to_human_cost_gain =0.5
        # if args.param == 1:
        #     self.to_human_cost_gain = 1.0 #lower = detour
        #     self.speed_cost_gain = 2.0 #lower = faster
        #     self.obs_cost_gain = 1.0 #lower z= fearless

        self.to_human_cost_gain = 1.0 #lower = detour
        self.speed_cost_gain = 1.0 #lower = faster
        self.obs_cost_gain = 1.0 #lower z= fearless
        # if args.param == 3:
        #     self.to_human_cost_gain = 2.0 #lower = detour
        #     self.speed_cost_gain = 1.0 #lower = faster
        #     self.obs_cost_gain = 1.0 #lower z= fearless
        # #############################
        self.robot_radius = 0.225  # [m]
        self.robot_radius_static = 0.21  # [m]

        self.obs_radius =0.15
        self.x = 0.0
        self.y = 0.0
        self.th = 0.0
        self.goalX = 9.0
        self.goalY = 0.0
        self.prev_x = 0.0
        self.prev_y = 0.0
        self.distance = 0.0
        self.first_time=True
        self.mindect=0.12
        self.maxdect=1.2
        self.prev_x0= 0.0
        self.prev_y0= 0.0
        self.distance0=0.0
        self.v=0.0
        self.w=0.0
        # self.xy = set()
        self.r = rospy.Rate(20)

    # Callback for Odometry
    def assignOdomCoords(self, msg):
        # X- and Y- coords and pose of robot fed back into the robot config
        
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        rot_q = msg.pose.pose.orientation
        (roll,pitch,theta) = \
            euler_from_quaternion ([rot_q.x,rot_q.y,rot_q.z,rot_q.w])
        self.th = theta
        self.v = msg.twist.twist.linear.x
        self.w = msg.twist.twist.angular.z
        # self.xy.add((self.x,self.y))


    # Callback for attaining goal co-ordinates from Rviz Publish Point #rviz中设置目标点
    def goalCB(self,msg):
        self.goalX = msg.point.x
        self.goalY = msg.point.y

# class Obstacles():
#     def __init__(self):
#         # Set of coordinates of obstacles in view
#         self.obst = set()
#         self.obs1 = set()
#         self.obs_last=set()
#         self.minx = 3.5
#         self.deltx = 0.15
#         self.index = 0
#         self.pattern = 0
#         # self.obstacle_list = []  # 先用列表存储，再一次性加入 set


#     def assignObs(self, msg, config):

#         deg = len(msg.ranges)   # Number of degrees - varies in Sim vs real world
#         # print("Laser degree length {}".format(deg))
#         self.obst = set()   # reset the obstacle set to only keep visible objects

#         for i in range(len(msg.ranges)):
#             if msg.ranges[i]>config.maxdect or msg.ranges[i] == float('Inf'):
#                 continue
#             elif np.isnan(msg.ranges[i]):
#                 distance = config.mindect
#             elif msg.ranges[i]<config.mindect:
#                 continue
#             else:
#                 distance = msg.ranges[i]

#             #degree and rad
#             deg = (360)/len(msg.ranges)
#             rad = (2*math.pi)/len(msg.ranges)
#             objTheta_rad = rad * i
#             objTheta_deg = deg * i

#             #local(robot)
#             obsX_robo = distance * math.cos(abs(objTheta_rad))
#             obsY_robo = distance * math.sin(abs(objTheta_rad))

#             #global
#             obsX = obsX_robo * math.cos(config.th) - obsY_robo * math.sin(config.th) + config.x
#             obsY = obsX_robo * math.sin(config.th) + obsY_robo * math.cos(config.th) + config.y

#             # all obastacle data
#             self.obst.add((obsX,obsY))
#         if config.distance0>self.deltx:
#             if self.index == 0:
#                 self.obs1 = self.obst.copy()
#             # elif self.index == 1:
#             #     self.obs2 = self.obst.copy()
#             self.index = (self.index + 1) % 1
#             config.prev_x0=config.x
#             config.prev_y0=config.y
#         if self.pattern==1:
#             self.obst= self.obs1 | self.obst

#         if len(self.obs_last) - len(self.obst)>50:
#             self.obst=self.obs_last | self.obst
#         else:
#             self.obs_last=self.obst.copy()


        
# Model to determine the expected position of the robot after moving along trajectory
def motion(x, u, dt):
    # motion model
    # x = [x(m), y(m), theta(rad), v(m/s), omega(rad/s)]
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[2] += u[1] * dt

    return x

# Determine the dynamic window from robot configurations
def calc_dynamic_window(x, config):

    # -----------------------------
    # Dynamic window limits
    # -----------------------------
    v_min = max(
        config.min_speed,
        x[3] - config.max_accel * config.predict_time 
    )

    v_max = min(
        config.max_speed,
        x[3] + config.max_accel * config.predict_time 
    )

    w_min = max(
        -config.max_yawrate,
        x[4] - config.max_dyawrate * config.predict_time 
    )

    w_max = min(
        config.max_yawrate,
        x[4] + config.max_dyawrate * config.predict_time 
    )

    # -----------------------------
    # Sampling number
    # -----------------------------
    num_v = max(
        2,
        int((v_max - v_min) / config.v_reso) + 1
    )

    num_w = max(
        2,
        int((w_max - w_min) / config.yawrate_reso) + 1
    )

    # -----------------------------
    # Velocity sampling
    # -----------------------------
    vs = np.linspace(v_min, v_max, num_v)
    ws = np.linspace(w_min, w_max, num_w)

    # -----------------------------
    # Ensure exact zero exists
    # -----------------------------
    EPS = 1e-6

    # linear velocity
    if v_min <= 0.0 <= v_max:
        if not np.any(np.abs(vs) < EPS):
            vs = np.concatenate((vs, [0.0]))

    # angular velocity
    if w_min <= 0.0 <= w_max:
        if not np.any(np.abs(ws) < EPS):
            ws = np.concatenate((ws, [0.0]))

    # -----------------------------
    # Angular velocity filtering
    # -----------------------------
    MIN_W = 0.05

    # floating-point correction
    ws[np.abs(ws) < EPS] = 0.0

    # keep:
    #   1. effective yawrate
    #   2. exact zero
    mask = (np.abs(ws) > MIN_W) | (ws == 0.0)

    ws = ws[mask]

    # -----------------------------
    # Sort after concatenate
    # -----------------------------
    vs.sort()
    ws.sort()

    return vs, ws

class TrajectoryPreview:
    """Keep a calculated trajectory and coordinate views for display."""

    def __init__(self):
        self.trajectory = None
        self.x_points = []
        self.y_points = []

    @property
    def point_count(self):
        return len(self.x_points)

    def set_trajectory(self, traj):
        """Retain the trajectory without copying or integrating it again."""
        self.trajectory = traj
        if traj is None:
            self.x_points = []
            self.y_points = []
        else:
            self.x_points = traj[:, 0]
            self.y_points = traj[:, 1]

    def update(self, xinit, v, w, config):
        self.set_trajectory(calc_trajectory_midpoint_vw(xinit, v, w, config))


robot_trajectory = TrajectoryPreview()
human_trajectory = TrajectoryPreview()

angle_robot=0.0
# Calculate a trajectory sampled across a prediction time
def motion_midpoint(x, u, dt):
    """Integrate using midpoint speeds and the interval's initial yaw rate."""
    v_mid, w_mid = u
    theta_mid = x[2] + 0.5 * dt * x[4]

    x[0] += v_mid * math.cos(theta_mid) * dt
    x[1] += v_mid * math.sin(theta_mid) * dt
    x[2] += w_mid * dt
    return x


def prepare_linear_motion(v0, v, config):
    """Prepare once per linear candidate for reuse by all yaw-rate candidates."""
    v0, v = float(v0), float(v)
    dv = v - v0
    a_max = config.max_accel
    a = a_max if dv > 0 else (-a_max if dv < 0 else 0.0)
    t_v = abs(dv) / a_max

    times = config.trajectory_times
    steps = config.trajectory_steps
    index = bisect_left(times, t_v)

    if 0 < index < len(times):
        t0, t1 = times[index - 1], times[index]
        # Treat floating-point neighbors as the same base sampling instant.
        if t_v - t0 > 1e-12 and t1 - t_v > 1e-12:
            steps = (
                steps[:index - 1]
                + (
                    (t_v, t_v - t0, t0 + 0.5 * (t_v - t0), False),
                    (t1, t1 - t_v, t_v + 0.5 * (t1 - t_v), True),
                )
                + steps[index:]
            )

    return v, t_v, a, steps


def calc_trajectory_midpoint(xinit, v, w, config, linear_plan=None):
    """Integrate through t_v, returning states only at the base sampling times."""
    x = [float(value) for value in xinit]
    v0, w0 = x[3], x[4]

    if linear_plan is None:
        linear_plan = prepare_linear_motion(v0, v, config)
    v, t_v, a, steps = linear_plan

    w = float(w)
    dw = w - w0
    alpha_max = config.max_dyawrate
    alpha = (
        alpha_max if dw > 0
        else (-alpha_max if dw < 0 else 0.0)
    )
    t_w = abs(dw) / alpha_max

    traj = np.empty((len(config.trajectory_times), len(x)), dtype=float)
    traj[0] = x
    row = 1

    for t1, h, t_mid, save_point in steps:
        v_mid = v0 + a * t_mid if t_mid < t_v else v
        w_mid = w0 + alpha * t_mid if t_mid < t_w else w
        x = motion_midpoint(x, (v_mid, w_mid), h)

        # Keep endpoint speeds even at the internal, unsaved t_v state.
        x[3] = v0 + a * t1 if t1 < t_v else v
        x[4] = w0 + alpha * t1 if t1 < t_w else w

        if save_point:
            traj[row] = x
            row += 1

    return traj


def calc_trajectory_midpoint_vw(xinit, v, w, config, linear_plan=None):
    """Integrate through t_v and t_w, saving only the base sampling states."""
    x = [float(value) for value in xinit]
    v0, w0 = x[3], x[4]

    if linear_plan is None:
        linear_plan = prepare_linear_motion(v0, v, config)
    v, t_v, a, steps = linear_plan

    w = float(w)
    dw = w - w0
    alpha_max = config.max_dyawrate
    alpha = (
        alpha_max if dw > 0
        else (-alpha_max if dw < 0 else 0.0)
    )
    t_w = abs(dw) / alpha_max

    # Locate t_w in the intervals already split at t_v.
    index = bisect_left(steps, (t_w,))
    if index < len(steps):
        t1, h, _, save_point = steps[index]
        t0 = t1 - h
        # Avoid duplicate boundaries and near-zero intervals.
        if t_w - t0 > 1e-12 and t1 - t_w > 1e-12:
            steps = (
                steps[:index]
                + (
                    (t_w, t_w - t0, t0 + 0.5 * (t_w - t0), False),
                    (t1, t1 - t_w, t_w + 0.5 * (t1 - t_w), save_point),
                )
                + steps[index + 1:]
            )

    traj = np.empty((len(config.trajectory_times), len(x)), dtype=float)
    traj[0] = x
    row = 1

    for t1, h, t_mid, save_point in steps:
        v_mid = v0 + a * t_mid if t_mid < t_v else v
        w_mid = w0 + alpha * t_mid if t_mid < t_w else w
        x = motion_midpoint(x, (v_mid, w_mid), h)

        # Keep endpoint speeds at both internal, unsaved switch states.
        x[3] = v0 + a * t1 if t1 < t_v else v
        x[4] = w0 + alpha * t1 if t1 < t_w else w

        if save_point:
            traj[row] = x
            row += 1

    return traj

def calc_angle_fromtraj(v, y, config):

    x = np.array([config.x,config.y,config.th,v,y])
    time = 0
    while time <= config.predict_time:
        # store each motion model along a trajectory
        x = motion(x, [v, y], config.dt)
        time += config.dt # next sample

    return x[0],x[1]


################################################################################


# Calculate trajectory, costings, and return velocities to apply to robot
def calc_final_input(x, u, vs, ws, config, ob,readob,step=None, log_writer=None):
    # global list_x,list_y,line_num
    xinit = x[:]######
    max_cost = 0.0
    best_traj = None
    max_u = u
    max_u[0] = 0.0 #no way can be chosen
    max_u[1] = human.angular.z
    global angle_robot
    # evaluate all trajectory with sampled input in dynamic window
    for v in vs:
        linear_plan = prepare_linear_motion(xinit[3], v, config)
        for w in ws:

            traj = calc_trajectory_midpoint_vw(xinit, v, w, config, linear_plan)

            to_human_cost = (1-calc_to_human_cost(v,w,config,traj,4)) * config.to_human_cost_gain

            speed_cost = config.speed_cost_gain *(1-abs(human.linear.x - v)/(config.max_speed-config.min_speed))

            ob_cost = calc_obstacle_cost(traj, ob, config, readob) * config.obs_cost_gain

            if np.isinf(ob_cost):
                continue

            final_cost = to_human_cost + speed_cost + ob_cost
            # if log_writer and RECORDER is not None:
            #     log_writer.writerow([
            #         round(step, 3),                   # ← 第几次 while
            #         round(v, 2),
            #         round(w, 2),
            #         to_human_cost,
            #         speed_cost,
            #         ob_cost,
            #         final_cost
            #     ])

            # search minimum trajectory     ##最大代价
            if max_cost <= final_cost:
                max_cost = final_cost
                max_u = [v, w]
                best_traj = traj
    # if max_u[0]==0.0:
    #     max_u[1]=human.angular.z
    robot_trajectory.set_trajectory(best_traj)
    human_trajectory.update(xinit, human.linear.x, human.angular.z, config)
    return max_u
#################################################################################


# Unified obstacle score: use the minimum distance after the current sample.
def calc_obstacle_cost(traj, ob, config, readob):
    if len(traj) <= 1:
        return 1.0

    minr = config.maxdect
    ox_arr, oy_arr = ob

    if ox_arr.size > 0 and oy_arr.size > 0:
        tx = traj[1:, 0:1]
        ty = traj[1:, 1:2]
        dist2 = (tx - ox_arr)**2 + (ty - oy_arr)**2

        if np.any(dist2 <= config.robot_radius**2):
            return float("-Inf")

        minr = min(minr, np.sqrt(dist2.min()))

    moving_future = readob.moving_future
    if moving_future is not None and moving_future.size > 0:
        # Skip t=0 in both predictions and keep the remaining samples aligned.
        num_steps = min(len(traj), moving_future.shape[1])
        if num_steps > 1:
            dx = traj[None, 1:num_steps, 0] - moving_future[:, 1:num_steps, 0]
            dy = traj[None, 1:num_steps, 1] - moving_future[:, 1:num_steps, 1]
            distances = np.hypot(dx, dy) - readob.obs_info[:, 4:5]

            if np.any(distances <= config.robot_radius):
                return float("-Inf")

            minr = min(minr, distances.min())

    return (minr - config.robot_radius) / (config.maxdect - config.robot_radius)

############################################################################
def calc_to_human_cost( v, w,config,traj,n):
    global human_angle
    # elif n==3:# final point distance method
    #     xr,yr=traj[-1,0],traj[-1,1]
    #     cost=math.sqrt((xh-xr)**2 + (yh-yr)**2)/(2*(config.max_speed-config.min_speed)*config.predict_time)
    if n==4:# w minus directly
        robot_angle=w
        cost = abs(human_angle-robot_angle)/(config.max_yawrate*2)

    return cost
##################################################################################

# Begin DWA calculations
def dwa_control(x, u, config, ob,readob,step,log_writer):
    # Dynamic Window control

    vs,ws = calc_dynamic_window(x, config)

    u = calc_final_input(x, u, vs,ws, config, ob,readob,step,log_writer)

    return u


human_angle=0

def share1(vel_msg):# human command get 获取人类指令
    global human
    global human_angle

    human.linear.x=vel_msg.linear.x
    human.angular.z=vel_msg.angular.z
    
    # if human.linear.x<-0.08:
    #     human.linear.x=-0.08

    # if human.linear.x<0.0:
    #     human.linear.x=0.0
    human_angle=human.angular.z
    if human.linear.x<=0.001 and human.linear.x>=-0.001:
        human.linear.x = 0.0
    # human_angle=cal_angle(human.linear.x,human.angular.z)


class TrajectoryMarkers:
    """Initialize the robot and human trajectory markers for RViz."""

    def __init__(self):
        self.marker = self._create_marker(0.0, 1.0, 0.0)
        self.h_marker = self._create_marker(0.0, 0.0, 1.0)

    @staticmethod
    def _create_marker(red, green, blue):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.LINE_STRIP
        marker.action = marker.ADD

        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        marker.color.a = 1.0
        marker.color.r = red
        marker.color.g = green
        marker.color.b = blue

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        return marker


trajectory_markers = TrajectoryMarkers()
marker = trajectory_markers.marker
h_marker = trajectory_markers.h_marker

def line(num, target_marker, x_points, y_points):
    """Replace a trajectory marker's points with the requested samples."""
    target_marker.points = []

    for i in range(num):
        p = Point()
        p.x = x_points[i]
        p.y = y_points[i]
        p.z = 0.0
        target_marker.points.append(p)

human=Twist()


def signal_handler(signal, frame):
    print("\nCtrl + C  is pressed,exit sys")
    if runtime_state.result_pending:
        delete_rosbag(runtime_state.rosbag_id)
        print("You have not arrive the goal point,rosbag deleted")
    if log_file is not None:
        log_file.close()
    sys.exit(0)  # 退出程序

# 绑定 SIGINT 信号（Ctrl + C）到 signal_handler
signal.signal(signal.SIGINT, signal_handler)

def listen_key():
    def on_press(key):
        try:
            if key.char == 'y':  # 检测键盘输入 y
                runtime_state.save_requested = True
                print("Key 'y' detected. Save requested.")
        except AttributeError:
            pass

    # 启动监听器
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


def main():
    runtime_state.rosbag_id = None
    print(__file__ + " start!!")
    print("human is 0,share is 1")
    runtime_state.control_mode = 1
    # inputs=input()
    # if inputs=="0":
    #         runtime_state.control_mode = 0
    # elif inputs=="1":
    #         runtime_state.control_mode = 1
    # else:
    #         rospy.loginfo("input error,run as human")
    #         runtime_state.control_mode = 0

    # robot specification
    # rand=RandomNumberGenerator()
    config = Config()
    # position of obstacles
    # obs = Obstacles()
    readob = GazeboObsReader(config)
    
    # if chizu=="4":
    #     config.robot_radius=0.106
    #     obs.pattern=1
    #     print("changed OK")
    model_name = "limo"
    # 指定目标位置和方向 (四元数)
    target_position = [0.0, 0.0, 0.148]  # x, y, z
    target_orientation = [0.0, 0.0, 0.0, 1.0]  # x, y, z, w
    set_robot_position(model_name, target_position, target_orientation)
    counter = StringMessageCounter()

    threading.Thread(target=listen_key, daemon=True).start()
    subOdom = rospy.Subscriber("/odom", Odometry, config.assignOdomCoords)
    # subLaser = rospy.Subscriber("/filtered_scan", LaserScan, obs.assignObs, config)
    # subLaser = rospy.Subscriber("/scan", LaserScan, obs.assignObs, config)
    sub_hum = rospy.Subscriber("/cmd_vel_human",Twist,share1,queue_size=1)

    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    processor = Mid360Processor(config.robot_radius)
    # sub_obs = rospy.Subscriber("/gazebo/base_collision",Contact,StringMessageCounter.callbackobs,queue_size=10)
    pub_line = rospy.Publisher('~line_list', Marker, queue_size=10)
    pub_line_human = rospy.Publisher('~line_list_human', Marker, queue_size=10)
    marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
    
    rospy.loginfo("Waiting for first LaserScan...")
    rospy.wait_for_message("/scan", LaserScan)
    rospy.loginfo("LaserScan received.")
    
    speed = Twist()
    # change_goal(config,rand.get_next())
    # goal_sphere(config)
    # initial state [x(m), y(m), theta(rad), v(m/s), omega(rad/s)]
    x = np.array([config.x, config.y, config.th, 0.0, 0.0])
    # initial linear and angular velocities
    u = np.array([0.0, 0.0])
    step = 0
    if RECORDER==True:
        runtime_state.rosbag_id = start_rosbag(
            cmd_vel_pub=pub, initial_position=target_position,
        )
    #     print("You can press y to stop and save rosbag when you need.")
    start_time = rospy.get_time()
    # runs until terminated externally
    while not rospy.is_shutdown():
        current_time = rospy.get_time() - start_time
        if runtime_state.control_mode == 1:
            x[0] = config.x
            x[1] = config.y
            x[2] = config.th
            x[3] = config.v
            x[4] = config.w

            u = dwa_control(x, u, config,processor.xy_set,readob,step=current_time,log_writer=log_writer)
            speed.linear.x = u[0]
            speed.angular.z = u[1]
            line(robot_trajectory.point_count, marker,
                 robot_trajectory.x_points, robot_trajectory.y_points)
            pub_line.publish(marker)
            line(human_trajectory.point_count, h_marker,
                 human_trajectory.x_points, human_trajectory.y_points)
            pub_line_human.publish(h_marker)
        
        else:
            # if 0 then do directly
            speed.linear.x = human.linear.x
            speed.angular.z = human.angular.z
        # if runtime_state.result_pending:
        #     marker_pub.publish(markers)
        publish_cmd_vel(pub, speed)

        if atGoal(config) == 1 or runtime_state.save_requested:
            if runtime_state.result_pending:
                print("YOU have arrive the goal point")
                stop_rosbag()
                # with open(f'/home/frienkie/cood/testold.txt', 'w') as f:
                #     json.dump(list(config.xy), f)

                # change_goal(config,rand.get_next())
                # goal_sphere(config)
                play_celebration_sound()
                save_result(runtime_state.rosbag_id,round(counter.count_time-start_time,2),2)
                runtime_state.result_pending = False
        if counter.send_count>= 1:
            if runtime_state.result_pending:
                print("You have been hit by the obstacle,please reset the robot.write file")
                stop_rosbag()
                save_result(runtime_state.rosbag_id,round(counter.count_time-start_time,2),2)
                runtime_state.result_pending = False
        config.r.sleep()


if __name__ == '__main__':
    rospy.init_node('shared_dwa')
    try:
        main()
    finally:
        stop_rosbag()

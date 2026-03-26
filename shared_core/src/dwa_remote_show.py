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
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion
from std_msgs.msg import Float32, Header
from visualization_msgs.msg import Marker, MarkerArray
from distancetime0 import *
import json
import signal
import sys
from pynput import keyboard  # for keyboard input detect
import simpleaudio as sa
import threading
# import argparse
# parser = argparse.ArgumentParser(description="设置参数")

# 添加参数
# parser.add_argument("--param", type=int, required=True, help="参数值")

# # 解析参数
# args = parser.parse_args()

RECORDER = True

class Config():
    # simulation parameters

    def __init__(self):
        # robot parameter
        #NOTE good params:
        #NOTE 0.55,0.1,1.0,1.6,3.2,0.15,0.05,0.1,1.7,2.4,0.1,3.2,0.18
        self.max_speed = 0.20  # [m/s]
        self.min_speed = 0.0  # [m/s]
        # self.max_yawrate = 1.2  # [rad/s]
        self.max_yawrate = 0.6  # [rad/s]
        self.max_accel = 2.5  # [m/ss]
        self.max_dyawrate = 3.2  # [rad/ss]
        ##################################################33
        self.v_reso = 0.1  # [m/s]
        # self.yawrate_reso = 0.1  # [rad/s]
        self.yawrate_reso = 0.1 # [rad/s]
        #######################################################
        self.dt = 0.4  # [s]
        self.predict_time = 2.4  # [s]
        self.showpredict_time = 3.0  # [s]
        self.showdt = 1.0
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
        self.obs_cost_gain = 2.0 #lower z= fearless
        # if args.param == 3:
        #     self.to_human_cost_gain = 2.0 #lower = detour
        #     self.speed_cost_gain = 1.0 #lower = faster
        #     self.obs_cost_gain = 1.0 #lower z= fearless
        # #############################
        self.robot_radius = 0.12  # [m]
        self.x = 0.0
        self.y = 0.0
        self.th = 0.0
        self.goalX = 0.0
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
        self.xy = set()
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
        self.xy.add((self.x,self.y))
        if self.first_time:  # 只在第一次执行
            self.prev_x = self.x
            self.prev_y = self.y
            self.prev_x0 =self.x
            self.prev_y0 =self.y
            self.first_time = False  # 之后不会再执行
        self.distance0=math.sqrt((self.x - self.prev_x0)**2 + (self.y - self.prev_y0)**2)
        odom_callback(self)



    # Callback for attaining goal co-ordinates from Rviz Publish Point #rviz中设置目标点
    def goalCB(self,msg):
        self.goalX = msg.point.x
        self.goalY = msg.point.y

class Obstacles():
    def __init__(self):
        # Set of coordinates of obstacles in view
        self.obst = set()
        self.obs1 = set()
        self.obs_last=set()
        self.minx = 3.5
        self.deltx = 0.15
        self.index = 0
        self.pattern = 0
        # self.obstacle_list = []  # 先用列表存储，再一次性加入 set


    def assignObs(self, msg, config):

        deg = len(msg.ranges)   # Number of degrees - varies in Sim vs real world
        # print("Laser degree length {}".format(deg))
        self.obst = set()   # reset the obstacle set to only keep visible objects

        for i in range(len(msg.ranges)):
            if msg.ranges[i]>config.maxdect or msg.ranges[i] == float('Inf'):
                continue
            elif np.isnan(msg.ranges[i]):
                distance = config.mindect
            elif msg.ranges[i]<0.11:
                continue
            else:
                distance = msg.ranges[i]

            #degree and rad
            deg = (360)/len(msg.ranges)
            rad = (2*math.pi)/len(msg.ranges)
            objTheta_rad = rad * i
            objTheta_deg = deg * i

            #local(robot)
            obsX_robo = distance * math.cos(abs(objTheta_rad))
            obsY_robo = distance * math.sin(abs(objTheta_rad))

            #global
            obsX = obsX_robo * math.cos(config.th) - obsY_robo * math.sin(config.th) + config.x
            obsY = obsX_robo * math.sin(config.th) + obsY_robo * math.cos(config.th) + config.y

            # all obastacle data
            self.obst.add((obsX,obsY))
        if config.distance0>self.deltx:
            if self.index == 0:
                self.obs1 = self.obst.copy()
            # elif self.index == 1:
            #     self.obs2 = self.obst.copy()
            self.index = (self.index + 1) % 1
            config.prev_x0=config.x
            config.prev_y0=config.y
        if self.pattern==1:
            self.obst= self.obs1 | self.obst

        if len(self.obs_last) - len(self.obst)>50:
            self.obst=self.obs_last | self.obst
        else:
            self.obs_last=self.obst.copy()


    def assignObs1(self, msg, config):
        self.minx=3.5
        for i in range(len(msg.ranges)):
            # if msg.ranges[i] == float('Inf'):
            if msg.ranges[i]>config.maxdect or msg.ranges[i] == float('Inf'):
                distance=config.maxdect
                # scan_range.append(3.5)

            elif msg.ranges[i]<config.mindect:
                distance=config.maxdect

            elif np.isnan(msg.ranges[i]):
                # scan_range.append(0)
                distance = 0.12

            else:
                distance = msg.ranges[i]
            if self.minx>distance:
                self.minx=distance

        
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

    # Dynamic window from robot specification
    Vs = [config.min_speed, config.max_speed,
          -config.max_yawrate, config.max_yawrate]

    # Dynamic window from motion model
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_dyawrate * config.dt,
          x[4] + config.max_dyawrate * config.dt]

    #  [vmin, vmax, yawrate min, yawrate max]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    return dw

list_x=[]
list_y=[]
line_num=0
list_x_human=[]
list_y_human=[]
line_num_human=0
trajectory_markers = []  # 存储所有轨迹的marker


def show_trajectory(xinit, v, y, config):
    global line_num
    x = np.array(xinit)
    time = 0
    list_x.clear()
    list_y.clear()

    while time <= config.showpredict_time:
        # store each motion model along a trajectory
        x = motion(x, [v, y], config.dt)
        list_x.append(x[0])
        list_y.append(x[1])
        time += config.dt # next sample
    line_num=len(list_x)

def show_trajectory_human(xinit, v, y, config):
    global line_num_human
    x = np.array(xinit)
    time = 0
    list_x_human.clear()
    list_y_human.clear()

    while time <= config.showpredict_time:
        # store each motion model along a trajectory
        x = motion(x, [v, y], config.dt)
        list_x_human.append(x[0])
        list_y_human.append(x[1])
        time += config.dt # next sample
    line_num_human=len(list_x_human)

angle_robot=0.0
# Calculate a trajectory sampled across a prediction time
def calc_trajectory(xinit, v, y, config):
    x = np.array(xinit)
    traj = np.array(x)  # many motion models stored per trajectory
    time = 0

    while time <= config.predict_time:
        # store each motion model along a trajectory
        x = motion(x, [v, y], config.dt)
        traj = np.vstack((traj, x))
        time += config.dt # next sample

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
def calc_final_input(x, u, dw, config, ob):
    # global list_x,list_y,line_num
    global trajectory_markers
    xinit = x[:]######
    max_cost = 0.0
    max_u = u
    max_u[0] = 0.0 #no way can be chosen
    max_u[1] = human.angular.z
    global angle_robot
    trajectory_markers = []  # 清空之前的轨迹marker
    marker_id = 0
    # evaluate all trajectory with sampled input in dynamic window
    for v in np.arange(dw[0], dw[1]+config.v_reso, config.v_reso):
        for w in np.arange(dw[2], dw[3]+config.yawrate_reso, config.yawrate_reso):
            w = np.round(w, 2)

            traj = calc_trajectory(xinit, v, w, config)

            to_human_cost = (1-calc_to_human_cost(v,w,config,traj,4)) * config.to_human_cost_gain

            speed_cost = config.speed_cost_gain *(1-abs(human.linear.x - v)/(config.max_speed-config.min_speed))

            ob_cost = calc_obstacle_cost(traj, ob, config) * config.obs_cost_gain

            # 只显示v大于等于0.15时的marker
            if v >= 0.15:
                # 为每个轨迹创建marker
                traj_marker = Marker()
                # 确保header正确初始化
                traj_marker.header = Header()
                traj_marker.header.frame_id = "odom"
                traj_marker.header.stamp = rospy.Time.now()
                traj_marker.ns = "trajectories"
                traj_marker.type = traj_marker.LINE_LIST  # 使用LINE_LIST来实现虚线效果
                traj_marker.action = traj_marker.ADD
                traj_marker.id = marker_id
                marker_id += 1
                
                # 设置marker属性 - 线条宽度减半
                traj_marker.scale.x = 0.025  # 原来0.05，现在减半
                traj_marker.scale.y = 0.025
                traj_marker.scale.z = 0.025
                traj_marker.color.a = 1.0
                traj_marker.pose.orientation.x = 0.0
                traj_marker.pose.orientation.y = 0.0
                traj_marker.pose.orientation.z = 0.0
                traj_marker.pose.orientation.w = 1.0
                traj_marker.pose.position.x = 0.0
                traj_marker.pose.position.y = 0.0
                traj_marker.pose.position.z = 0.0
                
                # 设置所有marker为黄色
                traj_marker.color.r = 1.0
                traj_marker.color.g = 1.0
                traj_marker.color.b = 0.0
                
                # 添加轨迹点 - 使用LINE_LIST创建虚线效果
                # LINE_LIST需要成对添加点，每对点形成一条线段
                traj_marker.points = []
                dash_length = 0.1  # 虚线段的长度
                gap_length = 0.05  # 虚线之间的间隔
                
                if len(traj) > 1:
                    for i in range(len(traj) - 1):
                        # 计算当前点到下一个点的距离
                        dx = traj[i+1, 0] - traj[i, 0]
                        dy = traj[i+1, 1] - traj[i, 1]
                        dist = math.sqrt(dx**2 + dy**2)
                        
                        if dist > 0:
                            # 归一化方向向量
                            dx_norm = dx / dist
                            dy_norm = dy / dist
                            
                            # 沿着路径创建虚线
                            current_dist = 0.0
                            while current_dist < dist:
                                # 虚线段的起点
                                start_point = Point()
                                start_point.x = traj[i, 0] + dx_norm * current_dist
                                start_point.y = traj[i, 1] + dy_norm * current_dist
                                start_point.z = 0.0
                                
                                # 虚线段的终点
                                end_dist = min(current_dist + dash_length, dist)
                                end_point = Point()
                                end_point.x = traj[i, 0] + dx_norm * end_dist
                                end_point.y = traj[i, 1] + dy_norm * end_dist
                                end_point.z = 0.0
                                
                                # 添加一对点形成一条线段
                                traj_marker.points.append(start_point)
                                traj_marker.points.append(end_point)
                                
                                # 移动到下一个虚线段的起点（跳过间隔）
                                current_dist += dash_length + gap_length
                
                # 只有当points不为空时才添加marker（LINE_LIST需要至少一对点）
                if len(traj_marker.points) > 0 and len(traj_marker.points) % 2 == 0:
                    trajectory_markers.append(traj_marker)

            if np.isinf(ob_cost):
                continue

            final_cost = to_human_cost + speed_cost + ob_cost
            

            # search minimum trajectory     ##最大代价
            if max_cost <= final_cost:
                max_cost = final_cost
                max_u = [v, w]
    # if max_u[0]==0.0:
    #     max_u[1]=human.angular.z
    show_trajectory(xinit, max_u[0], max_u[1], config)
    show_trajectory_human(xinit, human.linear.x, human.angular.z, config)
    return max_u
#################################################################################



# Calculate obstacle cost inf: collision, 0:free
def calc_obstacle_cost(traj, ob, config):
    skip_n = 1
    minr = config.maxdect
    
    # Loop through every obstacle in set and calc Pythagorean distance
    # Use robot radius to determine if collision
    for ii in range(1, len(traj[:, 1]), skip_n):
        for i in ob.copy():
            ox = i[0]           ##障害物
            oy = i[1]
            dx = traj[ii, 0] - ox    ##轨迹
            dy = traj[ii, 1] - oy

            r = math.sqrt(dx**2 + dy**2)  ##距离

            if r <= config.robot_radius:
                return float("-Inf")  # collision

            if minr >= r:
                minr = r


    return (minr-config.robot_radius)/(config.maxdect-config.robot_radius)

############################################################################333
def calc_to_human_cost( v, w,config,traj,n):
    global human_angle
    # elif n==3:# final point distance method
    #     xr,yr=traj[-1,0],traj[-1,1]
    #     cost=math.sqrt((xh-xr)**2 + (yh-yr)**2)/(2*(config.max_speed-config.min_speed)*config.showpredict_time)
    if n==4:# w minus directly
        robot_angle=w
        cost = abs(human_angle-robot_angle)/(config.max_yawrate*2)
    elif n==5:
        cost=abs(math.atan2(w,v)-human_r)/math.pi

    return cost
##################################################################################

# Begin DWA calculations
def dwa_control(x, u, config, ob):
    # Dynamic Window control

    dw = calc_dynamic_window(x, config)

    u = calc_final_input(x, u, dw, config, ob)

    return u

def atGoal(config):
    # check at goal
    if math.sqrt((config.x - config.goalX)**2 + (config.y - config.goalY)**2) \
        <= config.goal_radius:
        return 1
    return 0


human_angle=0
yici=1

def share1(vel_msg,config):# human command get 获取人类指令
    global human
    global human_r
    global inputkey
    global human_angle

    human.linear.x=vel_msg.linear.x
    human.angular.z=vel_msg.angular.z
    
    # if human.linear.x<-0.08:
    #     human.linear.x=-0.08

    if human.linear.x<0.0:
        human.linear.x=0.0
    human_angle=human.angular.z
    if human.linear.x<=0.001 and human.linear.x>=-0.001:
        human.linear.x = 0.0
    # human_angle=cal_angle(human.linear.x,human.angular.z)
    human_r=math.atan2(human.angular.z,human.linear.x)





def change_goal(config,n):
    global yici
    if n==0:
        config.goalX=-4.0
        config.goalY=10.0
    if n==1:
        config.goalX=-3.8
        config.goalY=8.5
    if n==2:
        config.goalX=-4.0
        config.goalY=-3.0
    if n==3:
        config.goalX=-3.7
        config.goalY=5.9
    if n==4:
        config.goalX=-6.0
        config.goalY=6.0
    if n==5:
        config.goalX=6.0
        config.goalY=6.0
    if yici==1:
        print(n)

human=Twist()
human_r=float("inf")
inputkey=0
write=0

def signal_handler(signal, frame):
    print("\nCtrl + C  is pressed,exit sys")
    sys.exit(0)  # 退出程序

# 绑定 SIGINT 信号（Ctrl + C）到 signal_handler
signal.signal(signal.SIGINT, signal_handler)

def listen_key():
    global write

    def on_press(key):
        global write
        try:
            if key.char == 'y':  # 检测键盘输入 y
                write = 1
                print("Key 'y' detected. Variable 'write' set to 1.")
        except AttributeError:
            pass

    # 启动监听器
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

def main():
    global yici
    global write
    print(__file__ + " start!!")
    print("human is 0,share is 1")
    #inputkey=1
    inputs=input()
    if inputs=="0":
            inputkey=0
    elif inputs=="1":
            inputkey=1
    else:
            rospy.loginfo("input error,run as human")
            inputkey = 0

    # robot specification
    # rand=RandomNumberGenerator()
    config = Config()
    # position of obstacles
    obs = Obstacles()
    # if chizu=="4":
    #     config.robot_radius=0.106
    #     obs.pattern=1
    #     print("changed OK")

    threading.Thread(target=listen_key, daemon=True).start()

    subOdom = rospy.Subscriber("/odom", Odometry, config.assignOdomCoords)
    subLaser1 = rospy.Subscriber("/scan", LaserScan, obs.assignObs1, config)
    subLaser = rospy.Subscriber("/filtered_scan", LaserScan, obs.assignObs, config)

    sub_hum = rospy.Subscriber("/cmd_vel_human",Twist,share1,config,queue_size=1)

    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

    x_value_pub = rospy.Publisher('/min_d', Float32, queue_size = 1)
    # sub_obs = rospy.Subscriber("/gazebo/base_collision",Contact,StringMessageCounter.callbackobs,queue_size=10)
    marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
    marker_array_pub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=1)
    
    speed = Twist()
    # change_goal(config,rand.get_next())
    # goal_sphere(config)
    # initial state [x(m), y(m), theta(rad), v(m/s), omega(rad/s)]
    x = np.array([config.x, config.y, config.th, 0.0, 0.0])
    # initial linear and angular velocities
    u = np.array([0.0, 0.0])

    if RECORDER==True:
        file_value=start_rosbag()
        print("You can press y to stop and save rosbag when you need.")
        start_time = rospy.get_time()
    # runs until terminated externally
    while not rospy.is_shutdown():

        if (inputkey== 1):
            u = dwa_control(x, u, config, obs.obst)
            x[0] = config.x
            x[1] = config.y
            x[2] = config.th
            x[3] = u[0]
            x[4] = u[1]
            speed.linear.x = x[3]
            speed.angular.z = x[4]
            # 清空之前的marker
            delete_marker = Marker()
            delete_marker.header = Header()
            delete_marker.header.frame_id = "odom"
            delete_marker.header.stamp = rospy.Time.now()
            delete_marker.ns = "trajectories"  # 只删除trajectories命名空间的marker
            delete_marker.action = delete_marker.DELETEALL
            marker_pub.publish(delete_marker)
            # 使用MarkerArray批量发布所有轨迹的marker，提高效率
            if len(trajectory_markers) > 0:
                marker_array = MarkerArray()
                current_time = rospy.Time.now()
                for traj_marker in trajectory_markers:
                    # 确保header正确设置
                    if not hasattr(traj_marker.header, 'frame_id') or not traj_marker.header.frame_id:
                        if not hasattr(traj_marker, 'header') or traj_marker.header is None:
                            traj_marker.header = Header()
                        traj_marker.header.frame_id = "odom"
                    traj_marker.header.stamp = current_time
                    # 确保points不为空且数量为偶数（LINE_LIST需要成对点）
                    if len(traj_marker.points) > 0 and len(traj_marker.points) % 2 == 0:
                        marker_array.markers.append(traj_marker)
                # 只有当marker_array中有marker时才发布
                if len(marker_array.markers) > 0:
                    marker_array_pub.publish(marker_array)
        
        else:
            # if 0 then do directly
            speed.linear.x = human.linear.x
            speed.angular.z = human.angular.z
        if yici>0:
            marker_pub.publish(markers)
        pub.publish(speed)
        x_value_pub.publish(obs.minx)
        if RECORDER==True:
            if write==1 or config.x>2.8:

                if yici>0:
                    print("YOU have arrive the goal point")
                    print("run time: %.2f s" % (rospy.get_time()-start_time))
                    print("distance in this time: %.2f m" % config.distance)
                    save(start_time,config.distance,inputkey)
                    # with open(f'/home/frienkie/cood/test{file_value}.txt', 'w') as f:
                    #     json.dump(list(config.xy), f)
                    stop_rosbag()
                    # change_goal(config,rand.get_next())
                    # goal_sphere(config)
                    play_celebration_sound()
                    yici=0
            
        config.r.sleep()


if __name__ == '__main__':
    rospy.init_node('shared_dwa')
    main()
#!/usr/bin/env python

import rospy
import math
from visualization_msgs.msg import Marker
from openpyxl import Workbook, load_workbook # type: ignore
import os
import simpleaudio as sa
import threading
from gazebo_model_collision_plugin.msg import Contact
import random
import numpy as np
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from sensor_msgs.msg import JoyFeedbackArray,JoyFeedback
import time,csv
from geometry_msgs.msg import Twist, Point
from rosbag_recorder import RecorderSession


markers = Marker()
def goal_sphere(config):
    
    global markers
    # 创建Marker消息

    markers.header.frame_id = "odom"  # 设置参考坐标系
    markers.header.stamp = rospy.Time.now()
    markers.ns = "sphere"  # 命名空间
    markers.id = 2  # ID
    markers.type = Marker.SPHERE  # 类型设置为球体
    markers.action = Marker.ADD  # 操作类型

    # 设置球体的中心坐标和大小
    markers.pose.position.x = config.goalX  # 圆心 x 坐标
    markers.pose.position.y = config.goalY  # 圆心 y 坐标
    markers.pose.position.z = 0.0  # 圆心 z 坐标
    markers.pose.orientation.x = 0.0
    markers.pose.orientation.y = 0.0
    markers.pose.orientation.z = 0.0
    markers.pose.orientation.w = 1.0

    markers.scale.x = config.robot_radius*4  # 球体在 x 方向的大小
    markers.scale.y = config.robot_radius*4  # 球体在 y 方向的大小
    markers.scale.z = config.robot_radius*4  # 球体在 z 方向的大小

    # 设置颜色
    markers.color.r = 0.0  # 红色分量
    markers.color.g = 1.0  # 绿色分量
    markers.color.b = 0.0  # 蓝色分量
    markers.color.a = 1.0  # 透明度

def get_time(start_time):

    end_time = rospy.get_time()  # 获取结束时间
    elapsed_time = end_time - start_time  # 计算运行时间
    print("spend time: %.2f 秒" % elapsed_time)
    return elapsed_time

def odom_callback(config):
    delta_distance = math.sqrt((config.x - config.prev_x)**2 + (config.y - config.prev_y)**2)
    if delta_distance<0.0001:
        delta_distance=0.0
    config.distance += delta_distance
    config.prev_x = config.x
    config.prev_y = config.y
    # 更新之前的位置
    # print("当前prev: %.2f , %.2f", config.prev_x,config.prev_y)
    # print("当前总里程: %.2f 米", config.distance)

def play_celebration_sound():
    """
    播放一个到达终点的庆祝音效（异步，不阻塞主线程）
    """
    def _play():
        notes = [
            (660, 0.3),  # E5
            (0,   0.1),  # pause
            (880, 0.3),  # A5
            (0,   0.1),
            (660, 0.5),  # E5 (长音)
        ]
        sample_rate = 44100
        for freq, dur in notes:
            if freq == 0:
                time.sleep(dur)
                continue
            t = np.linspace(0, dur, int(sample_rate * dur), False)
            wave = np.sin(freq * 2 * np.pi * t)
            audio = (wave * 32767).astype(np.int16)
            sa.play_buffer(audio, 1, 2, sample_rate).wait_done()
    
    threading.Thread(target=_play, daemon=True).start()

def save(time,distance,count_time,n,m,chizu):# n is direct switch,m is para
    file_name = "/home/pc/result/data.xlsx"

    # 检查文件是否存在
    if os.path.exists(file_name):
        # 如果文件存在，加载工作簿
        workbook = load_workbook(file_name)
        sheet = workbook.active
    else:
        # 如果文件不存在，创建新的工作簿
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Sheet1"
    row=1
    # 查找第一列中最后一行的行号

    # 需要写入的浮点数据（这里替换为你的数据）
    data1 = get_time(time)  # 示例浮点数
    data2 = distance
    if count_time==0:
        data3=0
    else:
        data3 = count_time
    # 格式化浮点数，保留两位小数
    formatted_data1 = round(data1, 2)
    formatted_data2 = round(data2, 2)
    # 将数据写入第一列的下一行
    if n==0:
        for cell in sheet['N']:
            if cell.value is not None:
                row += 1
            else:
                break
        sheet.cell(row=row, column=1, value=formatted_data1)
        sheet.cell(row=row, column=2, value=formatted_data2)
        sheet.cell(row=row, column=3, value=data3)
        sheet.cell(row=row, column=14, value=4)
        sheet.cell(row=row, column=15, value=chizu)
        sheet.cell(row=row, column=16, value=row-1)
    else:
        for cell in sheet['N']:
            if cell.value is not None:
                row += 1
            else:
                break
        sheet.cell(row=row, column=7, value=formatted_data1)
        sheet.cell(row=row, column=8, value=formatted_data2)
        sheet.cell(row=row, column=9, value=data3)
        sheet.cell(row=row, column=14, value=m)
        sheet.cell(row=row, column=15, value=chizu)
        sheet.cell(row=row, column=16, value=row-1)
    # 保存工作簿
    workbook.save(file_name)

class StringMessageCounter:

    def __init__(self):

        # -----------------------------
        # 参数
        # -----------------------------
        self.release_time = 0.3          # 多久没有Contact认为离开障碍物
        self.vibration_time = 1.0        # 震动持续时间

        # -----------------------------
        # 原来的统计变量（保留）
        # -----------------------------
        self.send_count = 0
        self.count_time = 0
        self.first_time = True

        # -----------------------------
        # Edge Trigger状态
        # -----------------------------
        self.in_collision = False
        self.last_contact_time = 0.0

        # -----------------------------
        # JoyFeedback
        # -----------------------------
        self.vibra = JoyFeedback()
        self.vibra.type = 1
        self.vibra.id = 1
        self.vibra.intensity = 1.0

        self.novibra = JoyFeedback()
        self.novibra.type = 1
        self.novibra.id = 1
        self.novibra.intensity = 0.0

        self.vibration = rospy.Publisher(
            "joy/set_feedback",
            JoyFeedbackArray,
            queue_size=1
        )

        rospy.Subscriber(
            "/gazebo/base_collision",
            Contact,
            self.callbackobs,
            queue_size=1
        )

        # 定时检测是否离开障碍物
        self.monitor_timer = rospy.Timer(
            rospy.Duration(0.05),
            self.monitor_collision
        )

        # 当前震动Timer
        self.vibration_timer = None

        # 程序退出时关闭震动
        rospy.on_shutdown(self.shutdown)

        rospy.loginfo("Collision node started.")

    # ---------------------------------------------------------
    # 收到Contact
    # ---------------------------------------------------------
    def callbackobs(self, msg):

        now = rospy.get_time()

        # 更新最近一次Contact时间
        self.last_contact_time = now

        # 已经处于碰撞状态，不重复计数
        if self.in_collision:
            return

        # ---------- 新碰撞 ----------
        self.in_collision = True

        if self.first_time:
            self.count_time = now
            self.first_time = False

        self.send_count += 1

        rospy.loginfo(
            "Collision #%d  Time=%.3f",
            self.send_count,
            now
        )

        # 开始震动
        self.vibration.publish(
            JoyFeedbackArray(array=[self.vibra])
        )

        # 若之前还有Timer，取消
        if self.vibration_timer is not None:
            self.vibration_timer.shutdown()

        # 新建停止震动Timer
        self.vibration_timer = rospy.Timer(
            rospy.Duration(self.vibration_time),
            self.stop_vibration,
            oneshot=True
        )

    # ---------------------------------------------------------
    # 停止震动
    # ---------------------------------------------------------
    def stop_vibration(self, event):

        self.vibration.publish(
            JoyFeedbackArray(array=[self.novibra])
        )

        self.vibration_timer = None

    # ---------------------------------------------------------
    # 检测是否已经离开障碍物
    # ---------------------------------------------------------
    def monitor_collision(self, event):

        if not self.in_collision:
            return

        now = rospy.get_time()

        # 超过release_time没有收到Contact
        if now - self.last_contact_time > self.release_time:

            self.in_collision = False

            rospy.loginfo("Collision released.")

    # ---------------------------------------------------------
    # 程序退出
    # ---------------------------------------------------------
    def shutdown(self):

        rospy.loginfo("Stopping vibration...")

        # 关闭震动Timer
        if self.vibration_timer is not None:
            self.vibration_timer.shutdown()
            self.vibration_timer = None

        # 连续发送几次停止震动，防止最后一帧丢失
        msg = JoyFeedbackArray(array=[self.novibra])

        rate = rospy.Rate(50)

        for _ in range(10):
            self.vibration.publish(msg)
            rate.sleep()

        rospy.loginfo("Vibration stopped.")
        

def set_robot_position(model_name, position, orientation):
    """
    瞬间移动机器人到指定坐标
    :param model_name: 模型名称 (例如 "robot")
    :param position: 坐标 [x, y, z]
    :param orientation: 四元数 [x, y, z, w]
    """
    # 等待服务
    rospy.wait_for_service('/gazebo/set_model_state')
    
    try:
        # 创建服务代理
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        
        # 定义模型状态
        state_msg = ModelState()
        state_msg.model_name = model_name
        state_msg.pose.position.x = position[0]
        state_msg.pose.position.y = position[1]
        state_msg.pose.position.z = position[2]
        state_msg.pose.orientation.x = orientation[0]
        state_msg.pose.orientation.y = orientation[1]
        state_msg.pose.orientation.z = orientation[2]
        state_msg.pose.orientation.w = orientation[3]
        state_msg.twist.linear.x = 0.0
        state_msg.twist.linear.y = 0.0
        state_msg.twist.linear.z = 0.0
        state_msg.twist.angular.x = 0.0
        state_msg.twist.angular.y = 0.0
        state_msg.twist.angular.z = 0.0
        # 调用服务
        resp = set_state(state_msg)
        if resp.success:
            rospy.loginfo(f"Successfully moved {model_name} to {position}")
        else:
            rospy.logwarn(f"Failed to move {model_name}: {resp.status_message}")
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
    cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    stop_cmd = Twist()
    rate = rospy.Rate(20)
    for _ in range(20):  # 连续发送一段时间，确保停止
        cmd_vel_pub.publish(stop_cmd)
        rate.sleep()

_rosbag_recorder = None
import pandas as pd

def read_last_value_from_column():
    # 读取Excel文件
    workbook = load_workbook("/home/pc/result/data.xlsx") # 替换为你的Excel文件名
    sheet = workbook['Sheet1']  # 替换为你的工作表名
    row=1
    for cell in sheet['P']:
            if cell.value is not None:
                row += 1
            else:
                break
    if row>1:
        last_row=row-1
    else:
        last_row=1
    # 获取第16列的最后一行的值
    valid_values = sheet.cell(row=last_row, column=16).value
    # 检查是否是数字
    if not isinstance(valid_values, (int, float)):
        return 1
    return valid_values+1

def get_next_rosbag_number(excel_path="/home/pc/result/data.xlsx"):
    """Choose the next Excel ID; existing bag files do not change the ID."""
    last_num = 0
    if os.path.exists(excel_path):
        wb = load_workbook(excel_path, data_only=True, read_only=True)
        try:
            for row in wb.active.iter_rows(min_row=2, min_col=1, max_col=1,
                                           values_only=True):
                try:
                    last_num = max(last_num, int(row[0]))
                except (TypeError, ValueError, OverflowError):
                    continue
        finally:
            wb.close()

    return last_num + 1

def start_rosbag(cmd_vel_pub=None, initial_position=None, require_obstacles=False):
    """Hold still until the initial /odom has been written and flushed.

    require_obstacles is ignored for compatibility with older controllers.
    """
    global _rosbag_recorder
    if _rosbag_recorder is not None:
        raise RuntimeError("A rosbag recording is already running")

    timeout = float(rospy.get_param("~rosbag_ready_timeout", 5.0))
    position_tolerance = float(rospy.get_param("~rosbag_position_tolerance", 0.02))
    velocity_tolerance = float(rospy.get_param("~rosbag_velocity_tolerance", 0.02))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("~rosbag_ready_timeout must be positive and finite")
    if any(not math.isfinite(value) or value < 0
           for value in (position_tolerance, velocity_tolerance)):
        raise ValueError("Rosbag position and velocity tolerances must be nonnegative and finite")
    initial_position = rospy.get_param("~rosbag_initial_position", initial_position)
    if initial_position is not None:
        if len(initial_position) < 2 or not all(math.isfinite(float(value))
                                                for value in initial_position[:2]):
            raise ValueError("Rosbag initial position must contain finite x and y coordinates")
        initial_position = [float(value) for value in initial_position[:2]]
    required_topics = ["/odom"]
    if cmd_vel_pub is None:
        cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

    stop_cmd = Twist()
    cmd_vel_pub.publish(stop_cmd)
    rosbag_dir = os.path.expanduser("~/rosbag")
    os.makedirs(rosbag_dir, exist_ok=True)
    count = get_next_rosbag_number()
    output_file = os.path.join(rosbag_dir, f"rosbag{count}.bag")
    _rosbag_recorder = RecorderSession(
        output_file, required_topics=required_topics,
        initial_position=initial_position,
        position_tolerance=position_tolerance,
        velocity_tolerance=velocity_tolerance,
        minimum_odom_stamp=rospy.Time.now().to_sec(),
    )
    rospy.on_shutdown(stop_rosbag)
    print(f"Starting rosbag recording: {output_file}")
    try:
        _rosbag_recorder.start()
        status = _rosbag_recorder.wait_until_ready(
            lambda: cmd_vel_pub.publish(stop_cmd), timeout, rospy.is_shutdown,
        )
    except BaseException:
        cmd_vel_pub.publish(stop_cmd)
        try:
            stop_rosbag()
        except Exception as exc:
            rospy.logerr("Failed to close recorder: %s", exc)
        raise
    rospy.loginfo("Recording ready: %s; initial odom: %s",
                  ", ".join(status["written_topics"]), status["last_odom"])
    return count


def publish_cmd_vel(publisher, command):
    """Stop the controller if its active recording fails after the handshake."""
    if _rosbag_recorder is not None:
        try:
            _rosbag_recorder.check()
        except RuntimeError:
            publisher.publish(Twist())
            raise
    publisher.publish(command)


def stop_rosbag():
    """Drain queued messages, close the bag, and finish the recorder process."""
    global _rosbag_recorder
    recorder = _rosbag_recorder
    _rosbag_recorder = None
    if recorder is not None:
        print("Stopping rosbag recording...")
        recorder.stop()
        print("rosbag recording stopped.")

def atGoal(config):
    # check at goal
    # if math.sqrt((config.x - config.goalX)**2 + (config.y - config.goalY)**2) \
    #     <= config.goal_radius:
    if config.x>= config.goalX:
        return 1
    return 0

def save_result(a, b, c, file_path="/home/pc/result/data.xlsx"):
    """
    将a、b写入A、B列，将c写入同一行F列
    如果文件不存在，则创建并写入表头
    """
    if b<0:
        b=0
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if not os.path.exists(file_path):
        wb = Workbook()
        ws = wb.active
        ws.append(["A", "B", "C", "D", "E", "F"])
    else:
        wb = load_workbook(file_path)
        ws = wb.active

    # 新增一行（A、B列）
    ws.append([a, b])

    # 当前新增的行号
    row = ws.max_row

    # F列写入c
    ws.cell(row=row, column=6).value = c

    wb.save(file_path)

import os

def delete_rosbag(count):
    """
    删除对应编号的 rosbag 文件，例如：
    count = 3
    删除：
        ~/rosbag/rosbag3.bag
        ~/rosbag/rosbag3.bag.active（如果存在）
    """

    stop_rosbag()
    if count is None:
        return
    rosbag_dir = os.path.expanduser("~/rosbag")

    bag_file = os.path.join(rosbag_dir, f"rosbag{count}.bag")
    active_file = bag_file + ".active"

    deleted = False

    if os.path.exists(active_file):
        os.remove(active_file)
        print(f"Deleted: {active_file}")
        deleted = True

    if os.path.exists(bag_file):
        os.remove(bag_file)
        print(f"Deleted: {bag_file}")
        deleted = True

    if not deleted:
        print(f"rosbag{count}.bag does not exist.")

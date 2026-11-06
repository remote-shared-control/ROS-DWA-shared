#!/usr/bin/env python
import rospy
import numpy as np
from pedsim_msgs.msg import TrackedPersons
import math


class Config:
    def __init__(self):
        self.robot_radius = 0.22         # 机器人半径
        self.cube_half_size =0.1        # 原始立方体半边长


class CubeReader:
    def __init__(self, config):
        self.robot_radius_cal = 0.2
        self.config = config
        self.data = None
        self.pose_data = None

        rospy.Subscriber(
            "/ped_visualization",
            TrackedPersons,
            self.callback
        )

    def get_yaw_from_quaternion(self, q):
        """
        从四元数提取 yaw（只计算平面旋转）
        """
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)

    def get_expanded_vertices_2d(self, pose):
        """
        完全在 2D 中计算膨胀后的 4 个顶点
        """
        expanded = self.config.cube_half_size + self.robot_radius_cal

        local = np.array([
            [-expanded, -expanded],
            [-expanded,  expanded],
            [ expanded,  expanded],
            [ expanded, -expanded]
        ])

        yaw = self.get_yaw_from_quaternion(pose.orientation)

        R = np.array([
            [math.cos(yaw), -math.sin(yaw)],
            [math.sin(yaw),  math.cos(yaw)]
        ])

        rotated = np.dot(local, R.T)

        rotated[:, 0] += pose.position.x
        rotated[:, 1] += pose.position.y

        return [(round(p[0], 4), round(p[1], 4)) for p in rotated]

    def callback(self, msg):

        new_data = []
        new_pose_data = []
        obs_index = 1

        for track in msg.tracks:

            # 从 person track 中读取
            pose = track.pose.pose
            twist = track.twist.twist
            # print(pose)
            # 计算膨胀矩形顶点
            vertices = self.get_expanded_vertices_2d(pose)

            # 速度
            vx = round(twist.linear.x, 3)
            vy = round(twist.linear.y, 3)

            # 存储格式
            row = [obs_index] + vertices + [vx, vy]
            pose_row = [obs_index] + [
            round(pose.position.x, 3),
            round(pose.position.y, 3)]

            new_pose_data.append(pose_row)
            # print(f"Obstacle {obs_index},Velocity=({vx}, {vy})")

            new_data.append(row)

            obs_index += 1

        if len(new_data) == 0:
            self.data = None
            self.pose_data = None
        else:
            self.data = new_data
            self.pose_data = new_pose_data
        # print(self.pose_data)
#############################################333


def get_rotation_center(x, y, theta, v, w):
    """
    计算差速机器人的瞬时旋转中心
    
    参数:
        x, y: 机器人当前位置
        theta: 机器人朝向角（弧度）
        v: 线速度
        w: 角速度
    
    返回:
        (cx, cy): 旋转中心坐标，若 w≈0 则返回 None（直线运动）
    """
    
    r = v / w  # 旋转半径
    
    # 旋转矩阵 R(theta + pi/2)
    angle = theta + np.pi / 2
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    
    # 计算旋转中心
    c = np.array([x, y]) + R @ np.array([r, 0])
    
    return r,(c[0], c[1])



def transform_rectangle_to_local(vertices, c, v, theta, r):
    """
    将长方形的相邻两边变换到局部坐标系
    
    参数:
        vertices: 长方形四个顶点 [(x0,y0), (x1,y1), (x2,y2), (x3,y3)]，按顺序排列
        c: 旋转中心 (cx, cy)
        v: 速度向量 (vx, vy)
        theta: 机器人朝向角（弧度）
        r: 旋转半径（带符号，正为左转，负为右转）
    
    返回:
        (edge1_result, edge2_result): 两个元组，每个元组包含 (p_prime, q_prime, v_prime, phi, m)
    """
    vertices = np.array(vertices)
    c = np.array(c)
    v = np.array(v)
    
    r_abs = abs(r)

    
    # 计算长方形的两条边长
    len_01 = np.linalg.norm(vertices[1] - vertices[0])
    len_12 = np.linalg.norm(vertices[2] - vertices[1])
    
    # 边: (p_idx, q_idx, 对边任一点idx, 垂直边长)
    edges = [
        (0, 1, 2, len_12),
        (1, 2, 3, len_01),
        (2, 3, 0, len_12),
        (3, 0, 1, len_01),
    ]
    
    # 选择离旋转中心最近的顶点
    distances = [np.linalg.norm(vertices[i] - c) for i in range(4)]
    nearest_vertex = np.argmin(distances)
    
    selected_edge_indices = [
        nearest_vertex,
        (nearest_vertex - 1) % 4
    ]
    
    results = []
    
    for edge_idx in selected_edge_indices:
        p_idx, q_idx, opp_idx, perp_len = edges[edge_idx]
        
        p = vertices[p_idx]
        q = vertices[q_idx]
        opp_point = vertices[opp_idx]
        
        # 计算 psi 角
        psi = np.arctan2(q[1] - p[1], q[0] - p[0])
        
        # 旋转矩阵 R(-psi)
        cos_psi = np.cos(-psi)
        sin_psi = np.sin(-psi)
        R_neg_psi = np.array([
            [cos_psi, -sin_psi],
            [sin_psi,  cos_psi]
        ])
        
        # 计算 p' 和 q'
        p_prime = (1 / r_abs) * R_neg_psi @ (p - c)
        q_prime = (1 / r_abs) * R_neg_psi @ (q - c)
        
        # 计算 v'
        v_prime = (1 / r_abs) * R_neg_psi @ v
        
        # 计算 phi
        phi = theta - psi - np.sign(r) * np.pi / 2
        phi = np.arctan2(np.sin(phi), np.cos(phi))
        
        # 计算对边一点变换后的 y 坐标，判断符号
        opp_prime = (1 / r_abs) * R_neg_psi @ (opp_point - c)
        sign = 1 if opp_prime[1] > p_prime[1] else -1
        
        m = sign * perp_len / r_abs
        
        results.append((p_prime, q_prime, v_prime, phi, m))
    
    return results[0], results[1]

# 示例
# if __name__ == "__main__":
#     q = (3, 4)
#     p = (1, 0)
#     c = (0, 1)
#     v = (1, 0.5)
#     theta = 0
#     r = 2.0  # 正值表示左转
    
#     p_prime, q_prime, v_prime, phi = transform_to_local(q, p, c, v, theta, r)
    
#     print(f"p' = ({p_prime[0]:.4f}, {p_prime[1]:.4f})")
#     print(f"q' = ({q_prime[0]:.4f}, {q_prime[1]:.4f})")
#     print(f"v' = ({v_prime[0]:.4f}, {v_prime[1]:.4f})")
#     print(f"phi = {phi:.4f} rad ({np.degrees(phi):.2f}°)")
# ```

# 输出：
# ```
# p' = (0.3201, -0.3841)
# q' = (1.4223, 0.7111)
# v' = (0.3841, -0.2401)
# phi = -1.0304 rad (-59.04°)



######################################
from math import sin, cos


def find_min_tc_fast(phi, omega, v_py_prime, p_y_prime, 
                     p_x_prime, v_px_prime, q_x_prime, m, T):
    """
    求两个方程满足条件的最小解中较小的那个，解在(0, T]范围内：
    方程1: sin(phi + omega * tc) - v_py_prime * tc - p_y_prime = 0
    方程2: sin(phi + omega * tc) - v_py_prime * tc - p_y_prime + m = 0
    
    条件: p_x_prime < cos(phi + omega * tc) - v_px_prime * tc < q_x_prime
    """
    
    # 自动计算采样次数: 4倍余量确保精度
    samples = int(omega * T * 0.6366197723675814) + 1  # 0.6366 ≈ 2/pi
    samples = max(samples << 2, 8)  # 乘4并确保最小值
    
    step = T / samples
    tc_prev = step
    base_prev = sin(phi + omega * tc_prev) - v_py_prime * tc_prev - p_y_prime
    f1_prev = base_prev
    f2_prev = base_prev + m
    
    for i in range(2, samples + 1):
        tc_curr = i * step
        base_curr = sin(phi + omega * tc_curr) - v_py_prime * tc_curr - p_y_prime
        f1_curr = base_curr
        f2_curr = base_curr + m
        
        # 检测方程1符号变化
        if f1_prev * f1_curr < 0:
            # Illinois算法
            a, fa, b, fb = tc_prev, f1_prev, tc_curr, f1_curr
            for _ in range(8):
                c = (a * fb - b * fa) / (fb - fa)
                fc = sin(phi + omega * c) - v_py_prime * c - p_y_prime
                if fc * fa < 0:
                    b, fb = c, fc
                    fa *= 0.5
                else:
                    a, fa = c, fc
                    fb *= 0.5
            tc_root = (a * fb - b * fa) / (fb - fa)
            x_val = cos(phi + omega * tc_root) - v_px_prime * tc_root
            if p_x_prime <= x_val <= q_x_prime:
                return tc_root
        
        # 检测方程2符号变化（fa, fb基于f1加m）
        if f2_prev * f2_curr < 0:
            # Illinois算法，fc = f1的fc + m
            a, fa, b, fb = tc_prev, f2_prev, tc_curr, f2_curr
            for _ in range(8):
                c = (a * fb - b * fa) / (fb - fa)
                fc = sin(phi + omega * c) - v_py_prime * c - p_y_prime + m
                if fc * fa < 0:
                    b, fb = c, fc
                    fa *= 0.5
                else:
                    a, fa = c, fc
                    fb *= 0.5
            tc_root = (a * fb - b * fa) / (fb - fa)
            x_val = cos(phi + omega * tc_root) - v_px_prime * tc_root
            if p_x_prime <= x_val <= q_x_prime:
                return tc_root
        
        tc_prev = tc_curr
        f1_prev = base_curr
        f2_prev = base_curr + m
    
    return T


def point_in_quad(vertices: tuple, x: float, y: float) -> bool:
    """
    检查点(x,y)是否在四边形内部（射线法）
    
    参数:
        vertices: 四边形四个顶点 ((x1,y1), (x2,y2), (x3,y3), (x4,y4))
        x, y: 待检测点坐标
    
    返回:
        True 如果点在内部，False 否则
    """
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]
    x4, y4 = vertices[3]
    
    inside = False
    
    # 边1: (x1,y1) -> (x2,y2)
    if ((y1 > y) != (y2 > y)):
        if x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    
    # 边2: (x2,y2) -> (x3,y3)
    if ((y2 > y) != (y3 > y)):
        if x < (x3 - x2) * (y - y2) / (y3 - y2) + x2:
            inside = not inside
    
    # 边3: (x3,y3) -> (x4,y4)
    if ((y3 > y) != (y4 > y)):
        if x < (x4 - x3) * (y - y3) / (y4 - y3) + x3:
            inside = not inside
    
    # 边4: (x4,y4) -> (x1,y1)
    if ((y4 > y) != (y1 > y)):
        if x < (x1 - x4) * (y - y4) / (y1 - y4) + x4:
            inside = not inside
    
    return inside


############################
def compute_intersection_time(
    vertices: tuple,  # ((x1,y1), (x2,y2), (x3,y3), (x4,y4))
    v_rel: tuple,     # (vx, vy) 相对速度 = 机器人速度 - 障碍物速度
    x: float,
    y: float,
    T: float
) -> float:
    """
    计算机器人与四边形障碍物的最短相交时间
    
    参数:
        vertices: 四边形四个顶点 ((x1,y1), (x2,y2), (x3,y3), (x4,y4))
        v_rel: 相对速度 (vx, vy) = 机器人速度 - 障碍物速度
        x, y: 机器人起始位置
        T: 最大时间
    
    返回:
        最短相交时间，不相交返回T，起点在内部返回0
    """
    
    # 解包
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]
    x4, y4 = vertices[3]
    vx, vy = v_rel
    
    # 预计算边向量
    dx1 = x2 - x1
    dy1 = y2 - y1
    dx2 = x3 - x2
    dy2 = y3 - y2
    dx3 = x4 - x3
    dy3 = y4 - y3
    dx4 = x1 - x4
    dy4 = y1 - y4
    
    
    # 2. 静止且在外部
    if vx == 0.0 and vy == 0.0:
        return T
    
    # 3. 检查四条边的相交
    min_t = T
    
    # 边1
    det1 = vx * dy1 - vy * dx1
    if det1 != 0.0:
        qx1 = x1 - x
        qy1 = y1 - y
        t_num1 = qx1 * dy1 - qy1 * dx1
        if det1 > 0.0:
            if t_num1 >= 0.0 and t_num1 < min_t * det1:
                s_num1 = qx1 * vy - qy1 * vx
                if s_num1 >= 0.0 and s_num1 <= det1:
                    min_t = t_num1 / det1
        else:
            if t_num1 <= 0.0 and t_num1 > min_t * det1:
                s_num1 = qx1 * vy - qy1 * vx
                if s_num1 <= 0.0 and s_num1 >= det1:
                    min_t = t_num1 / det1
    
    # 边2
    det2 = vx * dy2 - vy * dx2
    if det2 != 0.0:
        qx2 = x2 - x
        qy2 = y2 - y
        t_num2 = qx2 * dy2 - qy2 * dx2
        if det2 > 0.0:
            if t_num2 >= 0.0 and t_num2 < min_t * det2:
                s_num2 = qx2 * vy - qy2 * vx
                if s_num2 >= 0.0 and s_num2 <= det2:
                    min_t = t_num2 / det2
        else:
            if t_num2 <= 0.0 and t_num2 > min_t * det2:
                s_num2 = qx2 * vy - qy2 * vx
                if s_num2 <= 0.0 and s_num2 >= det2:
                    min_t = t_num2 / det2
    
    # 边3
    det3 = vx * dy3 - vy * dx3
    if det3 != 0.0:
        qx3 = x3 - x
        qy3 = y3 - y
        t_num3 = qx3 * dy3 - qy3 * dx3
        if det3 > 0.0:
            if t_num3 >= 0.0 and t_num3 < min_t * det3:
                s_num3 = qx3 * vy - qy3 * vx
                if s_num3 >= 0.0 and s_num3 <= det3:
                    min_t = t_num3 / det3
        else:
            if t_num3 <= 0.0 and t_num3 > min_t * det3:
                s_num3 = qx3 * vy - qy3 * vx
                if s_num3 <= 0.0 and s_num3 >= det3:
                    min_t = t_num3 / det3
    
    # 边4
    det4 = vx * dy4 - vy * dx4
    if det4 != 0.0:
        qx4 = x4 - x
        qy4 = y4 - y
        t_num4 = qx4 * dy4 - qy4 * dx4
        if det4 > 0.0:
            if t_num4 >= 0.0 and t_num4 < min_t * det4:
                s_num4 = qx4 * vy - qy4 * vx
                if s_num4 >= 0.0 and s_num4 <= det4:
                    min_t = t_num4 / det4
        else:
            if t_num4 <= 0.0 and t_num4 > min_t * det4:
                s_num4 = qx4 * vy - qy4 * vx
                if s_num4 <= 0.0 and s_num4 >= det4:
                    min_t = t_num4 / det4
    
    return min_t


def cal_time(cube,x, y, theta, v, w, T):

        # 直接使用 cube.data（不再通过 get_snapshot 复制）
        cube_data = cube.data
        if cube_data is None:
            return T
        else:
            if w==0 or v==0:

                minre_mid= T
                # 机器人速度，同样保留三位小数
                robot_vx = round(v * math.cos(theta), 3)
                robot_vy = round(v * math.sin(theta), 3)
                for row in cube_data:   # self.data 中每一行代表一个障碍物
                    vertices=(row[1],row[2],row[3],row[4])
                    if point_in_quad(vertices, x, y):
                        return 0.0
                    else:
                        obs_vx, obs_vy = row[5], row[6]
                        rel_vel = (robot_vx - obs_vx, robot_vy - obs_vy)
                        result=compute_intersection_time(vertices, rel_vel, x, y, T)
                        if minre_mid>result:
                            minre_mid=result
                return minre_mid
                #print(minre_mid)
            else:

                r,c=get_rotation_center(x, y, theta, v, w)
                minre_mid= T
                for row in cube_data:   # self.data 中每一行代表一个障碍物
                    # 四个顶点（每个点都是 tuple: (x, y)）
                    vertices=(row[1],row[2],row[3],row[4])
                    if point_in_quad(vertices, x, y):
                        return 0.0
                    else:
                        v_obs=(row[5],row[6])
                        # minre = T
                        for p_prime, q_prime, v_prime, phi, m in transform_rectangle_to_local(vertices, c, v_obs, theta, r):
                        # 处理每条边
                            result =  find_min_tc_fast(phi=phi, omega=w, v_py_prime= v_prime[1], p_y_prime=p_prime[1], p_x_prime=p_prime[0],v_px_prime = v_prime[0], q_x_prime=q_prime[0], m=-m, T=T)
                            if minre_mid>=result:
                                minre_mid=result
                        # if minre_mid>=minre:
                        #     minre_mid=minre
                return minre_mid
        # print(minre_mid)
        

if __name__ == "__main__":
    rospy.init_node("gazebo_obs_reader")
    rospy.Rate(20)
    config = Config()
    cube=CubeReader(config)
    i=0
    x,y=0.0,0.0
    v,w ,theta=0.0,0.0,0.0
    T=20.0
    while not rospy.is_shutdown():
        time=cal_time(cube,x, y, theta, v, w, T)

        # i+=1
        # if(i>1000):
        #     print(time)
        #     i=0
            

    
    










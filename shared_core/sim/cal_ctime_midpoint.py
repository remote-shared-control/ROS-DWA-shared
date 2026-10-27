#!/usr/bin/env python3
"""分段 TTC：先检查 midpoint-vw 过渡轨迹，再求匀速阶段的边界方程。

``cube.data`` 沿用 cal_ctime.py 的格式：
    [id, vertex0, vertex1, vertex2, vertex3, obstacle_vx, obstacle_vy]
顶点必须按顺序给出，并且已经膨胀了机器人半径。障碍物在预测期间只做
恒定速度的平移，不改变朝向；机器人因此作为点进行碰撞检测。

The transition samples come from the planner's existing midpoint-vw integrator.
Between samples, collision checks use continuous relative line segments. Once
both commanded speeds are reached, the remaining trajectory is an exact line
or circular arc. This module intentionally has no ROS dependency.
"""

import math

import numpy as np


_SPACE_EPS = 1e-10
_TIME_EPS = 1e-12


def _linear_collision_time(vertices, position, velocity, horizon):
    """Clip relative point motion against a convex rectangle; None means clear.

    Boundary contact, including a stationary point on an edge, is a collision.
    The vertices may be clockwise or counterclockwise.
    """
    vertices = np.asarray(vertices, dtype=float)
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    area_twice = sum(
        vertices[i, 0] * vertices[(i + 1) % 4, 1]
        - vertices[(i + 1) % 4, 0] * vertices[i, 1]
        for i in range(4)
    )
    orientation = 1.0 if area_twice >= 0.0 else -1.0
    enter, leave = 0.0, float(horizon)
    for i in range(4):
        edge = vertices[(i + 1) % 4] - vertices[i]
        length = float(np.linalg.norm(edge))
        if length == 0.0:
            raise ValueError("Obstacle rectangles must have nonzero edges")
        normal = orientation * np.array([-edge[1], edge[0]]) / length
        distance = float(np.dot(position - vertices[i], normal))
        rate = float(np.dot(velocity, normal))
        if rate == 0.0:
            if distance < -_SPACE_EPS:
                return None
            continue
        crossing = -distance / rate
        if rate > 0.0:
            enter = max(enter, crossing)
        else:
            leave = min(leave, crossing)
        if enter > leave + _TIME_EPS:
            return None
    if leave < -_TIME_EPS or enter > horizon + _TIME_EPS:
        return None
    return min(float(horizon), max(0.0, enter))


def _boundary_roots(phi, omega, radius, obstacle_vy, initial_y, horizon):
    """Yield all roots of a transformed edge equation in chronological order.

    This is the dimensional form of sin(phi+w*t)-v_y'*t-p_y'=0.
    Analytic derivative zeros partition it into monotone intervals; bisection
    then cannot skip the first interval, negative-w roots, or tangent contact.
    Evaluating the sine *difference* keeps small angular velocities stable.
    """
    def equation(t):
        half_angle = 0.5 * omega * t
        return (initial_y + 2.0 * radius * math.cos(phi + half_angle)
                * math.sin(half_angle) - obstacle_vy * t)

    knots = [0.0, float(horizon)]
    ratio = obstacle_vy / (radius * omega)
    if -1.0 <= ratio <= 1.0:
        angle = math.acos(ratio)
        phase_low, phase_high = sorted((phi, phi + omega * horizon))
        period = 2.0 * math.pi
        for base in (angle, -angle):
            first = math.ceil((phase_low - base) / period)
            last = math.floor((phase_high - base) / period)
            for k in range(first, last + 1):
                t = (base + period * k - phi) / omega
                if 0.0 < t < horizon:
                    knots.append(t)
    knots = sorted(set(knots))
    previous_root = -math.inf
    left = knots[0]
    f_left = equation(left)
    if abs(f_left) <= _SPACE_EPS:
        previous_root = left
        yield left
    for right in knots[1:]:
        f_right = equation(right)
        if (f_left * f_right < 0.0
                and abs(f_left) > _SPACE_EPS and abs(f_right) > _SPACE_EPS):
            a, b, fa = left, right, f_left
            for _ in range(60):
                midpoint = 0.5 * (a + b)
                fm = equation(midpoint)
                if fm == 0.0:
                    a = b = midpoint
                    break
                if (fa < 0.0) != (fm < 0.0):
                    b = midpoint
                else:
                    a, fa = midpoint, fm
                if b - a <= _TIME_EPS:
                    break
            root = 0.5 * (a + b)
            if root > previous_root + _TIME_EPS:
                previous_root = root
                yield root
        if abs(f_right) <= _SPACE_EPS and right > previous_root + _TIME_EPS:
            previous_root = right
            yield right
        left, f_left = right, f_right


def _constant_collision_time(vertices, obstacle_velocity, pose, v, w, horizon):
    """Solve the constant-speed phase, measured from its own starting pose."""
    vertices = np.asarray(vertices, dtype=float)
    position = np.asarray(pose[:2], dtype=float)
    theta = float(pose[2])
    obstacle_velocity = np.asarray(obstacle_velocity, dtype=float)
    if _linear_collision_time(vertices, position, (0.0, 0.0), 0.0) is not None:
        return 0.0
    if w == 0.0 or v == 0.0:
        robot_velocity = np.array([v * math.cos(theta), v * math.sin(theta)])
        return _linear_collision_time(
            vertices, position, robot_velocity - obstacle_velocity, horizon
        )

    signed_radius = v / w
    radius = abs(signed_radius)
    # c = (x,y) + R(theta + pi/2) (v/w, 0), evaluated at t_constant.
    center = position + signed_radius * np.array([-math.sin(theta), math.cos(theta)])
    radial = position - center
    # For tiny radii compared with the world coordinates, subtraction cannot
    # resolve the center accurately. The analytic radial direction is exact.
    if radius < 1e-4 * max(1.0, float(np.max(np.abs(position)))):
        radial = signed_radius * np.array([math.sin(theta), -math.cos(theta)])
    earliest = None
    for i in range(4):
        p, q = vertices[i], vertices[(i + 1) % 4]
        edge = q - p
        edge_length = float(np.linalg.norm(edge))
        tangent = edge / edge_length
        normal = np.array([-tangent[1], tangent[0]])
        phi = math.atan2(float(np.dot(radial, normal)), float(np.dot(radial, tangent)))
        # R(-psi) transforms positions and obstacle velocity into the frame
        # centered on c. Compute their differences before subtracting c to
        # avoid cancellation for very large turning radii.
        velocity_local = np.array([
            np.dot(obstacle_velocity, tangent), np.dot(obstacle_velocity, normal)
        ])
        initial_x = float(np.dot(position - p, tangent))
        initial_y = float(np.dot(position - p, normal))
        # initial_y = radius*sin(phi) - dot(p-center, normal), evaluated
        # without subtracting nearly equal large values. initial_x likewise
        # places the transformed edge interval at [0, edge_length].
        edge_horizon = horizon if earliest is None else earliest
        for t in _boundary_roots(phi, w, radius, velocity_local[1], initial_y, edge_horizon):
            half_angle = 0.5 * w * t
            x = (initial_x - 2.0 * radius * math.sin(phi + half_angle)
                 * math.sin(half_angle) - velocity_local[0] * t)
            if -_SPACE_EPS <= x <= edge_length + _SPACE_EPS:
                earliest = t
                break
    return earliest


def cal_time(cube, transition, v, w, T):
    """Return TTC from candidate evaluation time; no collision returns T + 1.

    Parameters
    ----------
    cube : object
        Its ``data`` contains already inflated translating rectangles.
    transition : dict
        ``t_constant = max(tv, tw)``; ``times`` starts at 0 and ends exactly at
        ``min(t_constant, T)``; ``states`` contains corresponding midpoint-vw
        states [x, y, theta, v, w]. A zero-length transition has one sample.
    v, w : float
        Candidate speeds reached at t_constant.
    T : float
        Prediction horizon measured from the initial state, in seconds;
        the controller supplies config.movetimemin.

    先逐时间段检查过渡轨迹。某段碰撞时，取该段所有障碍物的最早 TTC，
    立即返回。过渡阶段安全时，才把障碍物平移到 t_constant，使用该时刻
    的机器人位姿求圆心、坐标转换和边界方程；最终加回 t_constant。
    """
    T = float(T)
    if T < 0.0:
        raise ValueError("T must be nonnegative")
    no_collision = T + 1.0
    cube_data = cube.data
    if cube_data is None or len(cube_data) == 0:
        return no_collision
    times = transition["times"]
    states = transition["states"]
    t_constant = float(transition["t_constant"])
    if t_constant < 0.0 or len(times) == 0 or len(times) != len(states):
        raise ValueError("Invalid midpoint transition")
    if (abs(times[0]) > _TIME_EPS
            or abs(times[-1] - min(t_constant, T)) > 1e-9
            or any(b <= a for a, b in zip(times, times[1:]))):
        raise ValueError("Transition times must cover [0, min(t_constant, T)]")
    obstacles = [
        (np.asarray(row[1:5], dtype=float), np.asarray(row[5:7], dtype=float))
        for row in cube_data
    ]
    initial_position = np.asarray(states[0][:2], dtype=float)
    for vertices, _ in obstacles:
        if _linear_collision_time(vertices, initial_position, (0.0, 0.0), 0.0) is not None:
            return 0.0

    for i in range(len(times) - 1):
        start, end = float(times[i]), float(times[i + 1])
        duration = end - start
        position = np.asarray(states[i][:2], dtype=float)
        velocity = (np.asarray(states[i + 1][:2], dtype=float) - position) / duration
        earliest = None
        for vertices, obstacle_velocity in obstacles:
            hit = _linear_collision_time(
                vertices, position - obstacle_velocity * start,
                velocity - obstacle_velocity, duration,
            )
            if hit is not None and (earliest is None or hit < earliest):
                earliest = hit
        if earliest is not None:
            return min(T, start + earliest)

    if t_constant >= T:
        return no_collision
    remaining = T - t_constant
    result = no_collision
    for vertices, obstacle_velocity in obstacles:
        hit = _constant_collision_time(
            vertices + obstacle_velocity * t_constant,
            obstacle_velocity, states[-1], v, w, remaining,
        )
        if hit is not None:
            result = min(result, t_constant + hit)
    return result

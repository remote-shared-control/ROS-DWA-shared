#!/usr/bin/env python3
"""分段 TTC：先检查 midpoint-vw 过渡轨迹，再求匀速阶段的边界方程。

``cube.data`` 沿用 cal_ctime.py 的格式：
    [id, vertex0, vertex1, vertex2, vertex3, obstacle_vx, obstacle_vy]
顶点必须按顺序给出，并且已经按 robot_radius_cal 膨胀。折线过渡阶段
额外按 robot_radius_transition 与该半径的差值外移边界，匀速方程阶段
保持原矩形。障碍物只做恒速平移；机器人作为点进行碰撞检测。

The transition samples come from the planner's existing midpoint-vw integrator.
Between samples, collision checks use continuous relative line segments. Once
both commanded speeds are reached, the remaining trajectory is an exact line
or circular arc. This module intentionally has no ROS dependency.
"""

import math

_SPACE_EPS = 1e-10
_TIME_EPS = 1e-12
# Broad-phase bounds must err on the side of retaining an obstacle. Account
# for both the narrow phase's tolerances and roundoff in world coordinates.
_BOUNDS_SPACE_EPS = 4.0 * _SPACE_EPS
_BOUNDS_ROUND_EPS = 32.0 * 2.220446049250313e-16


class _Rectangle:
    """Copied scalar geometry shared by every candidate in one cycle."""

    __slots__ = ("edges", "vx", "vy", "bounds", "coordinate_scale")

    def __init__(self, vertices, vx=0.0, vy=0.0):
        points = tuple((float(p[0]), float(p[1])) for p in vertices)
        area_twice = sum(
            points[i][0] * points[(i + 1) % 4][1]
            - points[(i + 1) % 4][0] * points[i][1]
            for i in range(4)
        )
        orientation = 1.0 if area_twice >= 0.0 else -1.0
        edges = []
        for i in range(4):
            px, py = points[i]
            qx, qy = points[(i + 1) % 4]
            dx, dy = qx - px, qy - py
            length = math.sqrt(dx * dx + dy * dy)
            if length == 0.0:
                raise ValueError("Obstacle rectangles must have nonzero edges")
            tx, ty = dx / length, dy / length
            edges.append((px, py, tx, ty, -ty, tx, length, orientation))
        self.edges = tuple(edges)
        self.vx, self.vy = float(vx), float(vy)
        self.bounds = (min(p[0] for p in points), max(p[0] for p in points),
                       min(p[1] for p in points), max(p[1] for p in points))
        self.coordinate_scale = max(1.0, *(abs(value) for value in self.bounds))


class _PreparedObstacles:
    __slots__ = ("obstacles", "transition_padding")

    def __init__(self, obstacles, transition_padding=0.0):
        self.obstacles = tuple(obstacles)
        self.transition_padding = float(transition_padding)


def prepare_obstacles(cube):
    """Copy obstacle geometry once per planning cycle for reuse by candidates.

    The returned snapshot owns scalar copies of all coordinates and velocities;
    subsequent updates to ``cube.data`` cannot change a candidate's obstacles.
    Passing a snapshot again is inexpensive and returns it unchanged.
    CubeReader supplies two radii: its vertices already include robot_radius_cal;
    only transition clipping adds robot_radius_transition - robot_radius_cal.
    Plain data-only inputs retain their existing, already-inflated geometry.
    """
    if isinstance(cube, _PreparedObstacles):
        return cube
    rows = cube.data
    transition_padding = 0.0
    if hasattr(cube, "robot_radius_transition"):
        transition_padding = (float(cube.robot_radius_transition)
                              - float(cube.robot_radius_cal))
    return _PreparedObstacles(
        () if rows is None else
        (_Rectangle(row[1:5], row[5], row[6]) for row in rows),
        transition_padding,
    )


def _clip_rectangle(rectangle, px, py, vx, vy, horizon, shift_x=0.0, shift_y=0.0,
                    padding=0.0):
    """Clip against cached half-planes; positive padding moves edges outward."""
    enter, leave = 0.0, float(horizon)
    for ex, ey, _, _, nx, ny, _, orientation in rectangle.edges:
        distance = ((px - (ex + shift_x)) * nx
                    + (py - (ey + shift_y)) * ny) * orientation + padding
        rate = (vx * nx + vy * ny) * orientation
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


def _linear_collision_time(vertices, position, velocity, horizon):
    """Clip relative point motion against a convex rectangle; None means clear.

    Boundary contact, including a stationary point on an edge, is a collision.
    The vertices may be clockwise or counterclockwise.
    """
    rectangle = vertices if isinstance(vertices, _Rectangle) else _Rectangle(vertices)
    return _clip_rectangle(rectangle, float(position[0]), float(position[1]),
                           float(velocity[0]), float(velocity[1]), horizon)


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


def _constant_collision_time(vertices, obstacle_velocity, pose, v, w, horizon,
                             shift_x=0.0, shift_y=0.0):
    """Solve the constant-speed phase, measured from its own starting pose."""
    rectangle = vertices if isinstance(vertices, _Rectangle) else _Rectangle(vertices)
    v, w = float(v), float(w)
    px, py = float(pose[0]), float(pose[1])
    theta = float(pose[2])
    ovx, ovy = float(obstacle_velocity[0]), float(obstacle_velocity[1])
    if _clip_rectangle(rectangle, px, py, 0.0, 0.0, 0.0, shift_x, shift_y) is not None:
        return 0.0
    if w == 0.0 or v == 0.0:
        return _clip_rectangle(rectangle, px, py, v * math.cos(theta) - ovx,
                               v * math.sin(theta) - ovy, horizon, shift_x, shift_y)

    signed_radius = v / w
    radius = abs(signed_radius)
    # c = (x,y) + R(theta + pi/2) (v/w, 0), evaluated at t_constant.
    sin_theta, cos_theta = math.sin(theta), math.cos(theta)
    radial_x = px - (px - signed_radius * sin_theta)
    radial_y = py - (py + signed_radius * cos_theta)
    # For tiny radii compared with the world coordinates, subtraction cannot
    # resolve the center accurately. The analytic radial direction is exact.
    if radius < 1e-4 * max(1.0, abs(px), abs(py)):
        radial_x, radial_y = signed_radius * sin_theta, -signed_radius * cos_theta
    earliest = None
    for ex, ey, tx, ty, nx, ny, edge_length, _ in rectangle.edges:
        phi = math.atan2(radial_x * nx + radial_y * ny,
                         radial_x * tx + radial_y * ty)
        # R(-psi) transforms positions and obstacle velocity into the frame
        # centered on c. Compute their differences before subtracting c to
        # avoid cancellation for very large turning radii.
        velocity_x = ovx * tx + ovy * ty
        velocity_y = ovx * nx + ovy * ny
        dx, dy = px - (ex + shift_x), py - (ey + shift_y)
        initial_x = dx * tx + dy * ty
        initial_y = dx * nx + dy * ny
        # initial_y = radius*sin(phi) - dot(p-center, normal), evaluated
        # without subtracting nearly equal large values. initial_x likewise
        # places the transformed edge interval at [0, edge_length].
        edge_horizon = horizon if earliest is None else earliest
        for t in _boundary_roots(phi, w, radius, velocity_y, initial_y, edge_horizon):
            half_angle = 0.5 * w * t
            x = (initial_x - 2.0 * radius * math.sin(phi + half_angle)
                 * math.sin(half_angle) - velocity_x * t)
            if -_SPACE_EPS <= x <= edge_length + _SPACE_EPS:
                earliest = t
                break
    return earliest


def _constant_swept_bounds(pose, v, w, horizon):
    """Conservative world AABB for a line or an entire constant-speed arc.

    Each arc coordinate differs from its endpoint chord by at most
    abs(v*w)*horizon**2/8, since that bounds its second derivative. Intersect
    this box with the independent distance-from-start bound. Both bounds
    contain every point, including arcs spanning more than one revolution.
    The endpoint uses a sine difference, avoiding large turning centers.
    """
    px, py, theta = float(pose[0]), float(pose[1]), float(pose[2])
    v, w = float(v), float(w)
    half_angle = 0.5 * w * horizon
    distance = v * horizon
    travel = distance * (math.sin(half_angle) / half_angle if half_angle else 1.0)
    end_x = px + travel * math.cos(theta + half_angle)
    end_y = py + travel * math.sin(theta + half_angle)
    pad = abs(v * w) * horizon * horizon / 8.0
    reach = abs(distance)
    if w != 0.0:
        reach = min(reach, 2.0 * abs(v / w))
    return (max(min(px, end_x) - pad, px - reach),
            min(max(px, end_x) + pad, px + reach),
            max(min(py, end_y) - pad, py - reach),
            min(max(py, end_y) + pad, py + reach))


def _swept_bounds_overlap(robot_bounds, robot_scale, robot_speed, obstacle,
                          start, horizon):
    """Reject only strictly disjoint robot/translated-obstacle swept AABBs."""
    shift_x, shift_y = obstacle.vx * start, obstacle.vy * start
    dx, dy = obstacle.vx * horizon, obstacle.vy * horizon
    xmin, xmax, ymin, ymax = obstacle.bounds
    margin = (_BOUNDS_SPACE_EPS
              + _BOUNDS_ROUND_EPS * (robot_scale + obstacle.coordinate_scale
                                     + abs(shift_x) + abs(shift_y)
                                     + abs(dx) + abs(dy))
              + _TIME_EPS * (robot_speed + abs(obstacle.vx) + abs(obstacle.vy)))
    # Equality and near-equality reach the original continuous narrow phase;
    # in particular, an edge/corner touch at the horizon is never rejected.
    return not (robot_bounds[1] < xmin + shift_x + min(0.0, dx) - margin
                or robot_bounds[0] > xmax + shift_x + max(0.0, dx) + margin
                or robot_bounds[3] < ymin + shift_y + min(0.0, dy) - margin
                or robot_bounds[2] > ymax + shift_y + max(0.0, dy) + margin)


def cal_time(cube, transition, v, w, T):
    """Return TTC from candidate evaluation time; no collision returns T + 1.

    Parameters
    ----------
    cube : object
        Its ``data`` contains already inflated translating rectangles, or it
        is a reusable snapshot returned by ``prepare_obstacles``.
        When supplied, robot_radius_transition changes only the transition
        radius; the constant phase retains the robot_radius_cal geometry.
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
    prepared = prepare_obstacles(cube)
    obstacles = prepared.obstacles
    if not obstacles:
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
    transition_padding = prepared.transition_padding if t_constant > 0.0 else 0.0
    initial_x, initial_y = float(states[0][0]), float(states[0][1])
    for obstacle in obstacles:
        if _clip_rectangle(obstacle, initial_x, initial_y, 0.0, 0.0, 0.0,
                           padding=transition_padding) is not None:
            return 0.0

    for i in range(len(times) - 1):
        start, end = float(times[i]), float(times[i + 1])
        duration = end - start
        px, py = float(states[i][0]), float(states[i][1])
        vx = (float(states[i + 1][0]) - px) / duration
        vy = (float(states[i + 1][1]) - py) / duration
        earliest = None
        for obstacle in obstacles:
            hit = _clip_rectangle(
                obstacle, px - obstacle.vx * start, py - obstacle.vy * start,
                vx - obstacle.vx, vy - obstacle.vy, duration,
                padding=transition_padding,
            )
            if hit is not None and (earliest is None or hit < earliest):
                earliest = hit
        if earliest is not None:
            return min(T, start + earliest)

    if t_constant >= T:
        return no_collision
    remaining = T - t_constant
    result = no_collision
    robot_bounds = _constant_swept_bounds(states[-1], v, w, remaining)
    robot_scale = max(1.0, *(abs(value) for value in robot_bounds))
    robot_speed = abs(float(v))
    for obstacle in obstacles:
        # Once a hit is known, a later obstacle only matters up to that time.
        # The full-horizon robot box remains a conservative bound for this
        # shorter interval; obstacle motion can use the tighter interval.
        horizon = min(remaining, result - t_constant)
        if not _swept_bounds_overlap(robot_bounds, robot_scale, robot_speed,
                                     obstacle, t_constant, horizon):
            continue
        hit = _constant_collision_time(
            obstacle, (obstacle.vx, obstacle.vy), states[-1], v, w, horizon,
            obstacle.vx * t_constant, obstacle.vy * t_constant,
        )
        if hit is not None:
            result = min(result, t_constant + hit)
    return result

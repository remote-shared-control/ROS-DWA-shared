"""Regression oracle for continuous midpoint/constant TTC safety decisions.

The four _reference* functions are frozen from cal_ctime_midpoint.py at commit
c4f083a (before scalar/prepared-obstacle optimizations), with only function names changed. Keep their NumPy geometry independent of the production
solver: these tests must detect an optimization that skips a candidate collision.
No Git checkout or command is needed to execute this suite.
"""

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


_SPACE_EPS = 1e-10
_TIME_EPS = 1e-12


def _reference_linear_collision_time(vertices, position, velocity, horizon):
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


def _reference_boundary_roots(phi, omega, radius, obstacle_vy, initial_y, horizon):
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


def _reference_constant_collision_time(vertices, obstacle_velocity, pose, v, w, horizon):
    """Solve the constant-speed phase, measured from its own starting pose."""
    vertices = np.asarray(vertices, dtype=float)
    position = np.asarray(pose[:2], dtype=float)
    theta = float(pose[2])
    obstacle_velocity = np.asarray(obstacle_velocity, dtype=float)
    if _reference_linear_collision_time(vertices, position, (0.0, 0.0), 0.0) is not None:
        return 0.0
    if w == 0.0 or v == 0.0:
        robot_velocity = np.array([v * math.cos(theta), v * math.sin(theta)])
        return _reference_linear_collision_time(
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
        for t in _reference_boundary_roots(phi, w, radius, velocity_local[1], initial_y, edge_horizon):
            half_angle = 0.5 * w * t
            x = (initial_x - 2.0 * radius * math.sin(phi + half_angle)
                 * math.sin(half_angle) - velocity_local[0] * t)
            if -_SPACE_EPS <= x <= edge_length + _SPACE_EPS:
                earliest = t
                break
    return earliest


def _reference_cal_time(cube, transition, v, w, T):
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
        if _reference_linear_collision_time(vertices, initial_position, (0.0, 0.0), 0.0) is not None:
            return 0.0

    for i in range(len(times) - 1):
        start, end = float(times[i]), float(times[i + 1])
        duration = end - start
        position = np.asarray(states[i][:2], dtype=float)
        velocity = (np.asarray(states[i + 1][:2], dtype=float) - position) / duration
        earliest = None
        for vertices, obstacle_velocity in obstacles:
            hit = _reference_linear_collision_time(
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
        hit = _reference_constant_collision_time(
            vertices + obstacle_velocity * t_constant,
            obstacle_velocity, states[-1], v, w, remaining,
        )
        if hit is not None:
            result = min(result, t_constant + hit)
    return result


def _load_solver():
    path = Path(__file__).resolve().parents[1] / "cal_ctime_midpoint.py"
    spec = importlib.util.spec_from_file_location("hybrid_ttc_equivalence_target", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rectangle(center, half_size, angle=0.0, velocity=(0.0, 0.0), clockwise=False):
    cx, cy = center
    hx, hy = half_size
    cosine, sine = math.cos(angle), math.sin(angle)
    points = [(cx + cosine * x - sine * y, cy + sine * x + cosine * y)
              for x, y in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))]
    if clockwise:
        points.reverse()
    return [1, *points, *velocity]


def _midpoint_transition(initial, v, w, tv, tw, horizon, dt):
    """Build valid input independently of the production trajectory builder."""
    t_constant = max(tv, tw)
    end = min(t_constant, horizon)
    times = sorted({0.0, end, *(float(t) for t in np.arange(dt, end, dt)),
                    *(t for t in (tv, tw) if 0.0 < t < end)})
    x, y, theta, v0, w0 = initial
    states = [list(initial)]
    for start, stop in zip(times, times[1:]):
        duration = stop - start
        middle = (start + stop) / 2.0
        v_mid = v0 + (v - v0) * middle / tv if middle < tv else v
        w_mid = w0 + (w - w0) * middle / tw if middle < tw else w
        w_start = w0 + (w - w0) * start / tw if start < tw else w
        theta_mid = theta + 0.5 * duration * w_start
        x += v_mid * math.cos(theta_mid) * duration
        y += v_mid * math.sin(theta_mid) * duration
        theta += w_mid * duration
        v_end = v0 + (v - v0) * stop / tv if stop < tv else v
        w_end = w0 + (w - w0) * stop / tw if stop < tw else w
        states.append([x, y, theta, v_end, w_end])
    return {"times": times, "states": states, "t_constant": t_constant}


def _position_at(motion, v, w, time):
    """Point on the contract's polyline followed by its exact constant arc."""
    times, states = motion["times"], motion["states"]
    for index, end in enumerate(times[1:]):
        if time <= end:
            fraction = (time - times[index]) / (end - times[index])
            return (np.asarray(states[index][:2]) * (1.0 - fraction)
                    + np.asarray(states[index + 1][:2]) * fraction)
    x, y, theta = states[-1][:3]
    duration = time - times[-1]
    if w == 0.0:
        return np.array([x + v * math.cos(theta) * duration,
                         y + v * math.sin(theta) * duration])
    half = 0.5 * w * duration
    distance = 2.0 * (v / w) * math.sin(half)
    return np.array([x + distance * math.cos(theta + half),
                     y + distance * math.sin(theta + half)])


class HybridTTCEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solver = _load_solver()

    def assert_equivalent(self, rows, motion, v, w, horizon):
        cube = SimpleNamespace(data=rows)
        expected = _reference_cal_time(cube, motion, v, w, horizon)
        for obstacles in (cube, self.solver.prepare_obstacles(cube)):
            actual = self.solver.cal_time(obstacles, motion, v, w, horizon)
            # Check the safety decision separately: a near-horizon difference
            # must never disappear into the numerical TTC comparison tolerance.
            self.assertEqual(actual <= horizon, expected <= horizon,
                             f"missed/extra collision: expected {expected}, got {actual}")
            self.assertEqual(actual < horizon, expected < horizon,
                             f"strict movetimemin decision changed: {expected} vs {actual}")
            self.assertAlmostEqual(actual, expected, delta=2e-8)
        return expected

    def test_seeded_rotated_moving_and_stationary_rectangles(self):
        rng = np.random.default_rng(20261027)
        collisions = clear = clipped = constant = 0
        for case in range(960):
            horizon = float(rng.uniform(0.05, 6.0))
            v = float(rng.uniform(-1.8, 1.8))
            w = float(rng.uniform(-4.0, 4.0))
            # Include exact straight travel, pure rotation, tiny nonzero
            # angular rates, and both signs of forward/reverse arc motion.
            if case % 12 in (0, 6):
                v = 0.0
            w = {0: 0.0, 1: 1e-9, 2: -1e-9, 3: 1e-5,
                 4: -1e-5}.get(case % 12, w)
            initial = [float(rng.uniform(-3.0, 3.0)),
                       float(rng.uniform(-3.0, 3.0)),
                       float(rng.uniform(-math.pi, math.pi)),
                       float(rng.uniform(-1.8, 1.8)),
                       float(rng.uniform(-4.0, 4.0))]
            if case % 4 == 0:
                tv = tw = 0.0
                initial[3:] = [v, w]
            elif case % 4 == 1:
                tv = horizon * float(rng.uniform(1.05, 2.0))
                tw = horizon * float(rng.uniform(0.0, 1.5))
            else:
                tv, tw = horizon * rng.uniform(0.05, 0.95, 2)
            motion = _midpoint_transition(
                initial, v, w, float(tv), float(tw), horizon,
                float(rng.uniform(0.08, 0.5)))
            clipped += motion["t_constant"] >= horizon
            constant += motion["t_constant"] == 0.0
            rows = []
            for obstacle in range(1 + case % 5):
                velocity = (rng.uniform(-0.8, 0.8, 2)
                            if (case + obstacle) % 3 else np.zeros(2))
                time = float(rng.uniform(0.0, horizon))
                anchor = _position_at(motion, v, w, time) - velocity * time
                if (case + obstacle) % 2:
                    anchor += rng.uniform(-3.0, 3.0, 2)
                rows.append(_rectangle(
                    anchor, rng.uniform(0.025, 0.5, 2),
                    float(rng.uniform(-math.pi, math.pi)), velocity,
                    clockwise=(case + obstacle) % 2 == 0))
            with self.subTest(case=case, v=v, w=w, horizon=horizon):
                result = self.assert_equivalent(rows, motion, v, w, horizon)
                collisions += result <= horizon
                clear += result > horizon
        # Avoid a vacuous random corpus dominated entirely by one outcome.
        self.assertGreater(collisions, 300)
        self.assertGreater(clear, 50)
        self.assertEqual(clipped, 240)
        self.assertEqual(constant, 240)

    def test_tangent_contacts_for_all_speed_signs_and_transformed_poses(self):
        for v in (-1.0, 1.0):
            for w in (-1.0, 1.0):
                for angle, position in ((0.0, (0.0, 0.0)),
                                        (0.7, (3.0, -2.0))):
                    cosine, sine = math.cos(angle), math.sin(angle)
                    # The circle just touches the rectangle's near vertical
                    # edge at t=pi/2, with no sign change across that edge.
                    center_local = (1.1 * v, v / w)
                    center = (position[0] + cosine * center_local[0]
                              - sine * center_local[1],
                              position[1] + sine * center_local[0]
                              + cosine * center_local[1])
                    row = _rectangle(center, (0.1, 0.1), angle)
                    motion = {"times": [0.0],
                              "states": [[*position, angle, v, w]],
                              "t_constant": 0.0}
                    with self.subTest(v=v, w=w, angle=angle):
                        result = self.assert_equivalent([row], motion, v, w, 4.0)
                        self.assertAlmostEqual(result, math.pi / 2, places=7)

    def test_horizon_contact_and_immediately_after_contact(self):
        for horizon in (0.0, 0.25, 1.0, 5.0):
            for constant_time in (0.0, 2.0 * horizon + 0.5):
                end = min(constant_time, horizon)
                times = [0.0] if end == 0.0 else [0.0, end]
                states = [[time, 0.0, 0.0, 1.0, 0.0] for time in times]
                motion = {"times": times, "states": states,
                          "t_constant": constant_time}
                for offset in (0.0, 1e-6):
                    row = _rectangle((horizon + offset + 0.1, 0.0), (0.1, 0.2))
                    with self.subTest(horizon=horizon, constant_time=constant_time,
                                      offset=offset):
                        result = self.assert_equivalent([row], motion, 1.0, 0.0,
                                                        horizon)
                        self.assertAlmostEqual(result, horizon if offset == 0.0
                                               else horizon + 1.0, places=12)

    def test_no_obstacles_and_clear_paths_return_horizon_plus_one(self):
        for horizon in (0.0, 0.01, 1.0, 5.0):
            for rows in (None, [], [_rectangle((50.0, 50.0), (0.2, 0.2))]):
                motion = {"times": [0.0],
                          "states": [[0.0, 0.0, 0.0, 1.0, 0.0]],
                          "t_constant": 0.0}
                with self.subTest(horizon=horizon, rows=rows):
                    self.assertEqual(self.assert_equivalent(
                        rows, motion, 1.0, 0.0, horizon), horizon + 1.0)


if __name__ == "__main__":
    unittest.main()

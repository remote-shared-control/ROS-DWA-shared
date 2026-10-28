"""Legacy TTC keeps the horizon finite and distinguishes a clear trajectory.

Execute the real computational definitions without importing ROS.  In
particular, changing the clear sentinel to T + 1 must not extend the interval
in which collisions are accepted.
"""

import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "cal_ctime.py"


def load_solver():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = {"np": np, "math": math, "sin": math.sin, "cos": math.cos}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec"),
         namespace)
    return SimpleNamespace(**namespace)


def rectangle(xmin, xmax, ymin, ymax, vx=0.0, vy=0.0):
    return [1, (xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), vx, vy]


class LegacyTTCSentinelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solver = load_solver()

    def calculate(self, rows, v=1.0, w=0.0, horizon=1.0):
        return self.solver.cal_time(SimpleNamespace(data=rows),
                                    0.0, 0.0, 0.0, v, w, horizon)

    def test_no_obstacles_return_horizon_plus_one_for_every_motion(self):
        for rows in (None, []):
            for v, w in ((1.0, 0.0), (0.0, 0.0), (0.0, 1.0), (1.0, 1.0)):
                for horizon in (0.5, 1.0, 5.0):
                    with self.subTest(rows=rows, v=v, w=w, horizon=horizon):
                        self.assertEqual(self.calculate(rows, v, w, horizon),
                                         horizon + 1.0)

    def test_clear_straight_stationary_and_arc_return_horizon_plus_one(self):
        far_away = [rectangle(10.0, 11.0, 10.0, 11.0)]
        for v, w in ((1.0, 0.0), (0.0, 0.0), (0.0, 1.0), (1.0, 1.0)):
            with self.subTest(v=v, w=w):
                self.assertEqual(self.calculate(far_away, v, w), 2.0)

    def test_initial_collision_stays_zero_for_every_motion(self):
        overlapping = [rectangle(-0.2, 0.2, -0.2, 0.2)]
        for v, w in ((1.0, 0.0), (0.0, 0.0), (0.0, 1.0), (1.0, 1.0)):
            with self.subTest(v=v, w=w):
                self.assertEqual(self.calculate(overlapping, v, w), 0.0)

    def test_straight_collision_still_returns_actual_time(self):
        self.assertAlmostEqual(self.calculate([rectangle(0.7, 0.8, -0.1, 0.1)]),
                               0.7)

    def test_arc_collision_still_returns_actual_time(self):
        hit_time = 0.7
        rows = [rectangle(math.sin(hit_time), math.sin(hit_time) + 0.1, 0.0, 1.0)]
        self.assertAlmostEqual(self.calculate(rows, w=1.0), hit_time, places=4)

    def test_straight_collision_exactly_at_horizon_returns_horizon(self):
        self.assertEqual(self.calculate([rectangle(1.0, 1.1, -0.1, 0.1)]), 1.0)

    def test_arc_collision_exactly_at_horizon_returns_horizon(self):
        rows = [rectangle(math.sin(1.0), math.sin(1.0) + 0.1, 0.0, 1.0)]
        self.assertAlmostEqual(self.calculate(rows, w=1.0), 1.0, places=6)

    def test_later_straight_collision_is_clear_within_original_horizon(self):
        # Cover ordinary intersection and a boundary collinear with motion.
        for ymin in (-0.1, 0.0):
            with self.subTest(ymin=ymin):
                rows = [rectangle(1.5, 1.6, ymin, 0.1)]
                self.assertEqual(self.calculate(rows), 2.0)

    def test_later_arc_collision_is_clear_within_original_horizon(self):
        rows = [rectangle(math.sin(1.3), math.sin(1.3) + 0.1, 0.0, 1.0)]
        self.assertEqual(self.calculate(rows, w=1.0), 2.0)

    def test_multiple_clear_obstacles_share_clear_sentinel(self):
        rows = [rectangle(10.0, 11.0, -0.1, 0.1),
                rectangle(12.0, 13.0, 2.0, 3.0)]
        for w in (0.0, 1.0):
            with self.subTest(w=w):
                self.assertEqual(self.calculate(rows, w=w), 2.0)

    def test_multiple_obstacles_keep_earliest_actual_collision(self):
        straight = [rectangle(10.0, 11.0, 2.0, 3.0),
                    rectangle(0.7, 0.8, -0.1, 0.1),
                    rectangle(0.4, 0.5, -0.1, 0.1)]
        self.assertAlmostEqual(self.calculate(straight), 0.4)
        arcs = [rectangle(10.0, 11.0, 2.0, 3.0),
                rectangle(math.sin(0.7), math.sin(0.7) + 0.1, 0.0, 1.0),
                rectangle(math.sin(0.4), math.sin(0.4) + 0.1, 0.0, 1.0)]
        self.assertAlmostEqual(self.calculate(arcs, w=1.0), 0.4, places=4)

    def test_linear_helper_clear_and_stationary_return_horizon_plus_one(self):
        vertices = rectangle(10.0, 11.0, -0.1, 0.1)[1:5]
        for relative_velocity in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)):
            with self.subTest(relative_velocity=relative_velocity):
                result = self.solver.compute_intersection_time(
                    vertices, relative_velocity, 0.0, 0.0, 1.0)
                self.assertEqual(result, 2.0)

    def test_linear_helper_does_not_accept_hits_between_T_and_T_plus_one(self):
        for ymin in (-0.1, 0.0):
            with self.subTest(ymin=ymin):
                vertices = rectangle(1.5, 1.6, ymin, 0.1)[1:5]
                result = self.solver.compute_intersection_time(
                    vertices, (1.0, 0.0), 0.0, 0.0, 1.0)
                self.assertEqual(result, 2.0)

    def test_arc_helper_clear_returns_horizon_plus_one(self):
        result = self.solver.find_min_tc_fast(
            phi=0.0, omega=1.0, v_py_prime=0.0, p_y_prime=2.0,
            p_x_prime=-2.0, v_px_prime=0.0, q_x_prime=2.0, m=-1.0, T=1.0)
        self.assertEqual(result, 2.0)

    def test_arc_helper_keeps_actual_hit_and_exact_horizon_hit(self):
        for hit_time in (0.7, 1.0):
            with self.subTest(hit_time=hit_time):
                result = self.solver.find_min_tc_fast(
                    phi=0.0, omega=1.0, v_py_prime=0.0,
                    p_y_prime=math.sin(hit_time), p_x_prime=-2.0,
                    v_px_prime=0.0, q_x_prime=2.0, m=-2.0, T=1.0)
                self.assertAlmostEqual(result, hit_time, places=4)


if __name__ == "__main__":
    unittest.main()

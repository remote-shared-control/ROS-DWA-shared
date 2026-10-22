"""Check obstacle prediction endpoints and collision detection without ROS."""

import ast
from bisect import bisect_left
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


SIM_DIR = Path(__file__).resolve().parents[1]


def load_definitions(filename, names, namespace):
    path = SIM_DIR / filename
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    body = [
        node for node in parsed.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)


class ObstaclePredictionTests(unittest.TestCase):
    def setUp(self):
        self.namespace = {
            "np": np, "math": math, "bisect_left": bisect_left,
            "rospy": SimpleNamespace(Rate=lambda hz: None),
            "Twist": lambda: SimpleNamespace(
                linear=SimpleNamespace(x=0.0), angular=SimpleNamespace(z=0.0)
            ),
        }
        load_definitions("dwa_simwaffi_midpoint.py", {
            "Config", "motion_midpoint", "prepare_linear_motion",
            "calc_trajectory_midpoint_vw", "calc_obstacle_cost",
        }, self.namespace)
        load_definitions("detectmove.py", {"GazeboObsReader"}, self.namespace)
        self.config = self.namespace["Config"]()
        self.reader = object.__new__(self.namespace["GazeboObsReader"])
        self.reader.config = self.config
        self.reader.moving_future = None
        self.reader.obs_info = np.empty((0, 6))

    def score(self, trajectory, obstacles=None):
        if obstacles is None:
            obstacles = (np.empty(0), np.empty(0))
        return self.namespace["calc_obstacle_cost"](
            trajectory, obstacles, self.config, self.reader
        )

    def test_collision_at_seventh_point_is_rejected(self):
        self.reader.obs_info = np.array([[2.65, 0.0, -1.0, 0.0, 0.15, 0.0]])
        self.reader.moving_future = self.reader.predict_future(self.reader.obs_info)
        trajectory = self.namespace["calc_trajectory_midpoint_vw"](
            np.zeros(5), 0.0, 0.0, self.config
        )
        self.assertEqual(trajectory.shape[0], 7)
        self.assertEqual(self.reader.moving_future.shape, (1, 7, 2))
        np.testing.assert_allclose(self.reader.moving_future[0, -1], [0.25, 0.0])

        self.assertEqual(self.score(trajectory), float("-inf"))
        # Earlier samples are safe; the seventh point exposes the collision.
        self.reader.moving_future = self.reader.moving_future[:, :-1]
        self.assertTrue(np.isfinite(self.score(trajectory)))

    def test_static_clearance_is_normalized_and_rejects_contact(self):
        trajectory = np.zeros((3, 5))
        self.assertAlmostEqual(self.score(
            trajectory, (np.array([0.71]), np.array([0.0]))
        ), (0.71 - self.config.robot_radius) / (
            self.config.maxdect - self.config.robot_radius
        ))
        self.assertEqual(self.score(
            trajectory, (np.array([self.config.robot_radius_static]), np.array([0.0]))
        ), float("-inf"))
        self.assertEqual(self.score(
            trajectory, (np.array([5.0]), np.array([0.0]))
        ), 1.0)

    def test_moving_clearance_uses_matching_times_and_obstacle_radius(self):
        trajectory = np.zeros((3, 5))
        trajectory[:, 0] = [0.0, 1.0, 2.0]
        self.reader.moving_future = np.array([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
        self.reader.obs_info = np.array([[1.0, 0.0, 1.0, 0.0, 0.15, 0.0]])
        # Paths overlap spatially, but remain one metre apart at each instant.
        self.assertAlmostEqual(self.score(trajectory), (
            0.85 - self.config.robot_radius
        ) / (self.config.maxdect - self.config.robot_radius))

    def test_moving_clearance_uses_each_obstacles_radius(self):
        trajectory = np.zeros((2, 5))
        self.reader.moving_future = np.array([[[1.0, 0.0]] * 2, [[1.2, 0.0]] * 2])
        self.reader.obs_info = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.1, 0.0],
            [1.2, 0.0, 0.0, 0.0, 0.6, 1.0],
        ])
        self.assertAlmostEqual(self.score(trajectory), (
            0.6 - self.config.robot_radius
        ) / (self.config.maxdect - self.config.robot_radius))

    def test_moving_initial_sample_does_not_affect_collision_or_score(self):
        self.config.robot_radius = 0.25
        trajectory = np.zeros((2, 5))
        self.reader.moving_future = np.array([[[0.5, 0.0], [1.0, 0.0]]])
        self.reader.obs_info = np.array([[0.5, 0.0, 0.0, 0.0, 0.25, 0.0]])
        expected = (0.75 - 0.25) / (self.config.maxdect - 0.25)
        # Current contact must neither reject the path nor lower its score.
        self.assertAlmostEqual(self.score(trajectory), expected)
        self.reader.moving_future[0, 0, 0] = 0.85
        self.assertAlmostEqual(self.score(trajectory), expected)
        # The same contact at the first future sample must still be rejected.
        self.reader.moving_future[0, 1, 0] = 0.5
        self.assertEqual(self.score(trajectory), float("-inf"))

    def test_static_initial_sample_is_also_skipped(self):
        trajectory = np.zeros((2, 5))
        trajectory[1, 0] = 3.0
        self.assertEqual(self.score(
            trajectory, (np.array([0.0]), np.array([0.0]))
        ), 1.0)

    def test_static_and_moving_distances_share_one_minimum(self):
        trajectory = np.zeros((2, 5))
        obstacles = (np.array([0.7]), np.array([0.0]))
        self.reader.obs_info = np.array([[1.0, 0.0, 0.0, 0.0, 0.15, 0.0]])
        self.reader.moving_future = np.array([[[1.0, 0.0], [1.0, 0.0]]])
        denominator = self.config.maxdect - self.config.robot_radius
        self.assertAlmostEqual(self.score(trajectory, obstacles), (
            0.7 - self.config.robot_radius
        ) / denominator)
        self.reader.moving_future[0, 1, 0] = 0.75
        self.assertAlmostEqual(self.score(trajectory, obstacles), (
            0.6 - self.config.robot_radius
        ) / denominator)

    def test_missing_or_longer_moving_prediction_is_handled(self):
        trajectory = np.zeros((2, 5))
        for future in (None, np.empty((0, 1, 2)), np.empty((1, 0, 2)),
                       np.zeros((1, 1, 2))):
            with self.subTest(future=future):
                self.reader.moving_future = future
                self.assertEqual(self.score(trajectory), 1.0)
                # Missing future samples must not skip static obstacles.
                self.assertEqual(self.score(trajectory, (
                    np.array([self.config.robot_radius_static]), np.array([0.0])
                )), float("-inf"))
        # The extra obstacle sample lies beyond the robot prediction horizon.
        self.reader.moving_future = np.array([[[3.0, 0.0], [3.0, 0.0], [0.0, 0.0]]])
        self.reader.obs_info = np.array([[3.0, 0.0, 0.0, 0.0, 0.15, 0.0]])
        self.assertEqual(self.score(trajectory), 1.0)
        self.assertEqual(self.score(trajectory[:1]), 1.0)

    def test_legacy_controller_keeps_uniform_time_alignment(self):
        namespace = self.namespace.copy()
        load_definitions("dwa_simwaffi.py", {
            "Config", "motion", "calc_trajectory", "calc_obstacle_cost",
        }, namespace)
        config = namespace["Config"]()
        self.reader.config = config
        self.reader.obs_info = np.array([[10.0, 0.0, 1.0, 0.0, 0.15, 0.0]])
        self.reader.moving_future = self.reader.predict_future(self.reader.obs_info)
        trajectory = namespace["calc_trajectory"](
            np.array([0.0, 0.0, 0.0, 1.0, 0.0]), 1.0, 0.0, config
        )
        # Preserve the legacy prediction grid: 13 samples from 0 to 4.8 s.
        self.assertEqual(self.reader.moving_future.shape, (1, 13, 2))
        np.testing.assert_allclose(
            self.reader.moving_future[0, :, 0] - 10.0, trajectory[:13, 0],
            atol=1e-12,
        )
        self.assertTrue(np.isfinite(namespace["calc_obstacle_cost"](
            trajectory, (np.empty(0), np.empty(0)), config, self.reader
        )))

    def test_dttc_skips_initial_contact_and_keeps_future_collision_times(self):
        namespace = self.namespace.copy()
        load_definitions("dwa_simwaffi_new_midpoint_dttc.py", {
            "Config", "calc_obstacle_ttc",
        }, namespace)
        config = namespace["Config"]()
        score = namespace["calc_obstacle_ttc"]
        trajectory = np.zeros((3, 5))
        future = np.array([[[0.0, 0.0], [3.0, 0.0], [3.0, 0.0]]])

        # Only current contact exists: there is no unsafe future sample.
        self.assertEqual(score(trajectory, future, config), config.predict_time + 1.0)
        self.assertEqual(score(
            trajectory[:1], future[:, :1], config
        ), config.predict_time + 1.0)

        future[0, 1] = [0.0, 0.0]
        self.assertEqual(score(trajectory, future, config), config.trajectory_times[1])
        future[0, 1] = [3.0, 0.0]
        future[0, 2] = [0.0, 0.0]
        self.assertEqual(score(trajectory, future, config), config.trajectory_times[2])

    def test_dttc_controller_detects_collision_at_its_own_endpoint(self):
        namespace = self.namespace.copy()
        load_definitions("dwa_simwaffi_new_midpoint_dttc.py", {
            "Config", "motion_midpoint", "prepare_linear_motion",
            "calc_trajectory_midpoint_vw", "calc_obstacle_ttc",
        }, namespace)
        config = namespace["Config"]()
        self.reader.config = config
        times = config.trajectory_times
        self.assertEqual(times[-1], config.predict_time)
        safe_distance = config.robot_radius + config.cube_half_size
        # Approach at 1 m/s: enter the collision radius between the final
        # two samples, regardless of the configured prediction horizon.
        overlap = min(safe_distance / 2.0, (times[-1] - times[-2]) / 2.0)
        initial_x = config.predict_time + safe_distance - overlap
        obs_info = np.array([[initial_x, 0.0, -1.0, 0.0, config.obs_radius, 0.0]])
        future = self.reader.predict_future(obs_info)
        trajectory = namespace["calc_trajectory_midpoint_vw"](
            np.zeros(5), 0.0, 0.0, config
        )
        self.assertEqual(trajectory.shape[0], len(times))
        self.assertEqual(future.shape, (1, len(times), 2))
        np.testing.assert_allclose(future[0, :, 0], initial_x - np.asarray(times))
        # All earlier samples are safe; the final sample must expose the hit.
        self.assertEqual(namespace["calc_obstacle_ttc"](
            trajectory[:-1], future[:, :-1], config
        ), config.predict_time + 1.0)
        self.assertEqual(namespace["calc_obstacle_ttc"](
            trajectory, future, config
        ), config.predict_time)


if __name__ == "__main__":
    unittest.main()

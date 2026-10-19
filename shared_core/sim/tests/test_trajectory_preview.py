"""Check trajectory selection and display reuse without a ROS installation.

Only computational definitions are loaded; ROS startup and subscribers are
excluded. Run with Python and NumPy: python -B -m unittest discover -s
shared_core/sim/tests -v
"""

import ast
from bisect import bisect_left
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


SIM_DIR = Path(__file__).resolve().parents[1]
CONTROLLERS = sorted(
    path for path in SIM_DIR.glob("*.py")
    if "robot_trajectory.set_trajectory(best_traj)" in path.read_text(encoding="utf-8")
)
DEFINITIONS = {
    "Config", "TrajectoryPreview", "motion", "motion_midpoint",
    "prepare_linear_motion", "calc_trajectory", "calc_trajectory_midpoint",
    "calc_trajectory_midpoint_vw",
    "calc_final_input", "line", "h_line", "show_trajectory",
    "show_trajectory_human",
}


def twist():
    return SimpleNamespace(
        linear=SimpleNamespace(x=0.0), angular=SimpleNamespace(z=0.0)
    )


def load_controller(source, filename):
    """Load the real controller functions with lightweight ROS value stubs."""
    parsed = ast.parse(source, filename=filename)
    body = ast.parse("from __future__ import annotations").body
    body.extend(
        node for node in parsed.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name in DEFINITIONS
    )
    namespace = {
        "np": np, "math": math, "bisect_left": bisect_left,
        "Twist": twist, "Point": SimpleNamespace,
        "rospy": SimpleNamespace(Rate=lambda hz: None),
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), filename, "exec"), namespace)
    config = namespace["Config"]()
    if not hasattr(config, "human"):
        config.human = twist()
    config.human.linear.x = 0.3
    config.human.angular.z = 0.2
    namespace.update(
        human=config.human,
        # Controlled costs make the middle candidate the unique winner.
        calc_to_human_cost=lambda *args: abs(
            (args[1] if len(args) > 2 else args[0]) - config.human.angular.z
        ),
        calc_obstacle_cost=lambda *args: 0.8,
        cal_time=lambda *args: 100.0,
        calc_obstacle_ttc=lambda *args: 100.0,
        angle_robot=0.0,
        marker=SimpleNamespace(points=[]),
        h_marker=SimpleNamespace(points=[]),
    )
    if "TrajectoryPreview" in namespace:
        namespace["robot_trajectory"] = namespace["TrajectoryPreview"]()
        namespace["human_trajectory"] = namespace["TrajectoryPreview"]()
    else:
        namespace.update(
            list_x=[], list_y=[], line_num=0,
            list_x_human=[], list_y_human=[], line_num_human=0,
        )
    return namespace, config


def trajectory_function_name(namespace):
    return (
        "calc_trajectory_midpoint_vw"
        if "calc_trajectory_midpoint_vw" in namespace
        else "calc_trajectory"
    )


def run_selection(namespace, config, vs, ws):
    state = np.array([1.0, -2.0, 0.7, -0.1, -0.2])
    result = namespace["calc_final_input"](
        state, [0.0, 0.0], vs, ws, config, None,
        SimpleNamespace(moving_future=None)
    )
    return state, result


class TrajectoryPreviewTests(unittest.TestCase):
    def controllers(self):
        self.assertEqual(len(CONTROLLERS), 10)
        for path in CONTROLLERS:
            namespace, config = load_controller(path.read_text(encoding="utf-8"), str(path))
            yield path, namespace, config

    def test_winner_reuses_candidate_without_another_robot_prediction(self):
        for path, namespace, config in self.controllers():
            with self.subTest(controller=path.name):
                name = trajectory_function_name(namespace)
                calculate = namespace[name]
                calls = []

                def record(*args):
                    traj = calculate(*args)
                    calls.append(traj)
                    return traj

                namespace[name] = record
                state, result = run_selection(namespace, config, [0.1, 0.3, 0.5], [-0.4, 0.2, 0.6])
                self.assertEqual(result, [0.3, 0.2])
                self.assertEqual(len(calls), 10)  # Nine candidates and one human prediction.
                robot = namespace["robot_trajectory"]
                self.assertIs(robot.trajectory, calls[4])
                self.assertTrue(np.shares_memory(robot.x_points, calls[4]))
                self.assertTrue(np.shares_memory(robot.y_points, calls[4]))
                self.assertEqual(robot.point_count, len(calls[4]))
                np.testing.assert_array_equal(state, [1.0, -2.0, 0.7, -0.1, -0.2])

    def test_equal_cost_keeps_the_last_tied_candidate(self):
        for path, namespace, config in self.controllers():
            with self.subTest(controller=path.name):
                namespace["calc_to_human_cost"] = lambda *args: 0.0
                _, result = run_selection(namespace, config, [0.3], [-0.4, 0.2, 0.6])
                self.assertEqual(result, [0.3, 0.6])

    def test_rejected_or_empty_candidates_clear_previous_preview(self):
        for path, namespace, config in self.controllers():
            for vs, ws in (([0.3], [0.2]), ([], [0.2]), ([0.3], [])):
                with self.subTest(controller=path.name, vs=vs, ws=ws):
                    robot = namespace["robot_trajectory"]
                    robot.set_trajectory(np.ones((3, 5)))
                    namespace["calc_obstacle_cost"] = lambda *args: float("-inf")
                    _, result = run_selection(namespace, config, vs, ws)
                    self.assertEqual(result, [0.0, 0.2])
                    self.assertIsNone(robot.trajectory)
                    self.assertEqual(robot.point_count, 0)
                    self.assertEqual(len(robot.y_points), 0)

    def test_human_uses_the_same_motion_model_and_horizon(self):
        for path, namespace, config in self.controllers():
            with self.subTest(controller=path.name):
                name = trajectory_function_name(namespace)
                state, _ = run_selection(namespace, config, [0.3], [0.2])
                expected = namespace[name](state, 0.3, 0.2, config)
                human = namespace["human_trajectory"]
                np.testing.assert_array_equal(human.trajectory, expected)
                np.testing.assert_array_equal(human.x_points, expected[:, 0])
                np.testing.assert_array_equal(human.y_points, expected[:, 1])
                self.assertFalse(hasattr(config, "showpredict_time"))
                self.assertFalse(hasattr(config, "showdt"))
                if hasattr(config, "trajectory_times"):
                    horizon = getattr(config, "predict_time_static", config.predict_time)
                    self.assertEqual(config.trajectory_times[-1], horizon)

    def test_straight_trajectory_keeps_distinct_states_and_the_initial_pose(self):
        for path, namespace, config in self.controllers():
            with self.subTest(controller=path.name):
                name = trajectory_function_name(namespace)
                initial = np.array([1.0, -2.0, 0.0, 0.3, 0.0])
                traj = namespace[name](initial, 0.3, 0.0, config)
                times = np.asarray(config.trajectory_times) if hasattr(config, "trajectory_times") else np.arange(len(traj)) * config.dt
                np.testing.assert_allclose(traj[:, 0], 1.0 + 0.3 * times)
                np.testing.assert_array_equal(traj[:, 1], np.full(len(traj), -2.0))
                np.testing.assert_array_equal(traj[0], initial)
                np.testing.assert_array_equal(initial, [1.0, -2.0, 0.0, 0.3, 0.0])

    def test_midpoint_vw_integrates_switches_without_extra_output_rows(self):
        for path, namespace, config in self.controllers():
            if "calc_trajectory_midpoint_vw" not in namespace:
                continue
            times = np.asarray(config.trajectory_times)
            cached_times = config.trajectory_times
            cached_steps = config.trajectory_steps
            initial = np.array([1.0, -2.0, 0.7, -0.1, -0.2])
            horizon = times[-1]
            switches = (
                (0.1, 0.2), (0.2, 0.1), (0.2, 0.2),
                (0.0, 0.0), (config.dt, config.dt),
                (np.nextafter(config.dt, 0.0), config.dt),
                (config.dt, np.nextafter(config.dt, np.inf)),
                (horizon, horizon),
                (horizon + config.dt, horizon + config.dt),
            )
            for t_v, t_w in switches:
                for direction in (-1, 1):
                    with self.subTest(controller=path.name, t_v=t_v, t_w=t_w, direction=direction):
                        v = initial[3] + direction * config.max_accel * t_v
                        w = initial[4] - direction * config.max_dyawrate * t_w
                        plan = namespace["prepare_linear_motion"](initial[3], v, config)
                        traj = namespace["calc_trajectory_midpoint_vw"](
                            initial, v, w, config, plan
                        )
                        self.assertEqual(traj.shape, (len(times), 5))
                        ramp_time = abs(w - initial[4]) / config.max_dyawrate
                        alpha = np.sign(w - initial[4]) * config.max_dyawrate
                        ramp = np.minimum(times, ramp_time)
                        expected_theta = (
                            initial[2] + initial[4] * ramp
                            + 0.5 * alpha * ramp ** 2
                            + w * np.maximum(times - ramp_time, 0.0)
                        )
                        np.testing.assert_allclose(
                            traj[:, 2], expected_theta, rtol=1e-12, atol=1e-12
                        )
                        np.testing.assert_array_equal(traj[0], initial)
                        np.testing.assert_array_equal(initial, [1.0, -2.0, 0.7, -0.1, -0.2])
                        self.assertIs(config.trajectory_times, cached_times)
                        self.assertIs(config.trajectory_steps, cached_steps)

    def test_midpoint_vw_integrates_deceleration_and_reverse_motion(self):
        for path, namespace, config in self.controllers():
            if "calc_trajectory_midpoint_vw" not in namespace:
                continue
            with self.subTest(controller=path.name):
                initial = np.array([1.0, -2.0, 0.0, 0.2, 0.0])
                v = -0.1
                times = np.asarray(config.trajectory_times)
                t_v = abs(v - initial[3]) / config.max_accel
                ramp = np.minimum(times, t_v)
                expected_x = (
                    initial[0] + initial[3] * ramp
                    - 0.5 * config.max_accel * ramp ** 2
                    + v * np.maximum(times - t_v, 0.0)
                )
                traj = namespace["calc_trajectory_midpoint_vw"](
                    initial, v, 0.0, config
                )
                np.testing.assert_allclose(traj[:, 0], expected_x, rtol=1e-12, atol=1e-12)
                np.testing.assert_array_equal(traj[:, 1], np.full(len(times), -2.0))
                np.testing.assert_array_equal(traj[:, 2], np.zeros(len(times)))

    def test_midpoint_selection_and_preview_never_call_legacy_function(self):
        for path, namespace, config in self.controllers():
            if "calc_trajectory_midpoint_vw" not in namespace:
                continue
            with self.subTest(controller=path.name):
                def reject_legacy(*args, **kwargs):
                    raise AssertionError("Legacy midpoint function must not be called")

                namespace["calc_trajectory_midpoint"] = reject_legacy
                state, result = run_selection(
                    namespace, config, [0.1, 0.3, 0.5], [-0.4, 0.2, 0.6]
                )
                self.assertEqual(result, [0.3, 0.2])
                namespace["robot_trajectory"].update(state, 0.3, 0.2, config)
                namespace["human_trajectory"].update(state, 0.3, 0.2, config)

    def test_markers_accept_coordinate_views_and_empty_preview(self):
        for path, namespace, config in self.controllers():
            with self.subTest(controller=path.name):
                run_selection(namespace, config, [0.3], [0.2])
                robot = namespace["robot_trajectory"]
                marker = namespace["marker"]
                if path.name == "direct.py":
                    namespace["line"](robot.point_count)
                    namespace["h_line"](namespace["human_trajectory"].point_count)
                else:
                    namespace["line"](robot.point_count, marker, robot.x_points, robot.y_points)
                self.assertEqual(len(marker.points), robot.point_count)
                self.assertEqual(marker.points[0].x, robot.x_points[0])
                robot.set_trajectory(None)
                if path.name == "direct.py":
                    namespace["line"](robot.point_count)
                else:
                    namespace["line"](robot.point_count, marker, robot.x_points, robot.y_points)
                self.assertEqual(marker.points, [])


if __name__ == "__main__":
    unittest.main()

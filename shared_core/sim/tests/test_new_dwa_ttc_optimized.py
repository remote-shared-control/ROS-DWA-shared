"""Keep every feasible new-DWA candidate on the constant equation TTC path.

Load the real controller definitions and solver import without ROS startup.
The independent continuous-equation oracle is shared with the hybrid tests;
the controller's original forward-Euler static trajectory is checked separately.
"""

import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from test_midpoint_ttc_optimized import load_module, rectangle, twist


SIM_DIR = Path(__file__).resolve().parents[1]


def load_controller(solver):
    filename = SIM_DIR / "dwa_simwaffi_new.py"
    parsed = ast.parse(filename.read_text(encoding="utf-8"))
    names = {"Config", "motion", "calc_trajectory", "calc_final_input",
             "calc_obstacle_cost", "calc_to_human_cost"}
    namespace = {
        "np": np, "math": math, "Twist": twist,
        "rospy": SimpleNamespace(Rate=lambda hz: None),
        "robot_trajectory": mock.Mock(), "human_trajectory": mock.Mock(),
    }
    body = ast.parse("from __future__ import annotations").body
    for node in parsed.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            body.append(node)
        elif isinstance(node, ast.ImportFrom) and node.module == "cal_ctime_midpoint":
            for alias in node.names:
                namespace[alias.asname or alias.name] = getattr(solver, alias.name)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(filename), "exec"), namespace)
    return namespace


class NewDWASelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solver = load_module("new_dwa_ttc_selection_solver",
                                 SIM_DIR / "cal_ctime_midpoint.py")
        cls.oracle = load_module("new_dwa_ttc_equation_oracle",
                                 Path(__file__).with_name("test_hybrid_ttc_equivalence.py"))

    def setUp(self):
        self.controller = load_controller(self.solver)
        self.config = self.controller["Config"]()
        self.config.to_human_cost_gain = 1.0
        self.config.speed_cost_gain = 0.0
        self.config.obs_cost_gain = 1.0
        self.config.x, self.config.y, self.config.th = 2.0, -3.0, 0.4
        self.config.human.angular.z = 0.15
        self.config.human.linear.x = 0.3
        # Deliberately differ from config pose: TTC must use config pose.
        self.initial = np.array([9.0, 8.0, -0.7, -0.3, 0.6])
        self.cube = SimpleNamespace(data=[rectangle(20.0, 21.0, 20.0, 21.0)],
                                    robot_radius_cal=0.21,
                                    robot_radius_transition=0.225)
        self.robot = self.controller["robot_trajectory"]
        self.human = self.controller["human_trajectory"]

    def select(self, vs, ws, ttc=None, scores=None, static=None, real_ttc=False):
        self.ttc_calls, self.ttc_values = [], []
        self.candidate_trajectories, self.events = [], []
        calculate = self.controller["calc_trajectory"]

        def record_trajectory(*args):
            trajectory = calculate(*args)
            self.candidate_trajectories.append(trajectory)
            return trajectory

        def record_ttc(cube, x, y, theta, v, w, horizon):
            self.events.append("ttc")
            self.ttc_calls.append((cube, x, y, theta, v, w, horizon))
            if real_ttc:
                value = self.solver.cal_time_constant(cube, x, y, theta, v, w, horizon)
            else:
                value = horizon + 1.0 if ttc is None else ttc(v, w, horizon)
            self.ttc_values.append(value)
            return value

        def record_static(*args):
            self.events.append("static")
            return 0.0 if static is None else static(*args)

        self.prepare = mock.Mock(wraps=self.solver.prepare_obstacles)
        self.yaw_cost = mock.Mock(side_effect=lambda w, config: 1.0 - (
            0.6 if scores is None else scores[w]))
        with mock.patch.dict(self.controller, {
                "prepare_obstacles": self.prepare,
                "cal_time": record_ttc,
                "calc_trajectory": record_trajectory,
                "calc_obstacle_cost": record_static,
                "calc_to_human_cost": self.yaw_cost,
        }):
            return self.controller["calc_final_input"](
                self.initial, [0.7, -0.7], vs, ws, self.config,
                (np.empty(0), np.empty(0)), self.cube)

    def test_actual_import_uses_only_constant_speed_solver(self):
        self.assertIs(self.controller["cal_time"], self.solver.cal_time_constant)
        self.assertIs(self.controller["prepare_obstacles"], self.solver.prepare_obstacles)

    def test_all_feasible_candidates_get_ttc_including_low_or_negative_scores(self):
        result = self.select([0.2, 0.4], [-0.4, 0.2, 0.6],
                             scores={-0.4: 0.8, 0.2: 0.6, 0.6: -0.2})
        self.assertEqual(result, [0.4, -0.4])
        self.assertEqual([(call[4], call[5]) for call in self.ttc_calls],
                         [(v, w) for v in (0.2, 0.4) for w in (-0.4, 0.2, 0.6)])
        self.assertEqual(len(self.candidate_trajectories), 6)
        self.assertIs(self.robot.set_trajectory.call_args.args[0],
                      self.candidate_trajectories[3])
        self.human.update.assert_called_once()
        np.testing.assert_array_equal(self.initial, [9.0, 8.0, -0.7, -0.3, 0.6])
        self.assertEqual(self.yaw_cost.call_count, 3)

    def test_static_collision_skips_ttc_and_keeps_static_first_order(self):
        costs = iter((0.0, -math.inf, math.inf, 0.0))
        self.select([0.3], [-0.4, 0.1, 0.2, 0.6],
                    static=lambda *args: next(costs))
        self.assertEqual([call[5] for call in self.ttc_calls], [-0.4, 0.6])
        self.assertEqual(self.events,
                         ["static", "ttc", "static", "static", "static", "ttc"])

    def test_cycle_prepares_one_snapshot_and_freezes_ttc_pose(self):
        initial_pose = (self.config.x, self.config.y, self.config.th)

        def update_callback(v, w, horizon):
            self.config.x, self.config.y, self.config.th = 90.0, 80.0, 2.0
            self.cube.data[0][1] = (200.0, 200.0)
            return horizon + 1.0

        self.select([0.2], [-0.4, 0.2, 0.6], ttc=update_callback)
        self.prepare.assert_called_once_with(self.cube)
        snapshot = self.ttc_calls[0][0]
        self.assertTrue(all(call[0] is snapshot for call in self.ttc_calls))
        self.assertTrue(all(call[1:4] == initial_pose for call in self.ttc_calls))
        self.assertEqual(snapshot.obstacles[0].bounds, (20.0, 21.0, 20.0, 21.0))

    def test_threshold_contact_is_rejected_and_clear_sentinel_is_accepted(self):
        self.config.movetimemin = 2.0
        result = self.select(
            [0.3], [-0.4, 0.2, 0.6],
            ttc=lambda v, w, horizon: {-0.4: horizon - 0.1,
                                       0.2: horizon, 0.6: horizon + 1.0}[w])
        self.assertEqual(result, [0.3, 0.6])
        self.assertEqual(len(self.ttc_calls), 3)
        self.assertTrue(all(call[-1] == 2.0 for call in self.ttc_calls))

    def test_horizon_uses_movetimemin_independently_of_predict_time(self):
        self.config.predict_time, self.config.movetimemin = 1.0, 3.0
        result = self.select([0.3], [0.2], ttc=lambda v, w, horizon: 2.0)
        self.assertEqual(result, [0.0, self.config.human.angular.z])
        self.assertEqual(self.ttc_calls[0][-1], 3.0)
        self.robot.set_trajectory.assert_called_once_with(None)

    def test_all_blocked_or_negative_scores_keep_original_fallback(self):
        for ttc, score in ((lambda v, w, horizon: 0.0, 0.6),
                           (lambda v, w, horizon: horizon + 1.0, -0.2)):
            with self.subTest(score=score):
                self.robot.reset_mock()
                result = self.select([0.2, 0.4], [-0.4, 0.2], ttc=ttc,
                                     scores={-0.4: score, 0.2: score})
                self.assertEqual(result, [0.0, 0.15])
                self.assertEqual(len(self.ttc_calls), 4)
                self.robot.set_trajectory.assert_called_once_with(None)

    def test_equal_scores_keep_last_safe_candidate(self):
        result = self.select([0.2, 0.4], [-0.4, 0.2, 0.6])
        self.assertEqual(result, [0.4, 0.6])
        self.assertEqual(len(self.ttc_calls), 6)

    def test_original_motion_keeps_forward_euler_heading_and_velocity_fields(self):
        original = np.array([1.2, -0.7, 0.3, -0.4, 0.6])
        state = original.copy()
        result = self.controller["motion"](state, [0.8, -0.5], 0.4)
        self.assertIs(result, state)
        np.testing.assert_allclose(result, [
            original[0] + 0.8 * math.cos(original[2]) * 0.4,
            original[1] + 0.8 * math.sin(original[2]) * 0.4,
            original[2] - 0.5 * 0.4, original[3], original[4],
        ], rtol=0.0, atol=1e-15)

    def test_original_static_trajectory_samples_and_initial_state_are_kept(self):
        self.config.predict_time_static, self.config.dt = 0.8, 0.4
        original = self.initial.copy()
        actual = self.controller["calc_trajectory"](self.initial, 0.8, -0.5, self.config)
        expected = [original.copy()]
        for _ in range(3):  # Original loop includes time == predict_time_static.
            step = expected[-1].copy()
            step[0] += 0.8 * math.cos(step[2]) * 0.4
            step[1] += 0.8 * math.sin(step[2]) * 0.4
            step[2] -= 0.5 * 0.4
            expected.append(step)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)
        np.testing.assert_array_equal(self.initial, original)

    def test_static_cost_nan_does_not_hide_another_points_contact(self):
        trajectory = np.zeros((2, 5))
        calculate = self.controller["calc_obstacle_cost"]
        for x in (0.0, self.config.robot_radius):
            with self.subTest(contact_x=x):
                self.assertEqual(calculate(
                    trajectory, (np.array([np.nan, x]), np.zeros(2)), self.config),
                    -math.inf)
        self.assertTrue(math.isnan(calculate(
            trajectory, (np.array([np.nan, 2.0]), np.zeros(2)), self.config)))

    def test_real_equation_solver_rejects_early_arcs_for_both_turns(self):
        self.config.x = self.config.y = self.config.th = 0.0
        for w in (-1.0, 1.0):
            ymin, ymax = (-0.1, 0.0) if w < 0 else (0.0, 0.1)
            self.cube.data = [rectangle(0.05, 0.08, ymin, ymax)]
            with self.subTest(w=w):
                self.assertEqual(self.select([1.0], [w], real_ttc=True), [0.0, 0.15])
                self.assertAlmostEqual(self.ttc_values[0], math.asin(0.05), places=8)

    def test_real_equation_solver_rejects_tangency_for_both_turns(self):
        self.config.x = self.config.y = self.config.th = 0.0
        for w in (-1.0, 1.0):
            ymin, ymax = sorted((w * 0.9, w * 1.1))
            self.cube.data = [rectangle(1.0, 1.2, ymin, ymax)]
            with self.subTest(w=w):
                self.assertEqual(self.select([1.0], [w], real_ttc=True), [0.0, 0.15])
                self.assertAlmostEqual(self.ttc_values[0], math.pi / 2.0, places=8)

    def test_real_horizon_contact_and_empty_or_clear_paths_have_distinct_outputs(self):
        self.config.x = self.config.y = self.config.th = 0.0
        self.config.movetimemin = 1.0
        for rows, expected, safe in (([rectangle(1.0, 1.2, -0.1, 0.1)], 1.0, False),
                                     ([rectangle(1.000001, 1.2, -0.1, 0.1)], 2.0, True),
                                     ([], 2.0, True), (None, 2.0, True)):
            self.cube.data = rows
            with self.subTest(rows=rows):
                self.assertEqual(self.select([1.0], [0.0], real_ttc=True),
                                 [1.0, 0.0] if safe else [0.0, 0.15])
                self.assertEqual(self.ttc_values, [expected])

    def test_equation_geometry_keeps_current_reader_radius_without_transition_padding(self):
        filename = SIM_DIR / "cal_ctime.py"
        tree = ast.parse(filename.read_text(encoding="utf-8"))
        body = [node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == "CubeReader"]
        namespace = {"np": np, "math": math, "ModelStates": object,
                     "rospy": SimpleNamespace(Subscriber=mock.Mock())}
        exec(compile(ast.Module(body=body, type_ignores=[]), str(filename), "exec"), namespace)
        reader = namespace["CubeReader"](self.config)
        self.assertEqual(reader.robot_radius_cal, 0.21)
        pose = SimpleNamespace(position=SimpleNamespace(x=0.6, y=0.38),
                               orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0))
        reader.data = [[1, *reader.get_expanded_vertices_2d(pose), 0.0, 0.0]]
        self.cube = reader
        self.config.x = self.config.y = self.config.th = 0.0
        # The expanded rectangle starts at y=0.01. An added 0.015 m transition
        # pad would produce a false collision with the straight path y=0.
        self.assertEqual(self.select([1.0], [0.0], real_ttc=True), [1.0, 0.0])
        self.assertEqual(self.ttc_values, [self.config.movetimemin + 1.0])

    def test_every_real_candidate_matches_independent_equation_oracle(self):
        rng = np.random.default_rng(20261028)
        vs, ws = [-0.7, 0.0, 0.8], [-1.2, -1e-9, 0.0, 1e-9, 1.1]
        collisions = clear = 0
        for scene in range(8):
            self.config.x, self.config.y = rng.uniform(-1.0, 1.0, 2)
            self.config.th = float(rng.uniform(-math.pi, math.pi))
            self.config.movetimemin = float(rng.uniform(0.5, 3.5))
            pose = (self.config.x, self.config.y, self.config.th)
            rows = []
            for index in range(3):
                anchor = np.asarray(pose[:2]) + rng.uniform(-2.0, 2.0, 2)
                rows.append(self.oracle._rectangle(
                    anchor, rng.uniform(0.03, 0.4, 2),
                    float(rng.uniform(-math.pi, math.pi)), rng.uniform(-0.6, 0.6, 2),
                    clockwise=(scene + index) % 2 == 0))
            self.cube.data = rows
            expected_u = [0.0, self.config.human.angular.z]
            expected_times = []
            for v in vs:
                for w in ws:
                    motion = {"times": [0.0], "states": [[*pose, v, w]],
                              "t_constant": 0.0}
                    expected = self.oracle._reference_cal_time(
                        self.cube, motion, v, w, self.config.movetimemin)
                    expected_times.append(expected)
                    if expected > self.config.movetimemin:
                        clear += 1
                        expected_u = [v, w]
                    else:
                        collisions += 1
            with self.subTest(scene=scene):
                self.assertEqual(self.select(vs, ws, real_ttc=True), expected_u)
                self.assertEqual(len(self.ttc_calls), len(vs) * len(ws))
                np.testing.assert_allclose(self.ttc_values, expected_times,
                                           rtol=0.0, atol=2e-8)
                self.assertEqual([value <= self.config.movetimemin for value in self.ttc_values],
                                 [value <= self.config.movetimemin for value in expected_times])
        self.assertGreater(collisions, 10)
        self.assertGreater(clear, 10)


if __name__ == "__main__":
    unittest.main()

"""Safety and selection regressions for the constant-speed midpoint TTC path.

The static trajectory still includes acceleration, while this controller's TTC
model applies each candidate v/w immediately at config.x/y/th. Load its real
computational definitions without requiring ROS, and compare the new wrapper
against the independent, frozen continuous-equation oracle.
"""

import ast
from bisect import bisect_left
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np


SIM_DIR = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def twist():
    return SimpleNamespace(linear=SimpleNamespace(x=0.0),
                           angular=SimpleNamespace(z=0.0))


def load_controller(solver):
    """Honor the controller's actual solver imports, with ROS setup omitted."""
    filename = SIM_DIR / "dwa_simwaffi_new_midpoint.py"
    parsed = ast.parse(filename.read_text(encoding="utf-8"))
    names = {"Config", "motion_midpoint", "prepare_linear_motion",
             "calc_trajectory_midpoint_vw", "calc_final_input",
             "calc_obstacle_cost", "calc_to_human_cost"}
    namespace = {
        "np": np, "math": math, "bisect_left": bisect_left,
        "Twist": twist, "rospy": SimpleNamespace(Rate=lambda hz: None),
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


def rectangle(xmin, xmax, ymin, ymax, vx=0.0, vy=0.0):
    return [1, (xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), vx, vy]


class MidpointConstantSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solver = load_module("midpoint_constant_selection_solver",
                                 SIM_DIR / "cal_ctime_midpoint.py")

    def setUp(self):
        self.controller = load_controller(self.solver)
        self.config = self.controller["Config"]()
        self.config.to_human_cost_gain = 1.0
        self.config.speed_cost_gain = 0.0
        self.config.obs_cost_gain = 1.0
        self.config.x, self.config.y, self.config.th = 2.0, -3.0, 0.4
        self.config.human.angular.z = 0.15
        self.config.human.linear.x = 0.3
        self.initial = np.array([9.0, 8.0, -0.7, -0.3, 0.6])
        self.cube = SimpleNamespace(data=[rectangle(20.0, 21.0, 20.0, 21.0)])
        self.robot = self.controller["robot_trajectory"]
        self.human = self.controller["human_trajectory"]

    def select(self, vs, ws, ttc=None, scores=None, static=None):
        self.ttc_calls = []
        self.candidate_trajectories = []
        self.events = []
        calculate = self.controller["calc_trajectory_midpoint_vw"]

        def record_trajectory(*args):
            trajectory = calculate(*args)
            self.candidate_trajectories.append(trajectory)
            return trajectory

        def record_ttc(cube, x, y, theta, v, w, horizon):
            self.events.append("ttc")
            self.ttc_calls.append((cube, x, y, theta, v, w, horizon))
            return horizon + 1.0 if ttc is None else ttc(v, w, horizon)

        def record_static(*args):
            self.events.append("static")
            return 0.0 if static is None else static(*args)

        self.prepare = mock.Mock(wraps=self.solver.prepare_obstacles)
        with mock.patch.dict(self.controller, {
                "prepare_obstacles": self.prepare,
                "cal_time": record_ttc,
                "calc_trajectory_midpoint_vw": record_trajectory,
                "calc_obstacle_cost": record_static,
                "calc_to_human_cost": lambda w, config: 1.0 - (
                    0.6 if scores is None else scores[w]),
        }):
            return self.controller["calc_final_input"](
                self.initial, [0.7, -0.7], vs, ws,
                self.config, (np.empty(0), np.empty(0)), self.cube)

    def test_every_candidate_gets_ttc_even_after_a_better_score_or_negative_score(self):
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

    def test_static_collision_skips_ttc_but_other_candidates_are_all_checked(self):
        costs = iter((0.0, -math.inf, math.inf, 0.0))
        self.select([0.3], [-0.4, 0.1, 0.2, 0.6],
                    static=lambda *args: next(costs))
        self.assertEqual([call[5] for call in self.ttc_calls], [-0.4, 0.6])
        self.assertEqual(self.events,
                         ["static", "ttc", "static", "static", "static", "ttc"])

    def test_cycle_shares_one_snapshot_and_freezes_config_pose(self):
        initial_pose = (self.config.x, self.config.y, self.config.th)

        def update_pose(v, w, horizon):
            self.config.x, self.config.y, self.config.th = 90.0, 80.0, 2.0
            self.cube.data[0][1] = (200.0, 200.0)
            return horizon + 1.0

        self.select([0.2], [-0.4, 0.2, 0.6], ttc=update_pose)
        self.prepare.assert_called_once_with(self.cube)
        snapshot = self.ttc_calls[0][0]
        self.assertTrue(all(call[0] is snapshot for call in self.ttc_calls))
        self.assertTrue(all(call[1:4] == initial_pose for call in self.ttc_calls))
        self.assertEqual(snapshot.obstacles[0].bounds, (20.0, 21.0, 20.0, 21.0))

    def test_ttc_equal_to_safety_threshold_is_rejected(self):
        threshold = self.config.movetimemin
        result = self.select(
            [0.3], [-0.4, 0.2, 0.6],
            ttc=lambda v, w, horizon: {-0.4: threshold - 0.1,
                                       0.2: threshold, 0.6: horizon + 1.0}[w])
        self.assertEqual(result, [0.3, 0.6])
        self.assertEqual(len(self.ttc_calls), 3)

    def test_prediction_shorter_than_safety_threshold_still_checks_the_whole_threshold(self):
        self.config.predict_time, self.config.movetimemin = 1.0, 3.0
        result = self.select([0.3], [0.2], ttc=lambda v, w, horizon: 2.0)
        self.assertEqual(result, [0.0, self.config.human.angular.z])
        self.assertEqual(self.ttc_calls[0][-1], 3.0)
        self.robot.set_trajectory.assert_called_once_with(None)

    def test_all_blocked_keep_fallback_and_clear_preview(self):
        result = self.select([0.2, 0.4], [-0.4, 0.2],
                             ttc=lambda v, w, horizon: 0.0)
        self.assertEqual(result, [0.0, 0.15])
        self.assertEqual(len(self.ttc_calls), 4)
        self.robot.set_trajectory.assert_called_once_with(None)

    def test_equal_scores_keep_last_safe_candidate(self):
        result = self.select([0.2, 0.4], [-0.4, 0.2, 0.6])
        self.assertEqual(result, [0.4, 0.6])
        self.assertEqual(len(self.ttc_calls), 6)

    def test_original_normalized_ttc_score_is_kept_when_predict_time_is_longer(self):
        self.config.predict_time, self.config.movetimemin = 6.0, 2.0
        self.config.move_obs_gain = 2.0
        # Dynamic scores are 0.5 and 1.5; the lower human score wins overall.
        result = self.select([0.3], [-0.4, 0.2],
                             scores={-0.4: 0.8, 0.2: 0.2},
                             ttc=lambda v, w, horizon: 3.0 if w < 0.0 else 5.0)
        self.assertEqual(result, [0.3, 0.2])
        self.assertTrue(all(call[-1] == 6.0 for call in self.ttc_calls))

    def test_clear_sentinel_does_not_raise_score_above_horizon_contact(self):
        self.config.predict_time, self.config.movetimemin = 6.0, 2.0
        result = self.select([0.3], [-0.4, 0.2],
                             ttc=lambda v, w, horizon: horizon + (1.0 if w < 0 else 0.0))
        self.assertEqual(result, [0.3, 0.2])

    def test_short_prediction_does_not_add_a_dynamic_score(self):
        self.config.predict_time, self.config.movetimemin = 1.0, 3.0
        result = self.select([0.3], [-0.4, 0.2],
                             ttc=lambda v, w, horizon: horizon + (1.0 if w < 0 else 0.5))
        self.assertEqual(result, [0.3, 0.2])

    def test_static_nan_does_not_hide_a_contact_in_another_point(self):
        trajectory = np.zeros((2, 5))
        calculate = self.controller["calc_obstacle_cost"]
        for x in (0.0, self.config.robot_radius):
            with self.subTest(contact_x=x):
                self.assertEqual(calculate(
                    trajectory, (np.array([np.nan, x]), np.zeros(2)), self.config),
                    -math.inf)
        self.assertTrue(math.isnan(calculate(
            trajectory, (np.array([np.nan, 2.0]), np.zeros(2)), self.config)))


class MidpointConstantCollisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solver = load_module("midpoint_constant_collision_solver",
                                 SIM_DIR / "cal_ctime_midpoint.py")
        cls.oracle = load_module("midpoint_constant_oracle",
                                 Path(__file__).with_name("test_hybrid_ttc_equivalence.py"))

    def solve(self, rows, v=1.0, w=0.0, horizon=3.0, pose=(0.0, 0.0, 0.0)):
        cube = SimpleNamespace(data=rows, robot_radius_cal=0.21,
                               robot_radius_transition=0.225)
        result = self.solver.cal_time_constant(cube, *pose, v, w, horizon)
        cached = self.solver.cal_time_constant(
            self.solver.prepare_obstacles(cube), *pose, v, w, horizon)
        self.assertEqual(result, cached)
        return result

    def test_early_arc_collision_is_found_before_point_one_seconds_for_both_turns(self):
        for w in (-1.0, 1.0):
            ymin, ymax = (-0.1, 0.0) if w < 0 else (0.0, 0.1)
            with self.subTest(w=w):
                self.assertAlmostEqual(self.solve([rectangle(0.05, 0.08, ymin, ymax)],
                                                  w=w), math.asin(0.05), places=8)

    def test_arc_tangency_without_sign_change_is_detected_for_both_turns(self):
        for w in (-1.0, 1.0):
            ymin, ymax = sorted((w * 0.9, w * 1.1))
            with self.subTest(w=w):
                self.assertAlmostEqual(self.solve([rectangle(1.0, 1.2, ymin, ymax)],
                                                  w=w), math.pi / 2.0, places=8)

    def test_horizon_contact_is_different_from_contact_just_after_horizon(self):
        for horizon in (0.0, 0.25, 1.0, 5.0):
            for offset in (0.0, 1e-6):
                with self.subTest(horizon=horizon, offset=offset):
                    result = self.solve([rectangle(horizon + offset,
                                                   horizon + offset + 0.2, -0.1, 0.1)],
                                        horizon=horizon)
                    self.assertAlmostEqual(result, horizon if offset == 0.0
                                           else horizon + 1.0, places=10)

    def test_no_obstacle_or_clear_path_returns_horizon_plus_one(self):
        for rows in (None, [], [rectangle(30.0, 31.0, 30.0, 31.0)]):
            for horizon in (0.0, 0.25, 5.0):
                with self.subTest(rows=rows, horizon=horizon):
                    self.assertEqual(self.solve(rows, horizon=horizon), horizon + 1.0)

    def test_rotating_in_place_can_be_hit_by_a_translating_obstacle(self):
        for w in (-1.0, 0.0, 1.0):
            with self.subTest(w=w):
                self.assertAlmostEqual(self.solve([rectangle(1.0, 1.2, -0.1, 0.1,
                                                             vx=-1.0)], v=0.0, w=w), 1.0)

    def test_stationary_boundary_contact_is_immediate(self):
        self.assertEqual(self.solve([rectangle(0.0, 0.2, -0.1, 0.1)],
                                    v=0.0, w=1.0), 0.0)

    def test_constant_model_uses_equation_radius_without_transition_padding(self):
        # Vertices already include 0.21 m. Applying the extra 0.015 m would
        # falsely hit this path, which passes 0.01 m below the rectangle.
        self.assertEqual(self.solve([rectangle(0.24, 0.96, 0.01, 0.73)]), 4.0)
        self.assertEqual(self.solve([rectangle(-0.36, 0.36, 0.01, 0.73)],
                                    v=0.0, w=1.0), 4.0)

    def test_seeded_constant_ttc_matches_independent_continuous_equation_oracle(self):
        rng = np.random.default_rng(20261028)
        collisions = clear = 0
        for case in range(240):
            horizon = float(rng.uniform(0.05, 6.0))
            pose = tuple(float(value) for value in (*rng.uniform(-3.0, 3.0, 2),
                                                    rng.uniform(-math.pi, math.pi)))
            v = 0.0 if case % 12 == 0 else float(rng.uniform(-1.8, 1.8))
            w = {0: 0.0, 1: 1e-9, 2: -1e-9}.get(case % 12,
                                                              float(rng.uniform(-4.0, 4.0)))
            motion = {"times": [0.0], "states": [[*pose, v, w]], "t_constant": 0.0}
            rows = []
            for obstacle in range(1 + case % 4):
                time = float(rng.uniform(0.0, horizon))
                velocity = rng.uniform(-0.8, 0.8, 2)
                anchor = self.oracle._position_at(motion, v, w, time) - velocity * time
                if case % 3 == 0 or (case + obstacle) % 2:
                    anchor += rng.uniform(-3.0, 3.0, 2)
                rows.append(self.oracle._rectangle(
                    anchor, rng.uniform(0.025, 0.5, 2),
                    float(rng.uniform(-math.pi, math.pi)), velocity,
                    clockwise=(case + obstacle) % 2 == 0))
            expected = self.oracle._reference_cal_time(
                SimpleNamespace(data=rows), motion, v, w, horizon)
            with self.subTest(case=case, v=v, w=w, horizon=horizon):
                actual = self.solve(rows, v, w, horizon, pose)
                self.assertEqual(actual <= horizon, expected <= horizon)
                self.assertEqual(actual < horizon, expected < horizon)
                self.assertAlmostEqual(actual, expected, delta=2e-8)
            collisions += expected <= horizon
            clear += expected > horizon
        self.assertGreater(collisions, 80)
        self.assertGreater(clear, 15)


if __name__ == "__main__":
    unittest.main()

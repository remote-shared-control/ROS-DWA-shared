"""Check midpoint-to-constant TTC transitions without ROS dependencies."""

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


def load_motion(filename):
    parsed = ast.parse((SIM_DIR / filename).read_text(encoding="utf-8"))
    names = {"motion_midpoint", "prepare_linear_motion", "calc_trajectory_midpoint_vw",
             "calc_final_input", "calc_obstacle_cost"}
    future_imports = [node for node in parsed.body
                      if isinstance(node, ast.ImportFrom) and node.module == "__future__"]
    body = [node for node in parsed.body
            if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"np": np, "math": math, "bisect_left": bisect_left}
    exec(compile(ast.Module(body=future_imports + body, type_ignores=[]),
                 filename, "exec"), namespace)
    if filename == "dwa_simwaffi_new_midpoint_hybrid.py":
        namespace["prepare_obstacles"] = load_solver().prepare_obstacles
    return namespace


def config_for(static_horizon=0.8, ttc_horizon=3.0):
    times = tuple(np.append(np.arange(0.0, static_horizon, 0.4), static_horizon))
    return SimpleNamespace(
        dt=0.4, max_accel=1.0, max_dyawrate=1.0,
        predict_time_static=static_horizon, predict_time=7.0, movetimemin=ttc_horizon,
        trajectory_times=times,
        trajectory_steps=tuple((b, b - a, (a + b) / 2, True)
                               for a, b in zip(times, times[1:])),
    )


def rectangle(xmin, xmax, ymin, ymax, vx=0.0, vy=0.0):
    return [1, (xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), vx, vy]


def transition(times, xy, constant_time=None, theta=0.0):
    return {
        "times": times,
        "states": [[x, y, theta, 0.0, 0.0] for x, y in xy],
        "t_constant": times[-1] if constant_time is None else constant_time,
    }


def load_solver():
    spec = importlib.util.spec_from_file_location(
        "hybrid_ttc_under_test", SIM_DIR / "cal_ctime_midpoint.py")
    solver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solver)
    return solver


class HybridTrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hybrid = load_motion("dwa_simwaffi_new_midpoint_hybrid.py")
        cls.original = load_motion("dwa_simwaffi_new_midpoint.py")
        cls.solver = load_solver()

    def test_static_cost_keeps_contact_detection_when_another_point_is_nan(self):
        config = SimpleNamespace(robot_radius=0.21, maxdect=1.2)
        trajectory = np.zeros((2, 5))
        calculate = self.hybrid["calc_obstacle_cost"]
        for x in (0.0, config.robot_radius):
            self.assertEqual(calculate(
                trajectory, (np.array([np.nan, x]), np.zeros(2)), config), -math.inf)
        self.assertAlmostEqual(calculate(
            trajectory, (np.array([1.0]), np.zeros(1)), config),
            (1.0 - config.robot_radius) / (config.maxdect - config.robot_radius))
        self.assertTrue(math.isnan(calculate(
            trajectory, (np.array([np.nan, 2.0]), np.zeros(2)), config)))

    def test_reuses_switch_times_and_preserves_original_output(self):
        initial = np.array([1.0, -2.0, 0.7, -0.1, 0.2])
        for tv, tw in ((0.1, 0.2), (0.2, 0.1), (0.2, 0.2),
                       (0.0, 0.0), (0.8, 0.3), (1.1, 0.6)):
            with self.subTest(tv=tv, tw=tw):
                config = config_for()
                v, w = initial[3] + tv, initial[4] - tw
                plan = self.hybrid["prepare_linear_motion"](initial[3], v, config)
                data = {}
                # The cached linear candidate must be accepted without recomputation.
                with mock.patch.dict(self.hybrid, prepare_linear_motion=mock.Mock(
                        side_effect=AssertionError("linear candidate recomputed"))):
                    result = self.hybrid["calc_trajectory_midpoint_vw"](
                        initial, v, w, config, plan, data)
                expected = self.original["calc_trajectory_midpoint_vw"](
                    initial, v, w, config)
                np.testing.assert_array_equal(result, expected)
                self.assertAlmostEqual(data["t_constant"], max(tv, tw))
                self.assertAlmostEqual(data["times"][-1], max(tv, tw))
                self.assertTrue(all(b > a for a, b in zip(
                    data["times"], data["times"][1:])))
                for switch in (tv, tw):
                    self.assertTrue(any(abs(t - switch) < 1e-12 for t in data["times"]))
                np.testing.assert_allclose(data["states"][-1][3:], [v, w], atol=1e-12)
                np.testing.assert_array_equal(initial, [1.0, -2.0, 0.7, -0.1, 0.2])

    def test_transition_pose_uses_the_existing_midpoint_vw_rule(self):
        data = {}
        initial = np.zeros(5)
        self.hybrid["calc_trajectory_midpoint_vw"](
            initial, 0.3, 0.6, config_for(), None, data)
        # Integration boundaries: 0, tv=0.3, sample=0.4, tw=0.6.
        # Midpoint position uses the initial yaw rate of each interval.
        expected_x = (0.045 + 0.03 * math.cos(0.06)
                      + 0.06 * math.cos(0.12))
        expected_y = 0.03 * math.sin(0.06) + 0.06 * math.sin(0.12)
        np.testing.assert_allclose(data["times"], [0.0, 0.3, 0.4, 0.6])
        np.testing.assert_allclose(data["states"][-1],
                                   [expected_x, expected_y, 0.18, 0.3, 0.6], atol=1e-12)

    def test_short_ttc_horizon_clips_a_partial_midpoint_interval(self):
        config = config_for(ttc_horizon=0.18)
        initial = np.zeros(5)
        data = {}
        actual = self.hybrid["calc_trajectory_midpoint_vw"](
            initial, 1.0, 0.8, config, None, data)
        np.testing.assert_array_equal(actual, self.original["calc_trajectory_midpoint_vw"](
            initial, 1.0, 0.8, config))
        np.testing.assert_allclose(data["times"], [0.0, 0.18])
        np.testing.assert_allclose(data["states"][-1], [0.0162, 0, 0.0162, 0.18, 0.18])
        self.assertEqual(data["t_constant"], 1.0)

    def test_candidate_evaluation_uses_movetimemin_as_ttc_horizon(self):
        config = config_for(ttc_horizon=0.18)
        config.human = SimpleNamespace(
            linear=SimpleNamespace(x=1.0), angular=SimpleNamespace(z=0.8))
        config.to_human_cost_gain = config.speed_cost_gain = config.obs_cost_gain = 1.0
        config.move_obs_gain = 1.0
        config.min_speed, config.max_speed = -1.0, 1.0
        solver = mock.Mock(wraps=self.solver.cal_time)
        with mock.patch.dict(self.hybrid, {
                "cal_time": solver,
                "prepare_obstacles": self.solver.prepare_obstacles,
                "calc_to_human_cost": mock.Mock(return_value=0.0),
                "calc_obstacle_cost": mock.Mock(return_value=1.0),
                "robot_trajectory": mock.Mock(), "human_trajectory": mock.Mock()}):
            result = self.hybrid["calc_final_input"](
                np.zeros(5), [0.0, 0.0], [1.0], [0.8], config, None,
                SimpleNamespace(data=[rectangle(20, 21, 20, 21)]))
        self.assertEqual(result, [1.0, 0.8])
        solver.assert_called_once()
        self.assertEqual(solver.call_args.args[-1], config.movetimemin)
        self.assertEqual(solver.call_args.args[1]["times"][-1], config.movetimemin)

    def test_transition_after_static_horizon_continues_acceleration(self):
        config = config_for(static_horizon=0.8)
        data = {}
        output = self.hybrid["calc_trajectory_midpoint_vw"](
            np.zeros(5), 1.1, 0.0, config, None, data)
        self.assertEqual(output.shape, (3, 5))
        self.assertAlmostEqual(data["times"][-1], 1.1)
        self.assertAlmostEqual(data["states"][-1][0], 0.5 * 1.1 ** 2)
        self.assertAlmostEqual(output[-1, 0], 0.5 * 0.8 ** 2)

    def test_already_constant_keeps_only_the_initial_ttc_pose(self):
        initial = np.array([2.0, 3.0, 0.4, 0.2, -0.3])
        data = {}
        self.hybrid["calc_trajectory_midpoint_vw"](
            initial, 0.2, -0.3, config_for(), None, data)
        self.assertEqual(data["times"], [0.0])
        self.assertEqual(data["t_constant"], 0.0)
        np.testing.assert_array_equal(data["states"], [initial])

    def test_roundoff_at_switch_and_base_boundaries_keeps_strictly_ordered_times(self):
        cases = [
            (np.array([0.0, 0.0, 0.0, -1.0, -1.0]), -1.0, -0.88, 3.2),
            (np.array([0.0, 0.0, 0.0, -1.0, -1.0]), -1.0, -0.98, 3.2),
        ]
        for boundary in (0.4, 0.8):
            for cutoff in (np.nextafter(boundary, 0.0), boundary,
                           np.nextafter(boundary, math.inf)):
                cases.append((np.zeros(5), cutoff, cutoff, 1.0))
        for initial, v, w, acceleration in cases:
            with self.subTest(initial=initial.tolist(), v=v, w=w):
                config = config_for()
                config.max_dyawrate = acceleration
                data = {}
                self.hybrid["calc_trajectory_midpoint_vw"](
                    initial, v, w, config, None, data)
                self.assertTrue(all(b > a for a, b in zip(
                    data["times"], data["times"][1:])), data["times"])
                self.assertAlmostEqual(data["times"][-1], data["t_constant"])
                self.assertEqual(self.solver.cal_time(
                    SimpleNamespace(data=[rectangle(20, 21, 20, 21)]),
                    data, v, w, config.movetimemin), config.movetimemin + 1.0)


class HybridSelectionTests(unittest.TestCase):
    def setUp(self):
        self.hybrid = load_motion("dwa_simwaffi_new_midpoint_hybrid.py")
        self.config = config_for()
        self.config.human = SimpleNamespace(
            linear=SimpleNamespace(x=0.0), angular=SimpleNamespace(z=0.15))
        self.config.to_human_cost_gain = 1.0
        self.config.speed_cost_gain = self.config.obs_cost_gain = 0.0
        self.config.min_speed, self.config.max_speed = -1.0, 1.0
        self.cube = SimpleNamespace(data=[rectangle(20, 21, 20, 21)])
        self.robot, self.human = mock.Mock(), mock.Mock()
        self.candidates = []
        self.ttc_calls = []

    def select(self, scores, collision_times):
        calculate = self.hybrid["calc_trajectory_midpoint_vw"]

        def record_trajectory(*args):
            traj = calculate(*args)
            self.candidates.append(traj)
            return traj

        def record_ttc(cube, motion, v, w, horizon):
            self.ttc_calls.append((cube, w))
            self.assertEqual(horizon, self.config.movetimemin)
            return collision_times.get(w, horizon + 1.0)

        original_prepare = self.hybrid["prepare_obstacles"]
        self.prepare = mock.Mock(wraps=original_prepare)
        self.fallback = [0.7, -0.7]
        with mock.patch.dict(self.hybrid, {
                "prepare_obstacles": self.prepare,
                "calc_trajectory_midpoint_vw": record_trajectory,
                "cal_time": record_ttc,
                "calc_to_human_cost": lambda w, config: 1.0 - scores[w],
                "calc_obstacle_cost": lambda *args: 1.0,
                "robot_trajectory": self.robot,
                "human_trajectory": self.human}):
            return self.hybrid["calc_final_input"](
                np.zeros(5), self.fallback, [0.3], list(scores),
                self.config, None, self.cube)

    def test_every_candidate_runs_ttc_including_lower_scores(self):
        result = self.select({0.1: 0.2, 0.2: 0.9, 0.3: 0.6}, {})
        self.assertEqual(result, [0.3, 0.2])
        self.assertEqual([w for _, w in self.ttc_calls], [0.1, 0.2, 0.3])
        self.assertEqual(len(self.candidates), 3)
        self.assertIs(self.robot.set_trajectory.call_args.args[0], self.candidates[1])
        self.human.update.assert_called_once()

    def test_equal_scores_keep_the_last_safe_candidate(self):
        result = self.select({0.1: 0.6, 0.2: 0.6, 0.3: 0.6}, {})
        self.assertEqual(result, [0.3, 0.3])
        self.assertEqual([w for _, w in self.ttc_calls], [0.1, 0.2, 0.3])
        self.assertIs(self.robot.set_trajectory.call_args.args[0], self.candidates[2])

    def test_blocked_best_score_checks_next_feasible_candidate(self):
        result = self.select(
            {0.1: 0.2, 0.2: 0.9, 0.3: 0.6}, {0.2: self.config.movetimemin})
        self.assertEqual(result, [0.3, 0.3])
        self.assertEqual([w for _, w in self.ttc_calls], [0.1, 0.2, 0.3])
        self.assertIs(self.robot.set_trajectory.call_args.args[0], self.candidates[2])

    def test_all_blocked_preserve_fallback_and_share_one_prepared_snapshot(self):
        result = self.select(
            {0.1: 0.2, 0.2: 0.9, 0.3: 0.6},
            dict.fromkeys((0.1, 0.2, 0.3), self.config.movetimemin))
        self.assertIs(result, self.fallback)
        self.assertEqual(result, [0.0, self.config.human.angular.z])
        self.assertEqual([w for _, w in self.ttc_calls], [0.1, 0.2, 0.3])
        self.prepare.assert_called_once_with(self.cube)
        prepared = self.ttc_calls[0][0]
        self.assertIsNot(prepared, self.cube)
        self.assertTrue(all(cube is prepared for cube, _ in self.ttc_calls))
        self.robot.set_trajectory.assert_called_once_with(None)

    def test_zero_score_can_win_and_negative_scores_still_run_ttc(self):
        result = self.select({0.1: -0.2, 0.2: 0.0, 0.3: -0.1}, {})
        self.assertEqual(result, [0.3, 0.2])
        self.assertEqual([w for _, w in self.ttc_calls], [0.1, 0.2, 0.3])


class HybridCollisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solver = load_solver()

    def solve(self, rows, motion, v=1.0, w=0.0, horizon=5.0):
        return self.solver.cal_time(SimpleNamespace(data=rows), motion, v, w, horizon)

    def test_swept_ramp_hit_returns_before_constant_solver_and_selects_earliest(self):
        motion = transition([0, 1], [(0, 0), (2, 0)])
        rows = [rectangle(1.5, 1.6, -0.1, 0.1), rectangle(0.5, 0.6, -0.1, 0.1)]
        with mock.patch.object(self.solver, "_constant_collision_time",
                               side_effect=AssertionError("constant phase evaluated")):
            self.assertAlmostEqual(self.solve(rows, motion), 0.25)

    def test_swept_ramp_accounts_for_moving_obstacle(self):
        motion = transition([0, 1], [(0, 0), (2, 0)])
        self.assertAlmostEqual(self.solve([rectangle(1, 1.1, -0.1, 0.1, vx=-1)],
                                         motion), 1 / 3)

    def test_initial_overlap_and_boundary_contact_are_immediate(self):
        for xmin in (-0.1, 0.0):
            with self.subTest(xmin=xmin):
                self.assertEqual(self.solve([rectangle(xmin, 0.1, -0.1, 0.1)],
                                            transition([0], [(0, 0)]), v=0), 0.0)

    def test_collision_at_transition_boundary_does_not_call_constant_solver(self):
        with mock.patch.object(self.solver, "_constant_collision_time",
                               side_effect=AssertionError("constant phase evaluated")):
            self.assertAlmostEqual(self.solve([rectangle(1, 1.1, -0.1, 0.1)],
                                             transition([0, 1], [(0, 0), (1, 0)])), 1.0)

    def test_constant_phase_shifts_obstacle_and_returns_global_time(self):
        motion = transition([0, 1], [(0, 0), (0.5, 0)])
        self.assertAlmostEqual(self.solve([rectangle(2, 2.2, -0.1, 0.1, vx=0.25)],
                                         motion), 10 / 3, places=8)

    def test_horizon_during_ramp_returns_sentinel_without_constant_solver(self):
        motion = transition([0, 0.25], [(0, 0), (0.03, 0)], constant_time=1.0)
        with mock.patch.object(self.solver, "_constant_collision_time",
                               side_effect=AssertionError("constant phase evaluated")):
            self.assertEqual(self.solve([rectangle(2, 3, -1, 1)], motion,
                                        horizon=0.25), 1.25)

    def test_ramp_contact_at_horizon_differs_from_contact_after_horizon(self):
        motion = transition([0, 0.25], [(0, 0), (0.25, 0)], constant_time=1.0)
        self.assertEqual(self.solve([rectangle(0.25, 0.3, -0.1, 0.1)], motion,
                                    horizon=0.25), 0.25)
        self.assertEqual(self.solve([rectangle(0.2501, 0.3, -0.1, 0.1)], motion,
                                    horizon=0.25), 1.25)

    def test_constant_contact_at_horizon_differs_from_contact_after_horizon(self):
        motion = transition([0, 1], [(0, 0), (0.5, 0)])
        self.assertEqual(self.solve([rectangle(4.5, 4.6, -0.1, 0.1)], motion), 5.0)
        self.assertEqual(self.solve([rectangle(4.5001, 4.6, -0.1, 0.1)], motion), 6.0)

    def test_clear_ramp_and_constant_phase_return_horizon_plus_one(self):
        motion = transition([0, 1], [(0, 0), (0.5, 0)])
        for w in (0.0, 1.0):
            with self.subTest(w=w):
                self.assertEqual(self.solve([rectangle(20, 21, 20, 21)], motion,
                                            w=w), 6.0)

    def test_zero_horizon_distinguishes_initial_contact_from_clear(self):
        motion = transition([0], [(0, 0)], constant_time=1.0)
        self.assertEqual(self.solve([rectangle(0, 1, -1, 1)], motion, horizon=0), 0.0)
        self.assertEqual(self.solve([rectangle(1, 2, -1, 1)], motion, horizon=0), 1.0)
        self.assertEqual(self.solve([], motion, horizon=0), 1.0)

    def test_stationary_robot_can_be_hit_by_moving_obstacle(self):
        self.assertAlmostEqual(self.solve([rectangle(1, 1.2, -0.1, 0.1, vx=-1)],
                                         transition([0], [(0, 0)]), v=0, w=1), 1.0)

    def test_reverse_straight_motion(self):
        self.assertAlmostEqual(self.solve([rectangle(-2.2, -2, -0.1, 0.1)],
                                         transition([0], [(0, 0)]), v=-1), 2.0)

    def test_circular_contact_for_both_turn_and_travel_directions(self):
        for v in (-1, 1):
            for w in (-1, 1):
                with self.subTest(v=v, w=w):
                    xs, ys = sorted((v * 0.5, v * 0.7)), sorted((v * w * 0.1, v * w * 0.3))
                    row = rectangle(*xs, *ys)
                    self.assertAlmostEqual(self.solve([row], transition([0], [(0, 0)]),
                                                     v=v, w=w), math.pi / 6, places=7)

    def test_circular_tangency_is_detected_without_a_sign_change(self):
        self.assertAlmostEqual(self.solve([rectangle(1, 1.2, 0.9, 1.1)],
                                         transition([0], [(0, 0)]), w=1), math.pi / 2, places=7)

    def test_circular_contact_at_horizon_is_not_the_clear_sentinel(self):
        horizon = math.pi / 2
        motion = transition([0], [(0, 0)])
        row = rectangle(1, 1.2, 0.9, 1.1)
        self.assertAlmostEqual(self.solve([row], motion, w=1, horizon=horizon),
                               horizon, places=7)
        self.assertEqual(self.solve([row], motion, w=1, horizon=horizon - 0.01),
                         horizon - 0.01 + 1.0)

    def test_rotated_obstacle_and_nonzero_initial_pose(self):
        theta, position = 0.7, np.array([3.0, -2.0])
        rotation = np.array([[math.cos(theta), -math.sin(theta)],
                             [math.sin(theta), math.cos(theta)]])
        row = rectangle(0.5, 0.7, 0.1, 0.3)
        row[1:5] = [rotation @ point + position for point in row[1:5]]
        self.assertAlmostEqual(self.solve([row], transition([0], [position], theta=theta),
                                         w=1), math.pi / 6, places=7)

    def test_circular_motion_against_a_translating_edge(self):
        contact_time, vx = 0.8, 0.2
        xmin = math.sin(contact_time) - vx * contact_time
        row = rectangle(xmin, xmin + 0.2, 0, 0.8, vx=vx)
        self.assertAlmostEqual(self.solve([row], transition([0], [(0, 0)]), w=1),
                               contact_time, places=7)

    def test_no_obstacles_or_no_intersection_returns_horizon_plus_one(self):
        motion = transition([0], [(0, 0)])
        for rows in (None, [], [rectangle(20, 21, 20, 21)]):
            with self.subTest(rows=rows):
                self.assertEqual(self.solve(rows, motion, w=1), 6.0)

    def test_prepared_snapshot_matches_raw_for_multiple_candidates(self):
        cube = SimpleNamespace(data=[rectangle(0.8, 1.0, 0.1, 0.4, vx=-0.1),
                                     rectangle(-2.2, -2, -0.1, 0.1),
                                     rectangle(2, 2.2, -0.1, 0.1, vx=-0.25)])
        prepared = self.solver.prepare_obstacles(cube)
        self.assertIs(self.solver.prepare_obstacles(prepared), prepared)
        for motion in (transition([0], [(0, 0)]),
                       transition([0, 1], [(0, 0), (0.5, 0)])):
            for v in (-1.0, 0.0, 1.0):
                for w in (-0.7, 0.0, 0.7):
                    with self.subTest(v=v, w=w, times=motion["times"]):
                        self.assertEqual(
                            self.solver.cal_time(prepared, motion, v, w, 5.0),
                            self.solver.cal_time(cube, motion, v, w, 5.0))

    def test_prepared_snapshot_copies_rows_and_next_cycle_sees_updates(self):
        row = rectangle(1.0, 1.2, -0.1, 0.1)
        row[1:5] = [np.array(point) for point in row[1:5]]
        cube = SimpleNamespace(data=[row])
        motion = transition([0], [(0, 0)])
        prepared = self.solver.prepare_obstacles(cube)
        self.assertEqual(self.solver.cal_time(prepared, motion, 1.0, 0.0, 5.0), 1.0)

        for point in cube.data[0][1:5]:
            point[0] += 1.0
        cube.data[0][5] = 0.5
        next_cycle = self.solver.prepare_obstacles(cube)
        self.assertIsNot(next_cycle, prepared)
        self.assertEqual(self.solver.cal_time(prepared, motion, 1.0, 0.0, 5.0), 1.0)
        self.assertEqual(self.solver.cal_time(next_cycle, motion, 1.0, 0.0, 5.0), 4.0)
        self.assertEqual(self.solver.cal_time(cube, motion, 1.0, 0.0, 5.0), 4.0)


if __name__ == "__main__":
    unittest.main()

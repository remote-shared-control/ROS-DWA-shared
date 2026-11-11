"""Check the two crossing streams and their advertised velocities without ROS."""

import ast
import bisect
import math
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


DIRECTORY = Path(__file__).resolve().parents[1]


class Pose:
    def __init__(self):
        self.position = SimpleNamespace(x=0.0, y=0.0, z=0.0)
        self.orientation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0)


class Twist:
    def __init__(self):
        self.linear = SimpleNamespace(x=0.0, y=0.0, z=0.0)
        self.angular = SimpleNamespace(x=0.0, y=0.0, z=0.0)


class ModelState:
    def __init__(self):
        self.model_name = ""
        self.reference_frame = ""
        self.pose = Pose()
        self.twist = Twist()


def load_scene_functions():
    path = DIRECTORY / "dynamic_obstacle1.py"
    tree = ast.parse(path.read_text())
    config = next(node for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "CONFIG"
                          for target in node.targets))
    namespace = {
        "random": random, "bisect": bisect, "math": math,
        "Pose": Pose, "Twist": Twist, "ModelState": ModelState,
        "ModelStates": lambda: SimpleNamespace(name=[], pose=[], twist=[]),
    }
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name != "main"]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace, ast.literal_eval(config.value)


class CrossingSceneTests(unittest.TestCase):
    def setUp(self):
        self.ns, self.cfg = load_scene_functions()

    def plan(self, **changes):
        cfg = dict(self.cfg, **changes)
        tracks, starts = self.ns["plan_scene"](cfg)
        return cfg, tracks, starts

    def test_default_locations_and_requested_speed_range(self):
        self.assertEqual(self.cfg["right_crossing_x"], 3.0)
        self.assertEqual(self.cfg["left_crossing_x"], 6.0)
        self.assertEqual((self.cfg["min_speed"], self.cfg["max_speed"]), (0.6, 0.8))

    def test_zero_one_even_and_odd_quantities_have_stable_balanced_streams(self):
        for count in (0, 1, 7, 8, 40):
            with self.subTest(count=count):
                cfg, tracks, starts = self.plan(num_obstacles=count)
                self.assertEqual((len(tracks), len(starts)), (count, count))
                right = left = 0
                for index, (track, times) in enumerate(zip(tracks, starts)):
                    x, _y, vx, vy = self.ns["state_at"](track, times, 0.0, cfg)
                    self.assertEqual(vx, 0.0)
                    if index % 2 == 0:  # obstacle_1, obstacle_3, ...
                        self.assertEqual(x, 3.0)
                        self.assertLess(vy, 0.0)
                        right += 1
                    else:
                        self.assertEqual(x, 6.0)
                        self.assertGreater(vy, 0.0)
                        left += 1
                self.assertEqual((right, left), ((count + 1) // 2, count // 2))

    def test_each_model_keeps_its_speed_and_direction_after_reentry(self):
        cfg, tracks, starts = self.plan(num_obstacles=12, duration=70.0)
        self.assertTrue(any(len(track) > 1 for track in tracks))
        for index, (track, times) in enumerate(zip(tracks, starts)):
            with self.subTest(model=index + 1):
                self.assertEqual(times[0], 0.0)
                self.assertEqual(times, sorted(times))
                self.assertTrue(all(a < b for a, b in zip(times, times[1:])))
                self.assertTrue(all(t <= cfg["duration"] for t in times))
                original_vy = track[0][4]
                self.assertGreaterEqual(abs(original_vy), 0.6)
                self.assertLessEqual(abs(original_vy), 0.8)
                for start_t, x, _y0, vx, vy in track:
                    self.assertEqual(x, 3.0 if index % 2 == 0 else 6.0)
                    self.assertEqual((vx, vy), (0.0, original_vy))

    def test_same_seed_repeats_and_another_seed_changes_the_scene(self):
        _, tracks_a, starts_a = self.plan(seed=42)
        _, tracks_b, starts_b = self.plan(seed=42)
        _, tracks_c, starts_c = self.plan(seed=43)
        self.assertEqual((tracks_a, starts_a), (tracks_b, starts_b))
        self.assertNotEqual((tracks_a, starts_a), (tracks_c, starts_c))

    def test_planning_does_not_change_global_random_state(self):
        previous = random.getstate()
        try:
            random.seed(123)
            expected = random.Random(123).random()
            self.plan()
            self.assertEqual(random.random(), expected)
        finally:
            random.setstate(previous)

    def test_sparse_reverse_and_dense_sampling_produce_the_same_positions(self):
        cfg, tracks, starts = self.plan(num_obstacles=5)
        samples = [i * 0.05 for i in range(1401)]
        evaluate = self.ns["state_at"]
        for index, (track, times) in enumerate(zip(tracks, starts)):
            with self.subTest(model=index + 1):
                dense = {t: evaluate(track, times, t, cfg) for t in samples}
                for t in reversed(samples[::31]):
                    self.assertEqual(evaluate(track, times, t, cfg), dense[t])
                for x, y, vx, vy in dense.values():
                    self.assertEqual(x, 3.0 if index % 2 == 0 else 6.0)
                    self.assertGreaterEqual(y, cfg["y_min"] - 1e-9)
                    self.assertLessEqual(y, cfg["y_max"] + 1e-9)
                    self.assertEqual(vx, 0.0)
                    self.assertTrue(0.6 <= abs(vy) <= 0.8)

    def test_right_stream_reenters_at_positive_y_without_reversing(self):
        cfg = dict(self.cfg, duration=70.0)
        boundary_time = (2.0 - cfg["y_min"]) / 0.7
        track = [(0.0, 3.0, 2.0, 0.0, -0.7),
                 (boundary_time, 3.0, cfg["y_max"], 0.0, -0.7)]
        starts = [segment[0] for segment in track]
        before = self.ns["state_at"](track, starts, boundary_time - 1e-6, cfg)
        at = self.ns["state_at"](track, starts, boundary_time, cfg)
        after = self.ns["state_at"](track, starts, boundary_time + 1e-6, cfg)
        self.assertAlmostEqual(before[1], cfg["y_min"] + 0.7e-6)
        self.assertEqual(at, (3.0, cfg["y_max"], 0.0, -0.7))
        self.assertLess(after[1], at[1])
        self.assertEqual(after[2:], (0.0, -0.7))

    def test_left_stream_reenters_at_negative_y_without_reversing(self):
        cfg = dict(self.cfg, duration=70.0)
        boundary_time = (cfg["y_max"] + 2.0) / 0.8
        track = [(0.0, 6.0, -2.0, 0.0, 0.8),
                 (boundary_time, 6.0, cfg["y_min"], 0.0, 0.8)]
        starts = [segment[0] for segment in track]
        at = self.ns["state_at"](track, starts, boundary_time, cfg)
        after = self.ns["state_at"](track, starts, boundary_time + 1e-6, cfg)
        self.assertEqual(at, (6.0, cfg["y_min"], 0.0, 0.8))
        self.assertGreater(after[1], at[1])
        self.assertEqual(after[2:], (0.0, 0.8))

    def test_state_evaluation_clamps_before_start_and_after_time_limit(self):
        cfg, tracks, starts = self.plan(num_obstacles=2, duration=5.5)
        for track, times in zip(tracks, starts):
            self.assertEqual(self.ns["state_at"](track, times, -10.0, cfg),
                             self.ns["state_at"](track, times, 0.0, cfg))
            self.assertEqual(self.ns["state_at"](track, times, 100.0, cfg),
                             self.ns["state_at"](track, times, 5.5, cfg))

    def test_running_frames_publish_both_velocity_components_but_do_not_integrate_gazebo_twist(self):
        cfg, tracks, starts = self.plan(num_obstacles=4)
        setter = Mock(return_value=SimpleNamespace(success=True))
        publisher = Mock()
        self.ns["publish_frame"](1.0, tracks, starts, cfg, setter, publisher)
        states = publisher.publish.call_args.args[0]
        self.assertEqual(states.name, ["obstacle_1", "obstacle_2", "obstacle_3", "obstacle_4"])
        for index, twist in enumerate(states.twist):
            expected = self.ns["state_at"](tracks[index], starts[index], 1.0, cfg)
            self.assertEqual((twist.linear.x, twist.linear.y), expected[2:])
            self.assertEqual(states.pose[index].position.x, expected[0])
            self.assertEqual(states.pose[index].position.y, expected[1])
        for call in setter.call_args_list:
            twist = call.args[0].twist
            self.assertEqual((twist.linear.x, twist.linear.y, twist.linear.z,
                              twist.angular.x, twist.angular.y, twist.angular.z),
                             (0.0,) * 6)

    def test_waiting_and_finished_frames_publish_zero_xy_velocities(self):
        cfg, tracks, starts = self.plan(num_obstacles=3, duration=5.5)
        setter = Mock(return_value=SimpleNamespace(success=True))
        publisher = Mock()
        for elapsed, moving in ((0.0, False), (3.0, False), (5.5, True), (99.0, True)):
            with self.subTest(elapsed=elapsed, moving=moving):
                self.ns["publish_frame"](elapsed, tracks, starts, cfg, setter, publisher,
                                         moving=moving)
                states = publisher.publish.call_args.args[0]
                self.assertTrue(all((t.linear.x, t.linear.y) == (0.0, 0.0)
                                    for t in states.twist))

    def test_zero_obstacle_frame_clears_published_state_without_gazebo_calls(self):
        cfg, tracks, starts = self.plan(num_obstacles=0)
        setter, publisher = Mock(), Mock()
        self.ns["publish_frame"](0.0, tracks, starts, cfg, setter, publisher)
        states = publisher.publish.call_args.args[0]
        self.assertEqual((states.name, states.pose, states.twist), ([], [], []))
        setter.assert_not_called()


if __name__ == "__main__":
    unittest.main()

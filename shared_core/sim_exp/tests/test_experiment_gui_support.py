"""Launcher inputs and ownership rules do not require Qt or ROS."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIRECTORY))
from experiment_protocol import validate_seed, validate_obstacle_count, validate_duration
from gui_support import (belongs_to_launch, controller_command, controller_names,
                         normalize_namespace, state_is_fresh)


class LauncherTests(unittest.TestCase):
    def test_command_preserves_paths_and_transfers_seed_and_launch_id_without_a_shell(self):
        with tempfile.TemporaryDirectory(prefix="DWA with spaces ") as directory:
            Path(directory, "dwa_test.py").write_text("")
            token = "1" * 32  # YAML must retain even an all-numeric token as a string.
            command = controller_command(directory, "dwa_test.py", 2147483647, token, "/sim_exp")
            self.assertEqual(command[1], str(Path(directory, "run_experiment.py").resolve()))
            self.assertIn("--new-session", command)
            self.assertIn("_experiment_seed:=2147483647", command)
            encoded = next(value.split(":=", 1)[1] for value in command
                           if value.startswith("_experiment_launch_id:="))
            self.assertEqual(json.loads(encoded), token)

    def test_only_existing_controller_filenames_can_be_selected(self):
        self.assertEqual(len(controller_names(DIRECTORY)), 11)
        for name in ("../sim/dwa_static.py", "dynamic_obstacles.py", "missing.py"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                controller_command(DIRECTORY, name, 42, "a" * 32, "/sim_exp")

    def test_seed_is_a_nonnegative_int32(self):
        for seed in (0, 42, 2147483647):
            self.assertEqual(validate_seed(seed), seed)
        for seed in (-1, 2147483648, True, "42", 1.2):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                validate_seed(seed)

    def test_obstacle_count_and_time_limit_are_included_in_the_child_arguments(self):
        command = controller_command(DIRECTORY, "dwa_static.py", 42, "a" * 32,
                                     "/sim_exp", num_obstacles=10, duration=5.5)
        self.assertIn("_experiment_num_obstacles:=10", command)
        self.assertIn("_experiment_duration:=5.5", command)

    def test_invalid_scene_settings_are_rejected_before_launch(self):
        for count in (-1, 1001, True, 3.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                validate_obstacle_count(count)
        self.assertEqual(validate_obstacle_count(0), 0)
        for duration in (0, -1, float("nan"), float("inf"), True, "70", 86401):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                validate_duration(duration)
        self.assertEqual(validate_duration(5.5), 5.5)

    def test_stale_or_other_launch_state_cannot_be_used_to_stop_a_new_child(self):
        state = {"launch_id": "a" * 32, "phase": "FINISHED"}
        self.assertFalse(belongs_to_launch(state, "b" * 32))
        self.assertFalse(belongs_to_launch(state, ""))
        self.assertTrue(belongs_to_launch(state, "a" * 32))
        self.assertFalse(state_is_fresh(state, None, 10.0))
        self.assertFalse(state_is_fresh(state, 5.0, 10.0))
        self.assertTrue(state_is_fresh(state, 9.0, 10.0))

    def test_namespace_is_validated_before_becoming_an_argument(self):
        self.assertEqual(normalize_namespace("/sim_exp/"), "/sim_exp")
        for value in ("", "/", "sim_exp", "/foo bar", "/foo;cmd"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_namespace(value)


if __name__ == "__main__":
    unittest.main()

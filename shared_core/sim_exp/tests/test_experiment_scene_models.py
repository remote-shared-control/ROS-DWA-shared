"""Verify real model reconciliation against a fake Gazebo world, not just topics."""

import re
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from test_experiment_sync import definitions


class SceneModelsTests(unittest.TestCase):
    def setUp(self):
        self.world = {"limo", "ground_plane", "obstacle_static"}
        self.deleted, self.spawned = [], []
        self.cfg = {"num_obstacles": 10, "radius": 0.15, "length": 0.4, "height": 0.2}

        def pose():
            return SimpleNamespace(position=SimpleNamespace(), orientation=SimpleNamespace())

        namespace = definitions("dynamic_obstacles.py", {"SceneModels", "ensure_model", "make_pose"}, {
            "re": re, "time": time, "Pose": pose,
            "rospy": SimpleNamespace(loginfo=Mock()),
            "CYLINDER_SDF": "model {name} radius {radius} length {length}",
        })

        def spawn(name, _sdf, _namespace, _pose, _frame):
            self.world.add(name)
            self.spawned.append(name)
            return SimpleNamespace(success=True)

        def delete(name):
            self.world.remove(name)
            self.deleted.append(name)
            return SimpleNamespace(success=True)

        self.get_world = Mock(side_effect=lambda: SimpleNamespace(
            success=True, model_names=list(self.world)))
        self.delete = Mock(side_effect=delete)
        self.spawn = Mock(side_effect=spawn)
        self.models = namespace["SceneModels"](
            self.get_world, lambda name, frame: SimpleNamespace(success=name in self.world),
            self.spawn, self.delete, verify_timeout=0.0)

    def sync(self, count):
        self.cfg["num_obstacles"] = count
        self.models.sync(self.cfg, [[(0.0, float(idx), 0.0, 0.0, 0.8)] for idx in range(count)])

    def managed(self):
        return self.world - {"limo", "ground_plane", "obstacle_static"}

    def test_reducing_40_to_10_really_deletes_the_30_surplus_models(self):
        self.world.update("obstacle_{}".format(i) for i in range(1, 41))
        self.sync(10)
        self.assertEqual(set(self.deleted), {"obstacle_{}".format(i) for i in range(11, 41)})
        self.assertEqual(self.managed(), {"obstacle_{}".format(i) for i in range(1, 11)})
        self.spawn.assert_not_called()
        self.assertGreaterEqual(self.get_world.call_count, 2)

    def test_zero_removes_all_managed_models_and_preserves_robot_and_static_world(self):
        self.world.update({"obstacle_1", "obstacle_40", "obstacle_9999", "obstacle_0"})
        self.sync(0)
        self.assertEqual(self.world, {"limo", "ground_plane", "obstacle_static"})
        self.assertEqual(len(self.deleted), 4)

    def test_increasing_quantity_spawns_missing_models(self):
        self.world.update({"obstacle_1", "obstacle_2"})
        self.sync(5)
        self.assertEqual(self.spawned, ["obstacle_3", "obstacle_4", "obstacle_5"])
        self.assertEqual(len(self.managed()), 5)

    def test_restart_leftovers_are_discovered_from_the_world(self):
        self.world.update({"obstacle_1", "obstacle_003", "obstacle_9999"})
        self.sync(1)
        self.assertEqual(set(self.deleted), {"obstacle_003", "obstacle_9999"})
        self.assertEqual(self.managed(), {"obstacle_1"})

    def test_failed_delete_is_an_error_instead_of_a_successful_preparation(self):
        self.world.add("obstacle_40")
        self.delete.side_effect = None
        self.delete.return_value = SimpleNamespace(success=False, status_message="Gazebo refused")
        with self.assertRaisesRegex(RuntimeError, "Cannot delete surplus model"):
            self.sync(0)
        self.assertIn("obstacle_40", self.world)

    def test_success_response_without_actual_deletion_fails_world_verification(self):
        self.world.add("obstacle_40")
        self.delete.side_effect = None
        self.delete.return_value = SimpleNamespace(success=True)
        with self.assertRaisesRegex(RuntimeError, "verification failed"):
            self.sync(0)
        self.assertIn("obstacle_40", self.world)

    def test_world_query_failure_does_not_delete_or_spawn_anything(self):
        self.get_world.side_effect = None
        self.get_world.return_value = SimpleNamespace(success=False, status_message="unavailable")
        with self.assertRaisesRegex(RuntimeError, "Cannot query Gazebo"):
            self.sync(0)
        self.delete.assert_not_called()
        self.spawn.assert_not_called()


if __name__ == "__main__":
    unittest.main()

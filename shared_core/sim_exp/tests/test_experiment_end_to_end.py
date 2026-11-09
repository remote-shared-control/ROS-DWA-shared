"""Replay real client/server threads and 40 models with an in-memory ROS bus.

This verifies application behavior, not ROS transport or Gazebo physics.
"""

import bisect
from copy import deepcopy
import heapq
import json
import random
import re
import threading
import time
import uuid
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from test_experiment_finish import CONTROLLERS, Twist, components, load_controller
from test_experiment_sync import Message, definitions
from experiment_protocol import ACTIVE, NSEC, Session, StartGate, validate_seed, validate_obstacle_count, validate_duration


def until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for the simulated experiment")
        time.sleep(0.001)


class RosBus:
    def __init__(self):
        self.now_ns = 100 * NSEC
        self.stopped = False
        self.services, self.callbacks, self.latched = {}, {}, {}
        self.lock = threading.RLock()
        self.positions = {}
        self.other_models = {"limo", "ground_plane", "obstacle_static"}
        self.deleted = []
        self.spawned = []
        self.messages = {}
        self.commands = []
        self.threads = []
        self.params = {}
        self.ServiceException = RuntimeError
        self.Time = SimpleNamespace(now=lambda: SimpleNamespace(
            to_nsec=lambda: self.now_ns))
        self.loginfo = self.logwarn = self.logerr = self.logwarn_throttle = Mock()
        self.on_shutdown = Mock()

    def get_param(self, name, default=None):
        return True if name == "/use_sim_time" else self.params.get(name, default)

    def get_time(self):
        return self.now_ns / NSEC

    def is_shutdown(self):
        return self.stopped

    def Publisher(self, name, _kind, **options):
        def publish(message):
            with self.lock:
                message = deepcopy(message)
                self.messages[name] = message
                if options.get("latch"):
                    self.latched[name] = message
                if name == "/cmd_vel":
                    self.commands.append((self.now_ns, components(message)))
                callbacks = list(self.callbacks.get(name, []))
            for callback in callbacks:
                callback(deepcopy(message))
        return SimpleNamespace(publish=publish)

    def Subscriber(self, name, _kind, callback, **_options):
        with self.lock:
            self.callbacks.setdefault(name, []).append(callback)
            latched = self.latched.get(name)
        if latched is not None:
            callback(deepcopy(latched))
        return SimpleNamespace()

    def Service(self, name, _kind, handler):
        with self.lock:
            self.services[name] = handler
        def shutdown(_reason):
            with self.lock:
                if self.services.get(name) is handler:
                    self.services.pop(name)
        return SimpleNamespace(shutdown=shutdown)

    def wait_for_service(self, name, timeout):
        until(lambda: name in self.services, timeout)

    def ServiceProxy(self, name, _kind):
        def call(*args):
            with self.lock:
                handler = self.services[name]
            return handler(Message(config=args[0]) if args else Message())
        return call

    def get_world(self):
        return SimpleNamespace(success=True, model_names=[*self.other_models, *self.positions])

    def get_model(self, name, _frame):
        return SimpleNamespace(success=name in self.positions)

    def spawn_model(self, name, _sdf, _namespace, pose, _frame):
        self.positions[name] = (pose.position.x, pose.position.y, pose.position.z)
        self.spawned.append(name)
        return SimpleNamespace(success=True)

    def delete_model(self, name):
        self.positions.pop(name)
        self.deleted.append(name)
        return SimpleNamespace(success=True)

    def set_model_state(self, state):
        if state.model_name not in self.positions:
            return SimpleNamespace(success=False, status_message="Model is missing")
        self.positions[state.model_name] = (
            state.pose.position.x, state.pose.position.y, state.pose.position.z)
        return SimpleNamespace(success=True)


class RepeatedExperimentTests(unittest.TestCase):
    def setUp(self):
        self.ros = RosBus()
        self.clients = []
        def pose():
            return SimpleNamespace(position=SimpleNamespace(), orientation=SimpleNamespace())
        namespace = {
            "rospy": self.ros, "json": json, "threading": threading, "time": time,
            "Session": Session, "ACTIVE": ACTIVE, "NSEC": NSEC,
            "Pose": pose, "Twist": Twist, "ModelState": SimpleNamespace,
            "ModelStates": lambda: SimpleNamespace(name=[], pose=[], twist=[]),
            "String": Message, "Int32": Message, "Trigger": object(),
            "TriggerResponse": Message, "bisect": bisect, "random": random,
            "heapq": heapq, "re": re, "validate_seed": validate_seed, "Reconfigure": object(),
            "validate_obstacle_count": validate_obstacle_count, "validate_duration": validate_duration,
            "CYLINDER_SDF": "model {name} radius {radius} length {length}",
            "ReconfigureResponse": Message, "IntParameter": Message, "StrParameter": Message, "DoubleParameter": Message,
            "DynamicConfig": lambda: Message(ints=[], strs=[]),
        }
        definitions("dynamic_obstacles.py", {
            "plan_scene", "choose_respawn_y", "state_at", "make_pose",
            "publish_frame", "ensure_model", "SceneModels", "ObstacleExperiments",
        }, namespace)
        self.cfg = {
            "num_obstacles": 40, "x_min": 0.5, "x_max": 14.0,
            "y_min": -4.0, "y_max": 4.0, "min_speed": 0.6, "max_speed": 1.0,
            "height": 0.2, "radius": 0.15, "length": 0.4, "duration": 70.0, "seed": 42, "rate": 20,
            "respawn_time_window": 4.0, "min_y_gap": 0.8,
        }
        self.tracks, self.starts = namespace["plan_scene"](self.cfg)
        self.state_at = namespace["state_at"]
        self.server = namespace["ObstacleExperiments"](
            self.cfg, self.tracks, self.starts, self.ros.set_model_state,
            self.ros.Publisher("/dynamic_obstacle_states", object(), latch=True),
            self.ros.Publisher("/dynamic_obstacle_finished", object(), latch=True),
            namespace["SceneModels"](self.ros.get_world, self.ros.get_model,
                                     self.ros.spawn_model, self.ros.delete_model))
        self.server_thread = threading.Thread(target=self.server.run, daemon=True)
        self.server_thread.start()
        client_ns = definitions("experiment_sync.py", {"ExperimentClient"}, {
            "rospy": self.ros, "json": json, "threading": threading, "time": time,
            "NSEC": NSEC, "StartGate": StartGate, "Twist": Twist,
            "String": Message, "Int32": Message, "Odometry": object(),
            "Trigger": object(), "Reconfigure": object(), "validate_seed": validate_seed,
            "validate_obstacle_count": validate_obstacle_count, "validate_duration": validate_duration,
            "IntParameter": Message, "StrParameter": Message, "DoubleParameter": Message,
            "DynamicConfig": lambda: Message(ints=[], strs=[]),
        })
        self.Client = client_ns["ExperimentClient"]

    def tearDown(self):
        for client in self.clients:
            client.close()
        self.ros.stopped = True
        self.server_thread.join(timeout=3.0)
        self.assertFalse(self.server_thread.is_alive())

    def advance(self, now_ns):
        self.ros.now_ns = now_ns
        def updated():
            with self.server.lock:
                return self.server.last_frame_ns == now_ns
        until(updated)

    def start(self, path):
        _, ns, config = load_controller(path)
        runtime = ns["runtime_state"]
        publisher = self.ros.Publisher("/cmd_vel", Twist)
        client = self.Client(publisher, runtime)
        self.clients.append(client)
        self.ros.Publisher("/odom", object()).publish(Message())
        result, complete = {}, threading.Event()
        def launch():
            try:
                result["t0"] = client.wait_for_start(config, SimpleNamespace())
            except Exception as exc:
                result["error"] = exc
            finally:
                complete.set()
        threading.Thread(target=launch, daemon=True).start()
        until(lambda: client.gate.t0_ns != 0 or complete.is_set())
        if "error" in result:
            raise result["error"]
        t0 = client.gate.t0_ns
        initial = dict(self.ros.positions)
        command = Twist()
        command.linear.x = 0.6
        self.advance(t0 - 1)
        runtime.publish_velocity(publisher, command)
        self.assertEqual(self.ros.commands[-1][1], (0.0,) * 6)
        self.assertEqual(self.ros.positions, initial)
        self.assertFalse(complete.is_set())
        self.advance(t0)
        until(complete.is_set)
        self.assertNotIn("error", result)
        self.assertEqual(result["t0"], t0 / NSEC)
        runtime.publish_velocity(publisher, command)
        self.assertEqual(self.ros.commands[-1][1][0], 0.6)
        return client, runtime, publisher, command, initial

    def test_every_controller_can_start_stop_at_70s_and_repeat_without_server_restart(self):
        run_ids = set()
        for path in CONTROLLERS:
            with self.subTest(controller=path.name):
                client, runtime, publisher, command, initial = self.start(path)
                run_ids.add(client.gate.run_id)
                t0 = client.gate.t0_ns
                self.advance(t0 + NSEC)
                self.assertNotEqual(self.ros.positions, initial)
                self.advance(t0 + 70 * NSEC - 1)
                runtime.publish_velocity(publisher, command)
                self.assertEqual(self.ros.commands[-1][1][0], 0.6)
                self.advance(t0 + 70 * NSEC)
                runtime.publish_velocity(publisher, command)
                self.assertEqual(self.ros.commands[-1][1], (0.0,) * 6)
                self.assertEqual(self.server.session.phase, "FINISHED")
                self.assertEqual(self.server.session.elapsed_ns, 70 * NSEC)
                self.assertTrue(all(t.linear.x == 0.0 for t in
                                    self.ros.messages["/dynamic_obstacle_states"].twist))
                for idx, (track, starts) in enumerate(zip(self.tracks, self.starts)):
                    x, y, _ = self.state_at(track, starts, 70.0, self.cfg)
                    self.assertEqual(self.ros.positions[f"obstacle_{idx + 1}"], (x, y, 0.2))
                final = dict(self.ros.positions)
                self.advance(t0 + 71 * NSEC)
                self.assertEqual(self.ros.positions, final)
                self.assertEqual(self.ros.messages["/dynamic_obstacle_finished"].data, 1)
                client.close()
        self.assertEqual(len(run_ids), len(CONTROLLERS))

    def test_exit_before_70s_freezes_obstacles_and_next_launch_can_start_again(self):
        client, runtime, publisher, command, _ = self.start(CONTROLLERS[0])
        self.advance(client.gate.t0_ns + 10 * NSEC)
        client.close()
        self.assertEqual(self.server.session.phase, "CANCELLED")
        frozen = dict(self.ros.positions)
        self.advance(self.ros.now_ns + NSEC)
        self.assertEqual(self.ros.positions, frozen)
        runtime.publish_velocity(publisher, command)
        self.assertEqual(self.ros.commands[-1][1], (0.0,) * 6)
        replacement, *_ = self.start(CONTROLLERS[0])
        self.assertNotEqual(client.gate.run_id, replacement.gate.run_id)

    def test_robot_seed_reaches_master_and_same_seed_repeats_without_server_restart(self):
        initial_scenes = []
        for seed in (42, 42, 17):
            token = uuid.uuid4().hex
            self.ros.params.update({"~experiment_seed": seed,
                                    "~experiment_launch_id": token,
                                    "~experiment_controller": CONTROLLERS[0].name})
            client, _, _, _, initial = self.start(CONTROLLERS[0])
            state = self.server.session.snapshot()
            self.assertEqual((state["seed"], state["launch_id"]), (seed, token))
            self.assertEqual(self.server.cfg["seed"], seed)
            initial_scenes.append(initial)
            client.close()
        self.assertEqual(initial_scenes[0], initial_scenes[1])
        self.assertNotEqual(initial_scenes[0], initial_scenes[2])

    def test_count_and_time_limit_change_across_runs_and_surplus_models_are_deleted(self):
        previous_count = 40
        for count, duration in ((40, 70.0), (10, 5.5), (0, 0.5), (25, 20.0)):
            with self.subTest(count=count, duration=duration):
                self.ros.params.update({"~experiment_seed": 42,
                                        "~experiment_num_obstacles": count,
                                        "~experiment_duration": duration,
                                        "~experiment_launch_id": uuid.uuid4().hex})
                deleted_before = len(self.ros.deleted)
                client, runtime, publisher, command, initial = self.start(CONTROLLERS[0])
                self.assertEqual(len(initial), count)
                self.assertEqual(self.server.session.num_obstacles, count)
                duration_ns = round(duration * NSEC)
                self.assertEqual(client.gate.end_ns - client.gate.t0_ns, duration_ns)
                self.assertEqual(len(self.ros.deleted) - deleted_before,
                                 max(previous_count - count, 0))
                self.advance(client.gate.t0_ns + duration_ns - 1)
                runtime.publish_velocity(publisher, command)
                self.assertEqual(self.ros.commands[-1][1][0], 0.6)
                self.advance(client.gate.t0_ns + duration_ns)
                runtime.publish_velocity(publisher, command)
                self.assertEqual(self.ros.commands[-1][1], (0.0,) * 6)
                self.assertEqual(self.server.session.phase, "FINISHED")
                states = self.ros.messages["/dynamic_obstacle_states"]
                self.assertEqual(len(states.name), count)
                self.assertTrue(all(item.linear.x == 0.0 for item in states.twist))
                self.assertEqual(self.ros.other_models, {"limo", "ground_plane", "obstacle_static"})
                client.close()
                previous_count = count


if __name__ == "__main__":
    unittest.main()

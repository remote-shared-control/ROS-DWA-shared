"""Replay real client/server threads and 40 models with an in-memory ROS bus.

This verifies application behavior, not ROS transport or Gazebo physics.
"""

import bisect
from copy import deepcopy
import heapq
import json
import random
import threading
import time
import uuid
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from test_experiment_finish import CONTROLLERS, Twist, components, load_controller
from test_experiment_sync import Message, definitions
from experiment_protocol import ACTIVE, NSEC, Session, StartGate, validate_seed


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

    def set_model_state(self, state):
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
            "heapq": heapq, "validate_seed": validate_seed, "Reconfigure": object(),
            "ReconfigureResponse": Message, "IntParameter": Message, "StrParameter": Message,
            "DynamicConfig": lambda: Message(ints=[], strs=[]),
        }
        definitions("dynamic_obstacles.py", {
            "plan_scene", "choose_respawn_y", "state_at", "make_pose",
            "publish_frame", "ObstacleExperiments",
        }, namespace)
        self.cfg = {
            "num_obstacles": 40, "x_min": 0.5, "x_max": 14.0,
            "y_min": -4.0, "y_max": 4.0, "min_speed": 0.6, "max_speed": 1.0,
            "height": 0.2, "duration": 70.0, "seed": 42, "rate": 20,
            "respawn_time_window": 4.0, "min_y_gap": 0.8,
        }
        self.tracks, self.starts = namespace["plan_scene"](self.cfg)
        self.state_at = namespace["state_at"]
        self.server = namespace["ObstacleExperiments"](
            self.cfg, self.tracks, self.starts, self.ros.set_model_state,
            self.ros.Publisher("/dynamic_obstacle_states", object(), latch=True),
            self.ros.Publisher("/dynamic_obstacle_finished", object(), latch=True))
        self.server_thread = threading.Thread(target=self.server.run, daemon=True)
        self.server_thread.start()
        client_ns = definitions("experiment_sync.py", {"ExperimentClient"}, {
            "rospy": self.ros, "json": json, "threading": threading, "time": time,
            "NSEC": NSEC, "StartGate": StartGate, "Twist": Twist,
            "String": Message, "Int32": Message, "Odometry": object(),
            "Trigger": object(), "Reconfigure": object(), "validate_seed": validate_seed,
            "IntParameter": Message, "StrParameter": Message,
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


if __name__ == "__main__":
    unittest.main()

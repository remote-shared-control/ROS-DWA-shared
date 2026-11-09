"""Exercise timing, repeated trials and the actual ROS adapters without ROS."""

import ast
import bisect
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIRECTORY))
from experiment_protocol import ACTIVE, NSEC, Session, StartGate, validate_seed, validate_obstacle_count, validate_duration


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.server = Session(duration=70, start_delay=2)
        self.prepared = self.server.prepare(100 * NSEC, 10.0)
        self.gate = StartGate()
        self.gate.adopt(self.prepared)

    def arm(self):
        armed = self.server.arm(self.server.run_id, 100 * NSEC, 10.0)
        self.gate.schedule(armed, 100 * NSEC, NSEC // 5)
        self.gate.observe(armed, 10.0)
        return armed

    def test_both_sides_use_exactly_same_future_t0_and_end(self):
        self.arm()
        for now, expected in ((101 * NSEC, False), (102 * NSEC - 1, False),
                              (102 * NSEC, True), (103 * NSEC, True)):
            self.server.heartbeat(self.server.run_id, 10.0)
            state = self.server.tick(now, 10.0)
            self.gate.observe(state, 10.0)
            self.assertEqual(self.gate.allows(now, 10.0), expected)
            self.assertEqual(state["phase"] == "RUNNING", expected)
            if expected:
                self.assertEqual(state["elapsed_ns"], now - self.gate.t0_ns)
        self.server.heartbeat(self.server.run_id, 11.0)
        state = self.server.tick(172 * NSEC, 11.0)
        self.gate.observe(state, 11.0)
        self.assertEqual(state["phase"], "FINISHED")
        self.assertFalse(self.gate.allows(172 * NSEC, 11.0))

    def test_repeated_arm_preserves_t0_and_active_prepare_is_rejected(self):
        original = self.arm()["t0_ns"]
        repeated = self.server.arm(self.server.run_id, 101 * NSEC, 10.1)
        self.assertEqual(repeated["t0_ns"], original)
        with self.assertRaises(ValueError):
            self.server.prepare(101 * NSEC, 10.1)

    def test_old_latched_finish_and_old_cancel_cannot_affect_new_run(self):
        old = self.server.cancel(self.server.run_id)
        self.server.prepare(110 * NSEC, 11.0)
        self.gate = StartGate()
        self.gate.adopt(self.server.snapshot())
        self.assertFalse(self.gate.observe(old, 11.0))
        self.assertEqual(self.gate.reason, "")
        with self.assertRaises(ValueError):
            self.server.cancel(old["run_id"])
        self.assertEqual(self.server.phase, "PREPARED")

    def test_late_response_is_rejected_and_cancel_freezes_elapsed(self):
        armed = self.server.arm(self.server.run_id, 100 * NSEC, 10.0)
        with self.assertRaises(ValueError):
            self.gate.schedule(armed, 102 * NSEC, NSEC // 5)
        self.server.cancel(self.server.run_id)
        self.server.tick(150 * NSEC, 10.1)
        self.assertEqual(self.server.phase, "CANCELLED")
        self.assertEqual(self.server.elapsed_ns, 0)

    def test_clock_rewind_aborts_instead_of_silently_restarting(self):
        self.arm()
        self.server.tick(102 * NSEC, 10.0)
        self.gate.allows(102 * NSEC, 10.0)
        self.server.tick(NSEC, 10.1)
        self.assertEqual(self.server.phase, "ABORTED")
        self.assertFalse(self.gate.allows(NSEC, 10.1))
        self.assertIn("backwards", self.gate.reason)

    def test_pause_preserves_simulation_schedule_but_client_loss_aborts(self):
        self.arm()
        for wall in (11.0, 12.0, 13.0):
            self.server.heartbeat(self.server.run_id, wall)
            self.server.tick(101 * NSEC, wall)
            self.assertEqual(self.server.phase, "ARMED")
        self.server.tick(101 * NSEC, 16.1)
        self.assertEqual(self.server.phase, "ABORTED")

    def test_state_loss_blocks_robot_without_waiting_for_a_finish_message(self):
        self.arm()
        self.assertFalse(self.gate.allows(103 * NSEC, 13.1))
        self.assertIn("timed out", self.gate.reason)

    def test_abandoned_prepare_expires_and_allows_next_trial(self):
        self.server.tick(100 * NSEC, 41.0)
        self.assertEqual(self.server.phase, "ABORTED")
        self.server.prepare(100 * NSEC, 42.0)
        self.assertNotEqual(self.server.run_id, self.prepared["run_id"])


def definitions(filename, names, namespace):
    tree = ast.parse((DIRECTORY / filename).read_text())
    selected = [node for node in tree.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    exec(compile(ast.Module(body=selected, type_ignores=[]), filename, "exec"), namespace)
    return namespace


class Message:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class ObstacleAdapterTests(unittest.TestCase):
    def test_direct_start_selects_sim_clock_without_a_manual_parameter_command(self):
        tree = ast.parse((DIRECTORY / 'dynamic_obstacles.py').read_text())
        main = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'main')
        for initial in (None, False, True):
            with self.subTest(initial=initial):
                parameters = {} if initial is None else {'/use_sim_time': initial}
                selected_clocks = []
                ros = SimpleNamespace(
                    set_param=lambda key, value: parameters.__setitem__(key, value),
                    init_node=lambda name: selected_clocks.append(
                        parameters.get('/use_sim_time', False)),
                )
                exec(compile(ast.Module(body=main.body[:2], type_ignores=[]),
                             'startup', 'exec'), {'rospy': ros})
                self.assertEqual(selected_clocks, [True])

    def setUp(self):
        self.now = 100 * NSEC
        self.wall = 10.0
        ros = SimpleNamespace(
            get_param=lambda name, default: default,
            Time=SimpleNamespace(now=lambda: SimpleNamespace(to_nsec=lambda: self.now)),
            Publisher=Mock(side_effect=lambda *a, **k: Mock()),
            Subscriber=Mock(), Service=Mock(side_effect=lambda *a, **k: Mock()),
            loginfo=Mock(), logerr=Mock(), is_shutdown=Mock(),
            ServiceException=RuntimeError,
        )
        self.frame = Mock()
        self.ns = definitions("dynamic_obstacles.py", {"ObstacleExperiments"}, {
            "rospy": ros, "Session": Session, "ACTIVE": ACTIVE, "NSEC": NSEC,
            "threading": threading, "time": SimpleNamespace(
                monotonic=lambda: self.wall, sleep=Mock()),
            "json": json, "String": Message, "Int32": Message,
            "Trigger": object(), "TriggerResponse": Message, "Reconfigure": object(),
            "validate_seed": validate_seed, "validate_obstacle_count": validate_obstacle_count,
            "validate_duration": validate_duration,
            "IntParameter": Message, "StrParameter": Message, "DoubleParameter": Message,
            "ReconfigureResponse": Message,
            "DynamicConfig": lambda: Message(ints=[], strs=[]),
            "publish_frame": self.frame,
            "ModelStates": lambda: Message(name=[], pose=[], twist=[]),
        })
        self.server = self.ns["ObstacleExperiments"](
            {"duration": 70.0, "rate": 20, "seed": 42, "num_obstacles": 40},
            [], [], Mock(), Mock(), Mock(), Mock())

    def test_seed_request_plans_and_prepares_one_atomic_experiment(self):
        planner = self.ns["plan_scene"] = Mock(return_value=(["new tracks"], ["new starts"]))
        request = Message(config=Message(
            ints=[Message(name="seed", value=17)],
            strs=[Message(name="launch_id", value="a" * 32)]))
        response = self.server._prepare_seed(request)
        state = json.loads(response.config.strs[0].value)
        self.assertEqual((state["seed"], state["launch_id"], state["phase"]),
                         (17, "a" * 32, "PREPARED"))
        self.assertEqual(self.server.tracks, ["new tracks"])
        planner.assert_called_once()

    def test_seed_change_during_an_active_run_is_rejected_without_resetting_models(self):
        self.server._prepare(None)
        self.server._arm(self.server.session.run_id)
        self.frame.reset_mock()
        planner = self.ns["plan_scene"] = Mock()
        request = Message(config=Message(ints=[Message(name="seed", value=99)], strs=[]))
        with self.assertRaises(RuntimeError):
            self.server._prepare_seed(request)
        self.assertEqual(self.server.cfg["seed"], 42)
        self.frame.assert_not_called()
        planner.assert_not_called()

    def test_prepare_resets_before_t0_is_chosen_and_response_contains_run_id(self):
        self.frame.reset_mock()
        response = self.server._prepare(None)
        self.assertTrue(response.success)
        self.assertEqual(json.loads(response.message)["phase"], "PREPARED")
        self.assertEqual(self.server.session.t0_ns, 0)
        self.assertFalse(self.frame.call_args.kwargs["moving"])
        # 40-model reset latency is excluded from the eventual start delay.
        self.now = 120 * NSEC
        response = self.server._arm(self.server.session.run_id)
        self.assertEqual(json.loads(response.message)["t0_ns"], 122 * NSEC)

    def test_actual_loop_keeps_scene_still_before_t0_then_uses_elapsed(self):
        self.server._prepare(None)
        self.server._arm(self.server.session.run_id)
        for now, moving, elapsed in ((101 * NSEC, False, 0.0),
                                     (102 * NSEC, True, 0.0),
                                     (103 * NSEC, True, 1.0)):
            self.now = now
            self.ns["rospy"].is_shutdown.side_effect = [False, True]
            self.server.run()
            self.assertEqual(self.frame.call_args.kwargs["moving"], moving)
            self.assertEqual(self.frame.call_args.args[0], elapsed)

    def test_failed_reset_is_not_acknowledged(self):
        self.frame.side_effect = RuntimeError("Gazebo unavailable")
        response = self.server._prepare(None)
        self.assertFalse(response.success)
        self.assertEqual(self.server.session.t0_ns, 0)

    def test_model_cleanup_failure_aborts_and_cannot_arm_the_robot(self):
        self.server.models.sync.side_effect = RuntimeError("Surplus model deletion failed")
        self.frame.reset_mock()
        response = self.server._prepare(None)
        self.assertFalse(response.success)
        self.assertEqual(self.server.session.phase, "ABORTED")
        self.assertFalse(self.server.scene_ready)
        self.frame.assert_not_called()
        self.assertFalse(self.server._arm(self.server.session.run_id).success)
        self.ns["rospy"].is_shutdown.side_effect = [False, True]
        self.server.run()
        self.frame.assert_not_called()

    def test_old_cancel_does_not_reset_next_experiment_models(self):
        self.server._prepare(None)
        old_id = self.server.session.run_id
        self.server._cancel(old_id)
        self.server._prepare(None)
        self.frame.reset_mock()
        self.assertFalse(self.server._cancel(old_id).success)
        self.frame.assert_not_called()

    def test_cancel_at_horizon_preserves_exact_final_frame(self):
        self.server._prepare(None)
        self.server._arm(self.server.session.run_id)
        self.now = 172 * NSEC
        response = self.server._cancel(self.server.session.run_id)
        self.assertTrue(response.success)
        self.assertEqual(self.server.session.phase, "FINISHED")
        self.assertEqual(self.frame.call_args.args[0], 70.0)


class ObstacleFrameTests(unittest.TestCase):
    def test_waiting_and_finished_frames_publish_zero_twist(self):
        def pose():
            return SimpleNamespace(position=SimpleNamespace(), orientation=SimpleNamespace())

        def twist():
            return SimpleNamespace(linear=SimpleNamespace(x=0.0))

        ns = definitions("dynamic_obstacles.py", {"make_pose", "state_at", "publish_frame"}, {
            "Pose": pose, "Twist": twist, "ModelState": SimpleNamespace,
            "ModelStates": lambda: SimpleNamespace(name=[], pose=[], twist=[]),
            "bisect": bisect,
        })
        publisher, setter = Mock(), Mock(return_value=SimpleNamespace(success=True))
        cfg = {"duration": 70, "x_min": 0.5, "height": 0.2}
        args = (0.0, [[(0.0, 10.0, 1.0, 0.8)]], [[0.0]], cfg, setter, publisher)
        ns["publish_frame"](*args, moving=False)
        self.assertEqual(publisher.publish.call_args.args[0].twist[0].linear.x, 0.0)
        ns["publish_frame"](*args, moving=True)
        self.assertEqual(publisher.publish.call_args.args[0].twist[0].linear.x, -0.8)
        ns["publish_frame"](70.0, *args[1:], moving=True)
        self.assertEqual(publisher.publish.call_args.args[0].twist[0].linear.x, 0.0)


class ClientAdapterTests(unittest.TestCase):
    def test_state_before_arm_response_and_previous_run_finish_are_handled(self):
        now = 100 * NSEC
        ns = definitions("experiment_sync.py", {"ExperimentClient"}, {
            "json": json, "time": SimpleNamespace(monotonic=lambda: 10.0),
            "rospy": SimpleNamespace(logwarn_throttle=Mock()),
        })
        client = object.__new__(ns["ExperimentClient"])
        client.lock, client.gate = threading.RLock(), StartGate()
        client.state, client.scene_seen = None, False
        session = Session()
        client.gate.adopt(session.prepare(now, 10.0))
        stale = dict(session.snapshot(), run_id="a" * 32, phase="FINISHED")
        client._state_callback(Message(data=json.dumps(stale)))
        self.assertFalse(client.scene_seen)
        self.assertEqual(client.gate.reason, "")
        armed = session.arm(session.run_id, now, 10.0)
        client._state_callback(Message(data=json.dumps(armed)))
        client.gate.schedule(armed, now, NSEC // 5)
        self.assertTrue(client.scene_seen)
        self.assertFalse(client.gate.allows(102 * NSEC - 1, 10.0))
        self.assertTrue(client.gate.allows(102 * NSEC, 10.0))


class ClientWorkflowTests(unittest.TestCase):
    """Run the actual blocking client startup against the server state machine."""

    def setUp(self):
        self.now, self.wall = 100 * NSEC, 10.0
        self.late_reply = False
        ros = SimpleNamespace(
            get_param=lambda name, default: True if name == "/use_sim_time" else default,
            Time=SimpleNamespace(now=lambda: SimpleNamespace(to_nsec=lambda: self.now)),
            is_shutdown=lambda: False, loginfo=Mock(), logwarn=Mock(),
            logwarn_throttle=Mock(),
        )
        ns = definitions("experiment_sync.py", {"ExperimentClient"}, {
            "threading": threading, "json": json, "NSEC": NSEC,
            "time": SimpleNamespace(monotonic=lambda: self.wall), "rospy": ros,
            "Twist": lambda: object(),
        })
        self.client = object.__new__(ns["ExperimentClient"])
        self.client.gate, self.client.lock = StartGate(), threading.RLock()
        self.client.odom_seen, self.client.scene_seen = True, False
        self.client.state = None
        self.client.namespace = "/sim_exp"
        self.client.ready_timeout = 30.0
        self.client.margin_ns = NSEC // 5
        self.client.cancel_requested = False
        self.client.runtime = SimpleNamespace(publish_velocity=Mock())
        self.client.publisher = Mock()
        self.server = Session()
        self.operations = []

        def rpc(name, timeout=None):
            self.operations.append(name.rsplit("/", 1)[-1])
            if name.endswith("/prepare"):
                state = self.server.prepare(self.now, self.wall)
            elif name.endswith("/arm"):
                state = self.server.arm(self.server.run_id, self.now, self.wall)
                if self.late_reply:
                    self.now += 3 * NSEC
            elif name.endswith("/cancel"):
                state = self.server.cancel(self.server.run_id)
            else:
                raise AssertionError(name)
            # Real ROS may deliver the state before the service reply.
            self.client._state_callback(Message(data=json.dumps(state)))
            return state

        def wait(_seconds):
            self.now += NSEC // 100
            self.wall += 0.01
            self.server.heartbeat(self.server.run_id, self.wall)
            state = self.server.tick(self.now, self.wall)
            self.client._state_callback(Message(data=json.dumps(state)))

        self.client._rpc = rpc
        self.client.closed = SimpleNamespace(is_set=lambda: False, wait=wait)

    def test_successful_start_waits_for_t0_and_resets_only_trial_statistics(self):
        config = SimpleNamespace(xy=[(1, 2)], distance=10.0, distance0=10.0,
                                 first_time=False, _experiment_tracking=False, x=1.5)
        counter = SimpleNamespace(send_count=5, count_time=5.0, first_time=False)
        result = self.client.wait_for_start(config, counter)
        self.assertEqual(result, 102.0)
        self.assertEqual(self.now, 102 * NSEC)
        self.assertEqual(self.operations, ["prepare", "arm"])
        self.assertEqual((config.xy, config.distance, config.x), ([], 0.0, 1.5))
        self.assertTrue(config._experiment_tracking)
        self.assertEqual(counter.send_count, 0)
        self.assertEqual(self.server.phase, "RUNNING")

    def test_late_arm_reply_sends_cancel_and_never_enables_robot(self):
        self.late_reply = True
        with self.assertRaises(ValueError):
            self.client.wait_for_start(SimpleNamespace(), SimpleNamespace())
        self.assertEqual(self.operations, ["prepare", "arm", "cancel"])
        self.assertEqual(self.server.phase, "CANCELLED")
        self.assertFalse(self.client.allow_motion())
        self.client.runtime.publish_velocity.assert_called()


if __name__ == "__main__":
    unittest.main()

"""Exercise experiment-stop behavior in every simulation DWA control loop.

The controller runtime state, Config and control loop are executed from their real source with
small ROS/message doubles, so these checks do not require ROS or NumPy.
"""

import ast
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import os
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


SIM_DIR = Path(os.environ.get("SIM_DIR", Path(__file__).resolve().parents[1]))
CONTROLLERS = sorted([*SIM_DIR.glob("dwa*.py"), SIM_DIR / "direct.py"])


class Twist:
    def __init__(self):
        self.linear = SimpleNamespace(x=0.0, y=0.0, z=0.0)
        self.angular = SimpleNamespace(x=0.0, y=0.0, z=0.0)


def components(command):
    return tuple(getattr(vector, axis) for vector in
                 (command.linear, command.angular) for axis in ("x", "y", "z"))


class Samples(list):
    def tolist(self):
        return list(self)


# Config only needs the sampling-time preparation from NumPy.
NUMPY = SimpleNamespace(
    arange=lambda start, stop, step: Samples(
        start + i * step for i in range(round((stop - start) / step))),
    append=lambda samples, end: Samples([*samples, end]),
)


def load_controller(path, enabled=None):
    tree = ast.parse(path.read_text())
    classes = [node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name in {"RuntimeState", "Config"}]
    rospy = SimpleNamespace(
        get_param=Mock(side_effect=lambda name, default: default if enabled is None else enabled),
        Rate=Mock(return_value=SimpleNamespace(sleep=Mock())),
        get_time=Mock(return_value=1.0),
        is_shutdown=Mock(), Subscriber=Mock(),
    )
    namespace = {"rospy": rospy, "Twist": Twist, "threading": threading,
                 "np": NUMPY, "print": Mock()}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"), namespace)
    namespace["runtime_state"] = namespace["RuntimeState"]()
    main = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    parameter = next(node for node in main.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Attribute)
                             and target.attr == "enable_experiment_finish"
                             for target in node.targets))
    exec(compile(ast.Module(body=[parameter], type_ignores=[]), str(path), "exec"), namespace)
    return tree, namespace, namespace["Config"]()


def run_loop(tree, namespace, config, cycles=3, mode=1, during_dwa=None):
    main = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    loop = next(node for node in main.body if isinstance(node, ast.While))
    commands = []
    publisher = Mock()
    publisher.publish.side_effect = lambda velocity: commands.append(deepcopy(velocity))
    human = Twist()
    human.linear.x, human.angular.z = 0.3, 0.2
    config.human = human
    planner = Mock(return_value=[0.6, -0.4])
    if during_dwa is not None:
        def compute(*args, **kwargs):
            during_dwa()
            return [0.6, -0.4]
        planner.side_effect = compute
    namespace["rospy"].is_shutdown.side_effect = [False] * cycles + [True]
    preview = SimpleNamespace(point_count=0, x_points=[], y_points=[])
    state = namespace["runtime_state"]
    state.control_mode = mode
    state.save_requested = state.result_pending = False
    namespace.update(
        config=config, pub=publisher, start_time=0.0, x=[0.0] * 5,
        u=[0.0, 0.0], speed=Twist(), human=human, dwa_control=planner,
        counter=SimpleNamespace(send_count=0),
        processor=SimpleNamespace(xy_set=([], [])), cube=object(), readob=object(),
        log_writer=None, robot_trajectory=preview, human_trajectory=preview,
        marker=object(), h_marker=object(), line=Mock(),
        pub_line=Mock(), pub_line_human=Mock(), atGoal=Mock(return_value=0),
        RECORDER=False,
    )
    exec(compile(ast.Module(body=[loop], type_ignores=[]), "control_loop", "exec"), namespace)
    return commands, planner


class ExperimentFinishTests(unittest.TestCase):
    def test_switch_defaults_to_enabled_and_accepts_false(self):
        self.assertEqual(len(CONTROLLERS), 11)
        for path in CONTROLLERS:
            for enabled in (None, False):
                with self.subTest(script=path.name, enabled=enabled):
                    _, ns, config = load_controller(path, enabled)
                    ns["rospy"].get_param.assert_called_once_with("~enable_experiment_finish", True)
                    self.assertIs(ns["runtime_state"].enable_experiment_finish, enabled is not False)
                    self.assertFalse(ns["runtime_state"].experiment_finished)
                    ns["rospy"].Rate.assert_called_once_with(20)

    def test_only_one_latches_and_later_messages_cannot_clear_it(self):
        for path in CONTROLLERS:
            with self.subTest(script=path.name):
                _, ns, config = load_controller(path)
                for value in (0, -1, 2):
                    ns["runtime_state"].experiment_finish_callback(SimpleNamespace(data=value))
                self.assertFalse(ns["runtime_state"].experiment_finished)
                ns["print"].assert_not_called()
                for value in (1, 1, 0, -1, 1):
                    ns["runtime_state"].experiment_finish_callback(SimpleNamespace(data=value))
                self.assertTrue(ns["runtime_state"].experiment_finished)
                ns["print"].assert_called_once_with("experiment finished")

    def test_concurrent_finish_callbacks_print_exactly_once(self):
        for path in CONTROLLERS:
            with self.subTest(script=path.name):
                _, ns, config = load_controller(path)
                with ThreadPoolExecutor(max_workers=8) as pool:
                    list(pool.map(ns["runtime_state"].experiment_finish_callback,
                                  [SimpleNamespace(data=1) for _ in range(32)]))
                self.assertTrue(ns["runtime_state"].experiment_finished)
                ns["print"].assert_called_once_with("experiment finished")

    def test_finished_loops_repeat_full_zero_at_control_rate_in_both_modes(self):
        for path in CONTROLLERS:
            for mode in (0, 1):
                with self.subTest(script=path.name, mode=mode):
                    tree, ns, config = load_controller(path)
                    ns["runtime_state"].experiment_finish_callback(SimpleNamespace(data=1))
                    commands, planner = run_loop(tree, ns, config, cycles=4, mode=mode)
                    self.assertEqual([components(command) for command in commands], [(0.0,) * 6] * 4)
                    self.assertEqual(config.r.sleep.call_count, 4)
                    planner.assert_not_called()
                    ns["line"].assert_not_called()
                    ns["print"].assert_called_once_with("experiment finished")

    def test_finish_during_dwa_discards_pending_nonzero_command(self):
        for path in CONTROLLERS:
            with self.subTest(script=path.name):
                tree, ns, config = load_controller(path)
                commands, planner = run_loop(
                    tree, ns, config, during_dwa=lambda: ns["runtime_state"].experiment_finish_callback(
                        SimpleNamespace(data=1)))
                self.assertEqual(planner.call_count, 1)
                self.assertEqual([components(command) for command in commands], [(0.0,) * 6] * 3)
                self.assertEqual(config.r.sleep.call_count, 3)
                ns["print"].assert_called_once_with("experiment finished")

    def test_zero_topic_preserves_normal_dwa_and_manual_commands(self):
        for path in CONTROLLERS:
            for mode in (0, 1):
                with self.subTest(script=path.name, mode=mode):
                    tree, ns, config = load_controller(path)
                    ns["runtime_state"].experiment_finish_callback(SimpleNamespace(data=0))
                    commands, planner = run_loop(tree, ns, config, mode=mode)
                    expected = (0.6, 0.0, 0.0, 0.0, 0.0, -0.4) if mode else (0.3, 0.0, 0.0, 0.0, 0.0, 0.2)
                    self.assertEqual([components(command) for command in commands], [expected] * 3)
                    self.assertEqual(planner.call_count, 3 if mode else 0)
                    self.assertEqual(config.r.sleep.call_count, 3)
                    ns["print"].assert_not_called()

    def test_disabled_topic_preserves_normal_dwa_and_manual_commands(self):
        for path in CONTROLLERS:
            for mode in (0, 1):
                with self.subTest(script=path.name, mode=mode):
                    tree, ns, config = load_controller(path, enabled=False)
                    ns["runtime_state"].experiment_finish_callback(SimpleNamespace(data=1))
                    commands, planner = run_loop(
                        tree, ns, config, mode=mode,
                        during_dwa=lambda: ns["runtime_state"].experiment_finish_callback(SimpleNamespace(data=1)))
                    expected = (0.6, 0.0, 0.0, 0.0, 0.0, -0.4) if mode else (0.3, 0.0, 0.0, 0.0, 0.0, 0.2)
                    self.assertEqual([components(command) for command in commands], [expected] * 3)
                    self.assertFalse(ns["runtime_state"].experiment_finished)
                    self.assertEqual(planner.call_count, 3 if mode else 0)
                    self.assertEqual(config.r.sleep.call_count, 3)
                    ns["print"].assert_not_called()

    def test_experiment_client_replaces_legacy_finish_subscription(self):
        for path in CONTROLLERS:
            with self.subTest(script=path.name):
                tree, ns, config = load_controller(path)
                main = next(node for node in tree.body
                            if isinstance(node, ast.FunctionDef) and node.name == "main")
                subscriptions = [node for node in ast.walk(main)
                                 if isinstance(node, ast.Call) and node.args
                                 and isinstance(node.args[0], ast.Constant)
                                 and node.args[0].value == "/dynamic_obstacle_finished"]
                self.assertEqual(subscriptions, [])
                creation = next(node for node in main.body
                                if isinstance(node, ast.Assign)
                                and isinstance(node.value, ast.Call)
                                and isinstance(node.value.func, ast.Name)
                                and node.value.func.id == "ExperimentClient")
                client = Mock()
                client.wait_for_start.return_value = 123.0
                ns.update(ExperimentClient=Mock(return_value=client), pub=Mock(),
                          config=config, counter=Mock())
                exec(compile(ast.Module(body=[creation], type_ignores=[]),
                             str(path), "exec"), ns)
                ns["ExperimentClient"].assert_called_once_with(ns["pub"], ns["runtime_state"])
                timer = next(node for node in main.body
                             if isinstance(node, ast.Assign)
                             and any(isinstance(target, ast.Name) and target.id == "start_time"
                                     for target in node.targets))
                exec(compile(ast.Module(body=[timer], type_ignores=[]),
                             str(path), "exec"), ns)
                self.assertEqual(ns["start_time"], 123.0)
                client.wait_for_start.assert_called_once_with(config, ns["counter"])

    def test_publication_gate_blocks_both_modes_even_when_finish_switch_disabled(self):
        for path in CONTROLLERS:
            for mode in (0, 1):
                with self.subTest(script=path.name, mode=mode):
                    tree, ns, config = load_controller(path, enabled=False)
                    gate = SimpleNamespace(allow_motion=Mock(return_value=False),
                                           note_command=Mock())
                    ns["runtime_state"].experiment = gate
                    commands, _ = run_loop(tree, ns, config, mode=mode)
                    self.assertEqual([components(command) for command in commands], [(0.0,) * 6] * 3)
                    gate.note_command.assert_not_called()

    def test_finish_and_inflight_publication_are_ordered(self):
        for path in CONTROLLERS:
            with self.subTest(script=path.name):
                _, ns, _ = load_controller(path)
                state = ns["runtime_state"]
                publishing, release = threading.Event(), threading.Event()
                callback_started, callback_done = threading.Event(), threading.Event()
                commands = []

                def emit(command):
                    publishing.set()
                    if not release.wait(2):
                        raise RuntimeError("Timed out releasing test publisher")
                    commands.append((state.experiment_finished, components(command)))

                def finish():
                    callback_started.set()
                    state.experiment_finish_callback(SimpleNamespace(data=1))
                    callback_done.set()

                publisher = SimpleNamespace(publish=emit)
                command = Twist()
                command.linear.x = 1.0
                with ThreadPoolExecutor(max_workers=2) as pool:
                    publication = pool.submit(state.publish_velocity, publisher, command)
                    self.assertTrue(publishing.wait(2))
                    callback = pool.submit(finish)
                    self.assertTrue(callback_started.wait(2))
                    completed_while_publishing = callback_done.wait(0.01)
                    release.set()
                    publication.result(timeout=2)
                    callback.result(timeout=2)
                self.assertFalse(completed_while_publishing)
                state.publish_velocity(publisher, command)
                self.assertEqual(commands, [
                    (False, (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
                    (True, (0.0,) * 6),
                ])

    def test_final_publish_guard_zeros_all_six_twist_components(self):
        for path in CONTROLLERS:
            with self.subTest(script=path.name):
                _, ns, config = load_controller(path)
                command = Twist()
                for vector in (command.linear, command.angular):
                    vector.x, vector.y, vector.z = 1.0, 2.0, 3.0
                publisher = Mock()
                ns["runtime_state"].publish_velocity(publisher, command)
                self.assertIs(publisher.publish.call_args.args[0], command)
                ns["runtime_state"].experiment_finish_callback(SimpleNamespace(data=1))
                ns["runtime_state"].publish_velocity(publisher, command)
                self.assertEqual(components(publisher.publish.call_args.args[0]), (0.0,) * 6)
                self.assertEqual(components(command), (1.0, 2.0, 3.0, 1.0, 2.0, 3.0))


if __name__ == "__main__":
    unittest.main()

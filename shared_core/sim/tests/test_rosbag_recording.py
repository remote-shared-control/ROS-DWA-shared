"""Exercise the recording/motion boundary without requiring a ROS installation."""

import ast
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch


SIM_DIR = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sim_recorder", SIM_DIR / "rosbag_recorder.py")
recorder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recorder)


def stamp(value=10.0):
    return SimpleNamespace(to_sec=lambda: value)


def twist(v=0.0, w=0.0):
    return SimpleNamespace(
        linear=SimpleNamespace(x=v, y=0.0, z=0.0),
        angular=SimpleNamespace(x=0.0, y=0.0, z=w),
        _type="geometry_msgs/Twist", _md5sum="test", _full_text="test",
    )


def odom(x=0.0, y=0.0, v=0.0, at=10.0):
    return SimpleNamespace(
        pose=SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=x, y=y))),
        twist=SimpleNamespace(twist=twist(v)), header=SimpleNamespace(stamp=stamp(at)),
        _type="nav_msgs/Odometry", _md5sum="test", _full_text="test",
    )


class WriterTests(unittest.TestCase):
    def make_writer(self, obstacles=False, **kwargs):
        required = ["/odom", "/cmd_vel"]
        if obstacles:
            required.append("/dynamic_obstacle_states")
        return recorder.InitialRecordingWriter(Mock(), required, [0.0, 0.0], **kwargs)

    def test_first_bag_message_is_fresh_stationary_initial_odom(self):
        writer = self.make_writer(minimum_odom_stamp=9.0)
        samples = [
            ("/cmd_vel", twist()),
            ("/odom", odom(at=8.0)),
            ("/odom", odom(x=0.2)),
            ("/odom", odom(v=0.1)),
            ("/odom", odom(x=float("nan"))),
        ]
        for topic, msg in samples:
            self.assertFalse(writer.write(topic, msg, stamp()))
        writer.bag.write.assert_not_called()
        initial = odom(x=0.001)
        self.assertFalse(writer.write("/odom", initial, stamp()))
        self.assertIs(writer.bag.write.call_args.args[1], initial)
        self.assertEqual(writer.status()["waiting_for"], ["/cmd_vel"])

    def test_ack_waits_for_writes_and_flush_and_not_human_input(self):
        writer = self.make_writer(obstacles=True)
        writer.write("/odom", odom(), stamp())
        writer.write("/cmd_vel", twist(), stamp())
        self.assertFalse(writer.ready)
        self.assertEqual(writer.status()["waiting_for"], ["/dynamic_obstacle_states"])
        obstacle = SimpleNamespace(_type="gazebo_msgs/ModelStates", _md5sum="test", _full_text="test")
        writer.bag.write.side_effect = lambda *a, **k: self.assertFalse(writer.ready)
        writer.bag.flush.side_effect = lambda: self.assertFalse(writer.ready)
        self.assertTrue(writer.write("/dynamic_obstacle_states", obstacle, stamp()))
        self.assertTrue(writer.ready)
        writer.bag.flush.assert_called_once()
        # Motion and normal odometry are accepted once the initial data is flushed.
        writer.bag.write.side_effect = None
        self.assertFalse(writer.write("/cmd_vel", twist(0.4), stamp(11.0)))
        self.assertFalse(writer.write("/odom", odom(x=1.0, v=0.4, at=11.0), stamp(11.0)))

    def test_write_or_flush_failure_never_releases_motion(self):
        for failing_operation in ("write", "flush"):
            with self.subTest(operation=failing_operation):
                writer = self.make_writer()
                writer.write("/odom", odom(), stamp())
                getattr(writer.bag, failing_operation).side_effect = OSError("disk full")
                with self.assertRaisesRegex(OSError, "disk full"):
                    writer.write("/cmd_vel", twist(), stamp())
                self.assertFalse(writer.ready)

    def test_motion_during_preparation_aborts(self):
        for topic, msg in (("/cmd_vel", twist(0.1)), ("/odom", odom(x=0.3))):
            with self.subTest(topic=topic):
                writer = self.make_writer(obstacles=True)
                writer.write("/odom", odom(), stamp())
                with self.assertRaises(RuntimeError):
                    writer.write(topic, msg, stamp())
                self.assertFalse(writer.ready)

    def test_turtlebot_uses_its_own_initial_position(self):
        writer = recorder.InitialRecordingWriter(Mock(), ["/odom"], [0.0, 1.5])
        self.assertFalse(writer.write("/odom", odom(), stamp()))
        self.assertTrue(writer.write("/odom", odom(y=1.5), stamp()))

    def test_odom_alone_is_enough_after_successful_write_and_flush(self):
        writer = recorder.InitialRecordingWriter(Mock(), ["/odom"], [0.0, 0.0])
        writer.bag.write.side_effect = lambda *a, **k: self.assertFalse(writer.ready)
        writer.bag.flush.side_effect = lambda: self.assertFalse(writer.ready)
        self.assertTrue(writer.write("/odom", odom(x=0.002, y=0.002), stamp()))
        writer.bag.write.assert_called_once()
        writer.bag.flush.assert_called_once()
        self.assertEqual(writer.status()["written_topics"], ["/odom"])
        self.assertEqual(writer.status()["waiting_for"], [])

    def test_position_within_one_centimetre_passes_initial_pose_check(self):
        writer = self.make_writer()
        writer.write("/odom", odom(x=0.006, y=0.006), stamp())
        self.assertTrue(writer.write("/cmd_vel", twist(), stamp()))

    def test_receipt_timestamp_and_latching_header_are_preserved(self):
        writer = self.make_writer()
        msg = odom()
        msg._connection_header = {"callerid": "/sim", "latching": "1"}
        received = stamp(12.0)
        writer.write("/odom", msg, received)
        self.assertIs(writer.bag.write.call_args.kwargs["t"], received)
        header = writer.bag.write.call_args.kwargs["connection_header"]
        self.assertEqual(header["latching"], "1")
        self.assertEqual(header["callerid"], "/sim")
        self.assertEqual(header["type"], "nav_msgs/Odometry")


class SessionTests(unittest.TestCase):
    def make_session(self):
        session = recorder.RecorderSession("/tmp/test.bag", required_topics=["/odom", "/cmd_vel"])
        session.process = Mock(returncode=0)
        session.process.poll.return_value = None
        return session

    def test_only_this_process_run_can_acknowledge(self):
        session = self.make_session()
        unrelated = {"event": "ready", "run_id": "previous-run"}
        matching = {"event": "ready", "run_id": session.config["run_id"]}
        session.process.stdout = io.StringIO("".join(
            recorder.STATUS_PREFIX + json.dumps(event) + "\n"
            for event in (unrelated, matching)
        ))
        session._read_output()
        self.assertEqual(session.events.qsize(), 1)
        stops = Mock()
        with patch.object(recorder.time, "sleep", side_effect=AssertionError("fixed sleep")):
            session.wait_until_ready(stops, 1.0, lambda: False)
        stops.assert_called_once()
        self.assertTrue(session.ready)

    def test_timeout_keeps_publishing_stop_and_reports_missing_topic(self):
        session = self.make_session()
        stops = Mock()
        with self.assertRaisesRegex(RuntimeError, "timeout.*cmd_vel"):
            session.wait_until_ready(stops, 0.11, lambda: False)
        self.assertGreaterEqual(stops.call_count, 2)
        self.assertFalse(session.ready)

    def test_child_error_exit_and_ros_shutdown_prevent_start(self):
        for condition in ("error", "exit", "shutdown"):
            with self.subTest(condition=condition):
                session = self.make_session()
                if condition == "error":
                    session.events.put({"event": "error", "message": "disk full"})
                if condition == "exit":
                    session.process.poll.return_value = 1
                stops = Mock()
                with self.assertRaises(RuntimeError):
                    session.wait_until_ready(stops, 1.0, lambda: condition == "shutdown")
                stops.assert_called_once()

    def test_stop_requests_graceful_close_and_detects_failure(self):
        session = self.make_session()
        session.stop()
        session.process.send_signal.assert_called_once_with(signal.SIGINT)
        session.process.wait.assert_called_once_with(timeout=5.0)
        session.process.stdout.close.assert_called_once()
        session = self.make_session()
        session.opened = True
        session.process.returncode = 1
        session.process.poll.return_value = 1
        with self.assertRaisesRegex(RuntimeError, "active"):
            session.stop()

    def test_startup_error_before_open_is_not_reported_as_close_failure(self):
        session = self.make_session()
        session.process.poll.return_value = 1
        session.process.returncode = 1
        # The error can arrive after the process exit was first observed.
        session.reader = Mock()
        session.reader.join.side_effect = lambda **_: session.events.put({
            "event": "error", "message": "Permission denied: /tmp/test.bag.active",
        })
        with self.assertRaisesRegex(RuntimeError, "Permission denied.*bag.active"):
            session.check()
        self.assertFalse(session.opened)
        session.stop()
        session.process.send_signal.assert_not_called()

    def test_real_process_pipe_handshake_and_sigint_shutdown(self):
        child_source = '''
import argparse, json, os, signal, time
parser = argparse.ArgumentParser()
parser.add_argument('--config')
config = json.loads(parser.parse_args().config)
stopping = False
def stop(*args):
    global stopping
    stopping = True
signal.signal(signal.SIGINT, stop)
active = config['output_file'] + '.active'
with open(active, 'w') as output:
    output.write('initial data')
def emit(run_id):
    print('ROSBAG_STATUS ' + json.dumps({
        'event': 'ready', 'run_id': run_id,
        'written_topics': ['/odom', '/cmd_vel'], 'waiting_for': [],
    }), flush=True)
emit('unrelated-run')
time.sleep(0.06)
emit(config['run_id'])
while not stopping:
    time.sleep(0.01)
os.replace(active, config['output_file'])
'''
        with tempfile.TemporaryDirectory() as directory:
            child = Path(directory) / "child.py"
            child.write_text(child_source)
            output = Path(directory) / "run.bag"
            session = recorder.RecorderSession(output, required_topics=["/odom", "/cmd_vel"])
            stops = Mock()
            try:
                with patch.object(recorder, "__file__", str(child)):
                    session.start()
                session.wait_until_ready(stops, 5.0, lambda: False)
                self.assertTrue(session.ready)
                self.assertGreaterEqual(stops.call_count, 2)
            finally:
                session.stop()
            self.assertEqual(output.read_text(), "initial data")
            self.assertFalse(Path(str(output) + ".active").exists())


class RecorderLifecycleTests(unittest.TestCase):
    def run_with_fake_ros(self, directory, flush_failure=False, queue_size=10000,
                          stop_before_initial=False, odom_only=False):
        output = str(Path(directory) / "run.bag")
        handlers, callbacks, events = {}, {}, []

        class Bag:
            def __init__(self, filename, mode):
                self.file = open(filename, mode)

            def write(self, topic, msg, **kwargs):
                events.append(("write", topic))
                self.file.write(topic + "\n")

            def flush(self):
                if flush_failure:
                    raise OSError("disk full")
                self.file.flush()
                events.append(("flush",))

            def close(self):
                self.file.close()
                events.append(("close",))

        def subscribe(topic, msg_type, callback, callback_args, **kwargs):
            callbacks[topic] = lambda msg: callback(msg, callback_args)
            if len(callbacks) == 4:
                if stop_before_initial:
                    callbacks["/odom"](odom(x=1.0))
                    return Mock()
                if odom_only:
                    callbacks["/odom"](odom())
                    return Mock()
                callbacks["/cmd_vel"](twist())
                callbacks["/odom"](odom(x=1.0))  # Stale reset pose is not recorded.
                callbacks["/odom"](odom())
                callbacks["/cmd_vel"](twist())
                callbacks["/dynamic_obstacle_states"](SimpleNamespace(
                    _type="gazebo_msgs/ModelStates", _md5sum="test", _full_text="test"))
            return Mock()

        def emit(event, **payload):
            events.append((event,))
            if event == "progress" and stop_before_initial:
                handlers[signal.SIGINT](None, None)
            if event == "ready":
                # This queued command must still be drained on graceful shutdown.
                callbacks["/cmd_vel"](twist(0.2))
                handlers[signal.SIGINT](None, None)

        fake_rospy = SimpleNamespace(
            init_node=Mock(), Subscriber=subscribe, Time=SimpleNamespace(now=stamp),
            is_shutdown=lambda: False, signal_shutdown=Mock(),
        )
        modules = {
            "rosbag": SimpleNamespace(Bag=Bag), "rospy": fake_rospy,
            "gazebo_msgs.msg": SimpleNamespace(ModelStates=object),
            "geometry_msgs.msg": SimpleNamespace(Twist=object),
            "nav_msgs.msg": SimpleNamespace(Odometry=object),
        }
        config = {
            "output_file": output, "run_id": "test-run",
            "required_topics": ["/odom"],
            "initial_position": [0, 0], "position_tolerance": 0.02,
            "velocity_tolerance": 0.02, "minimum_odom_stamp": 9.0,
            "queue_size": queue_size,
        }
        with patch.dict(sys.modules, modules), patch.object(
                recorder.signal, "signal", side_effect=lambda sig, cb: handlers.update({sig: cb})):
            recorder.run_recorder(config, emit)
        return output, events

    def test_ready_follows_flush_and_shutdown_drains_before_final_rename(self):
        with tempfile.TemporaryDirectory() as directory:
            output, events = self.run_with_fake_ros(directory)
            self.assertLess(events.index(("flush",)), events.index(("ready",)))
            self.assertEqual(events[-1], ("close",))
            self.assertEqual(Path(output).read_text().splitlines(), [
                "/odom", "/cmd_vel", "/dynamic_obstacle_states", "/cmd_vel",
            ])
            self.assertFalse(Path(output + ".active").exists())

    def test_no_obstacle_or_initial_cmd_vel_messages_do_not_block_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            output, events = self.run_with_fake_ros(directory, odom_only=True)
            self.assertLess(events.index(("write", "/odom")), events.index(("flush",)))
            self.assertLess(events.index(("flush",)), events.index(("ready",)))
            # Motion is injected by the fixture only AFTER receiving ready.
            self.assertLess(events.index(("ready",)), events.index(("write", "/cmd_vel")))
            self.assertEqual(Path(output).read_text().splitlines(), ["/odom", "/cmd_vel"])
            self.assertNotIn(("write", "/dynamic_obstacle_states"), events)

    def test_disk_error_and_overflow_keep_partial_file_and_never_finalize(self):
        for settings, error in (({"flush_failure": True}, OSError),
                                ({"queue_size": 2}, RuntimeError)):
            with self.subTest(settings=settings), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(error):
                    self.run_with_fake_ros(directory, **settings)
                self.assertFalse((Path(directory) / "run.bag").exists())
                self.assertTrue((Path(directory) / "run.bag.active").exists())

    def test_cancellation_before_initial_pose_keeps_recording_unfinished(self):
        with tempfile.TemporaryDirectory() as directory:
            output, events = self.run_with_fake_ros(directory, stop_before_initial=True)
            self.assertNotIn(("ready",), events)
            self.assertFalse(Path(output).exists())
            self.assertTrue(Path(output + ".active").exists())

    def test_existing_bag_and_partial_file_are_overwritten(self):
        cases = (("run.bag",), ("run.bag.active",), ("run.bag", "run.bag.active"))
        for filenames in cases:
            with self.subTest(filenames=filenames), tempfile.TemporaryDirectory() as directory:
                for filename in filenames:
                    (Path(directory) / filename).write_text("old recording")
                unrelated = Path(directory) / "other.bag"
                unrelated.write_text("keep this recording")
                output, events = self.run_with_fake_ros(directory)
                self.assertIn(("ready",), events)
                self.assertEqual(Path(output).read_text().splitlines(), [
                    "/odom", "/cmd_vel", "/dynamic_obstacle_states", "/cmd_vel",
                ])
                self.assertFalse(Path(output + ".active").exists())
                self.assertEqual(unrelated.read_text(), "keep this recording")

    def test_failed_retry_keeps_old_completed_bag_and_rewrites_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.bag"
            active = Path(directory) / "run.bag.active"
            output.write_text("old completed recording")
            active.write_text("old unfinished recording")
            with self.assertRaises(OSError):
                self.run_with_fake_ros(directory, flush_failure=True)
            self.assertEqual(output.read_text(), "old completed recording")
            self.assertIn("/odom", active.read_text())
            self.assertNotIn("old unfinished recording", active.read_text())


class RecordingNumberTests(unittest.TestCase):
    def load_number_function(self, excel_ids=()):
        tree = ast.parse((SIM_DIR / "distancetime.py").read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "get_next_rosbag_number")
        workbook = Mock()
        workbook.active.iter_rows.return_value = ((value,) for value in excel_ids)
        loader = Mock(return_value=workbook)
        namespace = {"os": os, "load_workbook": loader}
        exec(compile(ast.Module(body=[function], type_ignores=[]),
                     "distancetime.py", "exec"), namespace)
        return namespace["get_next_rosbag_number"], loader

    def test_no_excel_reuses_one_despite_existing_recordings(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("rosbag1.bag", "rosbag4.bag.active", "rosbag99.bag.backup"):
                (Path(directory) / name).write_text("preserve me")
            get_number, loader = self.load_number_function()
            self.assertEqual(get_number(str(Path(directory) / "missing.xlsx")), 1)
            loader.assert_not_called()
            self.assertEqual((Path(directory) / "rosbag1.bag").read_text(), "preserve me")

    def test_only_excel_determines_number_even_if_disk_has_larger_ids(self):
        for disk_number in (4, 11, 20):
            with self.subTest(disk_number=disk_number), tempfile.TemporaryDirectory() as directory:
                excel = Path(directory) / "data.xlsx"
                excel.touch()
                (Path(directory) / ("rosbag%d.bag.active" % disk_number)).touch()
                get_number, loader = self.load_number_function((10, None, "invalid", 2))
                self.assertEqual(get_number(str(excel)), 11)
                loader.return_value.close.assert_called_once()

    def test_empty_inputs_start_at_one(self):
        with tempfile.TemporaryDirectory() as directory:
            get_number, loader = self.load_number_function()
            self.assertEqual(get_number(str(Path(directory) / "missing.xlsx")), 1)
            loader.assert_not_called()


def load_helpers():
    parsed = ast.parse((SIM_DIR / "distancetime.py").read_text())
    names = {"start_rosbag", "stop_rosbag", "publish_cmd_vel", "delete_rosbag"}
    body = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {
        "math": math, "os": Mock(), "rospy": Mock(), "Twist": twist,
        "RecorderSession": Mock(), "get_next_rosbag_number": lambda **_: 7,
        "_rosbag_recorder": None,
    }
    namespace["os"].path.expanduser.return_value = "/mock/rosbag"
    namespace["os"].path.join.side_effect = lambda *parts: "/".join(parts)
    namespace["rospy"].get_param.side_effect = lambda name, default: default
    namespace["rospy"].Time.now.return_value = stamp()
    namespace["RecorderSession"].return_value.wait_until_ready.return_value = {
        "written_topics": ["/odom", "/cmd_vel"], "last_odom": {"x": 0, "y": 0},
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), "distancetime.py", "exec"), namespace)
    return namespace


class ControllerBoundaryTests(unittest.TestCase):
    def test_handshake_uses_existing_publisher_and_initial_position(self):
        namespace = load_helpers()
        publisher = Mock()
        self.assertEqual(namespace["start_rosbag"](publisher, [0, 1.5, 0], True), 7)
        settings = namespace["RecorderSession"].call_args.kwargs
        self.assertEqual(settings["initial_position"], [0.0, 1.5])
        # The legacy True argument above must not restore obstacle gating.
        self.assertEqual(settings["required_topics"], ["/odom"])
        self.assertNotIn("~rosbag_require_obstacles",
                         [call.args[0] for call in namespace["rospy"].get_param.call_args_list])
        session = namespace["RecorderSession"].return_value
        session.wait_until_ready.call_args.args[0]()
        self.assertTrue(all(recorder.twist_is_zero(call.args[0], 0)
                            for call in publisher.publish.call_args_list))
        command = twist(0.5)
        namespace["publish_cmd_vel"](publisher, command)
        self.assertIs(publisher.publish.call_args.args[0], command)
        session.check.assert_called_once()

    def test_start_failure_stops_process_and_never_returns_an_id(self):
        namespace = load_helpers()
        session = namespace["RecorderSession"].return_value
        session.wait_until_ready.side_effect = RuntimeError("no initial odom")
        publisher = Mock()
        with self.assertRaisesRegex(RuntimeError, "initial odom"):
            namespace["start_rosbag"](publisher, [0, 0])
        session.stop.assert_called_once()
        self.assertIsNone(namespace["_rosbag_recorder"])
        self.assertTrue(all(recorder.twist_is_zero(call.args[0], 0)
                            for call in publisher.publish.call_args_list))

    def test_recorder_failure_replaces_motion_command_with_stop(self):
        namespace = load_helpers()
        namespace["_rosbag_recorder"] = Mock()
        namespace["_rosbag_recorder"].check.side_effect = RuntimeError("disk full")
        publisher = Mock()
        with self.assertRaisesRegex(RuntimeError, "disk full"):
            namespace["publish_cmd_vel"](publisher, twist(0.5))
        publisher.publish.assert_called_once()
        self.assertTrue(recorder.twist_is_zero(publisher.publish.call_args.args[0], 0))

    def test_deletion_closes_active_bag_before_removing_files(self):
        namespace = load_helpers()
        events = []
        namespace["_rosbag_recorder"] = Mock()
        namespace["_rosbag_recorder"].stop.side_effect = lambda: events.append("closed")
        namespace["os"].remove.side_effect = lambda path: events.append("removed")
        namespace["delete_rosbag"](7)
        self.assertEqual(events[0], "closed")
        self.assertIn("removed", events)

    def test_every_dwa_waits_before_timing_and_uses_guarded_publish(self):
        scripts = sorted(SIM_DIR.glob("dwa*.py"))
        self.assertEqual(len(scripts), 9)
        for path in scripts:
            with self.subTest(script=path.name):
                parsed = ast.parse(path.read_text())
                main = next(node for node in parsed.body
                            if isinstance(node, ast.FunctionDef) and node.name == "main")
                calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
                start = next(call for call in calls
                             if isinstance(call.func, ast.Name) and call.func.id == "start_rosbag")
                keywords = {kw.arg: kw.value for kw in start.keywords}
                self.assertEqual(keywords["cmd_vel_pub"].id, "pub")
                self.assertEqual(keywords["initial_position"].id, "target_position")
                self.assertNotIn("require_obstacles", keywords)
                timer = next(node for node in ast.walk(main) if isinstance(node, ast.Assign)
                             and any(isinstance(target, ast.Name) and target.id == "start_time"
                                     for target in node.targets))
                self.assertLess(start.lineno, timer.lineno)
                self.assertTrue(any(isinstance(call.func, ast.Name)
                                    and call.func.id == "publish_cmd_vel" for call in calls))
                self.assertFalse(any(isinstance(call.func, ast.Attribute)
                                     and isinstance(call.func.value, ast.Name)
                                     and call.func.value.id == "pub" and call.func.attr == "publish"
                                     for call in calls))
                self.assertTrue(any(isinstance(node, ast.Try) and node.finalbody
                                    for node in parsed.body[-1].body))


if __name__ == "__main__":
    unittest.main()

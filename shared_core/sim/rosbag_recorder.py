#!/usr/bin/env python3
"""Record simulation topics and acknowledge written initial data before motion."""

import argparse
import json
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid


RECORD_TOPICS = (
    "/cmd_vel", "/cmd_vel_human", "/odom", "/dynamic_obstacle_states",
)
STATUS_PREFIX = "ROSBAG_STATUS "


def twist_is_zero(twist, tolerance):
    """Check only the robot's forward speed and yaw rate."""
    values = (twist.linear.x, twist.angular.z)
    return all(math.isfinite(value) and abs(value) <= tolerance
               for value in values)


class InitialRecordingWriter:
    """Used only by the writer thread; readiness follows write AND flush."""

    def __init__(self, bag, required_topics, initial_position=None,
                 position_tolerance=0.02, velocity_tolerance=0.02,
                 minimum_odom_stamp=0.0):
        self.bag = bag
        self.required_topics = set(required_topics)
        self.initial_position = initial_position
        self.position_tolerance = position_tolerance
        self.velocity_tolerance = velocity_tolerance
        self.minimum_odom_stamp = minimum_odom_stamp
        self.written_topics = set()
        self.ready = False
        self.last_odom = None

    def status(self):
        return {
            "written_topics": sorted(self.written_topics),
            "waiting_for": sorted(self.required_topics - self.written_topics),
            "last_odom": self.last_odom,
        }

    def _initial_odom_is_valid(self, msg, received_at):
        position = msg.pose.pose.position
        stamp = msg.header.stamp.to_sec()
        self.last_odom = {
            "x": position.x, "y": position.y, "stamp": stamp,
            "linear_x": msg.twist.twist.linear.x,
            "angular_z": msg.twist.twist.angular.z,
        }
        if not all(math.isfinite(value) for value in (position.x, position.y, stamp)):
            return False
        if received_at.to_sec() <= 0 or stamp < self.minimum_odom_stamp:
            return False
        if self.initial_position is not None:
            distance = math.hypot(position.x - self.initial_position[0],
                                  position.y - self.initial_position[1])
            if distance > self.position_tolerance:
                return False
        return twist_is_zero(msg.twist.twist, self.velocity_tolerance)

    def write(self, topic, msg, received_at):
        """Return True only on the transition to a flushed, ready recording."""
        if not self.ready:
            if topic == "/odom":
                if not self._initial_odom_is_valid(msg, received_at):
                    if "/odom" in self.written_topics:
                        raise RuntimeError("Robot left its initial pose before recording was ready")
                    return False
            elif "/odom" not in self.written_topics:
                # Make the first bag message a fresh, stationary initial pose.
                return False
            if topic == "/cmd_vel" and not twist_is_zero(msg, 1e-9):
                raise RuntimeError("Nonzero /cmd_vel received before recording was ready")

        header = dict(getattr(msg, "_connection_header", None) or {})
        header.update(topic=topic)
        header.setdefault("type", msg._type)
        header.setdefault("md5sum", msg._md5sum)
        header.setdefault("message_definition", msg._full_text)
        self.bag.write(topic, msg, t=received_at, connection_header=header)
        self.written_topics.add(topic)
        if not self.ready and self.required_topics <= self.written_topics:
            self.bag.flush()
            self.ready = True
            return True
        return False


class RecorderSession:
    """Own one recorder process and its run-specific acknowledgement pipe."""

    def __init__(self, output_file, **settings):
        self.config = dict(settings, output_file=str(output_file), run_id=uuid.uuid4().hex)
        self.process = None
        self.opened = False
        self.ready = False
        self.events = queue.Queue()
        self.last_status = {"waiting_for": list(settings["required_topics"])}
        self.error = None
        self.reader = None

    def start(self):
        self.process = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).resolve()),
             "--config", json.dumps(self.config)],
            stdout=subprocess.PIPE, universal_newlines=True, bufsize=1,
            start_new_session=True,
        )
        self.reader = threading.Thread(target=self._read_output, daemon=True)
        self.reader.start()

    def _read_output(self):
        for line in self.process.stdout:
            if line.startswith(STATUS_PREFIX):
                try:
                    event = json.loads(line[len(STATUS_PREFIX):])
                except ValueError:
                    continue
                if event.get("run_id") == self.config["run_id"]:
                    self.events.put(event)
            else:
                print(line, end="", flush=True)

    def _handle_event(self, event):
        self.last_status.update(event)
        if event.get("event") in ("opened", "progress", "ready"):
            self.opened = True
        if event.get("event") == "ready":
            self.ready = True
        elif event.get("event") == "error":
            self.error = event["message"]

    def _drain_events(self):
        while True:
            try:
                self._handle_event(self.events.get_nowait())
            except queue.Empty:
                break

    def check(self):
        self._drain_events()
        exit_code = self.process.poll()
        if exit_code is not None:
            # Read the final error event before falling back to an exit code.
            if self.reader is not None:
                self.reader.join(timeout=1.0)
            self._drain_events()
        if self.error:
            raise RuntimeError("Rosbag recorder failed: " + self.error)
        if exit_code is not None:
            raise RuntimeError("Rosbag recorder exited with status %s" % exit_code)

    def wait_until_ready(self, publish_stop, timeout, is_shutdown):
        deadline = time.monotonic() + timeout
        while True:
            publish_stop()
            self.check()
            if is_shutdown():
                raise RuntimeError("ROS shut down while waiting for recording")
            if self.ready:
                return self.last_status
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    "Rosbag readiness timeout; waiting for %s; last odom: %s"
                    % (", ".join(self.last_status["waiting_for"]),
                       self.last_status.get("last_odom")))
            try:
                self._handle_event(self.events.get(timeout=min(0.05, remaining)))
            except queue.Empty:
                pass

    def stop(self):
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        if self.reader is not None:
            self.reader.join(timeout=1.0)
        self._drain_events()
        self.process.stdout.close()
        if self.process.returncode != 0 and self.opened:
            raise RuntimeError("Recorder did not close cleanly; inspect %s.active"
                               % self.config["output_file"])


def run_recorder(config, emit):
    # Lazy ROS imports also allow the handshake/writer to be tested without ROS.
    import rosbag
    import rospy
    from gazebo_msgs.msg import ModelStates
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry

    stop_requested = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop_requested.set())
    rospy.init_node("dwa_bag_recorder", anonymous=True, disable_signals=True)

    output_file = config["output_file"]
    active_file = output_file + ".active"

    messages = queue.Queue(maxsize=config.get("queue_size", 10000))
    callback_lock = threading.Lock()
    accepting = True
    overflow = threading.Event()
    subscribers = []
    bag = None

    def receive(msg, topic):
        with callback_lock:
            if not accepting:
                return
            try:
                messages.put_nowait((topic, msg, rospy.Time.now()))
            except queue.Full:
                overflow.set()
                stop_requested.set()

    def detach_subscribers():
        nonlocal accepting
        with callback_lock:
            accepting = False
        for subscriber in subscribers:
            subscriber.unregister()
        subscribers.clear()

    try:
        # Reuse this trial number, truncating an unfinished previous attempt.
        # The completed .bag is replaced only after the new recording closes.
        bag = rosbag.Bag(active_file, "w")
        emit("opened", active_file=active_file)
        writer = InitialRecordingWriter(
            bag, config["required_topics"], config.get("initial_position"),
            config["position_tolerance"], config["velocity_tolerance"],
            config["minimum_odom_stamp"],
        )
        topic_types = (Twist, Twist, Odometry, ModelStates)
        for topic, msg_type in zip(RECORD_TOPICS, topic_types):
            subscribers.append(rospy.Subscriber(
                topic, msg_type, receive, callback_args=topic, queue_size=1000,
            ))
        emit("progress", **writer.status())
        last_progress = time.monotonic()

        def write_message(item):
            if writer.write(*item):
                emit("ready", **writer.status())

        while not stop_requested.is_set() and not rospy.is_shutdown():
            try:
                write_message(messages.get(timeout=0.05))
            except queue.Empty:
                pass
            if not writer.ready and time.monotonic() - last_progress >= 0.5:
                emit("progress", **writer.status())
                last_progress = time.monotonic()

        detach_subscribers()
        if overflow.is_set():
            raise RuntimeError("Recording queue overflow; recording aborted")
        while not messages.empty():
            write_message(messages.get_nowait())
        bag.close()
        bag = None
        if writer.ready:
            os.replace(active_file, output_file)
    finally:
        detach_subscribers()
        if bag is not None:
            bag.close()
        rospy.signal_shutdown("Recording stopped")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    config = json.loads(parser.parse_args().config)

    def emit(event, **payload):
        print(STATUS_PREFIX + json.dumps(dict(
            payload, event=event, run_id=config["run_id"],
        )), flush=True)

    try:
        run_recorder(config, emit)
    except Exception as exc:
        emit("error", message=str(exc))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

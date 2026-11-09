"""DWA adapter for the remote obstacle node; no generated ROS types needed."""

import json
import threading
import time

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32, String
from std_srvs.srv import Trigger
from dynamic_reconfigure.msg import Config as DynamicConfig, IntParameter, StrParameter, DoubleParameter
from dynamic_reconfigure.srv import Reconfigure

from experiment_protocol import (NSEC, StartGate, validate_seed,
                                 validate_obstacle_count, validate_duration)


class ExperimentClient:
    def __init__(self, publisher, runtime):
        self.publisher, self.runtime = publisher, runtime
        self.namespace = rospy.get_param("~experiment_namespace", "/sim_exp").rstrip("/")
        self.timeout = float(rospy.get_param("~experiment_service_timeout", 30.0))
        self.ready_timeout = float(rospy.get_param("~experiment_ready_timeout", 30.0))
        self.margin_ns = round(float(rospy.get_param("~experiment_start_margin", 0.2)) * NSEC)
        self.gate = StartGate(float(rospy.get_param("~experiment_status_timeout", 3.0)))
        self.lock = threading.RLock()
        self.closed = threading.Event()
        self.odom_seen = False
        self.scene_seen = False
        self.finish_notified = False
        self.first_command_logged = False
        self.cancel_requested = False
        self.state = None
        self.seed = rospy.get_param("~experiment_seed", None)
        if self.seed is not None:
            self.seed = validate_seed(self.seed)
        self.num_obstacles = rospy.get_param("~experiment_num_obstacles", None)
        if self.num_obstacles is not None:
            self.num_obstacles = validate_obstacle_count(self.num_obstacles)
        self.duration = rospy.get_param("~experiment_duration", None)
        if self.duration is not None:
            self.duration = validate_duration(self.duration)
        self.launch_id = rospy.get_param("~experiment_launch_id", "")
        self.controller = rospy.get_param("~experiment_controller", "")
        runtime.experiment = self
        self.heartbeat = rospy.Publisher(self.namespace + "/heartbeat", String, queue_size=1)
        self.state_sub = rospy.Subscriber(self.namespace + "/state", String,
                                          self._state_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self._odom_callback, queue_size=1)
        rospy.on_shutdown(self.close)
        threading.Thread(target=self._watch, daemon=True).start()

    def _odom_callback(self, _message):
        self.odom_seen = True

    def _state_callback(self, message):
        try:
            state = json.loads(message.data)
            if not isinstance(state, dict):
                raise ValueError("Expected a state object")
            with self.lock:
                self.state = state
                self.gate.observe(state, time.monotonic())
                if self.gate.run_id and state.get("run_id") == self.gate.run_id:
                    self.scene_seen = True
        except (ValueError, TypeError, KeyError):
            rospy.logwarn_throttle(5.0, "Invalid sim_exp state message")

    def _rpc(self, name, timeout=None, seed=None, num_obstacles=None, duration=None):
        """Bound the whole RPC by wall time, including the service response."""
        timeout = self.timeout if timeout is None else timeout
        result, done = {}, threading.Event()

        def call():
            try:
                rospy.wait_for_service(name, timeout=timeout)
                if seed is None and num_obstacles is None and duration is None:
                    response = rospy.ServiceProxy(name, Trigger)()
                    if not response.success:
                        raise RuntimeError(response.message)
                    payload = response.message
                else:
                    config = DynamicConfig()
                    config.ints = []
                    if seed is not None:
                        config.ints.append(IntParameter(name="seed", value=seed))
                    if num_obstacles is not None:
                        config.ints.append(IntParameter(name="num_obstacles", value=num_obstacles))
                    if duration is not None:
                        config.doubles = [DoubleParameter(name="duration", value=duration)]
                    config.strs = [StrParameter(name="launch_id", value=self.launch_id),
                                   StrParameter(name="controller", value=self.controller)]
                    response = rospy.ServiceProxy(name, Reconfigure)(config)
                    payload = next(param.value for param in response.config.strs
                                   if param.name == "experiment")
                result["value"] = json.loads(payload)
            except Exception as exc:
                result["error"] = exc
            finally:
                done.set()

        threading.Thread(target=call, daemon=True).start()
        if not done.wait(timeout):
            raise RuntimeError("Experiment service timed out: " + name)
        if "error" in result:
            raise RuntimeError("{}: {}".format(name, result["error"]))
        return result["value"]

    def _run_service(self, operation):
        return "{}/runs/{}/{}".format(self.namespace, self.gate.run_id, operation)

    def allow_motion(self):
        with self.lock:
            return (not self.closed.is_set() and self.gate.allows(
                rospy.Time.now().to_nsec(), time.monotonic()))

    def note_command(self, command):
        if self.first_command_logged:
            return
        if any((command.linear.x, command.linear.y, command.linear.z,
                command.angular.x, command.angular.y, command.angular.z)):
            self.first_command_logged = True
            rospy.loginfo("sim_exp run=%s first nonzero /cmd_vel at %.9f (t0=%.9f)",
                          self.gate.run_id, rospy.get_time(), self.gate.t0_ns / NSEC)

    def _cancel(self):
        with self.lock:
            if not self.gate.run_id or self.cancel_requested:
                return
            self.cancel_requested = True
            service = self._run_service("cancel")
        try:
            self._rpc(service, timeout=2.0)
        except RuntimeError as exc:
            rospy.logwarn("Could not cancel remote experiment: %s", exc)

    def _watch(self):
        while not self.closed.is_set() and not rospy.is_shutdown():
            with self.lock:
                run_id = self.gate.run_id
            if run_id:
                self.heartbeat.publish(String(data=run_id))
            # Test and publish under the controller's publication lock: a
            # zero decided just before t0 must not overwrite its first command.
            with self.runtime._experiment_finish_lock:
                if not self.allow_motion():
                    self.publisher.publish(Twist())
            with self.lock:
                reason = self.gate.reason
            if reason and not self.finish_notified:
                self.finish_notified = True
                rospy.loginfo("sim_exp stopped: %s", reason)
                self.runtime.experiment_finish_callback(Int32(data=1))
                if reason != "FINISHED":
                    threading.Thread(target=self._cancel, daemon=True).start()
            self.closed.wait(0.02)

    def wait_for_start(self, config, counter):
        try:
            if not rospy.get_param("/use_sim_time", False):
                raise RuntimeError("Set /use_sim_time=true before starting the nodes")
            seed = getattr(self, "seed", None)
            count = getattr(self, "num_obstacles", None)
            duration = getattr(self, "duration", None)
            configured = any(value is not None for value in (seed, count, duration))
            if not configured:
                prepared = self._rpc(self.namespace + "/prepare")
            else:
                prepared = self._rpc(self.namespace + "/prepare_seed", seed=seed,
                                     num_obstacles=count, duration=duration)
            with self.lock:
                self.gate.adopt(prepared)
                if seed is not None and (prepared.get("seed") != seed or
                                          prepared.get("launch_id") != self.launch_id):
                    raise RuntimeError("Remote experiment seed/launch ID does not match")
                if count is not None and prepared.get("num_obstacles") != count:
                    raise RuntimeError("Remote obstacle count does not match")
                if duration is not None and prepared.get("duration_ns") != round(duration * NSEC):
                    raise RuntimeError("Remote experiment duration does not match")
                # A state callback might have preceded the prepare response.
                if self.state is not None:
                    self.gate.observe(self.state, time.monotonic())
                    self.scene_seen = self.state.get("run_id") == self.gate.run_id
            deadline = time.monotonic() + self.ready_timeout
            while not (self.odom_seen and self.scene_seen):
                if rospy.is_shutdown() or time.monotonic() >= deadline:
                    raise RuntimeError("Waiting for odometry / prepared scene timed out")
                self.closed.wait(0.01)
            armed = self._rpc(self._run_service("arm"))
            with self.lock:
                self.gate.schedule(armed, rospy.Time.now().to_nsec(), self.margin_ns)
                if self.state is not None:
                    self.gate.observe(self.state, time.monotonic())
            rospy.loginfo("sim_exp run=%s t0=%.9f; waiting with zero velocity",
                          self.gate.run_id, self.gate.t0_ns / NSEC)
            while not self.allow_motion():
                with self.lock:
                    reason = self.gate.reason
                if reason or rospy.is_shutdown():
                    raise RuntimeError(reason or "ROS shutdown while waiting for t0")
                self.closed.wait(0.001)
            # Exclude setup/waiting motion and contacts from trial statistics.
            config.xy.clear()
            config.distance = config.distance0 = 0.0
            config.first_time = True
            config._experiment_tracking = True
            counter.send_count = 0
            counter.count_time = 0
            counter.first_time = True
            counter.in_collision = False
            counter.last_contact_time = 0.0
            return self.gate.t0_ns / NSEC
        except Exception:
            with self.lock:
                self.gate.reason = "Experiment startup failed"
            self.runtime.publish_velocity(self.publisher, Twist())
            self._cancel()
            raise

    def close(self):
        self.closed.set()
        self.runtime.publish_velocity(self.publisher, Twist())
        self._cancel()

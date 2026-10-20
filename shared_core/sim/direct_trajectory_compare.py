#!/usr/bin/env python3
"""Relay /cmd_vel_human and compare the legacy and midpoint-v/w predictions.

Both predictions use the same odometry state and the relayed target v/w.
The numerical routines mirror dwa_simwaffi_new.py and
dwa_simwaffi_new_midpoint.py without importing their simulation side effects.
"""

from bisect import bisect_left
from copy import deepcopy
import math
import threading

import numpy as np


class PredictionConfig:
    def __init__(self, dt=0.4, predict_time=2.4, max_accel=2.5,
                 max_dyawrate=3.2):
        for name, value in (("dt", dt), ("predict_time_static", predict_time),
                            ("max_accel", max_accel),
                            ("max_dyawrate", max_dyawrate)):
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("{} must be finite and positive".format(name))
            setattr(self, name, value)
        times = np.append(
            np.arange(0.0, self.predict_time_static, self.dt),
            self.predict_time_static,
        )
        self.trajectory_times = tuple(times.tolist())
        self.trajectory_steps = tuple(
            (t1, t1 - t0, t0 + 0.5 * (t1 - t0), True)
            for t0, t1 in zip(self.trajectory_times, self.trajectory_times[1:])
        )


def motion(x, u, dt):
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[2] += u[1] * dt
    return x


def calc_trajectory(xinit, v, w, config):
    """Preserve the original Euler rollout, including its last extra step."""
    x = np.array(xinit, dtype=float)
    states = [x.copy()]
    time = 0.0
    while time <= config.predict_time_static:
        x = motion(x, (v, w), config.dt)
        states.append(x.copy())
        time += config.dt
    return np.asarray(states)


def motion_midpoint(x, u, dt):
    """Use midpoint speeds and the interval's initial yaw rate, as before."""
    v_mid, w_mid = u
    theta_mid = x[2] + 0.5 * dt * x[4]
    x[0] += v_mid * math.cos(theta_mid) * dt
    x[1] += v_mid * math.sin(theta_mid) * dt
    x[2] += w_mid * dt
    return x


def prepare_linear_motion(v0, v, config):
    v0, v = float(v0), float(v)
    dv = v - v0
    a_max = config.max_accel
    a = a_max if dv > 0 else (-a_max if dv < 0 else 0.0)
    t_v = abs(dv) / a_max
    times = config.trajectory_times
    steps = config.trajectory_steps
    index = bisect_left(times, t_v)
    if 0 < index < len(times):
        t0, t1 = times[index - 1], times[index]
        if t_v - t0 > 1e-12 and t1 - t_v > 1e-12:
            steps = (
                steps[:index - 1]
                + (
                    (t_v, t_v - t0, t0 + 0.5 * (t_v - t0), False),
                    (t1, t1 - t_v, t_v + 0.5 * (t1 - t_v), True),
                )
                + steps[index:]
            )
    return v, t_v, a, steps


def calc_trajectory_midpoint_vw(xinit, v, w, config, linear_plan=None):
    """Split integration at both speed switch times; save base samples only."""
    x = [float(value) for value in xinit]
    v0, w0 = x[3], x[4]
    if linear_plan is None:
        linear_plan = prepare_linear_motion(v0, v, config)
    v, t_v, a, steps = linear_plan
    w = float(w)
    dw = w - w0
    alpha_max = config.max_dyawrate
    alpha = alpha_max if dw > 0 else (-alpha_max if dw < 0 else 0.0)
    t_w = abs(dw) / alpha_max

    index = bisect_left(steps, (t_w,))
    if index < len(steps):
        t1, h, _, save_point = steps[index]
        t0 = t1 - h
        if t_w - t0 > 1e-12 and t1 - t_w > 1e-12:
            steps = (
                steps[:index]
                + (
                    (t_w, t_w - t0, t0 + 0.5 * (t_w - t0), False),
                    (t1, t1 - t_w, t_w + 0.5 * (t1 - t_w), save_point),
                )
                + steps[index + 1:]
            )

    traj = np.empty((len(config.trajectory_times), len(x)), dtype=float)
    traj[0] = x
    row = 1
    for t1, h, t_mid, save_point in steps:
        v_mid = v0 + a * t_mid if t_mid < t_v else v
        w_mid = w0 + alpha * t_mid if t_mid < t_w else w
        x = motion_midpoint(x, (v_mid, w_mid), h)
        x[3] = v0 + a * t1 if t1 < t_v else v
        x[4] = w0 + alpha * t1 if t1 < t_w else w
        if save_point:
            traj[row] = x
            row += 1
    return traj


def make_marker(trajectory, header, namespace, color, line_width):
    from geometry_msgs.msg import Point
    from visualization_msgs.msg import Marker

    marker = Marker()
    marker.header = deepcopy(header)
    marker.ns = namespace
    marker.id = 0
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = line_width
    marker.color.r, marker.color.g, marker.color.b = color
    marker.color.a = 1.0
    marker.points = [
        Point(x=float(x), y=float(y), z=0.05) for x, y in trajectory[:, :2]
    ]
    return marker


class DirectTrajectoryCompareNode:
    def __init__(self):
        import rospy
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from visualization_msgs.msg import Marker

        self.ros = rospy
        self.config = PredictionConfig(
            dt=rospy.get_param("~dt", 0.4),
            predict_time=rospy.get_param("~predict_time", 2.4),
            max_accel=rospy.get_param("~max_accel", 2.5),
            max_dyawrate=rospy.get_param("~max_dyawrate", 3.2),
        )
        rate = float(rospy.get_param("~rate", 20.0))
        self.line_width = float(rospy.get_param("~line_width", 0.05))
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("rate must be finite and positive")
        if not math.isfinite(self.line_width) or self.line_width <= 0.0:
            raise ValueError("line_width must be finite and positive")
        input_topic = rospy.resolve_name(rospy.get_param("~input_topic", "/cmd_vel_human"))
        output_topic = rospy.resolve_name(rospy.get_param("~output_topic", "/cmd_vel"))
        if input_topic == output_topic:
            raise ValueError("input_topic and output_topic must be different")
        odom_topic = rospy.get_param("~odom_topic", "/odom")

        self._lock = threading.Lock()
        self._command = None
        self._odometry = None
        self.cmd_pub = rospy.Publisher(output_topic, Twist, queue_size=1)
        self.legacy_pub = rospy.Publisher("~line_list", Marker, queue_size=1)
        self.midpoint_pub = rospy.Publisher("~line_list_midpoint_vw", Marker, queue_size=1)
        self.odom_sub = rospy.Subscriber(
            odom_topic, Odometry, self.odom_callback, queue_size=1,
        )
        self.human_sub = rospy.Subscriber(
            input_topic, Twist, self.human_callback, queue_size=1,
        )
        self.timer = rospy.Timer(rospy.Duration(1.0 / rate), self.publish_previews)
        rospy.on_shutdown(self.stop)
        rospy.loginfo(
            "Direct relay %s -> %s; trajectory markers: %s (green), %s (blue)",
            input_topic, output_topic, self.legacy_pub.resolved_name,
            self.midpoint_pub.resolved_name,
        )

    def human_callback(self, message):
        command = deepcopy(message)
        with self._lock:
            self._command = command
        # Forward the complete Twist immediately, even before odometry arrives.
        self.cmd_pub.publish(command)

    def odom_callback(self, message):
        from tf.transformations import euler_from_quaternion

        position = message.pose.pose.position
        q = message.pose.pose.orientation
        yaw = euler_from_quaternion((q.x, q.y, q.z, q.w))[2]
        state = np.array([
            position.x, position.y, yaw,
            message.twist.twist.linear.x, message.twist.twist.angular.z,
        ], dtype=float)
        header = deepcopy(message.header)
        header.frame_id = header.frame_id or "odom"
        with self._lock:
            self._odometry = (state, header)

    def publish_previews(self, _event):
        with self._lock:
            command, odometry = self._command, self._odometry
        if command is None or odometry is None:
            return
        state, header = odometry
        v, w = command.linear.x, command.angular.z
        legacy = calc_trajectory(state, v, w, self.config)
        midpoint = calc_trajectory_midpoint_vw(state, v, w, self.config)
        self.legacy_pub.publish(make_marker(
            legacy, header, "no_midpoint", (0.0, 1.0, 0.0), self.line_width,
        ))
        self.midpoint_pub.publish(make_marker(
            midpoint, header, "midpoint_vw", (0.0, 0.0, 1.0), self.line_width,
        ))

    def stop(self):
        from geometry_msgs.msg import Twist

        self.cmd_pub.publish(Twist())


def main():
    import rospy

    rospy.init_node("direct_trajectory_compare")
    DirectTrajectoryCompareNode()
    rospy.spin()


if __name__ == "__main__":
    main()

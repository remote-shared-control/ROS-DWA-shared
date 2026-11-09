"""ROS callbacks communicate with Qt exclusively through queued signals."""

import json
import threading

from qt_compat import QtCore


class RosBridge(QtCore.QObject):
    state_received = QtCore.pyqtSignal(dict)
    message = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)

    def __init__(self, namespace, parent=None):
        super().__init__(parent)
        self.namespace = namespace
        self.closed = threading.Event()
        self.ros = None
        self.zero_pub = None

    def start(self):
        threading.Thread(target=self._connect, daemon=True).start()

    def _connect(self):
        try:
            import rospy
            from geometry_msgs.msg import Twist
            from std_msgs.msg import String
            self.ros = rospy
            self.Twist = Twist
            # Set this before init_node, including when the GUI starts first.
            rospy.set_param("/use_sim_time", True)
            rospy.init_node("sim_exp_gui", anonymous=True, disable_signals=True)
            if self.closed.is_set():
                return
            self.zero_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
            self.state_sub = rospy.Subscriber(self.namespace + "/state", String,
                                              self._state_callback, queue_size=1)
            self.message.emit("ROS 已连接，等待 master 端障碍物程序。")
        except Exception as exc:
            if not self.closed.is_set():
                self.error.emit("ROS 连接失败：{}。请在已 source ROS/工作区的终端启动。".format(exc))

    def _state_callback(self, message):
        try:
            state = json.loads(message.data)
            if not isinstance(state, dict) or state.get("protocol") != 1:
                return
            if not self.closed.is_set():
                self.state_received.emit(state)
        except (ValueError, TypeError):
            pass

    def cancel(self, run_id):
        if self.ros is None:
            return
        def stop():
            try:
                from std_srvs.srv import Trigger
                # Only used while stopping our own child, never while idle.
                if self.zero_pub is not None:
                    self.zero_pub.publish(self.Twist())
                if run_id:
                    name = "{}/runs/{}/cancel".format(self.namespace, run_id)
                    self.ros.wait_for_service(name, timeout=2.0)
                    response = self.ros.ServiceProxy(name, Trigger)()
                    if not response.success:
                        raise RuntimeError(response.message)
            except Exception as exc:
                if not self.closed.is_set():
                    self.message.emit("远端取消未确认：{}；心跳超时仍会停止障碍物。".format(exc))
        threading.Thread(target=stop, daemon=True).start()

    def close(self):
        self.closed.set()
        if self.ros is not None:
            # rospy shutdown hooks can wait for transport threads.
            threading.Thread(target=lambda: self.ros.signal_shutdown("GUI closed"),
                             daemon=True).start()

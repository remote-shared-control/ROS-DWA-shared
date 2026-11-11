"""Native Qt interface; all widgets and QProcess operations stay on the UI thread."""

import os
from pathlib import Path
import signal
import time
import uuid

from qt_compat import QtCore, QtGui, QtWidgets

from experiment_protocol import ACTIVE, NSEC, TERMINAL, MAX_OBSTACLES, MAX_DURATION
from gui_ros_bridge import RosBridge
from gui_support import (belongs_to_launch, controller_command, controller_names,
                         state_is_fresh)


PHASE_LABELS = {"IDLE": "等待启动", "PREPARING": "正在准备障碍物场景", "PREPARED": "场景已准备",
                "ARMED": "等待共同起点", "RUNNING": "实验运行中",
                "FINISHED": "实验完成", "CANCELLED": "实验已停止", "ABORTED": "实验中止"}


class ExperimentWindow(QtWidgets.QMainWindow):
    def __init__(self, namespace="/sim_exp", connect_ros=True, directory=None,
                 settings=None, bridge=None, parent=None):
        super().__init__(parent)
        self.directory = Path(directory or Path(__file__).resolve().parent)
        self.namespace = namespace
        self.settings = settings or QtCore.QSettings("ROS-DWA", "SimExperiment")
        self.bridge = bridge or RosBridge(namespace, self)
        self.preview = not connect_ros
        self.remote = None
        self.received_at = None
        self.process = None
        self.launch_id = ""
        self.run_id = ""
        self.child_pid = 0
        self.stopping = False
        self.closing = False
        self.connection_error = ""
        self.setWindowTitle("DWA 实验控制台")
        self.resize(920, 780)
        self.setMinimumSize(760, 700)
        self._build_ui()
        self.bridge.state_received.connect(self._on_state)
        self.bridge.message.connect(self._log)
        self.bridge.error.connect(self._on_error)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(200)
        self.stop_timer = QtCore.QTimer(self)
        self.stop_timer.setSingleShot(True)
        self.stop_timer.timeout.connect(self._force_stop_current)
        if connect_ros:
            self.bridge.start()
        else:
            self._log("界面预览模式：未连接 ROS，启动按钮已禁用。")
        self._refresh()

    def _build_ui(self):
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(26, 22, 26, 22)
        layout.setSpacing(16)
        title_row = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("DWA 实验控制台")
        title.setStyleSheet("font-size: 24px; font-weight: 600; color: #162c40;")
        title_row.addWidget(title)
        title_row.addStretch()
        self.connection = QtWidgets.QLabel("等待连接")
        self.connection.setStyleSheet("color: #687889;")
        title_row.addWidget(self.connection)
        layout.addLayout(title_row)
        subtitle = QtWidgets.QLabel("选择算法与场景种子，一键启动同步避障实验。")
        subtitle.setStyleSheet("color: #667789;")
        layout.addWidget(subtitle)

        config_group = QtWidgets.QGroupBox("实验设置")
        form = QtWidgets.QFormLayout(config_group)
        form.setContentsMargins(18, 22, 18, 18)
        form.setVerticalSpacing(14)
        self.seed = QtWidgets.QSpinBox()
        self.seed.setRange(0, 2**31 - 1)
        self.seed.setValue(self.settings.value("seed", 42, type=int))
        self.seed.setMinimumHeight(34)
        self.seed.setToolTip("相同 seed、数量和时间限制可复现障碍物场景。")
        form.addRow("场景种子 seed", self.seed)
        self.num_obstacles = QtWidgets.QSpinBox()
        self.num_obstacles.setRange(0, MAX_OBSTACLES)
        self.num_obstacles.setValue(self.settings.value("num_obstacles", 40, type=int))
        self.num_obstacles.setMinimumHeight(34)
        self.num_obstacles.setToolTip("减少数量会删除多余模型；0 表示移除全部动态障碍物。")
        form.addRow("障碍物数量", self.num_obstacles)
        self.duration = QtWidgets.QDoubleSpinBox()
        self.duration.setRange(0.1, MAX_DURATION)
        self.duration.setDecimals(1)
        self.duration.setSuffix(" s")
        self.duration.setValue(self.settings.value("duration", 70.0, type=float))
        self.duration.setMinimumHeight(34)
        self.duration.setToolTip("从共同起点 t0 开始计算的 ROS 仿真时间。")
        form.addRow("时间限制", self.duration)
        self.controller = QtWidgets.QComboBox()
        self.controller.addItems(controller_names(self.directory))
        saved = self.settings.value("controller", "dwa_simwaffi_new_midpoint_hybrid.py")
        index = self.controller.findText(saved)
        if index >= 0:
            self.controller.setCurrentIndex(index)
        self.controller.setMinimumHeight(34)
        form.addRow("DWA 程序", self.controller)
        note = QtWidgets.QLabel("保持 seed、数量和时限一致，可对比不同 DWA 程序。")
        note.setStyleSheet("color: #667789;")
        form.addRow("", note)
        layout.addWidget(config_group)

        buttons = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("启动实验")
        self.start_button.setMinimumHeight(42)
        self.start_button.setStyleSheet(
            "QPushButton {background: #1565c0; color: white; border: 0; border-radius: 5px;"
            "font-weight: 600; padding: 8px 24px;}"
            "QPushButton:disabled {background: #b7c6d6; color: #f3f6f9;}")
        self.start_button.clicked.connect(self.start_experiment)
        self.stop_button = QtWidgets.QPushButton("停止实验")
        self.stop_button.setMinimumHeight(42)
        self.stop_button.clicked.connect(lambda: self.stop_experiment("手动停止实验"))
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        buttons.addStretch()
        self.duration_label = QtWidgets.QLabel("70 s · 仿真时间")
        self.duration_label.setStyleSheet("color: #667789;")
        buttons.addWidget(self.duration_label)
        layout.addLayout(buttons)

        status_group = QtWidgets.QGroupBox("实验状态")
        status_layout = QtWidgets.QVBoxLayout(status_group)
        self.phase = QtWidgets.QLabel("等待 master 端障碍物程序")
        self.phase.setStyleSheet("font-weight: 600; font-size: 15px;")
        self.detail = QtWidgets.QLabel("master 端运行所选场景脚本。")
        self.detail.setStyleSheet("color: #667789;")
        self.detail.setWordWrap(True)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setFormat("0.0 / {:g} s".format(self.duration.value()))
        self.progress.setMinimumHeight(24)
        status_layout.addWidget(self.phase)
        status_layout.addWidget(self.detail)
        status_layout.addWidget(self.progress)
        layout.addWidget(status_group)

        layout.addWidget(QtWidgets.QLabel("运行日志"))
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1500)
        self.log.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        self.log.setMinimumHeight(135)
        layout.addWidget(self.log, 1)
        master = os.environ.get("ROS_MASTER_URI", "http://localhost:11311")
        self.statusBar().showMessage("ROS Master: " + master)

    def _log(self, message):
        self.log.appendPlainText(message)

    def _on_error(self, message):
        self.connection_error = message
        self._log(message)
        self._refresh()

    def _on_state(self, state):
        self.remote, self.received_at = state, time.monotonic()
        self.connection_error = ""
        if belongs_to_launch(state, self.launch_id):
            self.run_id = state.get("run_id", "")
            if state.get("phase") in TERMINAL and self._process_running() and not self.stopping:
                self._log(PHASE_LABELS.get(state["phase"], state["phase"]))
                self._stop_local()
        self._refresh()

    def _process_running(self):
        return self.process is not None and self.process.state() != QtCore.QProcess.NotRunning

    def _refresh(self):
        fresh = state_is_fresh(self.remote, self.received_at, time.monotonic())
        supported = fresh and {"prepare_seed", "scene_settings"}.issubset(
            self.remote.get("capabilities", []))
        running = self._process_running()
        remote_phase = self.remote.get("phase") if fresh else None
        can_start = (supported and not self.preview and not running and
                     remote_phase not in ACTIVE and self.controller.count() > 0 and not self.closing)
        self.start_button.setEnabled(bool(can_start))
        self.stop_button.setEnabled(running and not self.stopping)
        self.seed.setEnabled(not running)
        self.num_obstacles.setEnabled(not running)
        self.duration.setEnabled(not running)
        self.controller.setEnabled(not running)
        self.duration_label.setText("{:g} s · 仿真时间".format(self.duration.value()))
        if self.preview:
            self.connection.setText("界面预览 · 未连接 ROS")
        elif not fresh:
            self.connection.setText("未连接 / 等待 master")
        elif not supported:
            self.connection.setText("master 脚本需要更新")
        else:
            self.connection.setText("● master 已连接")

        own = belongs_to_launch(self.remote, self.launch_id)
        if self.stopping and running:
            self.phase.setText("正在停止并清理 DWA…")
        elif running and not own:
            self.phase.setText("正在启动 DWA，等待场景准备…")
        elif fresh:
            label = PHASE_LABELS.get(remote_phase, remote_phase or "等待启动")
            if remote_phase in ACTIVE and not own:
                label = "其他实验正在运行"
            self.phase.setText(label)
        else:
            self.phase.setText("等待 master 端障碍物程序")
        if self.connection_error:
            self.detail.setText(self.connection_error)
        elif running and not own:
            self.progress.setValue(0)
            self.progress.setFormat("0.0 / {:g} s".format(self.duration.value()))
            self.detail.setText("正在准备 {}，seed={}，障碍物 {} 个，时限 {:g} s。".format(
                self.controller.currentText(), self.seed.value(),
                self.num_obstacles.value(), self.duration.value()))
        elif fresh:
            state = self.remote
            duration = state.get("duration_ns", 70 * NSEC) / NSEC
            elapsed = state.get("elapsed_ns", 0) / NSEC
            self.progress.setValue(round(1000 * min(elapsed / max(duration, 0.001), 1.0)))
            self.progress.setFormat("{:.1f} / {:g} s".format(elapsed, duration))
            text = "master seed：{}".format(state.get("seed", "—"))
            text += "  ·  障碍物：{}".format(state.get("num_obstacles", "—"))
            if state.get("run_id"):
                text += "  ·  实验：" + state["run_id"][:10]
            if remote_phase == "ARMED":
                seconds = max(0, (state.get("t0_ns", 0) - state.get("sim_time_ns", 0)) / NSEC)
                text += "  ·  距离开始 {:.1f} s".format(seconds)
            if state.get("reason"):
                text += "  ·  " + state["reason"]
            self.detail.setText(text)
        else:
            self.detail.setText("master 端运行所选的障碍物场景脚本；请确认两端 ROS 网络连接。")
        if running and self.received_at is not None and not fresh and not self.stopping:
            self.stop_experiment("与 master 的状态连接超时，停止本次实验")

    def start_experiment(self):
        if not self.start_button.isEnabled():
            return
        self.launch_id = uuid.uuid4().hex
        self.run_id = ""
        self.stopping = False
        self.child_pid = 0
        command = controller_command(self.directory, self.controller.currentText(),
                                     self.seed.value(), self.launch_id, self.namespace,
                                     num_obstacles=self.num_obstacles.value(),
                                     duration=self.duration.value())
        self.settings.setValue("seed", self.seed.value())
        self.settings.setValue("num_obstacles", self.num_obstacles.value())
        self.settings.setValue("duration", self.duration.value())
        self.settings.setValue("controller", self.controller.currentText())
        self._log("启动 {}，seed={}，障碍物={}，时间限制={:g} s".format(
            self.controller.currentText(), self.seed.value(),
            self.num_obstacles.value(), self.duration.value()))
        process = QtCore.QProcess(self)
        self.process = process
        process.setWorkingDirectory(str(self.directory))
        environment = QtCore.QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        process.setProcessEnvironment(environment)
        process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(lambda: self._read_output(process))
        process.started.connect(lambda: self._started(process))
        process.finished.connect(lambda code, status: self._finished(process, code, status))
        process.errorOccurred.connect(lambda error: self._process_error(process, error))
        process.start(command[0], command[1:])
        self._refresh()

    def _started(self, process):
        if process is self.process:
            self.child_pid = int(process.processId())
            if self.stopping:
                process.terminate()

    def _read_output(self, process):
        output = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if output:
            self.log.moveCursor(QtGui.QTextCursor.End)
            self.log.insertPlainText(output)
            self.log.moveCursor(QtGui.QTextCursor.End)

    def stop_experiment(self, reason):
        if not self._process_running() or self.stopping:
            return
        self._log(reason)
        self.bridge.cancel(self.run_id)
        self._stop_local()

    def _stop_local(self):
        self.stopping = True
        process = self.process
        if process is not None:
            process.terminate()  # rospy handles SIGTERM and cancels its own run.
            self.stop_timer.start(5000)
        self._refresh()

    def _kill_group(self, pid):
        if os.name == "posix" and pid > 0:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as exc:
                self._log("清理子进程组失败：" + str(exc))

    def _force_stop(self, process):
        if process is self.process and self._process_running():
            self._log("DWA 未及时退出，正在强制清理本次子进程。")
            self._kill_group(self.child_pid)
            process.kill()

    def _force_stop_current(self):
        if self.process is not None:
            self._force_stop(self.process)

    def _process_error(self, process, error):
        if process is self.process:
            if self.stopping and error == QtCore.QProcess.Crashed:
                return  # Expected after an intentional signal/forced stop.
            self._log("DWA 进程错误：" + process.errorString())
            if error == QtCore.QProcess.FailedToStart:
                self._finished(process, -1, QtCore.QProcess.CrashExit)

    def _finished(self, process, code, _status):
        if process is not self.process:
            return
        self.stop_timer.stop()
        self._read_output(process)
        if self.run_id and belongs_to_launch(self.remote, self.launch_id):
            if self.remote.get("phase") in ACTIVE:
                self.bridge.cancel(self.run_id)
        self._kill_group(self.child_pid)
        self._log("本次 DWA 已退出（退出码 {}），可以启动下一次实验。".format(code))
        self.process = None
        self.child_pid = 0
        self.stopping = False
        process.deleteLater()
        if self.closing:
            self.close()
        else:
            self._refresh()

    def closeEvent(self, event):
        if self._process_running():
            event.ignore()
            self.closing = True
            self.stop_experiment("关闭 GUI，停止本次实验")
            return
        self.timer.stop()
        self.stop_timer.stop()
        self.bridge.close()
        self.settings.sync()
        event.accept()

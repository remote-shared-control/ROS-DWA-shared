"""Exercise real Qt widgets and local QProcess children without robot access."""

import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIRECTORY))
try:
    from qt_compat import QtCore, QtWidgets
    from experiment_gui_window import ExperimentWindow
except ImportError:
    QtCore = QtWidgets = None


if QtCore is not None:
    class FakeBridge(QtCore.QObject):
        state_received = QtCore.pyqtSignal(dict)
        message = QtCore.pyqtSignal(str)
        error = QtCore.pyqtSignal(str)

        def __init__(self):
            super().__init__()
            self.cancelled, self.closed = [], False

        def start(self):
            pass

        def cancel(self, run_id):
            self.cancelled.append(run_id)

        def close(self):
            self.closed = True


@unittest.skipIf(QtWidgets is None, "Install PyQt5 to run the offscreen GUI tests")
class QtGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.app.setStyle("Fusion")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="DWA Qt test ")
        self.directory = Path(self.temporary.name)
        shutil.copyfile(DIRECTORY / "run_experiment.py", self.directory / "run_experiment.py")
        (self.directory / "dwa_test.py").write_text(
            "import sys,time\nprint(repr(sys.argv),flush=True)\ntime.sleep(60)\n")
        self.bridge = FakeBridge()
        self.window = ExperimentWindow(
            directory=self.directory, bridge=self.bridge,
            settings=QtCore.QSettings(str(self.directory / "settings.ini"), QtCore.QSettings.IniFormat))
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        if self.window._process_running():
            self.window.stop_experiment("test cleanup")
            self.until(lambda: not self.window._process_running())
        self.window.close()
        self.window.deleteLater()
        self.bridge.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        self.app.processEvents()
        self.temporary.cleanup()

    def until(self, predicate, timeout=3.0):
        deadline = time.monotonic() + timeout
        while not predicate():
            self.app.processEvents()
            if time.monotonic() > deadline:
                self.fail("Timed out waiting for Qt child/process event")
            time.sleep(0.002)
        self.app.processEvents()

    def remote(self, **fields):
        state = {"protocol": 1, "capabilities": ["prepare_seed"], "phase": "IDLE",
                 "launch_id": "", "seed": 42, "duration_ns": 70_000_000_000,
                 "elapsed_ns": 0, "run_id": "", "reason": ""}
        state.update(fields)
        self.bridge.state_received.emit(state)
        self.app.processEvents()

    def start(self, seed=17):
        self.remote()
        self.window.seed.setValue(seed)
        self.window.start_button.click()
        self.until(lambda: self.window.child_pid > 0)
        self.until(lambda: "_experiment_seed:=" in self.window.log.toPlainText())
        return self.window.process

    def test_start_is_blocked_until_fresh_seed_capable_remote_is_available(self):
        self.assertFalse(self.window.start_button.isEnabled())
        self.remote(capabilities=[])
        self.assertFalse(self.window.start_button.isEnabled())
        self.remote()
        self.assertTrue(self.window.start_button.isEnabled())
        self.window.received_at = time.monotonic() - 10
        self.window._refresh()
        self.assertFalse(self.window.start_button.isEnabled())

    def test_selected_seed_launches_real_child_and_old_finish_does_not_stop_it(self):
        process = self.start(seed=123)
        self.assertIn("_experiment_seed:=123", self.window.log.toPlainText())
        self.assertFalse(self.window.start_button.isEnabled())
        self.assertEqual(os.getpgid(self.window.child_pid), self.window.child_pid)
        self.window.start_experiment()
        self.assertIs(self.window.process, process)
        self.remote(phase="FINISHED", launch_id="f" * 32, run_id="old")
        self.assertTrue(self.window._process_running())
        self.assertEqual(self.window.settings.value("seed", type=int), 123)

    def test_matching_70s_finish_cleans_up_and_allows_next_run(self):
        self.start()
        token = self.window.launch_id
        self.remote(phase="FINISHED", launch_id=token, run_id="a" * 32,
                    elapsed_ns=70_000_000_000)
        self.until(lambda: not self.window._process_running())
        self.assertEqual(self.window.progress.value(), 1000)
        self.assertTrue(self.window.start_button.isEnabled())
        self.window.start_button.click()
        self.until(lambda: self.window.child_pid > 0)
        self.assertNotEqual(self.window.launch_id, token)

    def test_stop_cancels_only_our_remote_run_and_terminates_the_child(self):
        self.start()
        self.remote(phase="RUNNING", launch_id=self.window.launch_id,
                    run_id="b" * 32, elapsed_ns=5_000_000_000)
        self.window.stop_button.click()
        self.until(lambda: not self.window._process_running())
        self.assertTrue(self.bridge.cancelled)
        self.assertTrue(all(run_id == "b" * 32 for run_id in self.bridge.cancelled))

    def test_closing_window_stops_active_child_before_closing(self):
        self.start()
        self.remote(phase="RUNNING", launch_id=self.window.launch_id, run_id="c" * 32)
        self.window.close()
        self.until(lambda: not self.window._process_running() and not self.window.isVisible())
        self.assertTrue(self.bridge.closed)
        self.assertIn("c" * 32, self.bridge.cancelled)

    def test_connection_loss_stops_the_owned_child(self):
        self.start()
        self.remote(phase="RUNNING", launch_id=self.window.launch_id, run_id="d" * 32)
        self.window.received_at = time.monotonic() - 10
        self.window._refresh()
        self.until(lambda: not self.window._process_running())
        self.assertIn("d" * 32, self.bridge.cancelled)

    def test_unresponsive_child_can_be_force_stopped_without_a_late_timer(self):
        (self.directory / "dwa_test.py").write_text(
            "import sys,time,signal\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
            "print(repr(sys.argv),flush=True)\ntime.sleep(60)\n")
        self.start()
        self.remote(phase="RUNNING", launch_id=self.window.launch_id, run_id="e" * 32)
        self.window.stop_button.click()
        self.assertTrue(self.window._process_running())
        self.window._force_stop_current()
        self.until(lambda: not self.window._process_running())
        self.assertFalse(self.window.stop_timer.isActive())


if __name__ == "__main__":
    unittest.main()

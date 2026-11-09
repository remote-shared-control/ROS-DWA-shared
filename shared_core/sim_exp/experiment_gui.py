#!/usr/bin/env python3
"""Ubuntu/PyQt5 launcher for the robot-side synchronized DWA experiments."""

import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="/sim_exp")
    parser.add_argument("--preview", action="store_true",
                        help="Preview the window without ROS or robot control")
    args, unknown = parser.parse_known_args(argv)
    if any(":=" not in value for value in unknown):
        parser.error("Unrecognized arguments: " + " ".join(unknown))
    try:
        from qt_compat import QtWidgets
        from experiment_gui_window import ExperimentWindow
        from gui_support import normalize_namespace
    except ImportError as exc:
        print("GUI dependency unavailable: {}\n"
              "Ubuntu: sudo apt install python3-pyqt5".format(exc), file=sys.stderr)
        return 1
    try:
        namespace = normalize_namespace(args.namespace)
    except ValueError as exc:
        parser.error(str(exc))
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("DWA Experiment Console")
    app.setOrganizationName("ROS-DWA")
    app.setStyle("Fusion")
    window = ExperimentWindow(namespace=namespace, connect_ros=not args.preview)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())

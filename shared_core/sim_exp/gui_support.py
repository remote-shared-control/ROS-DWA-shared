"""Pure helpers used by the Qt launcher and its tests."""

import json
from pathlib import Path
import re
import sys

from experiment_protocol import validate_seed, validate_obstacle_count, validate_duration


def controller_names(directory):
    directory = Path(directory).resolve()
    names = [path.name for path in directory.glob("dwa*.py")
             if path.is_file() and path.resolve().parent == directory]
    if (directory / "direct.py").is_file():
        names.append("direct.py")
    return sorted(names)


def normalize_namespace(namespace):
    namespace = namespace.rstrip("/")
    if not re.fullmatch(r"/[A-Za-z_][A-Za-z0-9_/]*", namespace):
        raise ValueError("ROS namespace must look like /sim_exp")
    return namespace


def controller_command(directory, controller, seed, launch_id, namespace,
                       python=None, num_obstacles=40, duration=70.0):
    directory = Path(directory).resolve()
    if controller not in controller_names(directory):
        raise ValueError("Choose a controller from sim_exp")
    validate_seed(seed)
    validate_obstacle_count(num_obstacles)
    duration = validate_duration(duration)
    if not re.fullmatch(r"[0-9a-f]{32}", launch_id):
        raise ValueError("Invalid launch ID")
    namespace = normalize_namespace(namespace)
    return [
        python or sys.executable, str(directory / "run_experiment.py"),
        "--new-session", "--controller", controller,
        "__name:=shared_dwa", "_experiment_seed:=" + str(seed),
        "_experiment_num_obstacles:=" + str(num_obstacles),
        "_experiment_duration:=" + repr(duration),
        "_experiment_launch_id:=" + json.dumps(launch_id),
        "_experiment_controller:=" + json.dumps(controller),
        "_experiment_namespace:=" + namespace,
    ]


def state_is_fresh(state, received_at, now, timeout=3.0):
    return bool(state and received_at is not None and now - received_at <= timeout)


def belongs_to_launch(state, launch_id):
    return bool(launch_id and state and state.get("launch_id") == launch_id)

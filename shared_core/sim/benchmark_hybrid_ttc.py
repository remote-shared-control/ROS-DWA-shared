#!/usr/bin/env python3
"""Reproducible, ROS-free benchmark of the production hybrid DWA cycle.

Run from any directory with a Python interpreter that has NumPy installed::

    python3 shared_core/sim/benchmark_hybrid_ttc.py --repeats 7 --profile
    python3 shared_core/sim/benchmark_hybrid_ttc.py --baseline-ref <commit>

The benchmark loads the actual controller functions through AST, replacing only
ROS message/Rate construction. It times dynamic-window sampling, candidate
evaluation and both trajectory previews. ROS callbacks, publishing and Gazebo
are outside these measurements; measured Hz is a compute ceiling, not a claim
about the complete ROS node. Fixtures include nearby crossing rectangles and
dense static point clouds, rather than only distant, easily rejected objects.
"""

from __future__ import annotations

import argparse
import ast
from bisect import bisect_left
import cProfile
import io
import json
import math
from pathlib import Path
import pstats
import statistics
import subprocess
from time import perf_counter
from types import SimpleNamespace

import numpy as np


SIM_DIR = Path(__file__).resolve().parent
CONTROLLER = "dwa_simwaffi_new_midpoint_hybrid.py"
TTC_MODULE = "cal_ctime_midpoint.py"


class Twist:
    def __init__(self):
        self.linear = SimpleNamespace(x=0.0, y=0.0, z=0.0)
        self.angular = SimpleNamespace(x=0.0, y=0.0, z=0.0)


def _source(filename, ref):
    if ref is None:
        return (SIM_DIR / filename).read_text(encoding="utf-8")
    return subprocess.check_output(
        ["git", "show", f"{ref}:shared_core/sim/{filename}"],
        cwd=SIM_DIR, text=True,
    )


def load_controller(ref=None):
    """Execute production computational code without importing the ROS node."""
    ttc = {"__name__": "benchmark_ttc", "__file__": str(SIM_DIR / TTC_MODULE)}
    exec(compile(_source(TTC_MODULE, ref), TTC_MODULE, "exec"), ttc)
    namespace = {
        "__name__": "benchmark_controller",
        "np": np, "math": math, "bisect_left": bisect_left,
        "Twist": Twist, "rospy": SimpleNamespace(Rate=lambda hz: None),
    }
    tree = ast.parse(_source(CONTROLLER, ref), CONTROLLER)
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in ("Config", "TrajectoryPreview"):
            nodes.append(node)
        elif isinstance(node, ast.ImportFrom) and node.module == "__future__":
            nodes.insert(0, node)
        elif isinstance(node, ast.ImportFrom) and node.module == "cal_ctime_midpoint":
            for alias in node.names:
                namespace[alias.asname or alias.name] = ttc[alias.name]
        elif isinstance(node, ast.Assign):
            # Allow pure numeric constants added by computational optimizations.
            if all(isinstance(target, ast.Name) and target.id.isupper() for target in node.targets):
                try:
                    ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    continue
                nodes.append(node)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), CONTROLLER, "exec"), namespace)
    namespace["robot_trajectory"] = namespace["TrajectoryPreview"]()
    namespace["human_trajectory"] = namespace["TrajectoryPreview"]()
    calls = {"cal_time": 0, "static_feasible": 0}
    original = namespace["cal_time"]

    def counted_cal_time(*args, **kwargs):
        calls["cal_time"] += 1
        return original(*args, **kwargs)

    namespace["cal_time"] = counted_cal_time
    original_static_cost = namespace["calc_obstacle_cost"]

    def counted_static_cost(traj, obstacles, config):
        cost = original_static_cost(traj, obstacles, config)
        if not np.isinf(cost * config.obs_cost_gain):
            calls["static_feasible"] += 1
        return cost

    namespace["calc_obstacle_cost"] = counted_static_cost
    return namespace, calls


def _rectangle(identifier, x, y, vx, vy):
    # CubeReader supplies vertices already inflated by the robot radius.
    half = 0.16 + 0.21
    return [identifier, [x - half, y - half], [x + half, y - half],
            [x + half, y + half], [x - half, y + half], vx, vy]


def scenarios():
    empty = (np.empty(0), np.empty(0))
    crossing = [
        _rectangle(0, 1.3, -0.9, -0.05, 0.45),
        _rectangle(1, 2.3, 1.0, -0.15, -0.30),
        _rectangle(2, -0.9, 1.2, 0.05, -0.35),
        _rectangle(3, -1.4, -0.8, 0.10, 0.20),
    ]
    wall_x = np.linspace(-0.8, 3.0, 360)
    corridor = (np.concatenate((wall_x, wall_x)),
                np.concatenate((np.full_like(wall_x, 0.8), np.full_like(wall_x, -0.8))))
    rng = np.random.default_rng(20261027)
    cloud = np.column_stack((rng.uniform(-2.0, 3.5, 1200), rng.uniform(-2.3, 2.3, 1200)))
    # Leave the initial footprint clear, while retaining obstacles in candidate paths.
    cloud = cloud[np.hypot(cloud[:, 0], cloud[:, 1]) > 0.55]
    crowded = crossing + [
        _rectangle(4 + i, 1.0 + 0.45 * (i % 4),
                   (1.5 if i < 4 else -1.5),
                   -0.05 * (i % 3), (-0.15 if i < 4 else 0.15))
        for i in range(8)
    ]
    yield "clear_0", empty, []
    yield "crossing_4_corridor", corridor, crossing
    yield "crowded_12_cloud", (cloud[:, 0], cloud[:, 1]), crowded
    # A wide wall sweeps across every candidate before movetimemin. Include
    # distant rectangles to exercise the all-blocked case without relying on
    # an early winner to avoid TTC work for the remaining candidates.
    sweeping_wall = [100, [3.0, -10.0], [3.3, -10.0],
                     [3.3, 10.0], [3.0, 10.0], -2.0, 0.0]
    distant = [_rectangle(101 + i, 15.0 + i, (-1.0 if i % 2 else 1.0) * 12.0,
                          -0.1, 0.1) for i in range(15)]
    yield "sweep_16_all_blocked", empty, [sweeping_wall] + distant


def run_cycle(namespace, calls, static_points, rectangles):
    config = namespace["Config"]()
    config.human.linear.x = 0.8
    config.human.angular.z = 0.12
    state = [0.0, 0.0, 0.0, 0.35, 0.15]
    cube = SimpleNamespace(data=rectangles)
    vs, ws = namespace["calc_dynamic_window"](state, config)
    calls["cal_time"] = 0
    calls["static_feasible"] = 0
    started = perf_counter()
    command = namespace["dwa_control"](
        state, [0.0, 0.0], config, static_points, cube, None, None,
    )
    elapsed = perf_counter() - started
    preview = namespace["robot_trajectory"].trajectory
    human_preview = namespace["human_trajectory"].trajectory
    return {
        "elapsed_ms": elapsed * 1000.0,
        "candidates": len(vs) * len(ws),
        "cal_time_calls": calls["cal_time"],
        "static_feasible": calls["static_feasible"],
        "command": np.asarray(command).tolist(),
        "preview": None if preview is None else preview.copy(),
        "human_preview": None if human_preview is None else human_preview.copy(),
    }


def benchmark(ref, repeats, profile):
    namespace, calls = load_controller(ref)
    rows, outputs = [], {}
    profiler = cProfile.Profile() if profile else None
    for name, static_points, rectangles in scenarios():
        run_cycle(namespace, calls, static_points, rectangles)  # warm caches
        samples = [run_cycle(namespace, calls, static_points, rectangles)
                   for _ in range(repeats)]
        last = samples[-1]
        if ref is None and last["cal_time_calls"] != last["static_feasible"]:
            raise AssertionError(f"{name}: a statically feasible candidate skipped TTC")
        median_ms = statistics.median(row["elapsed_ms"] for row in samples)
        rows.append({
            "scenario": name, "static_points": len(static_points[0]),
            "rectangles": len(rectangles), "candidates": last["candidates"],
            "cal_time_calls": last["cal_time_calls"],
            "median_ms": round(median_ms, 3), "compute_hz": round(1000.0 / median_ms, 2),
            "command": last["command"],
        })
        outputs[name] = last
        if profiler is not None:
            profiler.runcall(run_cycle, namespace, calls, static_points, rectangles)
    if profiler is not None:
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(25)
        print(stream.getvalue())
    return rows, outputs


def compare_outputs(baseline, current):
    for name in baseline:
        for field in ("command", "preview", "human_preview"):
            left, right = baseline[name][field], current[name][field]
            if left is None or right is None:
                if left is not right:
                    raise AssertionError(f"{name}: {field} differs (None vs trajectory)")
            else:
                np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12,
                                           err_msg=f"{name}: {field}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--baseline-ref", help="Git revision of both production files")
    parser.add_argument("--profile", action="store_true", help="Print one-cycle cProfile summary per fixture")
    parser.add_argument("--json", type=Path, help="Also save the measurements as JSON")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    results = {}
    baseline_outputs = None
    if args.baseline_ref:
        results[args.baseline_ref], baseline_outputs = benchmark(args.baseline_ref, args.repeats, False)
    results["working_tree"], outputs = benchmark(None, args.repeats, args.profile)
    if baseline_outputs is not None:
        compare_outputs(baseline_outputs, outputs)
        print("Selected v/w and robot/human preview trajectories match the baseline (1e-12 tolerance).")
    for label, rows in results.items():
        print(f"\n{label}:")
        for row in rows:
            print(f"  {row['scenario']:<23} {row['median_ms']:>9.3f} ms  "
                  f"{row['compute_hz']:>7.2f} Hz  candidates={row['candidates']}  "
                  f"cal_time={row['cal_time_calls']}  v/w={row['command']}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""ROS-free benchmark of the constant-candidate DWA controllers.

    python3 shared_core/sim/benchmark_midpoint_ttc.py --repeats 7 --profile
    python3 shared_core/sim/benchmark_midpoint_ttc.py --baseline-ref HEAD
    python3 shared_core/sim/benchmark_midpoint_ttc.py --controller dwa_simwaffi_new.py \\
        --baseline-dir /tmp/saved-sim-sources

The production controller is loaded through AST, with only ROS construction
replaced. Fixtures and preview comparisons are shared with the hybrid benchmark.
Each statically feasible candidate must call TTC exactly once, including when
all candidates are blocked or a better-scoring candidate has already been found.
Reported Hz measures computation only; it excludes ROS callbacks and publishing.

The legacy solver misses some early and tangent collisions. Baseline output
differences are reported for review rather than asserted to be equivalent.
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
from types import SimpleNamespace

import numpy as np

from benchmark_hybrid_ttc import (
    SIM_DIR, Twist, compare_outputs, run_cycle as run_shared_cycle, scenarios,
)


CONTROLLER = "dwa_simwaffi_new_midpoint.py"


def _source(filename, ref, baseline_dir=None):
    if baseline_dir is not None:
        path = Path(baseline_dir) / filename
        if not path.is_file():
            path = Path(baseline_dir) / "shared_core" / "sim" / filename
        return path.read_text(encoding="utf-8")
    if ref is None:
        return (SIM_DIR / filename).read_text(encoding="utf-8")
    return subprocess.check_output(
        ["git", "show", f"{ref}:shared_core/sim/{filename}"],
        cwd=SIM_DIR, text=True,
    )


def _load_legacy_ttc(ref, baseline_dir=None):
    """Execute computational functions without cal_ctime's ROS imports."""
    namespace = {
        "__name__": "benchmark_legacy_ttc", "np": np, "math": math,
        "sin": math.sin, "cos": math.cos,
    }
    tree = ast.parse(_source("cal_ctime.py", ref, baseline_dir), "cal_ctime.py")
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    exec(compile(ast.Module(body=functions, type_ignores=[]),
                 "cal_ctime.py", "exec"), namespace)
    return namespace


def load_controller(ref=None, controller=CONTROLLER, baseline_dir=None):
    """Load either legacy cal_time or the production cal_time_constant alias."""
    namespace = {
        "__name__": "benchmark_midpoint_controller",
        "np": np, "math": math, "bisect_left": bisect_left,
        "Twist": Twist, "rospy": SimpleNamespace(Rate=lambda hz: None),
    }
    tree = ast.parse(_source(controller, ref, baseline_dir), controller)
    nodes, ttc_modules = [], {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in ("Config", "TrajectoryPreview"):
            nodes.append(node)
        elif isinstance(node, ast.ImportFrom) and node.module == "__future__":
            nodes.insert(0, node)
        elif isinstance(node, ast.ImportFrom) and node.module in ("cal_ctime", "cal_ctime_midpoint"):
            if node.module not in ttc_modules:
                if node.module == "cal_ctime":
                    ttc_modules[node.module] = _load_legacy_ttc(ref, baseline_dir)
                else:
                    module = {"__name__": "benchmark_midpoint_ttc",
                              "__file__": str(SIM_DIR / "cal_ctime_midpoint.py")}
                    exec(compile(_source("cal_ctime_midpoint.py", ref, baseline_dir),
                                 "cal_ctime_midpoint.py", "exec"), module)
                    ttc_modules[node.module] = module
            module = ttc_modules[node.module]
            for alias in node.names:
                if node.module == "cal_ctime" and alias.name == "CubeReader":
                    # Fixtures replace the ROS subscriber with a data snapshot.
                    continue
                if alias.name == "*":
                    namespace.update({key: value for key, value in module.items()
                                      if not key.startswith("_")})
                else:
                    namespace[alias.asname or alias.name] = module[alias.name]
        elif isinstance(node, ast.Assign):
            if all(isinstance(target, ast.Name) and target.id.isupper()
                   for target in node.targets):
                try:
                    ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    continue
                nodes.append(node)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), controller, "exec"), namespace)
    namespace["robot_trajectory"] = namespace["TrajectoryPreview"]()
    namespace["human_trajectory"] = namespace["TrajectoryPreview"]()
    counters = {"cal_time": 0, "static_feasible": 0, "static_examined": 0,
                "ttc_rejected": 0}
    tracking = {"pending_config": None, "candidates": set()}
    original_ttc = namespace["cal_time"]
    original_static = namespace["calc_obstacle_cost"]
    distinct_clear_value = original_ttc(
        SimpleNamespace(data=[]), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ) > 1.0

    def counted_static_cost(traj, obstacles, config):
        if tracking["pending_config"] is not None:
            raise AssertionError("A statically feasible candidate skipped TTC")
        counters["static_examined"] += 1
        cost = original_static(traj, obstacles, config)
        if not np.isinf(cost * config.obs_cost_gain):
            counters["static_feasible"] += 1
            tracking["pending_config"] = config
        return cost

    def counted_cal_time(*args, **kwargs):
        config = tracking["pending_config"]
        if config is None:
            raise AssertionError("TTC ran without a statically feasible candidate")
        candidate = (float(args[4]), float(args[5]))
        if candidate in tracking["candidates"]:
            raise AssertionError(f"Candidate {candidate} ran TTC more than once")
        tracking["candidates"].add(candidate)
        counters["cal_time"] += 1
        result = original_ttc(*args, **kwargs)
        # The legacy sentinel equals its horizon, so do not label every clear
        # legacy candidate as blocked when predict_time == movetimemin.
        if distinct_clear_value or config.predict_time > config.movetimemin:
            rejected = result <= config.movetimemin
        else:
            rejected = result < config.movetimemin
        if rejected:
            counters["ttc_rejected"] += 1
        tracking["pending_config"] = None
        return result

    namespace["cal_time"] = counted_cal_time
    namespace["calc_obstacle_cost"] = counted_static_cost
    return namespace, counters, tracking


def run_cycle(namespace, counters, tracking, static_points, rectangles):
    for key in counters:
        counters[key] = 0
    tracking["pending_config"] = None
    tracking["candidates"].clear()
    result = run_shared_cycle(namespace, counters, static_points, rectangles)
    if tracking["pending_config"] is not None:
        raise AssertionError("The final statically feasible candidate skipped TTC")
    if counters["static_examined"] != result["candidates"]:
        raise AssertionError("Some sampled candidates skipped static evaluation")
    if counters["cal_time"] != counters["static_feasible"]:
        raise AssertionError("Some statically feasible candidates skipped TTC")
    result.update({"static_rejected": result["candidates"] - counters["static_feasible"],
                   "ttc_rejected": counters["ttc_rejected"]})
    return result


def benchmark(ref, repeats, profile, controller=CONTROLLER, baseline_dir=None):
    namespace, counters, tracking = load_controller(ref, controller, baseline_dir)
    rows, outputs = [], {}
    profiler = cProfile.Profile() if profile else None
    for name, static_points, rectangles in scenarios():
        run_cycle(namespace, counters, tracking, static_points, rectangles)
        samples = [run_cycle(namespace, counters, tracking, static_points, rectangles)
                   for _ in range(repeats)]
        last = samples[-1]
        median_ms = statistics.median(row["elapsed_ms"] for row in samples)
        rows.append({
            "scenario": name, "static_points": len(static_points[0]),
            "rectangles": len(rectangles), "candidates": last["candidates"],
            "cal_time_calls": last["cal_time_calls"],
            "static_feasible": last["static_feasible"],
            "static_rejected": last["static_rejected"],
            "ttc_rejected": last["ttc_rejected"],
            "median_ms": round(median_ms, 3), "compute_hz": round(1000.0 / median_ms, 2),
            "command": last["command"],
        })
        outputs[name] = last
        if profiler is not None:
            profiler.runcall(run_cycle, namespace, counters, tracking, static_points, rectangles)
    if profiler is not None:
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(25)
        print(stream.getvalue())
    return rows, outputs


def output_differences(baseline, current):
    """Report genuine solver changes without requiring legacy false negatives."""
    differences = []
    for name in baseline:
        try:
            compare_outputs({name: baseline[name]}, {name: current[name]})
        except AssertionError:
            fields = []
            for field in ("command", "preview", "human_preview"):
                left, right = baseline[name][field], current[name][field]
                if left is None or right is None:
                    matches = left is right
                else:
                    matches = (np.shape(left) == np.shape(right)
                               and np.allclose(left, right, rtol=1e-12, atol=1e-12))
                if not matches:
                    fields.append(field)
            differences.append({"scenario": name, "fields": fields,
                                "baseline_command": baseline[name]["command"],
                                "current_command": current[name]["command"]})
    return differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--controller", default=CONTROLLER,
                        choices=(CONTROLLER, "dwa_simwaffi_new.py"),
                        help="Production controller to benchmark (default: %(default)s)")
    baseline = parser.add_mutually_exclusive_group()
    baseline.add_argument("--baseline-ref", help="Git revision of the production sources")
    baseline.add_argument("--baseline-dir", type=Path,
                          help="Saved sources directory, or repository root, used as the baseline")
    parser.add_argument("--profile", action="store_true", help="Print a cumulative production-cycle profile")
    parser.add_argument("--json", type=Path, help="Also save measurements and output comparisons as JSON")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    results, baseline_outputs = {}, None
    if args.baseline_ref:
        results[args.baseline_ref], baseline_outputs = benchmark(
            args.baseline_ref, args.repeats, False, args.controller,
        )
    elif args.baseline_dir is not None:
        results[str(args.baseline_dir)], baseline_outputs = benchmark(
            None, args.repeats, False, args.controller, args.baseline_dir,
        )
    results["working_tree"], outputs = benchmark(
        None, args.repeats, args.profile, args.controller,
    )
    differences = []
    if baseline_outputs is not None:
        differences = output_differences(baseline_outputs, outputs)
        if not differences:
            print("Selected v/w and robot/human previews match the baseline (1e-12 tolerance).")
        else:
            for difference in differences:
                print(f"Output differs in {difference['scenario']}: {', '.join(difference['fields'])}; "
                      f"v/w {difference['baseline_command']} -> {difference['current_command']}")
            print("Differences are reported because correcting legacy missed roots or threshold contacts can change choices.")
    for label, rows in results.items():
        print(f"\n{label}:")
        for row in rows:
            print(f"  {row['scenario']:<23} {row['median_ms']:>9.3f} ms  "
                  f"{row['compute_hz']:>7.2f} Hz  candidates={row['candidates']}  "
                  f"static_rejected={row['static_rejected']}  TTC={row['cal_time_calls']}  "
                  f"ttc_rejected={row['ttc_rejected']}  v/w={row['command']}")
    if args.json:
        args.json.write_text(json.dumps({"controller": args.controller,
                                        "measurements": results,
                                        "output_differences": differences}, indent=2) + "\n",
                             encoding="utf-8")


if __name__ == "__main__":
    main()

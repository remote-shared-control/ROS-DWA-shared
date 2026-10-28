"""Check that the larger midpoint TTC radius stays within the transition phase."""

import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


SIM_DIR = Path(__file__).resolve().parents[1]


def load_cube_reader():
    """Use the real reader defaults and vertex builder without importing ROS."""
    parsed = ast.parse((SIM_DIR / "cal_ctime.py").read_text(encoding="utf-8"))
    body = [node for node in parsed.body
            if isinstance(node, ast.ClassDef) and node.name == "CubeReader"]
    namespace = {
        "np": np, "math": math, "ModelStates": object,
        "rospy": SimpleNamespace(Subscriber=lambda *args: None),
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), "cal_ctime.py", "exec"),
         namespace)
    return namespace["CubeReader"]


def motion(times, points, constant_time=None, theta=0.0):
    return {
        "times": times,
        "states": [[x, y, theta, 0.0, 0.0] for x, y in points],
        "t_constant": times[-1] if constant_time is None else constant_time,
    }


class HybridPhaseRadiusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "hybrid_phase_radii_under_test", SIM_DIR / "cal_ctime_midpoint.py")
        cls.solver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.solver)
        cls.CubeReader = load_cube_reader()

    def reader(self):
        # This separate static-cost radius must not replace either TTC radius.
        return self.CubeReader(SimpleNamespace(cube_half_size=0.15, robot_radius=0.9))

    def obstacle(self, center, velocity=(0.0, 0.0)):
        reader = self.reader()
        pose = SimpleNamespace(
            position=SimpleNamespace(x=center[0], y=center[1]),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0))
        reader.data = [[1, *reader.get_expanded_vertices_2d(pose), *velocity]]
        return reader

    def solve(self, reader, transition, v=1.0, w=0.0, horizon=3.0):
        return self.solver.cal_time(reader, transition, v, w, horizon)

    def test_reader_defaults_keep_equation_geometry_at_original_radius(self):
        reader = self.obstacle((0.0, 0.0))
        self.assertEqual(reader.robot_radius_cal, 0.21)
        self.assertEqual(reader.robot_radius_transition, 0.225)
        self.assertEqual(reader.data[0][1:5],
                         [(-0.36, -0.36), (-0.36, 0.36),
                          (0.36, 0.36), (0.36, -0.36)])
        self.assertAlmostEqual(self.solver.prepare_obstacles(reader).transition_padding,
                               0.015)

    def test_transition_detects_obstacle_only_reached_by_the_larger_radius(self):
        # The robot clears the physical obstacle by 0.22 m in the y direction.
        reader = self.obstacle((0.6, 0.37))
        transition = motion([0.0, 1.0], [(0.0, 0.0), (1.0, 0.0)])
        self.assertAlmostEqual(self.solve(reader, transition), 0.225)
        # Callers that only supply already-inflated geometry retain the old API.
        self.assertEqual(self.solve(SimpleNamespace(data=reader.data), transition),
                         4.0)

    def test_initial_check_uses_larger_radius_only_when_a_transition_exists(self):
        reader = self.obstacle((0.0, 0.37))
        ramp = motion([0.0, 1.0], [(0.0, 0.0), (0.0, 0.0)])
        constant = motion([0.0], [(0.0, 0.0)])
        self.assertEqual(self.solve(reader, ramp, v=0.0), 0.0)
        self.assertEqual(self.solve(reader, constant, v=0.0, w=1.0), 4.0)
        # Existing initial contact at robot_radius_cal still counts as a hit.
        reader = self.obstacle((0.0, 0.36))
        self.assertEqual(self.solve(reader, constant, v=0.0, w=1.0), 0.0)

    def test_straight_constant_phase_does_not_use_the_larger_radius(self):
        reader = self.obstacle((1.6, 0.37))
        transition = motion([0.0, 1.0], [(0.0, 0.0), (0.1, 0.0)])
        self.assertEqual(self.solve(reader, transition), 4.0)

    def test_circular_constant_phase_does_not_use_the_larger_radius(self):
        transition = motion([0.0, 0.5], [(0.0, 0.0), (0.0, 0.5)])
        # The circle reaches x=1.0; the equation rectangle starts at x=1.01.
        # Applying the transition padding here would produce a false collision.
        for w in (-1.0, 1.0):
            with self.subTest(w=w):
                reader = self.reader()
                center_y = 0.5 + w
                reader.data = [[
                    1, (1.01, center_y - 0.1), (1.01, center_y + 0.1),
                    (1.2, center_y + 0.1), (1.2, center_y - 0.1), 0.0, 0.0,
                ]]
                self.assertEqual(self.solve(reader, transition, w=w), 4.0)

    def test_constant_phase_hits_keep_original_radius_and_global_ttc(self):
        straight = motion([0.0, 1.0], [(0.0, 0.0), (0.1, 0.0)])
        self.assertAlmostEqual(self.solve(self.obstacle((1.6, 0.0)), straight),
                               2.14)
        circular = motion([0.0, 0.2], [(0.0, 0.0), (0.0, 0.0)])
        for w in (-1.0, 1.0):
            with self.subTest(w=w):
                self.assertAlmostEqual(self.solve(
                    self.obstacle((1.16, w * 0.4)), circular, w=w),
                    0.2 + math.asin(0.8), places=8)

    def test_rotated_translating_obstacles_use_padding_on_each_relative_segment(self):
        angle, origin = 0.7, np.array([2.0, -1.0])
        rotation = np.array([[math.cos(angle), -math.sin(angle)],
                             [math.sin(angle), math.cos(angle)]])
        transition = motion(
            [0.0, 0.4, 1.0],
            [origin + rotation @ np.array([x, 0.0]) for x in (0.0, 0.4, 1.0)],
            theta=angle)
        for center_x in (0.6, 1.3):
            for reversed_winding in (False, True):
                with self.subTest(center_x=center_x, reversed=reversed_winding):
                    reader = self.obstacle((center_x, 0.37), (-0.2, 0.0))
                    points = [tuple(origin + rotation @ np.array(p))
                              for p in reader.data[0][1:5]]
                    if reversed_winding:
                        points.reverse()
                    velocity = rotation @ np.array([-0.2, 0.0])
                    reader.data = [[1, *points, *velocity]]
                    self.assertAlmostEqual(self.solve(reader, transition),
                                           (center_x - 0.375) / 1.2, places=9)

    def test_cached_snapshot_keeps_its_radius_when_reader_changes(self):
        reader = self.obstacle((0.6, 0.37))
        transition = motion([0.0, 1.0], [(0.0, 0.0), (1.0, 0.0)])
        prepared = self.solver.prepare_obstacles(reader)
        reader.robot_radius_transition = reader.robot_radius_cal
        self.assertIs(self.solver.prepare_obstacles(prepared), prepared)
        self.assertAlmostEqual(self.solve(prepared, transition), 0.225)
        self.assertEqual(self.solve(reader, transition), 4.0)

    def test_transition_padding_tracks_a_changed_equation_radius(self):
        reader = self.reader()
        reader.robot_radius_cal = 0.20
        pose = SimpleNamespace(
            position=SimpleNamespace(x=0.6, y=0.37),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0))
        reader.data = [[1, *reader.get_expanded_vertices_2d(pose), 0.0, 0.0]]
        prepared = self.solver.prepare_obstacles(reader)
        self.assertAlmostEqual(prepared.transition_padding, 0.025)
        transition = motion([0.0, 1.0], [(0.0, 0.0), (1.0, 0.0)])
        self.assertAlmostEqual(self.solve(prepared, transition), 0.225)

    def test_transition_contact_at_horizon_is_distinct_from_a_later_contact(self):
        transition = motion([0.0, 0.25], [(0.0, 0.0), (0.25, 0.0)],
                            constant_time=1.0)
        # With 0.225 m radius the first rectangle starts exactly at x=0.25.
        self.assertAlmostEqual(self.solve(self.obstacle((0.625, 0.37)), transition,
                                          horizon=0.25), 0.25)
        self.assertEqual(self.solve(self.obstacle((0.6251, 0.37)), transition,
                                    horizon=0.25), 1.25)

    def test_clear_and_empty_obstacles_still_return_horizon_plus_one(self):
        transition = motion([0.0, 0.25], [(0.0, 0.0), (0.25, 0.0)],
                            constant_time=1.0)
        for data in (None, []):
            with self.subTest(data=data):
                reader = self.reader()
                reader.data = data
                self.assertEqual(self.solve(reader, transition, horizon=0.25), 1.25)
        self.assertEqual(self.solve(self.obstacle((20.0, 20.0)), transition,
                                    horizon=0.25), 1.25)


if __name__ == "__main__":
    unittest.main()

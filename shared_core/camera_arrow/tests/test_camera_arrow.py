"""Test direction semantics, real rendering and the image callback without ROS.

Run: python3 -B -m unittest discover -s shared_core/camera_arrow/tests -v
Dependencies: NumPy and OpenCV (the same as the ROS image overlay node).
"""

import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dwa_camera_arrow import (
    CameraArrowNode, CommandState, MotionHint, arrow_outline, draw_hud,
    decode_compressed_image, project_vertices, resolve_input_transport, velocity_hint,
    _draw_hud_layer, _hud_layout,
)


class DirectionTests(unittest.TestCase):
    def test_bearing_matches_constant_velocity_endpoint_for_forward_and_reverse(self):
        for v in (-0.3, 0.3):
            for w in (-1.2, -0.4, 0.4, 1.2):
                for duration in (0.5, 1.0):
                    with self.subTest(v=v, w=w, duration=duration):
                        x = v / w * math.sin(w * duration)
                        y = v / w * (1.0 - math.cos(w * duration))
                        expected = np.array([x, y]) / math.hypot(x, y)
                        hint = velocity_hint(v, w, duration)
                        self.assertEqual(hint.kind, "move")
                        np.testing.assert_allclose(
                            [math.cos(hint.angle), math.sin(hint.angle)], expected,
                            atol=1e-12,
                        )

    def test_straight_and_reverse_are_opposite(self):
        self.assertEqual(velocity_hint(0.3, 0.0).angle, 0.0)
        self.assertAlmostEqual(abs(velocity_hint(-0.3, 0.0).angle), math.pi)

    def test_spin_and_stop_do_not_suggest_translation(self):
        self.assertEqual(velocity_hint(0.0, 0.0).kind, "stop")
        self.assertEqual(velocity_hint(0.01, 0.04).kind, "stop")
        self.assertEqual(velocity_hint(0.0, 0.4).label, "TURN LEFT")
        self.assertEqual(velocity_hint(0.0, -0.4).label, "TURN RIGHT")
        left = arrow_outline(velocity_hint(0.0, 0.4))
        right = arrow_outline(velocity_hint(0.0, -0.4))
        np.testing.assert_allclose(left[:, 0], right[:, 0])
        np.testing.assert_allclose(left[:, 1], -right[:, 1])

    def test_fast_spin_during_translation_cannot_wrap_into_forward(self):
        for w in (-100.0, 100.0):
            hint = velocity_hint(0.3, w)
            self.assertAlmostEqual(abs(hint.angle), math.pi / 3.0)

    def test_invalid_commands_and_parameters(self):
        for v, w in ((float("nan"), 0), (0, float("inf"))):
            self.assertEqual(velocity_hint(v, w).kind, "unavailable")
        for options in ({"preview_time": 0}, {"preview_time": float("nan")},
                        {"linear_deadband": -1}, {"angular_deadband": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                velocity_hint(0.3, 0.2, **options)

    def test_projection_agrees_with_robot_axes(self):
        center = (320, 140)
        points = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
        pixels, depths = project_vertices(points, center, 150)
        self.assertLess(pixels[0, 1], center[1])  # Forward -> up.
        self.assertGreater(pixels[1, 1], center[1])  # Reverse -> down.
        self.assertLess(pixels[2, 0], center[0])  # Robot left -> image left.
        self.assertGreater(pixels[3, 0], center[0])
        self.assertTrue(np.all(depths > 0.0))

    def test_reverse_distinguishes_back_left_straight_and_back_right(self):
        for w, side in ((-0.8, "LEFT"), (0.0, ""), (0.8, "RIGHT")):
            with self.subTest(w=w):
                hint = velocity_hint(-0.3, w)
                pixels, _ = project_vertices(
                    [[math.cos(hint.angle), math.sin(hint.angle), 0]], (320, 140), 150
                )
                self.assertGreater(pixels[0, 1], 140)
                if side == "LEFT":
                    self.assertLess(pixels[0, 0], 320)
                elif side == "RIGHT":
                    self.assertGreater(pixels[0, 0], 320)
                else:
                    self.assertAlmostEqual(pixels[0, 0], 320)
                self.assertEqual(hint.label, ("REVERSE " + side).strip())


class CommandAndImageTests(unittest.TestCase):
    def test_missing_stale_and_clock_reset_commands(self):
        state = CommandState()
        self.assertEqual(state.hint(10.0, 0.5).label, "NO COMMAND")
        state.update(0.3, 0.4, 10.0)
        self.assertEqual(state.hint(10.4, 0.5).kind, "move")
        self.assertEqual(state.hint(10.6, 0.5).label, "STALE COMMAND")
        self.assertEqual(state.hint(0.0, 0.5).label, "STALE COMMAND")
        state.update(0.0, 0.0, 0.0)
        self.assertEqual(state.hint(0.1, 0.5).kind, "stop")

    def test_all_states_start_at_ten_percent_of_image_height(self):
        hints = [velocity_hint(v, w) for v, w in (
            (0.3, 0), (0.3, 0.8), (0.3, -0.8), (-0.3, 0),
            (-0.3, 0.8), (0, 0.8), (0, -0.8), (0, 0),
        )] + [MotionHint("unavailable"), MotionHint("unavailable", label="STALE COMMAND")]
        for width, height in ((640, 480), (1280, 720), (320, 240), (1920, 1080)):
            for hint in hints:
                with self.subTest(size=(width, height), hint=hint):
                    image = np.full((height, width, 3), 80, dtype=np.uint8)
                    result = draw_hud(image, hint)
                    self.assertIs(result, image)
                    changed = np.any(image != 80, axis=2)
                    ys, xs = np.where(changed)
                    self.assertGreater(len(xs), 0)
                    self.assertAlmostEqual(int(ys.min()), round(height * 0.1), delta=4)
                    self.assertTrue(np.all(ys < height * 0.55))
                    self.assertTrue(np.all(xs > width * 0.2))
                    self.assertTrue(np.all(xs < width * 0.8))
                    self.assertEqual(image.dtype, np.uint8)

    def make_node(self):
        node = CameraArrowNode.__new__(CameraArrowNode)
        node.state = CommandState()
        node.options = {}
        node.timeout = 0.5
        node.anchor = (0.5, 0.28)
        node.top_margin_ratio = 0.1
        node.size_ratio = 0.32
        node.opacity = 0.9
        node.ros = SimpleNamespace(get_time=lambda: 10.0, logwarn_throttle=Mock())
        node.publisher = Mock()
        node.publisher.get_num_connections.return_value = 1
        node.compressed_publisher = Mock()
        node.compressed_publisher.get_num_connections.return_value = 0
        node.input_transport = "raw"
        source = np.full((480, 640, 3), 80, dtype=np.uint8)
        source.flags.writeable = False
        node.bridge = Mock()
        node.bridge.imgmsg_to_cv2.return_value = source
        node.bridge.cv2_to_imgmsg.return_value = SimpleNamespace(header=None)
        node.bridge.cv2_to_compressed_imgmsg.side_effect = lambda image, dst_format: SimpleNamespace(
            header=None, format=dst_format,
            data=cv2.imencode("." + dst_format, image)[1].tobytes(),
        )
        node.bridge_error = TypeError
        return node, source

    def test_final_twist_is_used_and_output_preserves_camera_header(self):
        node, source = self.make_node()
        node.velocity_callback(SimpleNamespace(
            linear=SimpleNamespace(x=0.3), angular=SimpleNamespace(z=-0.8)
        ))
        self.assertEqual(node.state.hint(10.0, 0.5).label, "FORWARD RIGHT")
        header = SimpleNamespace(stamp=9.95, frame_id="camera_optical_frame", seq=2)
        message = SimpleNamespace(header=header)
        node.image_callback(message)
        node.bridge.imgmsg_to_cv2.assert_called_once_with(message, desired_encoding="bgr8")
        rendered = node.bridge.cv2_to_imgmsg.call_args[0][0]
        self.assertTrue(np.any(rendered != source))
        self.assertTrue(np.all(source == 80))
        self.assertIs(node.publisher.publish.call_args[0][0].header, header)
        self.assertEqual(node.bridge.cv2_to_imgmsg.call_args[1], {"encoding": "bgr8"})

    def test_no_viewer_skips_image_conversion(self):
        node, _ = self.make_node()
        node.publisher.get_num_connections.return_value = 0
        node.image_callback(SimpleNamespace())
        node.bridge.imgmsg_to_cv2.assert_not_called()
        node.bridge.cv2_to_compressed_imgmsg.assert_not_called()

    def test_both_inputs_support_raw_compressed_and_dual_outputs(self):
        for transport in ("raw", "compressed"):
            for raw, compressed in ((1, 0), (0, 1), (1, 1)):
                with self.subTest(input=transport, raw=raw, compressed=compressed):
                    node, source = self.make_node()
                    node.input_transport = transport
                    node.publisher.get_num_connections.return_value = raw
                    node.compressed_publisher.get_num_connections.return_value = compressed
                    header = SimpleNamespace(stamp=9.95, frame_id="camera", seq=3)
                    message = SimpleNamespace(
                        header=header, format="png",
                        data=cv2.imencode(".png", source)[1].tobytes(),
                    )
                    node.state.update(-0.3, -0.8, 10.0)
                    node.image_callback(message)
                    if transport == "compressed":
                        node.bridge.imgmsg_to_cv2.assert_not_called()
                    else:
                        node.bridge.imgmsg_to_cv2.assert_called_once()
                    self.assertEqual(node.publisher.publish.call_count, raw)
                    self.assertEqual(node.compressed_publisher.publish.call_count, compressed)
                    self.assertEqual(node.bridge.cv2_to_imgmsg.call_count, raw)
                    self.assertEqual(node.bridge.cv2_to_compressed_imgmsg.call_count, compressed)
                    if raw:
                        self.assertIs(node.publisher.publish.call_args[0][0].header, header)
                    if compressed:
                        output = node.compressed_publisher.publish.call_args[0][0]
                        self.assertIs(output.header, header)
                        self.assertEqual(output.format, "jpg")
                        decoded = decode_compressed_image(output)
                        self.assertEqual(decoded.shape, source.shape)
                        self.assertTrue(np.any(decoded != source))
                    self.assertTrue(np.all(source == 80))

    def test_bad_image_is_reported_without_publishing(self):
        node, _ = self.make_node()
        node.bridge.imgmsg_to_cv2.side_effect = TypeError("unsupported image")
        node.image_callback(SimpleNamespace())
        node.ros.logwarn_throttle.assert_called_once()
        node.publisher.publish.assert_not_called()

    def test_corrupt_compressed_input_is_reported_without_publishing(self):
        node, _ = self.make_node()
        node.input_transport = "compressed"
        node.image_callback(SimpleNamespace(data=b"not an image"))
        node.ros.logwarn_throttle.assert_called_once()
        node.publisher.publish.assert_not_called()
        node.compressed_publisher.publish.assert_not_called()


class TransportTests(unittest.TestCase):
    def test_transport_detection_and_custom_topic_override(self):
        self.assertEqual(resolve_input_transport("/camera/image_raw"), "raw")
        self.assertEqual(resolve_input_transport("/camera/image_raw/compressed"), "compressed")
        self.assertEqual(resolve_input_transport("/camera/jpeg", "compressed"), "compressed")
        self.assertEqual(resolve_input_transport("/camera/compressed", "raw"), "raw")
        with self.assertRaises(ValueError):
            resolve_input_transport("/camera/image_raw", "unsupported")

    def test_jpeg_png_and_grayscale_are_decoded_to_bgr(self):
        color = np.full((24, 32, 3), (20, 100, 180), dtype=np.uint8)
        gray = np.full((24, 32), 80, dtype=np.uint8)
        rgba = np.full((24, 32, 4), (20, 100, 180, 120), dtype=np.uint8)
        for source in (color, gray, rgba):
            for extension in (".jpg", ".png"):
                with self.subTest(shape=source.shape, extension=extension):
                    data = cv2.imencode(extension, source)[1].tobytes()
                    decoded = decode_compressed_image(SimpleNamespace(data=data))
                    self.assertEqual(decoded.shape, (24, 32, 3))
                    self.assertEqual(decoded.dtype, np.uint8)
                    expected = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if source.ndim == 2 else source[:, :, :3]
                    np.testing.assert_allclose(decoded, expected, atol=2)

    def test_empty_or_invalid_compressed_image_is_rejected(self):
        for data in (b"", b"not a jpeg or png"):
            with self.subTest(data=data), self.assertRaises(ValueError):
                decode_compressed_image(SimpleNamespace(data=data))

    def create_node(self, parameters):
        class Image:
            pass

        class CompressedImage:
            pass

        ros = SimpleNamespace(
            get_param=lambda name, default: parameters.get(name, default),
            resolve_name=lambda name: name,
            Publisher=Mock(), Subscriber=Mock(), loginfo=Mock(),
        )
        modules = {
            "rospy": ros,
            "cv_bridge": SimpleNamespace(CvBridge=Mock(), CvBridgeError=TypeError),
            "geometry_msgs.msg": SimpleNamespace(Twist=object),
            "sensor_msgs.msg": SimpleNamespace(Image=Image, CompressedImage=CompressedImage),
        }
        with patch.dict(sys.modules, modules):
            node = CameraArrowNode()
        return node, ros, Image, CompressedImage

    def test_node_subscribes_with_the_correct_message_type(self):
        cases = (("/camera/image_raw", "auto", False),
                 ("/camera/image_raw/compressed", "auto", True),
                 ("/camera/jpeg", "compressed", True))
        for topic, transport, compressed in cases:
            with self.subTest(topic=topic, transport=transport):
                node, ros, image_type, compressed_type = self.create_node({
                    "~image_topic": topic, "~input_transport": transport,
                })
                subscription = ros.Subscriber.call_args_list[1]
                self.assertEqual(subscription.args[0], topic)
                self.assertIs(subscription.args[1], compressed_type if compressed else image_type)
                self.assertEqual(node.input_transport, "compressed" if compressed else "raw")
                self.assertEqual(ros.Publisher.call_args_list[0].args,
                                 ("/camera/image_dwa_arrow", image_type))
                self.assertEqual(ros.Publisher.call_args_list[1].args,
                                 ("/camera/image_dwa_arrow/compressed", compressed_type))

    def test_input_cannot_be_either_output_topic(self):
        for topic in ("/camera/image_dwa_arrow", "/camera/image_dwa_arrow/compressed"):
            with self.subTest(topic=topic), self.assertRaises(ValueError):
                self.create_node({"~image_topic": topic})

    def test_top_margin_default_override_and_legacy_placement(self):
        for value, expected in ((None, 0.1), (0.05, 0.05), (-1.0, None)):
            with self.subTest(top_margin=value):
                parameters = {} if value is None else {"~top_margin_ratio": value}
                node, _, _, _ = self.create_node(parameters)
                self.assertEqual(node.top_margin_ratio, expected)
        for value in (-0.2, 1.0, float("nan")):
            with self.subTest(top_margin=value), self.assertRaises(ValueError):
                self.create_node({"~top_margin_ratio": value})


class RegionRenderingTests(unittest.TestCase):
    def test_roi_matches_full_frame_render_including_text_shadows_and_clipped_edges(self):
        rng = np.random.default_rng(42)
        hints = [velocity_hint(v, w) for v, w in (
            (0.3, 0), (0.3, 0.8), (0.3, -0.8), (-0.3, 0),
            (-0.3, -0.8), (-0.3, 0.8), (0, 0.8), (0, -0.8), (0, 0),
        )] + [MotionHint("unavailable", label="INVALID COMMAND")]
        placements = (
            ((0.5, 0.28), 0.32, 0.1), ((0.5, 0.28), 0.32, None),
            ((0.01, 0.01), 0.6, None), ((0.99, 0.99), 0.6, None),
            ((0.5, 0.28), 0.05, 0.0), ((0.5, 0.28), 0.32, 0.95),
        )
        source = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
        for hint in hints:
            for anchor, size, margin in placements:
                with self.subTest(hint=hint, anchor=anchor, size=size, margin=margin):
                    layout = _hud_layout(source.shape, hint, anchor, size, margin)
                    full_layer = source.copy()
                    _draw_hud_layer(full_layer, hint, layout, np.zeros(2, dtype=np.int32))
                    expected = cv2.addWeighted(full_layer, 0.9, source, 0.1, 0.0)
                    actual = source.copy()
                    with patch("cv2.addWeighted", wraps=cv2.addWeighted) as blend:
                        draw_hud(actual, hint, anchor, size, 0.9, top_margin_ratio=margin)
                    if blend.called:
                        left, top, right, bottom = layout.bounds
                        expected_layer = full_layer[max(0, top):min(480, bottom),
                                                    max(0, left):min(640, right)]
                        np.testing.assert_array_equal(blend.call_args.args[0], expected_layer)
                    # OpenCV's SIMD/scalar blend paths can differ by one gray
                    # level when the ROI row width has a different alignment.
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=1)

    def test_only_small_region_is_blended_and_surrounding_pixels_are_preserved(self):
        rng = np.random.default_rng(7)
        source = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
        for hint in (velocity_hint(0.3, 0.8), velocity_hint(0, -0.8),
                     MotionHint("unavailable", label="INVALID COMMAND")):
            with self.subTest(hint=hint):
                actual = source.copy()
                with patch("cv2.addWeighted", wraps=cv2.addWeighted) as blend:
                    draw_hud(actual, hint)
                layer = blend.call_args.args[0]
                self.assertLess(layer.shape[0] * layer.shape[1], source.shape[0] * source.shape[1] * 0.15)
                left, top, right, bottom = _hud_layout(source.shape, hint, (0.5, 0.28), 0.32, 0.1).bounds
                outside = np.ones(source.shape[:2], dtype=bool)
                outside[top:bottom, left:right] = False
                np.testing.assert_array_equal(actual[outside], source[outside])


if __name__ == "__main__":
    unittest.main()

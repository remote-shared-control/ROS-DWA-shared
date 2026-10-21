#!/usr/bin/env python3
"""Overlay a robot-relative 3D direction arrow on a ROS camera image.

Only the final Twist is needed. The drawing is a fixed HUD, not a projection
onto the real ground, so camera calibration, odometry and TF are unnecessary.
Raw Image and JPEG/PNG CompressedImage inputs are supported. Outputs use the
usual raw base topic and its /compressed sibling. The geometry and rendering
helpers can also be used without ROS.
"""

from dataclasses import dataclass
from functools import lru_cache
import math
import threading

import numpy as np


@dataclass(frozen=True)
class MotionHint:
    kind: str  # move, turn, stop, unavailable
    angle: float = 0.0  # move: bearing in robot XY; turn: signed yaw rate
    label: str = "NO COMMAND"


def velocity_hint(v, w, preview_time=1.0, linear_deadband=0.02,
                  angular_deadband=0.05):
    """Estimate the bearing of a constant-v/w motion, without world coordinates.

    For a circular arc the displacement bearing is w*T/2 when moving forward,
    and w*T/2 + pi when reversing. Limit the preview sweep to +/-120 degrees
    so high yaw rates cannot make a full loop appear to be straight motion.
    """
    if (not math.isfinite(preview_time) or preview_time <= 0.0
            or not math.isfinite(linear_deadband) or linear_deadband < 0.0
            or not math.isfinite(angular_deadband) or angular_deadband < 0.0):
        raise ValueError("Preview time must be positive; deadbands must be nonnegative")
    if not math.isfinite(v) or not math.isfinite(w):
        return MotionHint("unavailable", label="INVALID COMMAND")
    if abs(v) <= linear_deadband:
        if abs(w) <= angular_deadband:
            return MotionHint("stop", label="STOP")
        return MotionHint("turn", w, "TURN LEFT" if w > 0.0 else "TURN RIGHT")

    sweep = 0.0 if abs(w) <= angular_deadband else float(np.clip(
        w * preview_time, -2.0 * math.pi / 3.0, 2.0 * math.pi / 3.0
    ))
    angle = sweep / 2.0 + (math.pi if v < 0.0 else 0.0)
    angle = math.atan2(math.sin(angle), math.cos(angle))
    label = "FORWARD" if v > 0.0 else "REVERSE"
    if abs(sweep) > math.radians(10.0):
        label += " LEFT" if math.sin(angle) > 0.0 else " RIGHT"
    return MotionHint("move", angle, label)


class CommandState:
    """Keep a thread-safe command snapshot and expire stopped publishers."""

    def __init__(self):
        self._lock = threading.Lock()
        self._command = None

    def update(self, v, w, received_at):
        with self._lock:
            self._command = (float(v), float(w), float(received_at))

    def hint(self, now, timeout, **options):
        with self._lock:
            command = self._command
        if command is None:
            return MotionHint("unavailable")
        v, w, received_at = command
        age = now - received_at
        if age < 0.0 or age > timeout:
            return MotionHint("unavailable", label="STALE COMMAND")
        return velocity_hint(v, w, **options)


def arrow_outline(hint):
    """Return an arrow polygon in robot XY (+X forward, +Y left)."""
    if hint.kind == "move":
        outline = np.array([
            [-0.52, -0.12], [0.06, -0.13], [0.06, -0.27],
            [0.14, -0.28], [0.58, -0.025], [0.60, 0.0],
            [0.58, 0.025], [0.14, 0.28], [0.06, 0.27],
            [0.06, 0.13], [-0.52, 0.12], [-0.56, 0.08],
            [-0.49, 0.0], [-0.56, -0.08],
        ])
        c, s = math.cos(hint.angle), math.sin(hint.angle)
        return outline @ np.array([[c, s], [-s, c]])
    if hint.kind == "turn":
        # An annular arrow shows rotation without suggesting translation.
        angles = np.linspace(-1.1, 3.5, 20)
        radial = np.column_stack((np.cos(angles), np.sin(angles)))
        end = radial[-1]
        tip = 0.36 * np.array([math.cos(4.1), math.sin(4.1)])
        outline = np.vstack((0.47 * radial, 0.60 * end, tip,
                             0.10 * end, 0.25 * radial[::-1]))
        if hint.angle < 0.0:
            outline[:, 1] *= -1.0
        return outline
    return np.empty((0, 2))


def project_vertices(vertices, center, size):
    """Perspective view from behind and above the robot, for a fixed HUD."""
    pitch = math.radians(38.0)
    x, y, z = np.asarray(vertices, dtype=float).T
    depth = 3.4 + math.cos(pitch) * x - math.sin(pitch) * z
    focal = size * 3.4 / 1.2
    u = center[0] - focal * y / depth
    v = center[1] - focal * (math.sin(pitch) * x + math.cos(pitch) * z) / depth
    return np.column_stack((u, v)), depth


@dataclass(frozen=True)
class _HudLayout:
    outline: np.ndarray
    pixels: np.ndarray
    side_layers: np.ndarray
    center: np.ndarray
    radius: int
    text_pos: np.ndarray
    font_scale: float
    bounds: tuple
    scale: float
    text_gap: int
    label_bounds: tuple


@lru_cache(maxsize=128)
def _hud_layout(image_shape, hint, anchor, size_ratio, top_margin_ratio,
                hud_scale=0.7, text_gap_ratio=0.5):
    """Cache exact geometry and text bounds; never quantize the direction."""
    import cv2

    height, width = image_shape[:2]
    center = np.array([width * anchor[0], height * anchor[1]])
    base_size = min(width, height) * size_ratio
    outline = arrow_outline(hint)
    base_radius = max(5, int(base_size * 0.14))
    radius = max(1, int(round(base_radius * hud_scale)))
    padding = max(2, int(math.ceil(3 * hud_scale)))
    if len(outline):
        count = len(outline)
        vertices = np.vstack((
            np.column_stack((outline, np.zeros(count))),
            np.column_stack((outline, np.full(count, 0.105))),
        ))
        projected, _ = project_vertices(vertices, center, base_size)
        reference_pixels = np.rint(projected).astype(np.int32)
        pixels = np.rint(center + (projected - center) * hud_scale).astype(np.int32)
        reference_bottom = max(reference_pixels[:, 1].max(),
                               reference_pixels[:count, 1].max() + 6) + 4
        lower = pixels.min(axis=0) - padding
        upper = pixels.max(axis=0) + padding + 1
    else:
        pixels = np.empty((0, 2), dtype=np.int32)
        marker_center = np.rint(center).astype(np.int32)
        reference_bottom = marker_center[1] + base_radius + 4
        lower = marker_center - radius - padding
        upper = marker_center + radius + padding + 1

    base_font_scale = max(0.3, base_size / 300.0)
    font_scale = base_font_scale * hud_scale
    # Halve the previous label-to-arrow clearance, then apply the same scale
    # as the rest of the HUD. The label now sits above the complete arrow.
    (_, reference_text_height), _ = cv2.getTextSize(
        hint.label, cv2.FONT_HERSHEY_DUPLEX, base_font_scale, 3
    )
    reference_text_top = int(center[1] + base_size * 0.62) - reference_text_height - 3
    text_gap = int(round(max(0, reference_text_top - reference_bottom)
                         * hud_scale * text_gap_ratio))
    text_size, _ = cv2.getTextSize(hint.label, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1)
    text_thickness = max(1, int(round(3 * hud_scale)))
    # Include the scaled black outline and the space below the text baseline.
    (text_width, text_height), baseline = cv2.getTextSize(
        hint.label, cv2.FONT_HERSHEY_DUPLEX, font_scale, text_thickness
    )
    text_pos = np.array([int(center[0] - text_size[0] / 2),
                         int(lower[1] - text_gap - baseline - padding - 1)], dtype=np.int32)
    label_padding = max(padding, int(round(9 * hud_scale)))
    label_lower = text_pos + (-label_padding, -text_height - padding)
    label_upper = text_pos + (text_width + label_padding + 1, baseline + padding + 1)
    # Leave room for the plate's anti-aliased border outside its polygon.
    lower = np.minimum(lower, label_lower - 2)
    upper = np.maximum(upper, label_upper + 3)
    marker_center = np.rint(center).astype(np.int32)
    if top_margin_ratio is not None:
        shift = int(round(height * top_margin_ratio)) - int(lower[1])
        pixels[:, 1] += shift
        marker_center[1] += shift
        text_pos[1] += shift
        label_lower[1] += shift
        label_upper[1] += shift
        lower[1] += shift
        upper[1] += shift
    if len(outline):
        # Three overlapping slices provide a thin extruded profile without
        # a Python loop, lighting calculation and draw call for every face.
        fractions = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0])[:, None, None]
        side_layers = np.rint(pixels[None, :count] + fractions
                              * (pixels[count:] - pixels[:count])[None]).astype(np.int32)
    else:
        side_layers = np.empty((0, 0, 2), dtype=np.int32)
    for array in (outline, pixels, side_layers, marker_center, text_pos):
        array.flags.writeable = False
    return _HudLayout(outline, pixels, side_layers, marker_center, radius, text_pos,
                      font_scale, (*lower.tolist(), *upper.tolist()), hud_scale, text_gap,
                      (*label_lower.tolist(), *label_upper.tolist()))


def _hud_palette(hint):
    """BGR colors: icy cyan for movement, amber for turning/stopping."""
    if hint.kind == "move":
        return ((235, 188, 48), (220, 154, 14), (255, 240, 166), (100, 65, 18))
    if hint.kind in ("turn", "stop"):
        return ((70, 195, 246), (25, 153, 233), (184, 237, 255), (30, 68, 110))
    return ((200, 208, 212), (123, 133, 143), (234, 238, 240), (65, 63, 64))


def _draw_label_plate(layer, bounds, color, radius):
    """Draw a compact, chamfered label plate, with anti-aliased edges."""
    import cv2

    left, top, right, bottom = bounds
    radius = min(radius, max(1, (right - left) // 2), max(1, (bottom - top) // 2))
    polygon = np.array([
        [left + radius, top], [right - radius, top], [right, top + radius],
        [right, bottom - radius], [right - radius, bottom],
        [left + radius, bottom], [left, bottom - radius], [left, top + radius],
    ], dtype=np.int32)
    cv2.fillPoly(layer, [polygon], color, cv2.LINE_AA)


def _draw_hud_layer(layer, hint, layout, offset):
    """Paint prepared geometry into a local layer, without copying the frame."""
    import cv2

    outline = layout.outline
    pixels = layout.pixels - offset
    palette = _hud_palette(hint)
    if len(outline):
        count = len(outline)
        side_colors = (palette[3], tuple(min(255, int(c * 1.15)) for c in palette[3]),
                       tuple(min(255, int(c * 1.3)) for c in palette[3]))
        for polygon, color in zip(layout.side_layers - offset, side_colors):
            cv2.fillPoly(layer, [polygon], color, cv2.LINE_AA)
        cv2.fillPoly(layer, [pixels[count:]], palette[0], cv2.LINE_AA)
        cv2.polylines(layer, [pixels[count:]], True, palette[2],
                      max(1, int(round(layout.scale))), cv2.LINE_AA)
    else:
        c = tuple((layout.center - offset).tolist())
        radius = layout.radius
        color = palette[1]
        cv2.circle(layer, c, radius, (27, 29, 25), -1, cv2.LINE_AA)
        cv2.circle(layer, c, radius, color, max(1, int(round(2 * layout.scale))), cv2.LINE_AA)
        if hint.kind == "stop":
            half = max(2, int(radius * 0.4))
            cv2.rectangle(layer, (c[0] - half, c[1] - half),
                          (c[0] + half, c[1] + half), color, -1)

    font_scale = layout.font_scale
    text_pos = tuple((layout.text_pos - offset).tolist())
    left, top, right, bottom = layout.label_bounds
    label_bounds = (left - int(offset[0]), top - int(offset[1]),
                    right - int(offset[0]), bottom - int(offset[1]))
    _draw_label_plate(layer, label_bounds, (26, 24, 19), max(2, int(round(5 * layout.scale))))
    accent_x = label_bounds[0] + max(2, int(round(3 * layout.scale)))
    cv2.line(layer, (accent_x, label_bounds[1] + 3), (accent_x, label_bounds[3] - 3),
              palette[1], max(1, int(round(2 * layout.scale))), cv2.LINE_AA)
    cv2.putText(layer, hint.label, text_pos, cv2.FONT_HERSHEY_DUPLEX,
                font_scale, (245, 248, 249), max(1, int(round(layout.scale))), cv2.LINE_AA)


def draw_hud(image, hint, anchor=(0.5, 0.28), size_ratio=0.32, opacity=0.9,
             top_margin_ratio=0.0, hud_scale=0.7, text_gap_ratio=0.5):
    """Blend a compact HUD at the top, with a label above its arrow.

    Set top_margin_ratio=None to use the original anchor_y center placement.
    """
    import cv2

    layout = _hud_layout(image.shape, hint, anchor, size_ratio, top_margin_ratio,
                         hud_scale, text_gap_ratio)
    height, width = image.shape[:2]
    left, top, right, bottom = layout.bounds
    left, top = max(0, left), max(0, top)
    right, bottom = min(width, right), min(height, bottom)
    if left >= right or top >= bottom:
        return image
    region = image[top:bottom, left:right]
    layer = region.copy()
    _draw_hud_layer(layer, hint, layout, np.array([left, top], dtype=np.int32))
    cv2.addWeighted(layer, opacity, region, 1.0 - opacity, 0.0, dst=region)
    return image


def resolve_input_transport(topic, transport="auto"):
    """Infer conventional compressed topics, allowing custom names explicitly."""
    if transport not in ("auto", "raw", "compressed"):
        raise ValueError("input_transport must be auto, raw or compressed")
    if transport == "auto":
        return "compressed" if topic.rstrip("/").endswith("/compressed") else "raw"
    return transport


def decode_compressed_image(message):
    """Decode normal camera JPEG/PNG bytes to BGR, including mono/RGBA inputs."""
    import cv2

    encoded = np.frombuffer(message.data, dtype=np.uint8)
    if not encoded.size:
        raise ValueError("Compressed image is empty")
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Compressed image is not a valid JPEG/PNG image")
    return image


class CameraArrowNode:
    def __init__(self):
        import rospy
        from cv_bridge import CvBridge, CvBridgeError
        from geometry_msgs.msg import Twist
        from sensor_msgs.msg import CompressedImage, Image

        self.ros = rospy
        self.bridge = CvBridge()
        self.bridge_error = CvBridgeError
        self.state = CommandState()
        self.options = {
            "preview_time": float(rospy.get_param("~preview_time", 1.0)),
            "linear_deadband": float(rospy.get_param("~linear_deadband", 0.02)),
            "angular_deadband": float(rospy.get_param("~angular_deadband", 0.05)),
        }
        velocity_hint(0.0, 0.0, **self.options)  # Validate before subscribing.
        self.timeout = float(rospy.get_param("~command_timeout", 0.5))
        self.anchor = (float(rospy.get_param("~anchor_x", 0.5)),
                       float(rospy.get_param("~anchor_y", 0.28)))
        top_margin = float(rospy.get_param("~top_margin_ratio", 0.0))
        if not math.isfinite(top_margin) or (top_margin != -1.0 and not 0.0 <= top_margin < 1.0):
            raise ValueError("top_margin_ratio must be in [0, 1), or -1 for anchor_y placement")
        self.top_margin_ratio = None if top_margin == -1.0 else top_margin
        self.hud_scale = float(rospy.get_param("~hud_scale", 0.7))
        self.text_gap_ratio = float(rospy.get_param("~text_gap_ratio", 0.5))
        if not math.isfinite(self.hud_scale) or not 0.0 < self.hud_scale <= 2.0:
            raise ValueError("hud_scale must be in (0, 2]")
        if not math.isfinite(self.text_gap_ratio) or self.text_gap_ratio < 0.0:
            raise ValueError("text_gap_ratio must be nonnegative")
        self.size_ratio = float(rospy.get_param("~size_ratio", 0.32))
        self.opacity = float(rospy.get_param("~opacity", 0.9))
        if not math.isfinite(self.timeout) or self.timeout <= 0.0:
            raise ValueError("command_timeout must be positive")
        if not all(math.isfinite(value) and 0.0 < value < 1.0 for value in self.anchor):
            raise ValueError("anchor_x and anchor_y must be between 0 and 1")
        if not 0.05 <= self.size_ratio <= 0.6 or not 0.0 < self.opacity <= 1.0:
            raise ValueError("size_ratio must be in [0.05, 0.6]; opacity in (0, 1]")

        image_topic = rospy.resolve_name(rospy.get_param("~image_topic", "/camera/image_raw"))
        output_topic = rospy.resolve_name(rospy.get_param("~output_topic", "/camera/image_dwa_arrow"))
        compressed_topic = rospy.resolve_name(output_topic.rstrip("/") + "/compressed")
        velocity_topic = rospy.get_param("~velocity_topic", "/cmd_vel")
        self.input_transport = resolve_input_transport(
            image_topic, rospy.get_param("~input_transport", "auto")
        )
        if image_topic in (output_topic, compressed_topic) or output_topic == compressed_topic:
            raise ValueError("Input, raw output and compressed output topics must be different")
        self.publisher = rospy.Publisher(output_topic, Image, queue_size=1)
        self.compressed_publisher = rospy.Publisher(compressed_topic, CompressedImage, queue_size=1)
        self.velocity_subscriber = rospy.Subscriber(
            velocity_topic, Twist, self.velocity_callback, queue_size=1
        )
        self.image_subscriber = rospy.Subscriber(
            image_topic, CompressedImage if self.input_transport == "compressed" else Image,
            self.image_callback, queue_size=1, buff_size=2 ** 24
        )
        rospy.loginfo("DWA camera arrow: %s (%s) + %s -> %s, %s",
                      image_topic, self.input_transport, velocity_topic,
                      output_topic, compressed_topic)

    def velocity_callback(self, message):
        self.state.update(message.linear.x, message.angular.z, self.ros.get_time())

    def image_callback(self, message):
        import cv2

        publish_raw = self.publisher.get_num_connections() > 0
        publish_compressed = self.compressed_publisher.get_num_connections() > 0
        if not publish_raw and not publish_compressed:
            return
        try:
            if self.input_transport == "compressed":
                image = decode_compressed_image(message)
            else:
                # cv_bridge may return a read-only view of the incoming image.
                image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8").copy()
            hint = self.state.hint(self.ros.get_time(), self.timeout, **self.options)
            draw_hud(image, hint, self.anchor, self.size_ratio, self.opacity,
                     top_margin_ratio=self.top_margin_ratio, hud_scale=self.hud_scale,
                     text_gap_ratio=self.text_gap_ratio)
            if publish_raw:
                output = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
                output.header = message.header
                self.publisher.publish(output)
            if publish_compressed:
                output = self.bridge.cv2_to_compressed_imgmsg(image, dst_format="jpg")
                output.header = message.header
                self.compressed_publisher.publish(output)
        except (self.bridge_error, cv2.error, ValueError) as error:
            self.ros.logwarn_throttle(5.0, "DWA image conversion failed: %s", error)


def main():
    import rospy

    rospy.init_node("dwa_camera_arrow")
    try:
        node = CameraArrowNode()
    except ValueError as error:
        rospy.logfatal("DWA camera arrow configuration: %s", error)
        raise SystemExit(1)
    rospy.spin()
    return node


if __name__ == "__main__":
    main()

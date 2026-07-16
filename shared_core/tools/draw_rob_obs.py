#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rosbag
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe

# ==========================================================
# IEEE Style
# ==========================================================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.linewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none"
})

# ==========================================================
# User Parameters
# ==========================================================

bag_file = "/home/frienkie/rosbag1/rosbag19.bag"

SAVE_FIG = False
SAVE_NAME = "trajectory"

# sampling interval (s)
sample_interval = 0.2

# obstacle radius (m)
obstacle_radius = 0.15

# robot diameter represented by line width
robot_radius = 0.20

# figure margin
margin = 0.5

# more distinguishable than viridis
cmap = plt.cm.turbo

# line width (point)
robot_linewidth = 9

# outline width
outline_width = 13

# obstacle alpha
obstacle_alpha = 0.75

# ==========================================================
# Open rosbag
# ==========================================================

print("Opening rosbag...")

bag = rosbag.Bag(bag_file)

bag_start_time = bag.get_start_time()

# ==========================================================
# Read odom
# ==========================================================

print("Reading /odom...")

odom_time = []
robot_x = []
robot_y = []

for topic, msg, t in bag.read_messages(topics=["/odom"]):

    sec = t.to_sec() - bag_start_time

    odom_time.append(sec)
    robot_x.append(msg.pose.pose.position.x)
    robot_y.append(msg.pose.pose.position.y)

odom_time = np.asarray(odom_time)
robot_x = np.asarray(robot_x)
robot_y = np.asarray(robot_y)

if len(odom_time) == 0:
    raise RuntimeError("No /odom found.")

# ==========================================================
# Read obstacle states
# ==========================================================

print("Reading /dynamic_obstacle_states...")

obs_records = []

robot_track = []

next_sample_time = 0.0

for topic, msg, t in bag.read_messages(
        topics=["/dynamic_obstacle_states"]):

    sec = t.to_sec() - bag_start_time

    if sec < next_sample_time:
        continue

    next_sample_time += sample_interval

    idx = np.argmin(np.abs(odom_time - sec))

    rx = robot_x[idx]
    ry = robot_y[idx]

    # robot trajectory
    robot_track.append([
        sec,
        ry,     # horizontal axis
        rx      # vertical axis
    ])

    # obstacle
    for name, pose in zip(msg.name, msg.pose):

        if "obstacle" not in name.lower():
            continue

        obs_records.append({
            "time": sec,
            "x": pose.position.x,
            "y": pose.position.y
        })

bag.close()

if len(obs_records) == 0:
    raise RuntimeError("No obstacle found.")

robot_track = np.asarray(robot_track)

print("Robot samples :", len(robot_track))
print("Obstacle samples :", len(obs_records))

# ==========================================================
# Time normalization
# ==========================================================

times = robot_track[:, 0]

norm = Normalize(
    vmin=0,
    vmax=times.max()
)

# ==========================================================
# Prepare robot trajectory
# ==========================================================

points = robot_track[:, 1:3]

segments = np.stack(
    (
        points[:-1],
        points[1:]
    ),
    axis=1
)

segment_time = times[:-1]

# ==========================================================
# Figure
# ==========================================================

fig, ax = plt.subplots(
    figsize=(3.5, 3.5),
    dpi=300
)

print("Drawing...")

# ==========================================================
# Draw obstacles
# ==========================================================

for r in obs_records:

    color = cmap(norm(r["time"]))

    obstacle = Circle(
        (r["y"], r["x"]),          # horizontal:Y vertical:X
        obstacle_radius,
        facecolor=color,
        edgecolor='none',
        alpha=obstacle_alpha,
        zorder=1
    )

    ax.add_patch(obstacle)

# ==========================================================
# Draw robot trajectory (gradient + single outline)
# ==========================================================
# ==========================================================
# Draw robot trajectory (true robot width)
# ==========================================================

def data_radius_to_linewidth(ax, radius):
    """
    Convert radius in data coordinates (m)
    to matplotlib linewidth (pt).
    """
    fig = ax.figure

    # 必须先绘制一次，建立坐标变换
    fig.canvas.draw()

    trans = ax.transData

    p1 = trans.transform((0.0, 0.0))
    p2 = trans.transform((radius, 0.0))

    pixel_radius = abs(p2[0] - p1[0])

    # pixel -> point
    point_radius = pixel_radius * 72.0 / fig.dpi

    # linewidth表示直径
    return point_radius * 2


# ==========================================================
# Auto scale
# ==========================================================

obs_x = np.array([r["y"] for r in obs_records])
obs_y = np.array([r["x"] for r in obs_records])

robot_plot_x = robot_track[:, 1]
robot_plot_y = robot_track[:, 2]

all_x = np.concatenate((obs_x, robot_plot_x))
all_y = np.concatenate((obs_y, robot_plot_y))

ax.set_xlim(
    all_x.min() - margin,
    all_x.max() + margin
)

ax.set_ylim(
    all_y.min() - margin,
    all_y.max() + margin
)
# ==========================================================
# Goal line (X = 9.0 m)
# ==========================================================

goal_x = 9.0

ax.axhline(
    y=goal_x,
    color="red",
    linewidth=2.0,
    linestyle="--",
    zorder=3
)

# 可选：添加文字
ax.text(
    ax.get_xlim()[1] - 0.05,
    goal_x + 0.05,
    "Goal",
    color="red",
    fontsize=10,
    ha="right",
    va="bottom"
)
# ==========================================================
# Axis style (IEEE)
# ==========================================================

ax.set_aspect("equal")

ax.set_xlabel("Y [m]")
ax.set_ylabel("X [m]")

ax.tick_params(
    direction="in",
    width=1.0,
    length=4,
    top=True,
    right=True
)

for spine in ax.spines.values():
    spine.set_linewidth(1.0)

ax.grid(False)

# ==========================================================
# Compute true robot linewidth
# ==========================================================

robot_linewidth = data_radius_to_linewidth(
    ax,
    robot_radius
)

# 绿色边框厚度（pt）
green_border = 1.2

inner_linewidth = max(
    robot_linewidth - 2 * green_border,
    0.5
)

print("--------------------------------")
print("Robot radius      :", robot_radius, "m")
print("Robot linewidth   :", robot_linewidth, "pt")
print("Inner linewidth   :", inner_linewidth, "pt")
print("--------------------------------")

# ==========================================================
# Draw robot body (green)
# ==========================================================

robot_body = LineCollection(
    segments,
    colors="green",
    linewidth=robot_linewidth,
    capstyle="round",
    joinstyle="round",
    zorder=4
)

ax.add_collection(robot_body)

# ==========================================================
# Draw trajectory (time gradient)
# ==========================================================

lc = LineCollection(
    segments,
    cmap=cmap,
    norm=norm,
    linewidth=inner_linewidth,
    capstyle="round",
    joinstyle="round",
    zorder=5
)

lc.set_array(segment_time)

ax.add_collection(lc)

# ==========================================================
# Colorbar
# ==========================================================

sm = ScalarMappable(
    cmap=cmap,
    norm=norm
)

sm.set_array([])

cbar = plt.colorbar(
    sm,
    ax=ax,
    fraction=0.045,
    pad=0.03
)

cbar.set_label("Time [s]")

cbar.ax.tick_params(
    direction="in",
    width=1,
    length=3,
    labelsize=9
)

cbar.outline.set_linewidth(1.0)

# ==========================================================
# Tight layout
# ==========================================================

plt.tight_layout()

# ==========================================================
# Save
# ==========================================================

if SAVE_FIG:

    plt.savefig(
        SAVE_NAME + ".pdf",
        dpi=600,
        bbox_inches="tight",
        transparent=True
    )

    plt.savefig(
        SAVE_NAME + ".png",
        dpi=600,
        bbox_inches="tight"
    )

    print("Saved:")
    print("  " + SAVE_NAME + ".pdf")
    print("  " + SAVE_NAME + ".png")

# ==========================================================
# Show
# ==========================================================

plt.show()
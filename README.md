# ROS‑DWA‑shared

## Overview
ROS implementation of the Dynamic Window Approach (DWA) local planner for robot shared navigation. Given human velocities, it outputs velocity commands to steer a mobile base, avoiding obstacles in real time.

## Environment
- Ubuntu 20.04  
- ROS Noetic  

## Installation and Build
```bash
cd ~/catkin_ws/src
git clone https://github.com/frienkie/ROS-DWA-shared.git
cd ~/catkin_ws
catkin build ROS-DWA-shared -DCMAKE_BUILD_TYPE=Release
```

## Usage
```bash
rosrun shared_core dwa_remote.py --param 1

```
or
```bash
rosrun shared_core dwa.py --param 1

```
This will launch the DWA-shared controller.
dwa.py includes the time , route distance, rosbag recorder,collison detector and so on.You can see the detail function in distancetime.py. 

## Running Demo with Simulator
If using TurtleBot3 in Gazebo:
```bash
# Install TurtleBot3 packages
git clone -b noetic-devel https://github.com/ROBOTIS-GIT/turtlebot3.git
git clone -b noetic-devel https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git
git clone -b noetic-devel https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git
cd ~/catkin_ws
export TURTLEBOT3_MODEL=burger
```

## Topics
- Subscribed: `scan`, `cmd_vel_human`,`/filtered_scan`,`/odom`
- Published: `/cmd_vel`,`line_list`,`line_list_human`

## Simulation recording before motion

All nine `shared_core/sim/dwa*.py` controllers use a recording handshake when
their existing `RECORDER` switch is enabled. The controller publishes zero
velocity at 20 Hz while a separate Python recorder starts. The recorder first
writes a fresh, stationary `/odom` sample at the script's initial x/y position,
then a zero `/cmd_vel`. Dynamic controllers also wait for the first
`/dynamic_obstacle_states` sample; static controllers do not require it.
`/cmd_vel_human` is recorded when available and never blocks startup.

The recorder flushes this initial data before acknowledging readiness through
its own process pipe. Only then does the controller start its experiment timer
and publish motion commands. There is no fixed 0.8-second startup sleep.
Missing topics, an incorrect initial pose, or a write failure abort startup;
an active recorder failure also stops the controller's velocity output.

Bag numbering still comes from `/home/pc/result/data.xlsx`, with output in
`~/rosbag/rosbagN.bag`. While recording, the file has a `.bag.active` suffix.
Stopping a ready recorder drains pending messages and closes the bag before renaming it.
Existing recordings are not overwritten; failed recordings retain `.active`
files for inspection. The recorder uses ROS time for message receipt stamps
and wall time for its startup timeout, including when Gazebo is paused.

Private ROS parameters:

| Parameter | Default | Purpose |
| --- | --- | --- |
| `~rosbag_ready_timeout` | `5.0` s | Maximum time to wait for written initial data |
| `~rosbag_position_tolerance` | `0.02` m | Allowed x/y distance from the initial position |
| `~rosbag_velocity_tolerance` | `0.02` | Allowed initial linear/angular velocity components, in m/s and rad/s |
| `~rosbag_initial_position` | Script's initial x/y | Expected position **in the `/odom` frame**; override if Gazebo world and odometry origins differ |
| `~rosbag_require_obstacles` | Dynamic: `true`; static: `false` | Whether obstacle data must be written before motion |

The TurtleBot variants use their configured `(0, 1.5)` initial position.
For example, with `RECORDER` enabled:

```bash
rosrun shared_core dwa_simwaffi_midpoint.py _rosbag_ready_timeout:=10.0
```

Run the handshake, failure handling and shutdown tests without ROS:

```bash
python3 -m unittest discover -s shared_core/sim/tests -p 'test_rosbag_recording.py' -v
```

In Gazebo, verify that the saved bag's first `/odom` is at the expected initial
pose and precedes its first nonzero `/cmd_vel`. The bag includes the short
stationary preparation interval; the controller's experiment timer excludes it.

## LIMO simulation: repeated wheel TF timestamps

If `dwa_simwaffi_new.py` prints `TF_REPEATED_DATA` for
`rear_left_wheel_link` or `rear_right_wheel_link`, its `lidar3d.py` TF listener
is receiving duplicate wheel transforms. The DWA node does not publish them.
Do not run a synthetic `joint_state_publisher` alongside Gazebo's
`limo_state_controller`, or multiple robot state publishers for the same frames.

`ped_simulation/config.launch` now starts RViz only by default and uses the
robot description and TF already provided by the simulation. After updating,
stop the old `ped_simulation/config.launch`, restart the LIMO simulation launch
(an earlier duplicate node name may have stopped its robot state publisher),
then run:

```bash
roslaunch ped_simulation config.launch
rosrun shared_core dwa_simwaffi_new.py
```

For model preview without Gazebo or another robot bringup, use
`roslaunch ped_simulation config.launch standalone_model:=true`.
The `limo_viz/view_four_diff_model.launch` and `view_ackerman_model.launch`
files also start synthetic joint publishers; use the RViz-only
`limo_viz/view_robot.launch` when the robot simulation is running.

Verify the running graph with `rostopic info /joint_states` and
`rostopic info /tf`: the wheel states should come from the simulation's
controller, and each wheel TF should have one publisher. Other TF publishers
for different frames are normal.

## Camera Direction Arrow

Display a 3D direction arrow in the upper middle of the camera image using the
final `/cmd_vel` command:

```bash
roslaunch shared_core camera_arrow.launch image_topic:=/camera/image_raw
```

For compressed input, set `image_topic:=/camera/image_raw/compressed`.
View `/camera/image_dwa_arrow` (raw) or `/camera/image_dwa_arrow/compressed` (JPEG).
See [setup, direction semantics and parameters](shared_core/camera_arrow/README.md).

## Direct human control with trajectory comparison

```bash
rosrun shared_core direct_trajectory_compare.py
```

The standalone node in `shared_core/sim/` immediately forwards each complete
`/cmd_vel_human` Twist to `/cmd_vel`, without DWA selection or velocity limiting.
Run it as the active velocity controller; stop other `/cmd_vel` publishers first.
It sends a zero Twist when the node shuts down.

After both a command and `/odom` arrive, two predictions are updated at 20 Hz
using the same odometry pose, measured v/w and commanded target v/w. In RViz,
set Fixed Frame to the odometry frame (normally `odom`) and add two Marker displays:

| Marker topic | Color | Prediction |
| --- | --- | --- |
| `/direct_trajectory_compare/line_list` | Green | Original method without midpoint |
| `/direct_trajectory_compare/line_list_midpoint_vw` | Blue | `calc_trajectory_midpoint_vw` with linear and angular acceleration limits |

Defaults are `~dt:=0.4`, `~predict_time:=2.4`, `~max_accel:=2.5` and
`~max_dyawrate:=3.2`, matching `dwa_simwaffi_new_midpoint.py`. The original
method preserves its `while time <= predict_time` loop, so its last point can
extend one step past the horizon; midpoint v/w ends exactly at the horizon.
These are predictions of the latest command, rather than a traveled-path history.
The node does not repeat or expire commands; `/cmd_vel` follows the input rate.

For example, change the prediction interval and horizon with:

```bash
rosrun shared_core direct_trajectory_compare.py _dt:=0.2 _predict_time:=3.0
```

Other private parameters: `~input_topic` (`/cmd_vel_human`), `~output_topic`
(`/cmd_vel`), `~odom_topic` (`/odom`), `~rate` (20 Hz, markers only) and
`~line_width` (0.05 m).

## Parameters
Refer to Class `connfig` for tuning:  
```bash
def __init__(self):
        # robot parameter
        #NOTE good params:
        #NOTE 0.55,0.1,1.0,1.6,3.2,0.15,0.05,0.1,1.7,2.4,0.1,3.2,0.18
        self.max_speed = 0.20  # [m/s]
        self.min_speed = 0.0  # [m/s]
        self.max_yawrate = 0.6  # [rad/s]

        self.max_accel = 2.5  # [m/ss]
        self.max_dyawrate = 3.2  # [rad/ss]
        ##################################################33
        self.v_reso = 0.04  # [m/s]
        self.yawrate_reso = 0.04  # [rad/s]
        #######################################################
        self.dt = 0.4  # [s]
        self.predict_time = 2.4  # [s]
        self.showpredict_time = 3.0  # [s]
        self.showdt = 1.0
        self.goal_radius=0.4
```

## How it works
1. Sample possible velocities within acceleration limits.  
2. Simulate forward trajectories.  
3. Score trajectories based on obstacle proximity, path adherence, goal approaching, and speed.  
4. Send the highest-scoring velocity command.  
5. Repeat continuously.

## References
- D. Fox, W. Burgard, S. Thrun, *The dynamic window approach to collision avoidance*, IEEE RAM (1997)  

## License
BSD‑3‑Clause (same as ROS navigation stack)

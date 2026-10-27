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

## Simulation recording and trial results

All `shared_core/sim/dwa*.py` controllers start the standard
`rosbag record -O` command directly when `RECORDER` is enabled. The controller
continues immediately after launching it; there is no recording-readiness
handshake, topic/pose/velocity check, or additional startup sleep. Velocity
commands are published directly.

Robot coordinates are collected independently in each controller's `/odom`
callback, following the method in `shared_core/src/dwa.py`. Coordinates are
kept in sampling order, including return visits, while consecutive duplicate
points are omitted. Distance is the sum of successive x/y displacements;
steps below 0.0001 m are ignored as in the original helper. The first odometry
sample initializes the previous position, so a nonzero initial position does
not add an artificial distance from `(0, 0)`.

With `RECORDER` enabled, reaching the goal, requesting a save, or detecting a
collision saves the trial results once. The TurtleBot variants retain their
manual-save trigger. The ordered coordinates are written as a JSON list to
`~/cood/testN.txt`, for example `[[0.0, 0.0], [0.1, 0.0]]`, independently of
when rosbag connects to `/odom`. Bag output remains `~/rosbag/rosbagN.bag`.
Numbering uses the largest ID in `/home/pc/result/data.xlsx`, then adds one
(or starts at one when no IDs exist). Same-name files are overwritten.

`save_result` preserves the existing A/B/F fields and adds duration and distance
in C/D. Existing result rows are preserved when appending new trials.

| Excel column | Value |
| --- | --- |
| A | Bag/trial ID |
| B | First collision time relative to trial start, or 0 without a collision |
| C | Elapsed ROS time in seconds, rounded to two decimals |
| D | Accumulated distance in metres, rounded to two decimals |
| E | Reserved |
| F | Method ID, as before |

The timer starts immediately after launching rosbag. Duration and distance
are captured before saving files and waiting for rosbag to close.

All simulation DWA scripts have independent `RECORDER` (rosbag/results)
and `RECORDER_CSV` (CSV) switches. CSV initialization always defines `log_file`
and `log_writer`, setting both to `None` when disabled. Existing enabled/disabled
states are preserved. CSV currently writes its header; the per-candidate cost
row statements in the controllers remain commented out.

Run the direct-launch, coordinate and Excel result tests with:

```bash
python3 -m unittest discover -s shared_core/sim/tests -p 'test_rosbag_recording.py' -v
```

The Excel tests require `openpyxl`; the other tests do not require ROS.

## Midpoint transition followed by equation-based TTC

`shared_core/sim/dwa_simwaffi_new_midpoint_hybrid.py` is a separate controller
based on `dwa_simwaffi_new_midpoint.py`. It uses the new
`shared_core/sim/cal_ctime_midpoint.py` TTC solver. Run it in the existing ROS
simulation environment from the repository root:

```bash
python3 shared_core/sim/dwa_simwaffi_new_midpoint_hybrid.py
```

For each candidate `(v, w)`, reuse `t_v = abs(v-v0)/max_accel` and
`t_w = abs(w-w0)/max_dyawrate`; both velocities are constant at
`t_constant = max(t_v, t_w)`. Retain all midpoint-vw substep states through
that exact instant, including the internal velocity-switch states. Check
each consecutive segment against the translating, already inflated obstacle
rectangles and return the first collision time immediately. Collision times
within a substep use linear interpolation of its midpoint-generated positions.

If the transition is collision-free, advance each rectangle by its velocity
times `t_constant`, determine the rotation center from the robot's switch
pose, and transform the rectangle edges into the local coordinate frames.
Solve the boundary equations over `config.movetimemin - t_constant` and add
`t_constant` to the earliest valid root. Straight-line and stationary-center
motion use relative linear intersection. The obstacle model retains constant
translation and fixed orientation. Contact counts as collision.

The dynamic prediction horizon is `config.movetimemin`. If it ends during
acceleration, check only the transition through that horizon. With no collision
or no obstacles, the solver returns `config.movetimemin + 1`. Static-obstacle
evaluation and RViz retain the existing midpoint trajectory and static prediction
horizon. The two original programs are unchanged.

The hybrid controller checks TTC for every candidate that passes static-obstacle
collision detection, including lower-scoring candidates. It rejects
`TTC <= config.movetimemin`; a contact exactly at the horizon remains unsafe.
Obstacle edge geometry is copied and prepared once per control cycle. During
the constant-speed phase, conservative swept bounding boxes skip only obstacle
pairs that cannot meet; possible contacts still use the continuous equations
and the original root precision. After finding a collision, later obstacles
only need to be searched up to that earliest time. Candidate spacing and the
midpoint sampling interval are unchanged.

A ROS-free benchmark covers 440 candidates, crossing moving obstacles, dense
static clouds and a moving wall that blocks every candidate:

```bash
python3 shared_core/sim/benchmark_hybrid_ttc.py --repeats 9 --profile
```

Add `--baseline-ref c4f083a` to compare with the version before optimization and
verify matching commands and robot/human preview trajectories. Each current
benchmark scenario also checks that no statically feasible candidate skips TTC.
Reported Hz measures planning computation only; it excludes ROS callbacks,
publishing and Gazebo. The node still targets `rospy.Rate(20)`.

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

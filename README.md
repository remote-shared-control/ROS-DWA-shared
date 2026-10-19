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

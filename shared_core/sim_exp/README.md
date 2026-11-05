# 同步实验副本

此目录从相邻的 `sim` 完整复制（不含 Python 缓存）。10 个 DWA 控制器和
`direct.py` 均通过共同 ROS 仿真起点放行，DWA/TTC 计算、候选速度和已有记录开关
保持原值。`sim` 和包外的构建配置未修改。

## 两台机器怎样启动

先保留原来的 Gazebo、机器人和传感器启动流程。两端连接同一个 ROS Master，
配置各自可被另一端访问的 `ROS_IP` 或 `ROS_HOSTNAME`，并接收同一个 Gazebo `/clock`。
障碍物脚本会在节点初始化前自动设置 `/use_sim_time=true`；两个 launch 文件也已
包含该参数，不需要手动运行 `rosparam set`。

在 **master/Gazebo 端**，停止原来的障碍物移动脚本，再运行：

```bash
roslaunch shared_core sim_exp_obstacles.launch seed:=42 start_delay:=2.0
```

如果 master 端没有这个仓库，只需把本目录的 `dynamic_obstacles.py` 和
`experiment_protocol.py` 放在同一个目录，然后在已 source ROS Noetic 的终端运行：

```bash
python3 dynamic_obstacles.py _seed:=42 _start_delay:=2.0
```

等待日志 `Scene ready; waiting for a DWA experiment request`。
障碍物已生成，此时保持静止。

在 **DWA 所在机器**，从已加载工作区环境的终端运行：

```bash
roslaunch shared_core sim_exp_controller.launch controller:=dwa_simwaffi_new_midpoint_hybrid.py
```

`controller` 可替换为本目录中任意 `dwa*.py` 或 `direct.py`。
先启动 master 端障碍物脚本后，也可在仓库根目录直接运行 DWA：

```bash
python3 shared_core/sim_exp/dwa_simwaffi_new_midpoint_hybrid.py
```

复制的同名脚本不设置可执行位，使用 Python、launch 或 `run_experiment.py` 选择副本，
因此原有 `rosrun shared_core dwa_*.py` 仍选择原目录的程序。每次只保留一个速度控制器；停止上次的 DWA 后再启动
下一次。master 端障碍物节点可以持续运行，每次请求都会恢复相同 seed 的初始场景。

## 同步流程

1. DWA 初始化后持续发送零速度，调用 `/sim_exp/prepare`。
2. 障碍物端复位全部模型、发布静止初始帧，再分配本次 `run_id`。
3. DWA 收到里程计和本次准备状态后调用本次的 `arm` 服务。
4. master 端预约 `t0 = 当前 ROS 时间 + start_delay` 并返回；重复 arm 不会改变 t0。
5. 两边到 t0 才允许运动。DWA 的实验时间和障碍物轨迹时间均为 `now - t0`。
6. 到达 70 秒，双方停止；DWA 退出、心跳丢失、服务错误或时钟回退也会中止实验。

服务采用 ROS 自带的 `std_srvs/Trigger`，响应的 `message` 是 JSON，包含 run_id、
t0（整数纳秒）、时长和阶段，无需生成自定义服务或修改相邻的 CMake/package.xml。
每个 run_id 有独立的 `/sim_exp/runs/<run_id>/arm` 和 `cancel` 服务，旧客户端不能
取消下一次实验。`/sim_exp/state` 发布带编号的锁存状态，DWA 忽略上次的完成状态。
`/dynamic_obstacle_states`、`/dynamic_obstacle_finished` 仍保留，实验副本使用前者
读取障碍物，使用带编号的 state 判断本次结束。

服务响应超过 `~experiment_service_timeout`（默认 30 秒）会拒绝启动；如果 arm
响应到达时离 t0 少于 `~experiment_start_margin`（默认 0.2 秒），DWA 会请求取消。
prepare 响应丢失时，未认领的预约在 30 秒墙上时间后失效。DWA 发送本次编号的心跳；
master 默认在 3 秒无心跳后冻结障碍物。DWA 同样在 3 秒无状态后持续输出零速度。
网络完全断开时，远端停止由该超时保障，无法保证即时取消。

原来的 `RECORDER` / `RECORDER_CSV` 开关保留。启用 rosbag 时，记录仍在等待起点前
启动，因此 bag 可包含准备阶段；结果计时和坐标距离从本次 t0 放行处开始计算。

## 精度和核对

共同 t0 对齐的是控制使能及轨迹时间基准。DWA 第一条速度需要规划计算，Python 调度、
网络、传感器和 Gazebo 设置模型服务也会引入延迟。现有障碍物更新是 20 Hz，40 个
模型通过服务逐个设置，所以这不保证同一物理步发生运动。

日志会记录本次编号、t0、第一条非零 cmd_vel 的发布时间，以及第一帧移动障碍物
更新的开始时间。用它们比较软件放行偏差；若要测实际运动，应再核对 Gazebo 模型
状态或 odom。需要同一个物理步时，应在 Gazebo 插件内统一应用控制。

发生 `/reset_simulation` 后，本次作废、双方停止；重新启动 DWA 发起新实验。
不要在运行中同时启动旧 `movetest.py` 或原来的自动运动脚本。

## 本地验证

无需 ROS 的同步状态及控制器测试：

```bash
python3 -m unittest discover -s shared_core/sim_exp/tests -p 'test_experiment*.py' -v
```

完整的副本算法回归测试还需要 NumPy，Excel 测试需要 openpyxl：

```bash
python3 -m unittest discover -s shared_core/sim_exp/tests -v
```

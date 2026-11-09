# 同步实验副本

此目录从相邻的 `sim` 完整复制（不含 Python 缓存）。10 个 DWA 控制器和
`direct.py` 均通过共同 ROS 仿真起点放行，DWA/TTC 计算、候选速度和已有记录开关
保持原值。`sim` 和包外的构建配置未修改。

## Robot 端 Qt GUI

Ubuntu 20.04 / ROS Noetic 推荐使用系统 Python 和 PyQt5。robot 端安装一次界面依赖：

```bash
sudo apt install python3-pyqt5
```

两端需要 ROS 标准包 `dynamic_reconfigure`（它只提供已有的消息/服务类型，不需要
单独运行服务节点，也不需要生成自定义消息）。若当前 ROS 安装不含此包，安装一次：

```bash
sudo apt install ros-noetic-dynamic-reconfigure
```

先启动原来的 Gazebo、机器人、传感器及人工速度输入节点。在 master 端更新本目录的
`dynamic_obstacles.py` 和 `experiment_protocol.py`，将它们放在同一个目录，然后只运行：

```bash
python3 dynamic_obstacles.py
```

master 端不需要 GUI，也不需要每次修改 seed 参数或重启障碍物程序。

在 robot 端已经 source ROS 和工作区的终端，从仓库根目录打开 GUI：

```bash
python3 shared_core/sim_exp/experiment_gui.py
```

也可以使用：

```bash
roslaunch shared_core sim_exp_gui.launch
```

1. 等待界面显示 `master 已连接`。
2. 输入 seed（0～2147483647，默认 42），选择一个 DWA 程序。
3. 点击“启动实验”。seed 随本次服务请求原子地传给 master，master 重新生成/复位
   对应场景；DWA 与障碍物按共同 t0 放行。
4. GUI 显示 master 实际接受的 seed、阶段、70 秒仿真进度和 DWA 输出日志。
5. 70 秒结束后 GUI 自动关闭本次 DWA 子进程，master 保持待命；可选择新 seed/算法
   再次启动。命令行直接启动的 DWA 仍保留原来的“结束后持续发布零速度”行为。

“停止实验”和关闭 GUI 都会停止 GUI 自己启动的 DWA，并取消它对应的远端实验。
子进程有独立进程组，超时清理也覆盖其 rosbag 等子进程。GUI 不会取消其他编号的实验；
master 正在运行另一场实验时，启动按钮不可用。GUI 与 master 的状态连接超时，也会
停止本次子进程。不要同时运行另一个手动 DWA / cmd_vel 控制器。

界面会记住上次的 seed 和程序选择。`--namespace /sim_exp` 可选择实验命名空间；
`--preview` 可在不连接 ROS、不启动 DWA 的情况下预览界面。较新 Python 环境也可使用
PySide6 核心窗口组件作为兼容绑定；Ubuntu Noetic 上优先使用上面的 PyQt5 安装方式。

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

无 seed 的命令行启动采用 ROS 自带的 `std_srvs/Trigger`。GUI 启动使用标准
`dynamic_reconfigure/Reconfigure` 调用 `/sim_exp/prepare_seed`，在同一次请求中传送
seed、GUI 启动编号和程序名，并完成场景准备；运行中拒绝改 seed。两种响应都携带
JSON 实验状态（包括 run_id、seed、启动编号、t0、时长和阶段），无需生成自定义服务
或修改相邻的 CMake/package.xml。旧命令行启动沿用 master 当前选中的 seed。
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

安装 PyQt5 后，上面的命令也会运行真实 Qt 控件和 QProcess 子进程的离屏测试；
缺少 Qt 时只跳过界面测试。可单独运行界面测试：

```bash
QT_QPA_PLATFORM=offscreen python3 -m unittest discover -s shared_core/sim_exp/tests -p 'test_experiment_gui_qt.py' -v
```

完整的副本算法回归测试还需要 NumPy，Excel 测试需要 openpyxl：

```bash
python3 -m unittest discover -s shared_core/sim_exp/tests -v
```

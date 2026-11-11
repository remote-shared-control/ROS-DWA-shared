# 同步实验副本

此目录从相邻的 `sim` 完整复制（不含 Python 缓存）。10 个 DWA 控制器和
`direct.py` 均通过共同 ROS 仿真起点放行，DWA/TTC 计算、候选速度和已有记录开关
保持原值。`sim` 和包外的构建配置未修改。

## 选择场景脚本

| master 端脚本 | 场景 | launch 入口 |
| --- | --- | --- |
| `dynamic_obstacles.py` | 原场景，障碍物沿 `-x` 移动，默认速度 0.6～1.0 m/s | `sim_exp_obstacles.launch` |
| `dynamic_obstacle1.py` | 新场景，3 m 处向右横穿、6 m 处向左横穿，默认速度 0.6～0.8 m/s | `sim_exp_obstacles1.launch` |

原来的 `dynamic_obstacles.py` 保留横穿修改之前的实现。两份脚本都支持现有 GUI、
seed、数量、时间限制和多余模型删除。Robot 端 GUI 不需要调整，master 运行的脚本决定场景。
切换前停止当前实验和场景程序，两份 master 脚本二选一运行，使用同一组服务和模型名。

## 新场景 dynamic_obstacle1.py：两处反向横穿

以机器人初始朝世界坐标 `+x` 前进为参照，左侧为 `+y`，右侧为 `-y`：

| 分组 | 横穿位置 | 方向 | 发布速度 |
| --- | --- | --- | --- |
| `obstacle_1`、`obstacle_3` 等奇数编号 | `x=3 m` | 向右，沿 `-y` | `vx=0`，`vy=-0.6～-0.8 m/s` |
| `obstacle_2`、`obstacle_4` 等偶数编号 | `x=6 m` | 向左，沿 `+y` | `vx=0`，`vy=+0.6～+0.8 m/s` |

GUI 设置的数量是两组总数，奇数数量时 3 m 组多一个。每个模型的速度由 seed
生成，在本次实验中保持恒定；seed 也决定初始 y 分布。模型只沿 y 移动，x 保持不变。
默认 y 范围为 `[-7,7] m`，离开边界后从另一侧重新进入，方向和速度不变。
所有段按 ROS 经过时间预先计算，因此循环延迟和采样顺序不会改变场景轨迹。

master 参数 `~right_crossing_x`、`~left_crossing_x` 可调整两个横穿位置，
`~y_min`、`~y_max` 调整循环边界；默认速度参数为 `~min_speed=0.6`、`~max_speed=0.8`。
等待开始、取消或到达时间限制时，发布的 vx/vy 都为零。Gazebo 模型由绝对位置更新，
其内部 twist 保持零，DWA 从 `/dynamic_obstacle_states` 获取真实规划速度。

数量增减、seed、时间限制和多余模型删除仍由原 GUI/服务流程管理。横穿场景的 master
运行 `python3 dynamic_obstacle1.py`，或 `roslaunch shared_core sim_exp_obstacles1.launch`。
场景的通行性需要结合数量、seed 和控制器进行闭环验证。

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

先启动原来的 Gazebo、机器人、传感器及人工速度输入节点。在 master 端将选中的场景
脚本和 `experiment_protocol.py` 放在同一个目录。原场景运行：

```bash
python3 dynamic_obstacles.py
```

新横穿场景则运行：

```bash
python3 dynamic_obstacle1.py
```

master 端不需要 GUI，也不需要每次修改 seed、数量、时限参数或重启障碍物程序。

在 robot 端已经 source ROS 和工作区的终端，从仓库根目录打开 GUI：

```bash
python3 shared_core/sim_exp/experiment_gui.py
```

也可以使用：

```bash
roslaunch shared_core sim_exp_gui.launch
```

1. 等待界面显示 `master 已连接`。
2. 输入 seed（0～2147483647，默认 42）、障碍物数量（0～1000，默认 40）、时间限制
   （0.1～86400 秒，默认 70 秒），选择一个 DWA 程序。时间限制采用 ROS 仿真时间。
3. 点击“启动实验”。这些设置在同一次服务请求中传给 master。master 查询实际世界，
   删除多余模型、补建缺少模型、复位场景并核对数量；准备成功后按共同 t0 放行。
4. GUI 显示 master 接受的 seed、数量、阶段、所设时限对应的进度和 DWA 输出日志。
5. 到达设置的时间限制后 GUI 自动关闭本次 DWA 子进程，master 保持待命；可选择新配置
   再次启动。命令行直接启动的 DWA 仍保留原来的“结束后持续发布零速度”行为。

“停止实验”和关闭 GUI 都会停止 GUI 自己启动的 DWA，并取消它对应的远端实验。
子进程有独立进程组，超时清理也覆盖其 rosbag 等子进程。GUI 不会取消其他编号的实验；
master 正在运行另一场实验时，启动按钮不可用。GUI 与 master 的状态连接超时，也会
停止本次子进程。不要同时运行另一个手动 DWA / cmd_vel 控制器。

界面会记住上次的 seed、数量、时间限制和程序选择。运行时输入框锁定，下一次实验再应用新值。
旧版 master 不支持数量/时限控制时，GUI 会禁用启动按钮，需同时更新上述两个 master 文件。
`--namespace /sim_exp` 可选择实验命名空间；
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
2. 障碍物端分配本次 `run_id`，进入 PREPARING；核对实际模型、删除/补建并复位后，
   发布静止初始帧，进入 PREPARED。准备期间持续更新状态，尚未允许运动。
3. DWA 收到里程计和本次准备状态后调用本次的 `arm` 服务。
4. master 端预约 `t0 = 当前 ROS 时间 + start_delay` 并返回；重复 arm 不会改变 t0。
5. 两边到 t0 才允许运动。DWA 的实验时间和障碍物轨迹时间均为 `now - t0`。
6. 到达本次时间限制（默认 70 秒），双方停止；DWA 退出、心跳丢失、服务错误或时钟回退也会中止实验。

无 seed 的命令行启动采用 ROS 自带的 `std_srvs/Trigger`。GUI 启动使用标准
`dynamic_reconfigure/Reconfigure` 调用 `/sim_exp/prepare_seed`，在同一次请求中传送
seed、障碍物数量、时限、GUI 启动编号和程序名，并完成场景准备；运行中拒绝改配置。
整数参数使用 `seed`、`num_obstacles`，浮点参数使用 `duration`。两种响应都携带
JSON 实验状态（包括 run_id、seed、数量、启动编号、t0、时长和阶段），无需生成自定义服务
或修改相邻的 CMake/package.xml。旧命令行启动沿用 master 当前选中的场景配置。
每个 run_id 有独立的 `/sim_exp/runs/<run_id>/arm` 和 `cancel` 服务，旧客户端不能
取消下一次实验。`/sim_exp/state` 发布带编号的锁存状态，DWA 忽略上次的完成状态。
`/dynamic_obstacle_states`、`/dynamic_obstacle_finished` 仍保留，实验副本使用前者
读取障碍物，使用带编号的 state 判断本次结束。

服务响应超过 `~experiment_service_timeout`（默认 30 秒）会拒绝启动；如果 arm
响应到达时离 t0 少于 `~experiment_start_margin`（默认 0.2 秒），DWA 会请求取消。
prepare 响应丢失时，未认领的预约在 30 秒墙上时间后失效。DWA 发送本次编号的心跳；
master 默认在 3 秒无心跳后冻结障碍物。DWA 同样在 3 秒无状态后持续输出零速度。
网络完全断开时，远端停止由该超时保障，无法保证即时取消。

## 多余障碍物的删除

master 启动时及每次实验准备时，都通过 `/gazebo/get_world_properties` 查询实际模型，
管理名称匹配 `obstacle_<整数>` 的模型。若本次数量是 N，保留 `obstacle_1` 到
`obstacle_N`，通过 `/gazebo/delete_model` 删除其余受管理模型，并补建缺少的模型。
这个过程能清理上次程序退出后遗留在 Gazebo 中的模型，不依赖进程内记录的旧数量。

例如从 40 个改为 10 个，会真正删除 `obstacle_11` 到 `obstacle_40`；设为 0 会删除
全部受管理动态障碍物。机器人、地面和 `obstacle_static` 等非数字后缀模型会保留。
删除/补建后再次查询世界，确认实际模型集合与目标一致，才允许开始计时和移动。
任何删除失败、缺失模型或残留模型都会阻止启动，界面和日志显示失败原因。

到达时限后，本次保留的 N 个模型按原方式冻结；下一次调低数量时会删除多出来的模型。
master 端只需运行所选的一个场景脚本，所有 Gazebo 服务调用都在该进程内完成。

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

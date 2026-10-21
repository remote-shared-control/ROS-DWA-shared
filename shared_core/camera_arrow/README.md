# 在相机图像上显示 DWA 方向箭头

`camera_arrow/dwa_camera_arrow.py` 直接订阅最终输出的 `Twist`，取
`linear.x = v` 和 `angular.z = w`，在相机图像的顶部中央
绘制具有透视、厚度和明暗面的 3D 箭头。现有 DWA 程序已经向 `/cmd_vel`
发布最终选择的速度，因此无需改动 `robot_trajectory.set_trajectory(best_traj)`，
也不需要重新预测轨迹或转换 `odom` 坐标。

程序、测试和本说明统一位于与 `sim` 平级的 `camera_arrow/` 目录；
启动文件仍位于 `launch/camera_arrow.launch`，内容保持不变。

默认将整组提示贴近图像顶部，仅保留防止抗锯齿边缘被裁切的少量像素。
文字位于箭头上方，文字与箭头的间距先减为原来的 50%，再随整体缩放；
箭头、文字和线条整体按原来的 70% 显示。
各方向均按完整边界对齐，避免裁切倒车或转向箭头。
绘制时只复制并透明混合提示所在的小区域，不再为叠加效果复制和混合整幅图像。

视觉设计借鉴 [《Crazy Taxi》的顶部方向箭头](https://store.steampowered.com/manual/71230)
和 [《Dead Space》的蓝色全息指引](https://www.mobygames.com/game/37332/dead-space/screenshots/xbox360/404524/)，
由 OpenCV 实时绘制独立的箭头网格。移动方向使用青蓝色顶面、细高光边和较薄的立体侧面；
原地转向及停止使用琥珀色，命令不可用时使用灰色。文字继续位于箭头上方，
深色切角底板和状态色标改善明暗背景中的可读性。

当前为速度优先的简化样式：使用三层轮廓绘制立体侧面，省去逐面光照排序、
高斯阴影、逐像素渐变和多轮遮罩混合，只在最后对局部区域做一次透明混合。
最近 128 种精确布局会被缓存，键包含方向、文字、分辨率及布局参数；方向不做量化。
目标为 720p 下单帧绘制约 0.25 ms，实际取决于硬件与负载，不包含图像解码、编码或 ROS 传输。

## 方向计算

以机器人自身为参考：`+X` 向前，`+Y` 向左，`w > 0` 表示左转。
假设选中的 `v、w` 在预览时间 `T` 内保持不变，圆弧位移的方位角为：

```text
前进：angle = w * T / 2
倒车：angle = w * T / 2 + pi
```

这个角度表示预览位移相对机器人的方向。它不同于预览结束时机器人的航向
`w*T`。默认 `T = 1.0 s`，预览扫角 `w*T` 限制在 ±120°，用于粗略提示方向。
画面上的箭头大小固定，不表示速度大小或真实距离。

| 速度状态 | 显示 |
| --- | --- |
| `v > 0, w = 0` | 箭头向画面上方延伸，表示前进 |
| `v > 0, w > 0 / w < 0` | 箭头向左上 / 右上偏转 |
| `v < 0, w < 0` | 左后方箭头 |
| `v < 0, w = 0` | 正后方箭头 |
| `v < 0, w > 0` | 右后方箭头 |
| `abs(v) <= 0.02, abs(w) > 0.05` | 环形 3D 箭头，表示原地左转 / 右转 |
| 两者均在死区内 | `STOP` 标记 |
| 尚无命令 / 命令超过 0.5 秒未更新 | 灰色状态标记，清除移动箭头 |

例如 `v < 0, w > 0` 时，机器人航向向左旋转，但后退位移朝右后方，
所以显示 `REVERSE RIGHT`。

这是固定在图像上的机器人方向 HUD，不与真实地面配准；无需相机内参或 TF。
若要箭头与地面或物体对齐，则需要另做相机标定和空间投影。

## 启动

在 ROS Noetic / Python 3 环境中安装依赖并重新构建本包：

```bash
sudo apt-get install ros-noetic-cv-bridge python3-opencv python3-numpy
cd ~/catkin_ws
catkin build shared_core
source devel/setup.bash
```

运行现有相机和 DWA 节点，再启动叠加节点。按实际相机话题替换
`/camera/image_raw`。支持原始彩色 / 灰度 `sensor_msgs/Image`，
以及普通 JPEG / PNG `sensor_msgs/CompressedImage`；不支持深度图或 `compressedDepth`。

raw 输入：

```bash
roslaunch shared_core camera_arrow.launch image_topic:=/camera/image_raw
```

compressed 输入：

```bash
roslaunch shared_core camera_arrow.launch image_topic:=/camera/image_raw/compressed
```

默认根据输入话题是否以 `/compressed` 结尾自动选择消息类型，
因此上述两种方式都不需要修改 launch 文件。使用了 ROS 话题重映射时，
根据重映射后的完整话题名判断。

也可以单独运行：

```bash
rosrun shared_core dwa_camera_arrow.py _image_topic:=/camera/image_raw
```

如果压缩图像的话题名没有 `/compressed` 后缀，可显式指定类型：

```bash
rosrun shared_core dwa_camera_arrow.py \
  _image_topic:=/camera/jpeg _input_transport:=compressed
```

两种输入均可生成以下输出，例如在已有的 `rqt_image_view` 中选择：

| 输出话题 | 消息类型 / 编码 |
| --- | --- |
| `/camera/image_dwa_arrow` | `sensor_msgs/Image`，`bgr8` |
| `/camera/image_dwa_arrow/compressed` | `sensor_msgs/CompressedImage`，JPEG |

输出保留原图的 `header`、时间戳、`frame_id` 和分辨率。
各输出仅在有订阅者时转换或编码；两者同时被订阅时只绘制一次箭头。
源图像和源话题保持不变。

如果最终速度经过其他节点处理后发布到不同话题，指定相应话题：

```bash
roslaunch shared_core camera_arrow.launch \
  image_topic:=/camera/color/image_raw \
  velocity_topic:=/cmd_vel \
  preview_time:=1.2
```

节点显示订阅到的最终命令。如果 DWA 切换到人工模式且继续向同一话题发速度，
箭头也会显示人工模式下的实际命令。`Twist` 本身没有时间戳，超时按接收时刻
计算，HUD 使用最新命令，不做与图像曝光时刻的严格同步。

## 参数

以下均为节点私有参数，可通过 `rosrun ... _参数名:=值` 设置。
已有 launch 文件提供 `image_topic`、`velocity_topic`、`output_topic`、`preview_time`
四个启动参数，其余参数使用默认值：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `image_topic` | `/camera/image_raw` | raw 或 compressed 输入图像的完整话题名 |
| `input_transport` | `auto` | `auto` / `raw` / `compressed`；自动模式按话题后缀判断 |
| `velocity_topic` | `/cmd_vel` | 最终 `geometry_msgs/Twist` |
| `output_topic` | `/camera/image_dwa_arrow` | raw 输出；compressed 输出自动添加 `/compressed` 后缀；两者均不能与输入相同 |
| `preview_time` | `1.0` | 方向预览时间，单位秒 |
| `anchor_x` | `0.5` | 水平方向的相对中心位置 |
| `top_margin_ratio` | `0.0` | 整组提示距顶部的留白占图像高度的比例；`-1` 使用原来的 `anchor_y` 中心定位 |
| `anchor_y` | `0.28` | 仅在 `top_margin_ratio=-1` 时决定垂直中心位置 |
| `size_ratio` | `0.32` | 基准尺寸相对图像短边的比例，再乘以 `hud_scale` |
| `hud_scale` | `0.7` | 箭头、文字和阴影的整体缩放比例 |
| `text_gap_ratio` | `0.5` | 文字与箭头的间距相对原来间距的比例，再随整体缩放 |
| `opacity` | `0.9` | 箭头不透明度 |
| `command_timeout` | `0.5` | 速度命令超时，单位秒 |
| `linear_deadband` | `0.02` | 线速度死区，单位 m/s |
| `angular_deadband` | `0.05` | 角速度死区，单位 rad/s |

例如，使用当前的紧凑布局：

```bash
rosrun shared_core dwa_camera_arrow.py _hud_scale:=0.7 _text_gap_ratio:=0.5
```

## 离线验证

无需 ROS，安装 NumPy / OpenCV 后可运行方向计算、渲染、命令超时、raw / compressed
图像转换、订阅类型及局部绘制边界测试；也可单独运行已有轨迹选择回归测试：

```bash
python3 -B -m unittest discover -s shared_core/camera_arrow/tests -v
python3 -B -m unittest discover -s shared_core/sim/tests -v
```

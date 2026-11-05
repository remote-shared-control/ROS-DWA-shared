#!/usr/bin/env python3
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState, SpawnModel
from geometry_msgs.msg import Pose
import os

DEFAULT_CYLINDER_SDF = """<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{model_name}">
    <static>false</static>
    <link name="link">
      <pose>0 0 0 0 0 0</pose>
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>{radius}</radius>
            <length>{length}</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>{radius}</radius>
            <length>{length}</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.1 0.5 0.5 1</ambient>
          <diffuse>0.1 0.8 0.8 1</diffuse>
        </material>
      </visual>
      <inertial>
        <mass>1.0</mass>
      </inertial>
    </link>
  </model>
</sdf>
"""

HARDCODED_MODELS = [
    {
        'name': 'linear_cylinder1',
        'height': 0.15,
        'init_x': 10.5,
        'init_y': 0.0,
        'linear_vx': -1.0,
        'linear_vy': 0.0,
        'length': 0.3,
        'cylinder_radius': 0.1,
        'model_sdf_path': '',
    },
    # 可以在这里继续添加更多模型配置
    {
        'name': 'linear_cylinder2',
        'init_x': 10.5,
        'init_y': -0.70,
        'linear_vx': -1.0,
        'linear_vy': 0.0,
        'height': 0.2,
        'length': 0.3,
        'cylinder_radius': 0.1,
        'model_sdf_path': '',
    },
]


def load_model_xml(model_name, length, radius, sdf_path):
    if sdf_path:
        if not os.path.isfile(sdf_path):
            raise FileNotFoundError("Provided model_sdf_path does not exist: {}".format(sdf_path))
        with open(sdf_path, 'r', encoding='utf-8') as sdf_file:
            return sdf_file.read()
    return DEFAULT_CYLINDER_SDF.format(model_name=model_name, radius=radius, length=length)


def ensure_model_exists(model_name, pose, length, radius, sdf_path):
    rospy.wait_for_service('/gazebo/get_model_state')
    get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

    try:
        resp = get_model_state(model_name, 'world')
        if resp.success:
            rospy.loginfo("Model '%s' already exists, waiting for stabilization...", model_name)
            rospy.sleep(0.5)  # 等待模型状态稳定
            return
    except rospy.ServiceException as exc:
        rospy.logwarn("Failed to check model %s: %s", model_name, exc)

    rospy.loginfo("Model '%s' not found, spawning...", model_name)
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

    model_xml = load_model_xml(model_name, length, radius, sdf_path)

    try:
        spawn_model(model_name, model_xml, '', pose, 'world')
        rospy.loginfo("Model '%s' spawned successfully.", model_name)
        rospy.sleep(0.5)  # 等待模型生成完成
    except rospy.ServiceException as exc:
        rospy.logerr("Failed to spawn model '%s': %s", model_name, exc)
        raise


def load_models_config():
    models_param = rospy.get_param('~models', None)
    models = []

    if models_param:
        for idx, cfg in enumerate(models_param):
            model_cfg = HARDCODED_MODELS[0].copy()
            model_cfg.update(cfg)
            model_cfg['name'] = cfg.get('name', f'linear_cylinder_{idx + 1}')
            models.append(model_cfg)
    else:
        for cfg in HARDCODED_MODELS:
            models.append(cfg.copy())

    return models


def main():
    rospy.init_node('cylinder_linear_controller')
    rospy.loginfo("Waiting for Gazebo clock...")
    while rospy.Time.now().to_sec() == 0 and not rospy.is_shutdown():
        rospy.sleep(0.1)
    rospy.loginfo("Gazebo clock ready.")
    rospy.sleep(2.0)
    models = load_models_config()

    rospy.wait_for_service('/gazebo/set_model_state')
    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

    for model in models:
        spawn_pose = Pose()
        spawn_pose.position.x = model['init_x']
        spawn_pose.position.y = model['init_y']
        spawn_pose.position.z = model['height']

        ensure_model_exists(
            model['name'],
            spawn_pose,
            model['length'],
            model['cylinder_radius'],
            model['model_sdf_path'],
        )

    # 等待所有模型准备就绪
    rospy.loginfo("All models ready, starting control loop...")
    rospy.sleep(0.5)
    
    rate = rospy.Rate(20)
    start_time = rospy.get_time()

    while not rospy.is_shutdown():
        t = rospy.get_time() - start_time

        for model in models:
            state = ModelState()
            state.model_name = model['name']
            state.reference_frame = 'world'  # 设置参考坐标系
            state.pose.position.x = model['init_x'] + model['linear_vx'] * t
            state.pose.position.y = model['init_y'] + model['linear_vy'] * t
            state.pose.position.z = model['height']
            
            # 设置默认姿态（无旋转）
            state.pose.orientation.x = 0.0
            state.pose.orientation.y = 0.0
            state.pose.orientation.z = 0.0
            state.pose.orientation.w = 1.0

            state.twist.linear.x = model['linear_vx']
            state.twist.linear.y = model['linear_vy']
            state.twist.linear.z = 0.0
            state.twist.angular.x = 0.0
            state.twist.angular.y = 0.0
            state.twist.angular.z = 0.0

            try:
                resp = set_state(state)
                if not resp.success:
                    rospy.logwarn("Failed to update model %s: service returned success=False", model['name'])
            except rospy.ServiceException as exc:
                rospy.logwarn("Failed to update model %s: %s", model['name'], exc)

        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        pass
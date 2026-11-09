#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configurable Gazebo Classic / ROS Noetic moving-obstacle experiments.

Use the same ~seed to produce exactly the same obstacle trajectories as a
function of elapsed ROS simulation time. Changing ~seed changes the scene.
"""

import bisect
import heapq
import random
import re
import json
import threading
import time

import rospy
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import (GetModelState, SetModelState, SpawnModel,
                             GetWorldProperties, DeleteModel)
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import Int32, String
from std_srvs.srv import Trigger, TriggerResponse
from dynamic_reconfigure.msg import Config as DynamicConfig, IntParameter, StrParameter, DoubleParameter
from dynamic_reconfigure.srv import Reconfigure, ReconfigureResponse

from experiment_protocol import (ACTIVE, NSEC, Session, validate_seed,
                                 validate_obstacle_count, validate_duration)

CONFIG = {
    'num_obstacles': 40,
    'x_min': 0.5,
    'x_max': 14.0,
    'y_min': -4.0,
    'y_max': 4.0,
    'min_speed': 0.6,
    'max_speed': 1.0,
    'radius': 0.15,
    'length': 0.4,
    'height': 0.2,
    'rate': 20,
    'duration': 70.0,
    'seed': 42,
    'respawn_time_window': 4.0,
    'min_y_gap': 0.8,
}

CYLINDER_SDF = """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>false</static>
    <kinematic>true</kinematic>
    <link name="link">
      <gravity>false</gravity>
      <collision name="collision">
        <geometry>
          <cylinder><radius>{radius}</radius><length>{length}</length></cylinder>
        </geometry>
        <surface><contact><collide_without_contact>true</collide_without_contact></contact></surface>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder><radius>{radius}</radius><length>{length}</length></cylinder>
        </geometry>
        <material>
          <ambient>0.2 0.8 0.2 1</ambient>
          <diffuse>0.2 0.8 0.2 1</diffuse>
        </material>
      </visual>
      <inertial><mass>1.0</mass></inertial>
    </link>
  </model>
</sdf>"""


def make_pose(x, y, z):
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    pose.orientation.w = 1.0
    return pose


def choose_respawn_y(rng, history, t, cfg):
    """Separate recent entry lanes, selecting best available if saturated."""
    window = cfg['respawn_time_window']
    recent = [y for previous_t, y in history if t - previous_t < window]
    low, high = cfg['y_min'], cfg['y_max']

    for _ in range(200):
        y = rng.uniform(low, high)
        if all(abs(y - other) >= cfg['min_y_gap'] for other in recent):
            history.append((t, y))
            return y

    # At high respawn density no 0.8 m lane may exist. Use the safest gap.
    candidates = [low + (high - low) * i / 160.0 for i in range(161)]
    rng.shuffle(candidates)  # deterministic tie-breaking under the seed
    y = max(candidates, key=lambda cand: min(
        (abs(cand - old) for old in recent), default=float('inf')))
    history.append((t, y))
    return y


def plan_scene(cfg):
    """Precompute every trajectory segment so callback delays cannot affect it.

    Segment tuple: (start_elapsed_time, start_x, y, speed).
    """
    rng = random.Random(cfg['seed'])
    n = cfg['num_obstacles']
    if n == 0:
        return [], []
    nx = 5 if n >= 20 else max(1, round(n ** 0.5))
    ny = (n + nx - 1) // nx
    dx = (cfg['x_max'] - cfg['x_min']) / nx
    dy = (cfg['y_max'] - cfg['y_min']) / ny

    cells = [(col, row) for row in range(ny) for col in range(nx)]
    rng.shuffle(cells)
    cells = cells[:n]

    tracks = []
    for idx, (col, row) in enumerate(cells):
        # Keep one obstacle in each grid cell, jitter within its center 60%.
        x = cfg['x_min'] + (col + 0.5 + rng.uniform(-0.30, 0.30)) * dx
        y = cfg['y_min'] + (row + 0.5 + rng.uniform(-0.30, 0.30)) * dy
        speed = rng.uniform(cfg['min_speed'], cfg['max_speed'])
        tracks.append([(0.0, x, y, speed)])

    # Generate respawns in globally chronological order. Their y-values are
    # independent of ROS loop timing, service latency, and update frequency.
    queue = []
    for idx, track in enumerate(tracks):
        _, x, _, speed = track[-1]
        heapq.heappush(queue, ((x - cfg['x_min']) / speed, idx))

    history = []
    while queue:
        t, idx = heapq.heappop(queue)
        if t > cfg['duration']:
            break
        y = choose_respawn_y(rng, history, t, cfg)
        speed = rng.uniform(cfg['min_speed'], cfg['max_speed'])
        tracks[idx].append((t, cfg['x_max'], y, speed))
        next_t = t + (cfg['x_max'] - cfg['x_min']) / speed
        heapq.heappush(queue, (next_t, idx))

    starts = [[segment[0] for segment in track] for track in tracks]
    return tracks, starts


def state_at(track, start_times, t, cfg):
    index = bisect.bisect_right(start_times, t) - 1
    start_t, x0, y, speed = track[max(0, index)]
    x = max(cfg['x_min'], x0 - speed * (t - start_t))
    return x, y, -speed


def ensure_model(name, x, y, cfg, get_state, spawn):
    response = get_state(name, 'world')
    if response.success:
        rospy.loginfo('%s exists; its position will be reset.', name)
        return

    sdf = CYLINDER_SDF.format(
        name=name, radius=cfg['radius'], length=cfg['length'])
    result = spawn(name, sdf, '', make_pose(x, y, cfg['height']), 'world')
    if not result.success:
        raise RuntimeError('Cannot spawn {}: {}'.format(name, result.status_message))


def publish_frame(t, tracks, starts, cfg, set_state, states_pub,
                  moving=True, update_gazebo=True):
    states = ModelStates()
    finished = t >= cfg['duration']

    for idx, (track, times) in enumerate(zip(tracks, starts)):
        x, y, vx = state_at(track, times, t, cfg)
        name = 'obstacle_{}'.format(idx + 1)
        pose = make_pose(x, y, cfg['height'])

        # Gazebo models are driven directly by absolute poses, not twist
        # integration. Keep Gazebo's internal velocity zero to avoid drift.
        gazebo_state = ModelState()
        gazebo_state.model_name = name
        gazebo_state.reference_frame = 'world'
        gazebo_state.pose = pose
        if update_gazebo:
            result = set_state(gazebo_state)
            if not result.success:
                raise RuntimeError('set_model_state failed for {}: {}'.format(
                    name, result.status_message))

        twist = Twist()
        twist.linear.x = vx if moving and not finished else 0.0
        states.name.append(name)
        states.pose.append(pose)
        states.twist.append(twist)

    states_pub.publish(states)


class SceneModels:
    """Reconcile actual Gazebo models, including leftovers from previous nodes."""

    def __init__(self, get_world, get_state, spawn, delete, verify_timeout=5.0):
        self.get_world, self.get_state = get_world, get_state
        self.spawn, self.delete = spawn, delete
        self.verify_timeout = verify_timeout

    def _managed_names(self):
        response = self.get_world()
        if not response.success:
            raise RuntimeError('Cannot query Gazebo model list: ' + response.status_message)
        return {name for name in response.model_names
                if re.fullmatch(r'obstacle_[0-9]+', name)}

    def sync(self, cfg, tracks, progress=None):
        if len(tracks) != cfg['num_obstacles']:
            raise ValueError('Trajectory count does not match num_obstacles')
        desired = {'obstacle_{}'.format(idx + 1) for idx in range(len(tracks))}
        existing = self._managed_names()
        for name in sorted(existing - desired):
            if progress:
                progress()
            result = self.delete(name)
            if not result.success:
                raise RuntimeError('Cannot delete surplus model {}: {}'.format(
                    name, result.status_message))
            rospy.loginfo('Deleted surplus model %s', name)
        for idx, track in enumerate(tracks):
            if progress:
                progress()
            _, x, y, _ = track[0]
            ensure_model('obstacle_{}'.format(idx + 1), x, y, cfg, self.get_state, self.spawn)
        # A successful deletion response alone is not enough: inspect the world.
        deadline = time.monotonic() + self.verify_timeout
        while True:
            actual = self._managed_names()
            if actual == desired:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError('Gazebo obstacle count verification failed; missing={}, extra={}'.format(
                    sorted(desired - actual)[:10], sorted(actual - desired)[:10]))
            if progress:
                progress()
            time.sleep(0.01)


class ObstacleExperiments:
    """Serialize scene updates and service callbacks; Gazebo stays remote."""

    def __init__(self, cfg, tracks, starts, set_state, states_pub, done_pub, model_manager):
        self.cfg, self.tracks, self.starts = cfg, tracks, starts
        self.set_state, self.states_pub, self.done_pub = set_state, states_pub, done_pub
        self.models = model_manager
        self.scene_ready = False
        self.namespace = rospy.get_param('~experiment_namespace', '/sim_exp').rstrip('/')
        self.session = Session(
            duration=cfg['duration'],
            start_delay=float(rospy.get_param('~start_delay', 2.0)),
            heartbeat_timeout=float(rospy.get_param('~heartbeat_timeout', 3.0)),
            prepare_timeout=float(rospy.get_param('~prepare_timeout', 30.0)),
        )
        self.session.seed = cfg['seed']
        self.session.num_obstacles = cfg['num_obstacles']
        self.lock = threading.RLock()
        self.run_services = []
        self.last_frame_ns = None
        self.last_phase = 'IDLE'
        self.last_status_wall = 0.0
        self.state_pub = rospy.Publisher(self.namespace + '/state', String,
                                         queue_size=1, latch=True)
        self.models.sync(cfg, tracks)
        self._frame(0.0, moving=False)
        self.scene_ready = True
        self._status()
        self.heartbeat_sub = rospy.Subscriber(self.namespace + '/heartbeat', String,
                                              self._heartbeat, queue_size=1)
        # Advertising prepare means the models and initial frame are ready.
        self.prepare_service = rospy.Service(self.namespace + '/prepare', Trigger,
                                              self._prepare)
        self.seed_service = rospy.Service(self.namespace + '/prepare_seed', Reconfigure,
                                         self._prepare_seed)
        rospy.loginfo('Scene ready; waiting for a DWA experiment request')

    def _frame(self, elapsed, moving=False, update_gazebo=True):
        publish_frame(elapsed, self.tracks, self.starts, self.cfg, self.set_state,
                      self.states_pub, moving=moving, update_gazebo=update_gazebo)

    def _status(self):
        self.state_pub.publish(String(data=json.dumps(self.session.snapshot())))
        self.last_status_wall = time.monotonic()

    def _response(self):
        self._status()
        return TriggerResponse(success=True, message=json.dumps(self.session.snapshot()))

    def _heartbeat(self, message):
        with self.lock:
            self.session.heartbeat(message.data, time.monotonic())

    def _prepare(self, _request):
        return self._prepare_configured()

    def _prepare_seed(self, request):
        """Atomically select the seed and prepare this run using a standard srv."""
        try:
            seeds = [param.value for param in request.config.ints if param.name == 'seed']
            counts = [param.value for param in request.config.ints if param.name == 'num_obstacles']
            durations = [param.value for param in getattr(request.config, 'doubles', [])
                         if param.name == 'duration']
            if any(len(values) > 1 for values in (seeds, counts, durations)):
                raise ValueError('Duplicate scene settings are not allowed')
            if not (seeds or counts or durations):
                raise ValueError('At least one scene setting is required')
            seed = validate_seed(seeds[0]) if seeds else None
            count = validate_obstacle_count(counts[0]) if counts else None
            duration = validate_duration(durations[0]) if durations else None
            metadata = {param.name: param.value for param in request.config.strs}
            launch_id = metadata.get('launch_id', '')
            if launch_id and (len(launch_id) != 32 or
                              any(c not in '0123456789abcdef' for c in launch_id)):
                raise ValueError('Invalid launch ID')
            response = self._prepare_configured(seed, launch_id,
                                                 metadata.get('controller', ''), count, duration)
            if not response.success:
                raise ValueError(response.message)
            config = DynamicConfig()
            state = json.loads(response.message)
            config.ints = [IntParameter(name='seed', value=state['seed']),
                           IntParameter(name='num_obstacles', value=state['num_obstacles'])]
            config.doubles = [DoubleParameter(name='duration', value=state['duration_ns'] / NSEC)]
            config.strs = [StrParameter(name='experiment', value=response.message)]
            return ReconfigureResponse(config=config)
        except (ValueError, TypeError, AttributeError) as exc:
            raise rospy.ServiceException(str(exc))

    def _prepare_configured(self, seed=None, launch_id="", controller="",
                            num_obstacles=None, duration=None):
        with self.lock:
            if self.session.phase in ACTIVE:
                return TriggerResponse(success=False, message='Another experiment is active')
            try:
                # Preparation and seed selection share the same lock; another
                # GUI cannot overwrite the seed between configuration and arm.
                candidate = dict(self.cfg)
                if seed is not None:
                    candidate['seed'] = seed
                if num_obstacles is not None:
                    candidate['num_obstacles'] = num_obstacles
                if duration is not None:
                    candidate['duration'] = duration
                tracks, starts = self.tracks, self.starts
                if candidate != self.cfg:
                    tracks, starts = plan_scene(candidate)
                self.session.prepare(rospy.Time.now().to_nsec(), time.monotonic(),
                                     seed=candidate['seed'], launch_id=launch_id,
                                     controller=controller, num_obstacles=candidate['num_obstacles'],
                                     duration=candidate['duration'], preparing=True)
                self.scene_ready = False
                self._status()
                # Deletion/spawn may take time. Send preparation progress so the
                # GUI does not mistake a busy Gazebo for a lost connection.
                self.models.sync(candidate, tracks, progress=self._status)
                self.cfg, self.tracks, self.starts = candidate, tracks, starts
                self._frame(0.0, moving=False)
                self.scene_ready = True
                self.session.ready(rospy.Time.now().to_nsec(), time.monotonic())
                for service in self.run_services:
                    service.shutdown('Replaced by a new experiment')
                prefix = self.namespace + '/runs/' + self.session.run_id
                run_id = self.session.run_id
                self.run_services = [
                    rospy.Service(prefix + '/arm', Trigger,
                                  lambda req: self._arm(run_id)),
                    rospy.Service(prefix + '/cancel', Trigger,
                                  lambda req: self._cancel(run_id)),
                ]
                self.last_frame_ns = None
                self.done_pub.publish(Int32(data=0))
                return self._response()
            except Exception as exc:
                self.session.abort(str(exc))
                self.scene_ready = False
                self.states_pub.publish(ModelStates())
                self._status()
                return TriggerResponse(success=False, message=str(exc))

    def _arm(self, run_id):
        with self.lock:
            try:
                self.session.arm(run_id, rospy.Time.now().to_nsec(), time.monotonic())
                rospy.loginfo('run=%s scheduled t0=%.9f', run_id,
                              self.session.t0_ns / NSEC)
                return self._response()
            except ValueError as exc:
                return TriggerResponse(success=False, message=str(exc))

    def _cancel(self, run_id):
        with self.lock:
            try:
                self.session._check_run(run_id)
                # A cancellation arriving at/after the horizon must retain
                # the exact final frame rather than the previous loop's pose.
                self.session.tick(rospy.Time.now().to_nsec(), time.monotonic())
                self.session.cancel(run_id)
                if self.scene_ready:
                    self._frame(self.session.elapsed_ns / NSEC, moving=False)
                self.done_pub.publish(Int32(data=1))
                return self._response()
            except Exception as exc:
                return TriggerResponse(success=False, message=str(exc))

    def run(self):
        interval_ns = round(NSEC / self.cfg['rate'])
        while not rospy.is_shutdown():
            with self.lock:
                now = rospy.Time.now().to_nsec()
                self.session.tick(now, time.monotonic())
                phase = self.session.phase
                changed = phase != self.last_phase
                due = (self.last_frame_ns is None or now < self.last_frame_ns or
                       now - self.last_frame_ns >= interval_ns)
                if changed or due:
                    try:
                        moving = phase == 'RUNNING'
                        if moving and self.session.first_obstacle_update_ns is None:
                            self.session.first_obstacle_update_ns = now
                            rospy.loginfo('run=%s first obstacle frame at %.9f (t0=%.9f)',
                                          self.session.run_id, now / NSEC,
                                          self.session.t0_ns / NSEC)
                        if self.scene_ready:
                            self._frame(self.session.elapsed_ns / NSEC, moving=moving,
                                        update_gazebo=moving or changed)
                        self.last_frame_ns = now
                    except Exception as exc:
                        self.session.abort('Gazebo scene update failed: ' + str(exc))
                        rospy.logerr('Experiment aborted: %s', exc)
                    if self.session.phase in {'FINISHED', 'CANCELLED', 'ABORTED'}:
                        self.done_pub.publish(Int32(data=1))
                if changed or time.monotonic() - self.last_status_wall >= 0.05:
                    self._status()
                self.last_phase = self.session.phase
            # Wall time keeps cancellation/heartbeat expiry responsive while paused.
            time.sleep(0.002)


def main():
    # rospy chooses its clock during init_node, so configure it first.
    rospy.set_param('/use_sim_time', True)
    rospy.init_node('sim_exp_dynamic_obstacles')
    cfg = dict(CONFIG)
    for key in CONFIG:
        cfg[key] = rospy.get_param('~' + key, CONFIG[key])
    cfg['seed'] = validate_seed(cfg['seed'])
    cfg['num_obstacles'] = validate_obstacle_count(cfg['num_obstacles'])
    cfg['duration'] = validate_duration(cfg['duration'])

    if cfg['rate'] <= 0:
        raise ValueError('rate must be positive')
    if cfg['min_speed'] <= 0 or cfg['max_speed'] < cfg['min_speed']:
        raise ValueError('Invalid speed range')
    if cfg['x_max'] <= cfg['x_min'] or cfg['y_max'] <= cfg['y_min']:
        raise ValueError('Invalid position ranges')

    rospy.loginfo('Planning scene: seed=%s, duration=%.1f s, models=%d',
                  cfg['seed'], cfg['duration'], cfg['num_obstacles'])
    tracks, starts = plan_scene(cfg)

    states_pub = rospy.Publisher('/dynamic_obstacle_states', ModelStates,
                                 queue_size=1, latch=True)
    done_pub = rospy.Publisher('/dynamic_obstacle_finished', Int32,
                               queue_size=1, latch=True)
    done_pub.publish(Int32(data=0))

    rospy.loginfo('Waiting for ROS simulation clock and Gazebo services...')
    while not rospy.is_shutdown() and rospy.Time.now().to_sec() == 0:
        rospy.sleep(0.1)
    rospy.wait_for_service('/gazebo/get_model_state')
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    rospy.wait_for_service('/gazebo/set_model_state')
    rospy.wait_for_service('/gazebo/get_world_properties')
    rospy.wait_for_service('/gazebo/delete_model')

    get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    spawn = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

    models = SceneModels(rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties),
                         get_state, spawn, rospy.ServiceProxy('/gazebo/delete_model', DeleteModel))

    coordinator = ObstacleExperiments(cfg, tracks, starts, set_state,
                                       states_pub, done_pub, models)
    coordinator.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

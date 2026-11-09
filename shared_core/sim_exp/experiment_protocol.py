"""ROS-independent experiment state and start gate (times are integer ns)."""

import math
import uuid


ACTIVE = {"PREPARING", "PREPARED", "ARMED", "RUNNING"}
TERMINAL = {"FINISHED", "CANCELLED", "ABORTED"}
NSEC = 1_000_000_000
MAX_OBSTACLES = 1000
MAX_DURATION = 86400.0


def validate_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**31 - 1:
        raise ValueError("seed must be an integer between 0 and 2147483647")
    return seed


def validate_obstacle_count(count):
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= MAX_OBSTACLES:
        raise ValueError("num_obstacles must be an integer between 0 and {}".format(MAX_OBSTACLES))
    return count


def validate_duration(duration):
    if (isinstance(duration, bool) or not isinstance(duration, (int, float)) or
            not math.isfinite(duration) or not 0.1 <= duration <= MAX_DURATION):
        raise ValueError("duration must be finite and between 0.1 and 86400 seconds")
    return float(duration)


class Session:
    """Owned by the obstacle node; its adapter serializes access with a lock."""

    def __init__(self, duration=70.0, start_delay=2.0,
                 heartbeat_timeout=3.0, prepare_timeout=30.0):
        if min(duration, start_delay, heartbeat_timeout, prepare_timeout) <= 0:
            raise ValueError("Experiment durations must be positive")
        self.duration_ns = round(duration * NSEC)
        self.delay_ns = round(start_delay * NSEC)
        self.heartbeat_timeout = heartbeat_timeout
        self.prepare_timeout = prepare_timeout
        self.run_id = ""
        self.phase = "IDLE"
        self.t0_ns = 0
        self.elapsed_ns = 0
        self.reason = ""
        self.previous_ns = None
        self.last_heartbeat = 0.0
        self.first_obstacle_update_ns = None
        self.seed = None
        self.launch_id = ""
        self.controller = ""
        self.num_obstacles = None

    def prepare(self, now_ns, wall_now, seed=None, launch_id="", controller="",
                num_obstacles=None, duration=None, preparing=False):
        if self.phase in ACTIVE:
            raise ValueError("Another experiment is already active")
        if now_ns <= 0:
            raise ValueError("Waiting for a valid Gazebo /clock")
        if num_obstacles is not None:
            num_obstacles = validate_obstacle_count(num_obstacles)
        if duration is not None:
            self.duration_ns = round(validate_duration(duration) * NSEC)
        self.run_id = uuid.uuid4().hex
        self.phase = "PREPARING" if preparing else "PREPARED"
        self.t0_ns = self.elapsed_ns = 0
        self.reason = ""
        self.previous_ns = now_ns
        self.last_heartbeat = wall_now
        self.first_obstacle_update_ns = None
        self.seed, self.launch_id, self.controller = seed, launch_id, controller
        self.num_obstacles = num_obstacles
        return self.snapshot()

    def ready(self, now_ns, wall_now):
        if self.phase != "PREPARING":
            raise ValueError("No scene preparation is in progress")
        if now_ns < self.previous_ns:
            self.abort("ROS clock moved backwards during scene preparation")
            raise ValueError(self.reason)
        self.phase, self.previous_ns = "PREPARED", now_ns
        self.last_heartbeat = wall_now

    def _check_run(self, run_id):
        if not run_id or run_id != self.run_id:
            raise ValueError("This request belongs to a different experiment")

    def heartbeat(self, run_id, wall_now):
        if run_id == self.run_id and self.phase in ACTIVE:
            self.last_heartbeat = wall_now

    def arm(self, run_id, now_ns, wall_now):
        self._check_run(run_id)
        self.tick(now_ns, wall_now)
        if self.phase == "PREPARED":
            self.t0_ns = now_ns + self.delay_ns
            self.phase = "ARMED"
        elif self.phase not in {"ARMED", "RUNNING"}:
            raise ValueError("Experiment is no longer ready to start")
        # A repeated arm request must never move the original t0.
        return self.snapshot()

    def cancel(self, run_id, reason="DWA cancelled the experiment"):
        self._check_run(run_id)
        if self.phase in ACTIVE:
            self.phase, self.reason = "CANCELLED", reason
        return self.snapshot()

    def abort(self, reason):
        if self.phase in ACTIVE:
            self.phase, self.reason = "ABORTED", reason

    def tick(self, now_ns, wall_now):
        if self.previous_ns is not None and now_ns < self.previous_ns:
            self.abort("ROS clock moved backwards; start a new experiment")
        self.previous_ns = now_ns
        if self.phase == "PREPARED":
            if wall_now - self.last_heartbeat > self.prepare_timeout:
                self.abort("Prepared experiment expired without a client")
        elif self.phase in {"ARMED", "RUNNING"}:
            if wall_now - self.last_heartbeat > self.heartbeat_timeout:
                self.abort("DWA heartbeat timed out")
            elif now_ns >= self.t0_ns:
                self.elapsed_ns = min(now_ns - self.t0_ns, self.duration_ns)
                self.phase = ("FINISHED" if self.elapsed_ns >= self.duration_ns
                              else "RUNNING")
        return self.snapshot()

    def snapshot(self):
        return {
            "protocol": 1, "run_id": self.run_id, "phase": self.phase,
            "t0_ns": self.t0_ns, "duration_ns": self.duration_ns,
            "elapsed_ns": self.elapsed_ns, "reason": self.reason,
            "first_obstacle_update_ns": self.first_obstacle_update_ns,
            "seed": self.seed, "launch_id": self.launch_id,
            "controller": self.controller, "sim_time_ns": self.previous_ns or 0,
            "num_obstacles": self.num_obstacles,
            "capabilities": ["prepare_seed", "scene_settings"],
        }


class StartGate:
    """Client-side gate. The ROS adapter holds its lock while accessing it."""

    def __init__(self, status_timeout=3.0):
        self.status_timeout = status_timeout
        self.run_id = ""
        self.t0_ns = 0
        self.end_ns = 0
        self.last_status_wall = None
        self.phase = ""
        self.reason = ""
        self.previous_ns = None

    def adopt(self, prepared):
        run_id = prepared["run_id"]
        if len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
            raise ValueError("Invalid experiment ID")
        if prepared["protocol"] != 1 or prepared["phase"] != "PREPARED":
            raise ValueError("Obstacle node did not prepare an experiment")
        self.run_id = run_id

    def schedule(self, armed, now_ns, margin_ns):
        if armed["run_id"] != self.run_id or armed["phase"] != "ARMED":
            raise ValueError("Start response belongs to an invalid experiment")
        if armed["t0_ns"] - now_ns < margin_ns:
            raise ValueError("Start reply arrived too late; cancel this experiment")
        if armed["duration_ns"] <= 0:
            raise ValueError("Invalid experiment duration")
        self.t0_ns = armed["t0_ns"]
        self.end_ns = self.t0_ns + armed["duration_ns"]
        self.previous_ns = now_ns

    def observe(self, state, wall_now):
        if state.get("protocol") != 1 or state.get("run_id") != self.run_id:
            return False  # Ignore the previous experiment's latched state.
        self.phase = state["phase"]
        self.last_status_wall = wall_now
        if self.phase in TERMINAL:
            self.reason = state.get("reason") or self.phase
        elif self.t0_ns and state.get("t0_ns") not in (0, self.t0_ns):
            self.reason = "Obstacle node changed the agreed start time"
        return True

    def allows(self, now_ns, wall_now):
        if self.previous_ns is not None and now_ns < self.previous_ns:
            self.reason = "ROS clock moved backwards"
        self.previous_ns = now_ns
        if not self.t0_ns or self.reason:
            return False
        if self.last_status_wall is None:
            return False
        if wall_now - self.last_status_wall > self.status_timeout:
            self.reason = "Obstacle state timed out"
            return False
        if now_ns >= self.end_ns:
            self.reason = "FINISHED"
            return False
        return (now_ns >= self.t0_ns and self.phase in {"ARMED", "RUNNING"})

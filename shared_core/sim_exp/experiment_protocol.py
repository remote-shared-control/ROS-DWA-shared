"""ROS-independent experiment state and start gate (times are integer ns)."""

import uuid


ACTIVE = {"PREPARED", "ARMED", "RUNNING"}
TERMINAL = {"FINISHED", "CANCELLED", "ABORTED"}
NSEC = 1_000_000_000


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

    def prepare(self, now_ns, wall_now):
        if self.phase in ACTIVE:
            raise ValueError("Another experiment is already active")
        if now_ns <= 0:
            raise ValueError("Waiting for a valid Gazebo /clock")
        self.run_id = uuid.uuid4().hex
        self.phase = "PREPARED"
        self.t0_ns = self.elapsed_ns = 0
        self.reason = ""
        self.previous_ns = now_ns
        self.last_heartbeat = wall_now
        self.first_obstacle_update_ns = None
        return self.snapshot()

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

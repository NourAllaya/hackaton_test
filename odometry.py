"""Sensor-based planar dead reckoning for the simulated rover.

The estimator receives only wheel-encoder ticks and an IMU yaw increment. The
simulator owns the hidden truth pose and turns its motion into noisy sensor
readings; the estimator never reads that truth directly.
"""
import math
import numpy as np


WHEEL_BASE_M = 0.42
TICKS_PER_METER = 5000.0


def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class SimulatedMotionSensors:
    """Generate noisy encoder and gyro observations from simulated motion."""

    def __init__(self, seed=6, wheel_base=WHEEL_BASE_M):
        self.rng = np.random.default_rng(seed)
        self.wheel_base = wheel_base
        self.left_scale = self.rng.normal(1.0, 0.002)
        self.right_scale = self.rng.normal(1.0, 0.002)
        self.gyro_bias = self.rng.normal(0.0, math.radians(0.025))

    def sample(self, previous_truth, current_truth, dt):
        """Return sensor-like readings; truth is used only inside this simulator."""
        x0, y0, yaw0 = previous_truth
        x1, y1, yaw1 = current_truth
        dx, dy = x1 - x0, y1 - y0
        yaw_delta = wrap_angle(yaw1 - yaw0)
        mid_yaw = yaw0 + yaw_delta * 0.5
        forward = dx * math.cos(mid_yaw) + dy * math.sin(mid_yaw)

        # Skid-steer wheel travel with independent scale error, slip and quantization.
        left_m = (forward - yaw_delta * self.wheel_base * 0.5) * self.left_scale
        right_m = (forward + yaw_delta * self.wheel_base * 0.5) * self.right_scale
        slip_sigma = 0.0015 + abs(forward) * 0.012
        left_m += self.rng.normal(0.0, slip_sigma)
        right_m += self.rng.normal(0.0, slip_sigma)
        left_ticks = int(round(left_m * TICKS_PER_METER))
        right_ticks = int(round(right_m * TICKS_PER_METER))
        gyro_delta = yaw_delta + self.gyro_bias * dt + self.rng.normal(0.0, math.radians(0.015) * math.sqrt(dt))
        return left_ticks, right_ticks, gyro_delta


class DeadReckoner:
    """Fuse differential wheel travel with the higher-rate IMU yaw increment."""

    def __init__(self, initial_pose, wheel_base=WHEEL_BASE_M):
        self.x, self.y, self.yaw = map(float, initial_pose)
        self.wheel_base = wheel_base

    @property
    def pose(self):
        return self.x, self.y, self.yaw

    def update(self, left_ticks, right_ticks, gyro_delta):
        left_m = left_ticks / TICKS_PER_METER
        right_m = right_ticks / TICKS_PER_METER
        forward = (left_m + right_m) * 0.5
        encoder_yaw = (right_m - left_m) / self.wheel_base
        # Gyro captures rapid turns; wheel differential gently bounds gyro drift.
        yaw_delta = 0.85 * gyro_delta + 0.15 * encoder_yaw
        mid_yaw = self.yaw + 0.5 * yaw_delta
        self.x += forward * math.cos(mid_yaw)
        self.y += forward * math.sin(mid_yaw)
        self.yaw = wrap_angle(self.yaw + yaw_delta)
        return self.pose

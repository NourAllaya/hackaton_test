"""Shared constants: camera model, ground projection, hazard classes."""
import math
import numpy as np

# Rover-mounted pinhole camera (forward-looking, pitched down), flat-ground assumption.
CAM = dict(w=320, h=240, f=250.0, cx=160.0, cy=120.0,
           height=1.0, pitch=math.radians(30.0))
RMAX = 8.0            # max usable ground range (m): beyond this 1 px >> 0.2 m of depth error
CLASSES = ["rock", "crater", "ice", "other"]
CLS_ID = {n: i for i, n in enumerate(CLASSES)}
UNIT = 0.05           # wire quantisation step: 5 cm


def pixel_to_rover(u, v, cam=CAM):
    """Pixel -> ground point in the rover frame (X forward, Y left). Returns X, Y, valid."""
    th, h, f = cam["pitch"], cam["height"], cam["f"]
    a = (np.asarray(u, float) - cam["cx"]) / f
    b = (np.asarray(v, float) - cam["cy"]) / f
    den = math.sin(th) + b * math.cos(th)
    valid = den > 1e-3
    t = np.where(valid, h / np.where(valid, den, 1.0), 0.0)
    X = t * (math.cos(th) - b * math.sin(th))
    Y = t * (-a)
    return X, Y, valid


def rover_to_world(X, Y, pose):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return x + X * c - Y * s, y + X * s + Y * c


def pixel_to_world(u, v, pose, cam=CAM):
    X, Y, valid = pixel_to_rover(u, v, cam)
    xw, yw = rover_to_world(X, Y, pose)
    return xw, yw, valid


def rover_to_pixel(X, Y, cam=CAM):
    """Inverse of pixel_to_rover (used by tests)."""
    th, h, f = cam["pitch"], cam["height"], cam["f"]
    z = X * math.cos(th) + h * math.sin(th)
    a = -Y / z
    b = (-X * math.sin(th) + h * math.cos(th)) / z
    return cam["cx"] + f * a, cam["cy"] + f * b

"""Software-emulated lunar scene + rover camera (100% synthetic, no external data)."""
import math
import cv2
import numpy as np
from common import CAM, pixel_to_rover, rover_to_world

HALF = 20.0   # world is [-20,20]^2 metres, origin = landing site
RES = 0.05    # m / texel


def path_pose(t):
    """Rover true pose along a smooth survey loop starting at the landing site."""
    x = 11.0 * math.sin(0.055 * t)
    y = 9.0 * math.sin(0.11 * t)
    dx = 11.0 * 0.055 * math.cos(0.055 * t)
    dy = 9.0 * 0.11 * math.cos(0.11 * t)
    return (x, y, math.atan2(dy, dx))


class World:
    def __init__(self, seed=7, n_haz=26):
        rng = np.random.default_rng(seed)
        self.rng = rng
        n = int(2 * HALF / RES)
        base = np.zeros((n, n), np.float32)
        for sigma, amp in ((40, 8), (12, 7), (4, 5), (1.2, 4)):
            noise = rng.standard_normal((n, n)).astype(np.float32)
            base += amp * cv2.GaussianBlur(noise, (0, 0), sigma) * (sigma ** 0.5) / 1.5
        gray = np.clip(98 + base, 70, 118)
        self.tex = np.dstack([gray, gray, gray]).astype(np.uint8)  # BGR
        self.truth = []
        self._place(n_haz)

    def _px(self, x, y):
        return ((x + HALF) / RES, (y + HALF) / RES)

    def _place(self, n_haz):
        rng, placed = self.rng, []
        classes = ["rock", "crater", "ice"]
        tries = 0
        while len(placed) < n_haz and tries < 5000:
            tries += 1
            t = rng.uniform(0, 110)
            px, py, yaw = path_pose(t)
            off = rng.uniform(-3.2, 3.2)
            x, y = px - off * math.sin(yaw), py + off * math.cos(yaw)
            if math.hypot(x, y) < 3.0 or abs(x) > 17 or abs(y) > 17:
                continue
            if any(math.hypot(x - p["x"], y - p["y"]) < 3.0 for p in placed):
                continue
            cls = classes[len(placed) % 3]
            r = {"rock": rng.uniform(0.25, 0.45), "crater": rng.uniform(0.7, 1.3),
                 "ice": rng.uniform(0.5, 0.9)}[cls]
            placed.append(dict(cls=cls, x=x, y=y, r=r))
        for h in placed:
            self._draw(h)
        self.truth = placed

    def _draw(self, h):
        cx, cy = self._px(h["x"], h["y"])
        rp = h["r"] / RES
        if h["cls"] == "rock":
            cv2.ellipse(self.tex, (int(cx + 0.7 * rp), int(cy + 0.4 * rp)), (int(rp * 1.1), int(rp * 0.7)),
                        20, 0, 360, (66, 66, 66), -1, cv2.LINE_AA)      # soft shadow
            cv2.ellipse(self.tex, (int(cx), int(cy)), (int(rp), int(rp * 0.85)), 20, 0, 360,
                        (168, 168, 168), -1, cv2.LINE_AA)
        elif h["cls"] == "crater":
            cv2.circle(self.tex, (int(cx), int(cy)), int(rp + 0.15 / RES), (125, 125, 125), -1, cv2.LINE_AA)
            cv2.circle(self.tex, (int(cx), int(cy)), int(rp), (30, 30, 30), -1, cv2.LINE_AA)
        else:  # ice: bright, bluish patch
            cv2.ellipse(self.tex, (int(cx), int(cy)), (int(rp), int(rp * 0.75)), -15, 0, 360,
                        (255, 228, 192), -1, cv2.LINE_AA)


class RoverCamera:
    def __init__(self, world, seed=3):
        self.world = world
        self.rng = np.random.default_rng(seed)
        H, W = CAM["h"], CAM["w"]
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        self.X, self.Y, self.valid = pixel_to_rover(u + 0.5, v + 0.5)
        self.n = world.tex.shape[0]

    def render(self, pose):
        xw, yw = rover_to_world(self.X, self.Y, pose)
        ix = np.clip(((xw + HALF) / RES).astype(int), 0, self.n - 1)
        iy = np.clip(((yw + HALF) / RES).astype(int), 0, self.n - 1)
        img = self.world.tex[iy, ix].astype(np.float32)
        img[~self.valid] = 0
        img += self.rng.normal(0, 4, img.shape)            # sensor noise
        return np.clip(img, 0, 255).astype(np.uint8)

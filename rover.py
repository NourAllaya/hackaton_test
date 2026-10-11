"""Onboard rover software: camera -> perception -> localization -> track map -> telemetry (UDP)."""
import math
import socket
import threading
import time
import cv2
import perception
import telemetry
from localize import localize, HazardMap
from sim import RoverCamera, path_pose
from odometry import DeadReckoner, SimulatedMotionSensors


class Rover:
    def __init__(self, world, uplink_port, ack_port, t0, fps=10, tx_period=0.5, seed=5, save_dir=None):
        self.cam, self.t0, self.fps, self.tx_period = RoverCamera(world), t0, fps, tx_period
        self.map = HazardMap()
        self.up = ("127.0.0.1", uplink_port)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ack = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ack.bind(("127.0.0.1", ack_port))
        self.ack.settimeout(0.001)
        # The initial pose is the only absolute pose supplied to the estimator.
        initial_pose = path_pose(0.0)
        self.odometry = DeadReckoner(initial_pose)
        self.motion_sensors = SimulatedMotionSensors(seed=seed + 1)
        self.previous_truth = initial_pose  # simulator-only input to sensor emulation
        self.seq = 0
        self.m = dict(frames=0, perc_ms=[], pkts=0, bytes=0, json_bytes=0, pos_err=[], saved=0)
        self.save_dir, self.stop = save_dir, False
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        dt, last_tx, i = 1.0 / self.fps, 0.0, 0
        nxt = time.time()
        while not self.stop:
            t = time.time() - self.t0
            true = path_pose(i * dt)
            if i == 0:
                est = self.odometry.pose
            else:
                readings = self.motion_sensors.sample(self.previous_truth, true, dt)
                est = self.odometry.update(*readings)
            self.previous_truth = true
            self.m["pos_err"].append(math.hypot(est[0] - true[0], est[1] - true[1]))
            frame = self.cam.render(true)
            p0 = time.perf_counter()
            dets = perception.detect(frame)
            locs = [l for l in (localize(d, est) for d in dets) if l]
            self.m["perc_ms"].append((time.perf_counter() - p0) * 1000)
            for l in locs:
                self.map.update(l, t)
            self.m["frames"] += 1
            if self.save_dir and i in (60, 200, 330) and self.m["saved"] < 3:
                vis = frame.copy()
                for d in dets:
                    cv2.rectangle(vis, (d["x"], d["y"]), (d["x"] + d["w"], d["y"] + d["h"]), (0, 255, 255), 1)
                    cv2.putText(vis, d["cls"], (d["x"], max(d["y"] - 3, 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                cv2.imwrite(f"{self.save_dir}/frame_{i}.png", cv2.resize(vis, None, fx=2, fy=2))
                self.m["saved"] += 1
            if t - last_tx >= self.tx_period:
                last_tx = t
                batch = self.map.to_send(t, 8)
                pkt = telemetry.encode_up(self.seq, int(t * 1000), est, batch, t)
                self.tx.sendto(pkt, self.up)
                self.m["pkts"] += 1
                self.m["bytes"] += len(pkt)
                self.m["json_bytes"] += telemetry.json_baseline_size(self.seq, int(t * 1000), est, batch)
                self.seq += 1
                for h in batch:
                    h["last_tx"] = t
                    h["sent"] = (h["x"], h["y"])
            try:
                while True:
                    pairs = telemetry.decode_ack(self.ack.recvfrom(256)[0])
                    if pairs:
                        self.map.ack(pairs)
            except socket.timeout:
                pass
            i += 1
            nxt += dt
            time.sleep(max(0.0, nxt - time.time()))

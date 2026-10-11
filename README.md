# Project C.R.A.T.E.R. — ATEC'26 prototype

C.R.A.T.E.R. is a fully software-emulated lunar rover pipeline:
**Perceive → Localize → Transmit → Visualize.** It uses no GPS or cloud service.

```bash
python run_demo.py --duration 60 --blackout 20:32 --hold
# Open http://localhost:8000
python tests.py
```

Dependencies: `numpy` and `opencv-python`. Run metrics and annotated camera
frames are written to `out/`.

## Current prototype

| Stage | Implementation | Current scope |
|---|---|---|
| Simulated world and camera | `sim.py` | Synthetic regolith, rocks, craters and ice; pinhole camera (320×240, 1 m height, 30° pitch). |
| Perception | `perception.py` | Deterministic classical computer vision: colour/intensity masks and connected components. No trained AI model is included. |
| Pose estimation | `odometry.py`, `rover.py` | Starts from a known initial pose, then integrates simulated wheel-encoder ticks and a noisy gyro yaw increment. The estimator does not read the simulator's true pose after initialization. |
| Hazard mapping | `localize.py`, `common.py` | Projects camera detections onto a flat ground plane and combines them with estimated rover pose. Tracks are confirmed after repeated sightings and transmitted as updates. |
| Telemetry and link | `telemetry.py`, `link.py` | Compact binary packets with CRC-8 over an emulated lossy, delayed, bandwidth-limited UDP link with scheduled blackouts. |
| Ground station and dashboard | `ground.py`, `dashboard.html` | Local-only dashboard showing the rover track, hazard locations, data age, uncertainty, link status and link metrics. |

## Important limits

- The camera currently detects hazards; camera feature tracking is **not yet fused into pose estimation**. Pose is wheel-encoder plus gyro dead reckoning, so error grows over time.
- LiDAR, radar, sun-sensor and UWB measurements are not implemented in this baseline. CADRE's paper describes stereo visual-inertial-solar odometry and UWB-based global pose optimization. Its ground-penetrating radar is for subsurface science, not the primary forward obstacle sensor.
- The rover currently follows a scripted survey path. Hazards are detected and mapped, but autonomous path replanning around them is not yet implemented.
- The flat-ground camera projection and synthetic colour thresholds are prototype assumptions. Replace or extend the perception and mapping interfaces when adding stereo depth, slope maps, a range sensor or a learned detector.

These boundaries are deliberate: metrics describe the software-emulated prototype and should not be presented as flight-qualified navigation or a trained AI system.

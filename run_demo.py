"""C.R.A.T.E.R. end-to-end demo / benchmark.
   python3 run_demo.py --duration 60 --blackout 20:32 --hold      # open http://localhost:8000"""
import argparse
import json
import math
import os
import time
from ground import Ground
from link import LossyLink
from rover import Rover
from sim import World

P_UP, P_G, P_DOWN, P_ROVER = 9101, 9102, 9103, 9104


def evaluate(world, ground_state):
    tr, gh = world.truth, ground_state["hazards"]
    used, errs, tp = set(), [], 0
    for g in gh:
        best, bd = None, 1.2
        for k, t in enumerate(tr):
            d = math.hypot(t["x"] - g["x"], t["y"] - g["y"])
            if t["cls"] == g["cls"] and d < bd and k not in used:
                best, bd = k, d
        if best is not None:
            used.add(best)
            errs.append(bd)
            tp += 1
    return dict(reported=len(gh), true_positive=tp, false_positive=len(gh) - tp,
                precision=tp / max(1, len(gh)), mean_pos_err_m=sum(errs) / max(1, len(errs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--blackout", default="20:32", help="start:end seconds, '' for none")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--hold", action="store_true", help="keep dashboard up after the run")
    a = ap.parse_args()
    os.makedirs("out", exist_ok=True)
    t0 = time.time()
    world = World(seed=a.seed)
    bo = [tuple(map(float, a.blackout.split(":")))] if a.blackout else []
    up = LossyLink(P_UP, P_G, t0, bo, seed=1, name="up").start()
    down = LossyLink(P_DOWN, P_ROVER, t0, bo, seed=2, name="down", bandwidth_bps=1200).start()
    gnd = Ground(P_G, P_DOWN, t0, a.port, truth=world.truth)
    gnd.serve()
    rv = Rover(world, P_UP, P_ROVER, t0, fps=a.fps, save_dir="out")
    rv.thread.start()
    print(f"Mission Control: http://localhost:{a.port}   (blackout {a.blackout or 'none'}, {a.duration:.0f}s)")
    while time.time() - t0 < a.duration:
        time.sleep(0.5)
    time.sleep(2.0)
    rv.stop = True
    st = gnd.state()
    ev = evaluate(world, st)
    pm = rv.m["perc_ms"]
    sent = rv.m["pkts"]
    res = dict(
        frames=rv.m["frames"], perception_ms_mean=sum(pm) / len(pm), perception_fps_capacity=1000 / (sum(pm) / len(pm)),
        mean_pose_error_m=sum(rv.m["pos_err"]) / max(1, len(rv.m["pos_err"])),
        packets_sent=sent, packets_received=st["stats"]["rx"], crc_rejected=st["stats"]["crc_fail"],
        delivery_ratio=st["stats"]["rx"] / max(1, sent), bytes_sent=rv.m["bytes"],
        avg_packet_bytes=rv.m["bytes"] / max(1, sent), json_equivalent_bytes=rv.m["json_bytes"],
        compression_vs_json=rv.m["json_bytes"] / max(1, rv.m["bytes"]), avg_bps=rv.m["bytes"] * 8 / a.duration,
        hazards_in_truth=len(world.truth), link_up=up.stats, link_down=down.stats, detection=ev,
        unacked_hazards=sum(1 for h in rv.map.h.values() if h["confirmed"] and h["pending"]))
    json.dump(res, open("out/metrics.json", "w"), indent=2)
    print(json.dumps(res, indent=2))
    if a.hold:
        print("Holding dashboard — Ctrl-C to quit")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()

"""Spatial awareness: pixel detections -> metric grid coords relative to the landing site (0,0).
Pinhole + flat-ground ray intersection, composed with the (drifting) odometry pose estimate."""
import math
from common import CAM, pixel_to_world


def localize(det, pose_est):
    x, y, ok = pixel_to_world(det["u"], det["v"], pose_est)
    if not ok:
        return None
    # physical size: horizontal extent of the blob at its centroid row, projected on the ground
    xl, yl, _ = pixel_to_world(det["x"], det["v"], pose_est)
    xr, yr, _ = pixel_to_world(det["x"] + det["w"], det["v"], pose_est)
    diam = math.hypot(float(xr - xl), float(yr - yl))
    return dict(cls=det["cls"], x=float(x), y=float(y), r=max(0.05, min(3.0, diam / 2)), conf=det["conf"])


class HazardMap:
    """Onboard track map: associates repeated sightings, fuses them (running mean), confirms after
    2 sightings (kills one-frame false positives) and schedules (re)transmission."""
    ASSOC = 0.9      # m
    MOVE_RESEND = 0.25
    CONFIRM = 2

    def __init__(self):
        self.h = {}
        self.next_id = 0

    def update(self, loc, t):
        best, bd = None, self.ASSOC
        for hz in self.h.values():
            d = math.hypot(hz["x"] - loc["x"], hz["y"] - loc["y"])
            if hz["cls"] == loc["cls"] and d < bd:
                best, bd = hz, d
        if best is None:
            if self.next_id > 255:
                return
            self.h[self.next_id] = dict(id=self.next_id, cls=loc["cls"], x=loc["x"], y=loc["y"], r=loc["r"],
                                        conf=loc["conf"], n=1, ver=0, sent=(loc["x"], loc["y"]), pending=True,
                                        last_tx=None, t_obs=t, confirmed=False)
            self.next_id += 1
            return
        n = min(best["n"], 12)
        best["x"] += (loc["x"] - best["x"]) / (n + 1)
        best["y"] += (loc["y"] - best["y"]) / (n + 1)
        best["r"] += (loc["r"] - best["r"]) / (n + 1)
        best["conf"] = max(best["conf"], loc["conf"])
        best["n"] += 1
        best["t_obs"] = t
        if best["n"] >= self.CONFIRM:
            best["confirmed"] = True
        sx, sy = best["sent"]
        if math.hypot(best["x"] - sx, best["y"] - sy) > self.MOVE_RESEND and not best["pending"]:
            best["ver"] = (best["ver"] + 1) & 3
            best["pending"] = True

    def to_send(self, t, k, retry=1.5):
        c = [h for h in self.h.values() if h["confirmed"] and h["pending"]
             and (h["last_tx"] is None or t - h["last_tx"] > retry)]
        c.sort(key=lambda h: (h["last_tx"] is not None, h["last_tx"] or 0, -h["conf"]))
        return c[:k]

    def ack(self, pairs):
        for hid, ver in pairs:
            h = self.h.get(hid)
            if h and h["ver"] == ver:
                h["pending"] = False

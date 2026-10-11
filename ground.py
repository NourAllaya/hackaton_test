"""Mission Control ground station: UDP receiver + ACKs + HTTP dashboard server."""
import json
import os
import socket
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import telemetry

HERE = os.path.dirname(os.path.abspath(__file__))


class Ground:
    def __init__(self, port, ack_port, t0, http_port=8000, truth=None):
        self.t0, self.truth = t0, truth or []
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", port))
        self.sock.settimeout(0.1)
        self.ack_to = ("127.0.0.1", ack_port)
        self.hz, self.trail, self.pose = {}, deque(maxlen=600), None
        self.recent_acks = deque(maxlen=12)
        self.s = dict(rx=0, crc_fail=0, bytes=0, lost=0, last_rx=None, last_seq=None, lat=0.0, first_rx=None)
        self.events = deque(maxlen=50)
        self.lock = threading.Lock()
        self.stop = False
        self.http_port = http_port
        threading.Thread(target=self._rx, daemon=True).start()
        self.httpd = None

    def _rx(self):
        while not self.stop:
            try:
                data, _ = self.sock.recvfrom(2048)
            except socket.timeout:
                continue
            now = time.time()
            p = telemetry.decode_up(data)
            with self.lock:
                if p is None:
                    self.s["crc_fail"] += 1
                    continue
                s = self.s
                s["rx"] += 1
                s["bytes"] += len(data)
                s["first_rx"] = s["first_rx"] or now
                if s["last_seq"] is not None:
                    gap = (p["seq"] - s["last_seq"]) & 0xFFFF
                    if 1 < gap < 1000:
                        s["lost"] += gap - 1
                if s["last_seq"] is None or ((p["seq"] - s["last_seq"]) & 0xFFFF) < 1000:
                    s["last_seq"] = p["seq"]
                    self.pose = dict(x=p["pose"][0], y=p["pose"][1], yaw=p["pose"][2], t=p["t"])
                    self.trail.append((p["pose"][0], p["pose"][1]))
                s["lat"] = now - (self.t0 + p["t"])
                if s["last_rx"] is not None and now - s["last_rx"] > 4:
                    self.events.append(f"t={now-self.t0:5.1f}s link re-acquired after {now-s['last_rx']:.1f}s silence")
                s["last_rx"] = now
                pairs = []
                for h in p["hazards"]:
                    obs = self.t0 + p["t"] - h["age"]
                    cur = self.hz.get(h["id"])
                    if cur is None or obs >= cur["obs"]:
                        new = cur is None
                        self.hz[h["id"]] = dict(id=h["id"], cls=h["cls"], x=h["x"], y=h["y"], r=h["r"],
                                                conf=h["conf"], obs=obs, rx=now, ver=h["ver"])
                        if new:
                            self.events.append(f"t={now-self.t0:5.1f}s NEW {h['cls']} #{h['id']} @({h['x']:.1f},{h['y']:.1f})")
                    pairs.append((h["id"], h["ver"]))
                for pr in pairs:
                    if pr not in self.recent_acks:
                        self.recent_acks.append(pr)
                ack = list(self.recent_acks)
            if pairs:   # ACK with redundancy (last 12) so a lost ACK is healed by the next one
                self.sock.sendto(telemetry.encode_ack(ack), self.ack_to)

    def state(self):
        now = time.time()
        with self.lock:
            hz = []
            for h in self.hz.values():
                age = now - h["obs"]
                sigma = 0.15 + (15 - h["conf"]) * 0.04 + 0.03 * age
                hz.append(dict(h, age=age, sigma=sigma))
            s = dict(self.s)
            silence = None if s["last_rx"] is None else now - s["last_rx"]
            link = "NO CONTACT" if silence is None else ("BLACKOUT" if silence > 4 else ("DEGRADED" if silence > 1.8 else "LIVE"))
            return dict(t=now - self.t0, pose=self.pose, trail=list(self.trail)[-300:], hazards=hz, link=link,
                        silence=silence, stats=s, events=list(self.events)[-8:])

    def serve(self):
        g = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path.startswith("/api/state"):
                    body, ct = json.dumps(g.state()).encode(), "application/json"
                elif self.path.startswith("/api/truth"):   # SIM-ONLY debug overlay, never on the radio link
                    body, ct = json.dumps(g.truth).encode(), "application/json"
                else:
                    with open(os.path.join(HERE, "dashboard.html"), "rb") as f:
                        body, ct = f.read(), "text/html; charset=utf-8"
                self.send_response(200)
                self.send_header("Content-Type", ct)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

        # The dashboard is local-only; the rover and ground station communicate over loopback.
        self.httpd = ThreadingHTTPServer(("127.0.0.1", self.http_port), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

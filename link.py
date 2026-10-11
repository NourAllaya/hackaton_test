"""Emulated degraded radio link (software UDP proxy), team-implemented:
Gilbert-Elliott burst loss, latency+jitter, bandwidth cap (token bucket), bit-flips, scheduled blackouts."""
import heapq
import random
import socket
import threading
import time


class LossyLink:
    def __init__(self, listen_port, dest_port, t0, blackouts=(), delay=0.4, jitter=0.15,
                 p_gb=0.06, p_bg=0.30, loss_good=0.05, loss_bad=0.75, bitflip=0.02,
                 bandwidth_bps=2400, seed=1, name="link"):
        self.name, self.t0, self.blackouts = name, t0, list(blackouts)
        self.dest = ("127.0.0.1", dest_port)
        self.delay, self.jitter = delay, jitter
        self.p_gb, self.p_bg, self.lg, self.lb, self.bitflip = p_gb, p_bg, loss_good, loss_bad, bitflip
        self.rate = bandwidth_bps / 8.0
        self.tokens, self.tlast = self.rate * 2, time.time()
        self.rng = random.Random(seed)
        self.bad = False
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.bind(("127.0.0.1", listen_port))
        self.rx.settimeout(0.1)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.q, self.cv, self.stop = [], threading.Condition(), False
        self.stats = dict(in_pkts=0, in_bytes=0, out_pkts=0, dropped_loss=0, dropped_blackout=0,
                          dropped_bw=0, flipped=0)
        self.threads = [threading.Thread(target=self._in, daemon=True), threading.Thread(target=self._out, daemon=True)]

    def start(self):
        for t in self.threads:
            t.start()
        return self

    def close(self):
        self.stop = True
        with self.cv:
            self.cv.notify_all()

    def in_blackout(self, now=None):
        s = (now or time.time()) - self.t0
        return any(a <= s < b for a, b in self.blackouts)

    def _in(self):
        while not self.stop:
            try:
                data, _ = self.rx.recvfrom(2048)
            except socket.timeout:
                continue
            now = time.time()
            st = self.stats
            st["in_pkts"] += 1
            st["in_bytes"] += len(data)
            if self.in_blackout(now):
                st["dropped_blackout"] += 1
                continue
            self.tokens = min(self.rate * 2, self.tokens + (now - self.tlast) * self.rate)
            self.tlast = now
            if self.tokens < len(data):
                st["dropped_bw"] += 1
                continue
            self.tokens -= len(data)
            self.bad = (self.rng.random() > self.p_bg) if self.bad else (self.rng.random() < self.p_gb)
            if self.rng.random() < (self.lb if self.bad else self.lg):
                st["dropped_loss"] += 1
                continue
            if self.rng.random() < self.bitflip:
                b = bytearray(data)
                b[self.rng.randrange(len(b))] ^= 1 << self.rng.randrange(8)
                data = bytes(b)
                st["flipped"] += 1
            due = now + max(0.0, self.delay + self.rng.uniform(-self.jitter, self.jitter))
            with self.cv:
                heapq.heappush(self.q, (due, id(data), data))
                self.cv.notify()

    def _out(self):
        while not self.stop:
            with self.cv:
                if not self.q:
                    self.cv.wait(0.2)
                    continue
                due = self.q[0][0]
                wait = due - time.time()
                if wait > 0:
                    self.cv.wait(wait)
                    continue
                _, _, data = heapq.heappop(self.q)
            self.tx.sendto(data, self.dest)
            self.stats["out_pkts"] += 1

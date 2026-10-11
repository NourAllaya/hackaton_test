"""Automated tests:  python3 tests.py"""
import math, random, time, urllib.request, json
import telemetry
from common import pixel_to_rover, rover_to_pixel, pixel_to_world, rover_to_world
from link import LossyLink
import socket


def test_packet_roundtrip():
    hz = [dict(id=3, cls="ice", ver=2, conf=11, x=-7.35, y=12.4, r=0.8, t_obs=9.0)]
    p = telemetry.decode_up(telemetry.encode_up(65535, 123456, (1.23, -4.56, math.radians(90)), hz, 10.0))
    h = p["hazards"][0]
    assert p["seq"] == 65535 and abs(p["pose"][0] - 1.25) < .03 and abs(h["x"] + 7.35) < .03
    assert (h["cls"], h["ver"], h["conf"], h["id"]) == ("ice", 2, 11, 3) and abs(h["age"] - 1.0) < .3


def test_crc_rejects_every_single_bitflip():
    pkt = telemetry.encode_up(1, 1000, (0, 0, 0), [dict(id=1, cls="rock", ver=0, conf=9, x=1, y=2, r=.3, t_obs=0)], 0)
    for i in range(len(pkt)):
        for b in range(8):
            bad = bytearray(pkt); bad[i] ^= 1 << b
            assert telemetry.decode_up(bytes(bad)) is None


def test_ack_roundtrip():
    assert telemetry.decode_ack(telemetry.encode_ack([(5, 1), (200, 3)])) == [(5, 1), (200, 3)]


def test_projection_inverse_and_pose():
    for u, v in ((10, 100), (160, 239), (300, 60), (160, 30)):
        X, Y, ok = pixel_to_rover(u, v); assert ok
        u2, v2 = rover_to_pixel(X, Y); assert abs(u - u2) < 1e-6 and abs(v - v2) < 1e-6
    X, Y, _ = pixel_to_rover(160, 150)           # centre column -> straight ahead
    assert abs(Y) < 1e-9 and X > 0
    xw, yw = rover_to_world(2.0, 0.0, (1.0, 1.0, math.pi / 2)); assert abs(xw - 1) < 1e-9 and abs(yw - 3) < 1e-9


def test_link_blackout_and_delay():
    t0 = time.time()
    lk = LossyLink(9201, 9202, t0, blackouts=[(0.0, 1.0)], delay=0.2, jitter=0, loss_good=0, p_gb=0, bitflip=0).start()
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); rx.bind(("127.0.0.1", 9202)); rx.settimeout(0.1)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tx.sendto(b"during", ("127.0.0.1", 9201)); time.sleep(1.1)
    try: rx.recvfrom(10); assert False, "blackout leaked"
    except socket.timeout: pass
    s = time.time(); tx.sendto(b"after", ("127.0.0.1", 9201))
    rx.settimeout(1); d = rx.recvfrom(10)[0]; assert d == b"after" and time.time() - s >= 0.19
    lk.close()


def test_end_to_end_with_blackout():
    import run_demo, subprocess, sys
    out = subprocess.run([sys.executable, "run_demo.py", "--duration", "40", "--blackout", "12:22", "--port", "8011"],
                         capture_output=True, text=True, timeout=120).stdout
    m = json.loads(out[out.index("{"):])
    assert m["detection"]["precision"] >= 0.9, m["detection"]
    assert m["detection"]["mean_pos_err_m"] < 0.5
    assert m["detection"]["true_positive"] >= 8
    assert m["unacked_hazards"] <= 1, "hazards must be recovered after blackout"
    assert m["avg_packet_bytes"] < 80 and m["compression_vs_json"] > 5
    assert m["perception_fps_capacity"] > 30


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)

"""Ultra-compact binary wire format (big-endian, struct-packed, CRC-8).

UPLINK  (rover -> ground):  header 12 B + 8 B / hazard + CRC 1 B
  B  type(4b)=1 | n_hazards(4b)
  H  seq                    (loss detection)
  I  t_ms                   (rover mission clock)
  h  pose_x, pose_y         (5 cm units, relative to landing site)
  B  yaw                    (360/256 deg)
  per hazard:
    B  id
    B  cls(2b) | ver(2b) | conf(4b)
    h  x, y                 (5 cm units)
    B  radius               (5 cm units)
    B  obs_age              (0.25 s units: how old the observation is at send time)
DOWNLINK (ground -> rover): B type(4b)=2|n(4b), n x (B id, B ver), CRC  -> cumulative-style ACKs
"""
import math
import struct
from common import UNIT, CLS_ID, CLASSES

HDR = struct.Struct(">BHIhhB")
HAZ = struct.Struct(">BBhhBB")


def crc8(data: bytes, poly=0x07):
    c = 0
    for b in data:
        c ^= b
        for _ in range(8):
            c = ((c << 1) ^ poly) & 0xFF if c & 0x80 else (c << 1) & 0xFF
    return c


def _q(v, lo=-32768, hi=32767):
    return max(lo, min(hi, int(round(v / UNIT))))


def encode_up(seq, t_ms, pose, hazards, now_s):
    hz = hazards[:15]
    out = HDR.pack((1 << 4) | len(hz), seq & 0xFFFF, t_ms & 0xFFFFFFFF, _q(pose[0]), _q(pose[1]),
                   int((math.degrees(pose[2]) % 360) / 360 * 256) & 0xFF)
    for h in hz:
        age = max(0, min(255, int((now_s - h["t_obs"]) / 0.25)))
        out += HAZ.pack(h["id"], (CLS_ID[h["cls"]] << 6) | ((h["ver"] & 3) << 4) | (h["conf"] & 15),
                        _q(h["x"]), _q(h["y"]), _q(h["r"], 0, 255), age)
    return out + bytes([crc8(out)])


def decode_up(buf: bytes):
    if len(buf) < HDR.size + 1 or crc8(buf[:-1]) != buf[-1]:
        return None
    tn, seq, t_ms, px, py, yaw = HDR.unpack_from(buf)
    n = tn & 15
    if tn >> 4 != 1 or len(buf) != HDR.size + n * HAZ.size + 1:
        return None
    hz, off = [], HDR.size
    for _ in range(n):
        i, cvc, x, y, r, age = HAZ.unpack_from(buf, off)
        off += HAZ.size
        hz.append(dict(id=i, cls=CLASSES[cvc >> 6], ver=(cvc >> 4) & 3, conf=cvc & 15,
                       x=x * UNIT, y=y * UNIT, r=r * UNIT, age=age * 0.25))
    return dict(seq=seq, t=t_ms / 1000.0, pose=(px * UNIT, py * UNIT, yaw * 360 / 256), hazards=hz)


def encode_ack(pairs):
    pairs = pairs[:15]
    out = bytes([(2 << 4) | len(pairs)]) + b"".join(bytes([i, v & 3]) for i, v in pairs)
    return out + bytes([crc8(out)])


def decode_ack(buf: bytes):
    if len(buf) < 2 or crc8(buf[:-1]) != buf[-1] or buf[0] >> 4 != 2:
        return None
    n = buf[0] & 15
    if len(buf) != 2 + 2 * n:
        return None
    return [(buf[1 + 2 * k], buf[2 + 2 * k]) for k in range(n)]


def json_baseline_size(seq, t_ms, pose, hazards):
    import json
    return len(json.dumps(dict(seq=seq, t=t_ms, pose=pose, hazards=[
        dict(id=h["id"], cls=h["cls"], x=round(h["x"], 3), y=round(h["y"], 3), r=round(h["r"], 3),
             conf=h["conf"]) for h in hazards])).encode())

"""Reference oracle for AFDX traces.

Reads the per-network pcaps and re-derives anomalies and receiver decisions from the
wire bytes alone. It never sees the generator's labels, so agreement between the two
is a real cross-check (the same idea as the Wireshark cross-check, but in-repo).

Receiver model (premises are restated in every scenario preamble):
  * Integrity check (IC), per virtual link AND per network: the first frame is accepted;
    later frames are accepted only if SN == previous+1 or previous+2 (255 wraps to 1,
    0 is reserved). On a drop, 'drop_with_resync' sets previous := received SN,
    'drop_without_resync' leaves previous unchanged.
  * Redundancy management (RM), per virtual link: of the IC-accepted copies, the first
    is delivered; a later copy with the same SN is discarded. This is normal, not an alarm.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from scapy.utils import rdpcap

from bench.afdx import afdx_synth as afdx

TOL = 2e-6  # seconds; pcap timestamps have microsecond resolution
RESYNC_RULES = ("drop_with_resync", "drop_without_resync")
DETECTORS = ("bag_violation", "skew_violation", "spoofed_source",
             "unknown_vl", "lmax_exceeded", "bad_udp_checksum")


def nxt(sn: int, k: int = 1) -> int:
    """Sequence number k steps after sn, skipping 0 (255 -> 1)."""
    return (sn - 1 + k) % 255 + 1


def in_window(prev: int, sn: int) -> bool:
    return sn in (nxt(prev, 1), nxt(prev, 2))


def udp_checksum_ok(raw: bytes) -> bool:
    """Verify the UDP checksum from raw bytes (IPv4, 20-byte header, no options)."""
    seg_len = int.from_bytes(raw[38:40], "big")
    seg = raw[34:34 + seg_len]
    pseudo = raw[26:34] + bytes([0, 17]) + seg_len.to_bytes(2, "big")
    data = pseudo + seg + (b"\x00" if len(seg) % 2 else b"")
    total = sum(int.from_bytes(data[i:i + 2], "big") for i in range(0, len(data), 2))
    while total >> 16:
        total = (total & 0xFFFF) + (total >> 16)
    return total == 0xFFFF


def load_frames(pcap_a: Path, pcap_b: Path) -> list[dict]:
    frames = []
    for net, path in (("A", pcap_a), ("B", pcap_b)):
        for pos, pkt in enumerate(rdpcap(str(path))):
            raw = bytes(pkt)
            frames.append({
                "id": f"{net}{pos}", "net": net, "pos": pos, "t": float(pkt.time),
                "vl_id": int.from_bytes(raw[4:6], "big"), "sn": raw[-1],
                "src": raw[6:12], "len": len(raw), "payload": raw[42:],
                "udp_ok": udp_checksum_ok(raw),
            })
    frames.sort(key=lambda f: (f["t"], f["net"], f["pos"]))
    return frames


def analyze(frames: list[dict], vls: dict[int, dict]) -> dict:
    """Anomaly detectors that need no sequence-number state."""
    found: dict[str, Any] = {k: set() for k in DETECTORS}
    last_t: dict = {}
    for f in frames:
        vl = vls.get(f["vl_id"])
        if vl is None:
            found["unknown_vl"].add(f["id"])
            continue
        key = (f["vl_id"], f["net"])
        if key in last_t and f["t"] - last_t[key] < vl["bag_ms"] / 1000 - TOL:
            found["bag_violation"].add(f["id"])
        last_t[key] = f["t"]
        if f["len"] + afdx.FCS_LEN > vl["lmax_bytes"]:
            found["lmax_exceeded"].add(f["id"])
        if f["src"] != bytes.fromhex(afdx.src_mac(vl["src_mac_user_id"], vl["interface_id"]).replace(":", "")):
            found["spoofed_source"].add(f["id"])
        if not f["udp_ok"]:
            found["bad_udp_checksum"].add(f["id"])

    # pair the j-th A copy with the j-th B copy of each (vl, sn)
    occ = defaultdict(lambda: {"A": [], "B": []})
    for f in frames:
        if f["vl_id"] in vls:
            occ[(f["vl_id"], f["sn"])][f["net"]].append(f)
    mismatch_pairs = []
    for (vl_id, _), copies in occ.items():
        for a, b in zip(copies["A"], copies["B"]):
            skew = vls[vl_id]["skew_max_ms"] / 1000
            if abs(a["t"] - b["t"]) > skew + TOL:
                found["skew_violation"].add((a if a["t"] > b["t"] else b)["id"])
            if a["payload"] != b["payload"]:
                mismatch_pairs.append((a["id"], b["id"]))
    found["ab_mismatch_pairs"] = mismatch_pairs
    return found


def decisions(frames: list[dict], vls: dict[int, dict], resync_rule: str) -> dict[str, dict]:
    """Per-frame IC and RM decision under the given resync rule."""
    if resync_rule not in RESYNC_RULES:
        raise ValueError(resync_rule)
    last: dict = {}
    delivered: dict = {}
    out: dict[str, dict] = {}
    for f in frames:
        if f["vl_id"] not in vls:
            out[f["id"]] = {"ic": "n/a", "rm": "n/a"}
            continue
        key, sn = (f["vl_id"], f["net"]), f["sn"]
        if key not in last or in_window(last[key], sn):
            ic = "accept"
            last[key] = sn
        else:
            ic = "drop"
            if resync_rule == "drop_with_resync":
                last[key] = sn
        rm = "n/a"
        if ic == "accept":
            rm = "discard_redundant" if delivered.get(f["vl_id"]) == sn else "deliver"
            delivered[f["vl_id"]] = sn
        out[f["id"]] = {"ic": ic, "rm": rm}
    return out

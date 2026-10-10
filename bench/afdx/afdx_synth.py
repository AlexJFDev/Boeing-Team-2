"""ARINC 664 Part 7 (AFDX) synthetic frame / virtual-link tooling.

- validate_config(): JSON Schema (afdx_schema.json) + cross-field spec rules
- SequenceCounter:   AFDX sequence numbers (1..255, wraps 255 -> 1, 0 reserved)
- build_frame():     scapy frame for one VL / sequence number
- generate_frames(): schedule frames per VL at BAG spacing on networks A/B
- parse_frame():     recover VL ID and SN from raw bytes (used as the test oracle)

Encoding assumptions come from secondary sources (Abaco / UEI tutorials), not the
paywalled ARINC 664 P7 text. Cross-check against Wireshark/tshark (separate task).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from jsonschema import Draft202012Validator
from scapy.layers.inet import IP, UDP
from scapy.layers.l2 import Ether
from scapy.packet import Packet, Raw

SCHEMA_PATH = Path(__file__).with_name("afdx_schema.json")

VALID_BAG_MS = (1, 2, 4, 8, 16, 32, 64, 128)
SKEW_MAX_BAG_MULTIPLE = 253          # catalogue C2: SkewMax <= 253 x BAG

SN_RESET, SN_MIN, SN_MAX = 0, 1, 255

ETH_HDR, IP_HDR, UDP_HDR, SN_LEN, FCS_LEN = 14, 20, 8, 1, 4
MIN_FRAME_NO_FCS = 60                # 64-byte minimum Ethernet frame minus FCS

DST_MAC_PREFIX = bytes([0x03, 0x00, 0x00, 0x00])   # constant field + 16-bit VL ID
SRC_MAC_PREFIX = bytes([0x02, 0x00, 0x00])         # constant field


# validation
@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


class SchemaValidationError(ValueError):
    def __init__(self, issues: list[ValidationIssue]):
        self.issues = issues
        super().__init__("; ".join(str(i) for i in issues))


@lru_cache(maxsize=1)
def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def _path(parts) -> str:
    return "/".join(str(p) for p in parts) or "(root)"


def max_app_payload(lmax_bytes: int) -> int:
    """Largest application payload that fits in Lmax (FCS and 1-byte SN included)."""
    return lmax_bytes - FCS_LEN - ETH_HDR - IP_HDR - UDP_HDR - SN_LEN


def validate_config(cfg: dict) -> list[ValidationIssue]:
    """Return all issues; an empty list means the config is valid."""
    validator = Draft202012Validator(load_schema())
    issues = [
        ValidationIssue(_path(e.absolute_path), e.message)
        for e in sorted(validator.iter_errors(cfg), key=lambda e: _path(e.absolute_path))
    ]
    if issues:  # cross-field checks assume a structurally sound config
        return issues

    seen: set[int] = set()
    for i, vl in enumerate(cfg["virtual_links"]):
        base = f"virtual_links/{i}"
        if vl["vl_id"] in seen:
            issues.append(ValidationIssue(f"{base}/vl_id", f"duplicate vl_id {vl['vl_id']}"))
        seen.add(vl["vl_id"])
        limit = SKEW_MAX_BAG_MULTIPLE * vl["bag_ms"]
        if vl["skew_max_ms"] > limit:
            issues.append(ValidationIssue(
                f"{base}/skew_max_ms",
                f"skew_max_ms {vl['skew_max_ms']} exceeds {SKEW_MAX_BAG_MULTIPLE} x BAG = {limit}"))
        cap = max_app_payload(vl["lmax_bytes"])
        if vl["app_payload_bytes"] > cap:
            issues.append(ValidationIssue(
                f"{base}/app_payload_bytes",
                f"{vl['app_payload_bytes']} bytes does not fit lmax_bytes={vl['lmax_bytes']} (max {cap})"))
    return issues


def assert_valid(cfg: dict) -> None:
    issues = validate_config(cfg)
    if issues:
        raise SchemaValidationError(issues)


# sequence numbers
class SequenceCounter:
    """AFDX sequence numbers: 1..255, wrapping 255 -> 1. 0 is reserved for reset."""

    def __init__(self, initial_sn: int = SN_MIN):
        if not SN_MIN <= initial_sn <= SN_MAX:
            raise ValueError(f"initial_sn must be {SN_MIN}..{SN_MAX}, got {initial_sn}")
        self._next = initial_sn

    def next(self) -> int:
        sn = self._next
        self._next = SN_MIN if sn == SN_MAX else sn + 1
        return sn


# frame building
def vl_dst_mac(vl_id: int) -> str:
    return ":".join(f"{b:02x}" for b in DST_MAC_PREFIX + vl_id.to_bytes(2, "big"))


def vl_dst_ip(vl_id: int) -> str:
    return f"224.224.{vl_id >> 8}.{vl_id & 0xFF}"


def src_mac(user_id: int, interface_id: int) -> str:
    raw = SRC_MAC_PREFIX + user_id.to_bytes(2, "big") + bytes([interface_id << 5])
    return ":".join(f"{b:02x}" for b in raw)


def _default_payload(vl_id: int, n: int) -> bytes:
    return bytes((vl_id + i) % 256 for i in range(n))  # deterministic, no RNG


def build_frame(vl: dict, sn: int, payload: bytes | None = None) -> Packet:
    """Build one AFDX frame. UDP payload = app bytes + zero padding + 1-byte SN (last byte)."""
    if not SN_RESET <= sn <= SN_MAX:
        raise ValueError(f"sequence number out of range: {sn}")
    app = payload if payload is not None else _default_payload(vl["vl_id"], vl["app_payload_bytes"])
    pad = max(0, MIN_FRAME_NO_FCS - (ETH_HDR + IP_HDR + UDP_HDR + len(app) + SN_LEN))
    pkt = (
        Ether(dst=vl_dst_mac(vl["vl_id"]), src=src_mac(vl["src_mac_user_id"], vl["interface_id"]), type=0x0800)
        / IP(src=vl["src_ip"], dst=vl_dst_ip(vl["vl_id"]))
        / UDP(sport=vl["udp_src_port"], dport=vl["udp_dst_port"])
        / Raw(app + b"\x00" * pad + bytes([sn]))
    )
    if len(pkt) + FCS_LEN > vl["lmax_bytes"]:
        raise ValueError(f"frame of {len(pkt) + FCS_LEN} bytes exceeds lmax_bytes={vl['lmax_bytes']}")
    return Ether(bytes(pkt))  # round-trip so lengths and checksums are populated


@dataclass(frozen=True)
class FrameRecord:
    network: str
    t_s: float
    vl_id: int
    sn: int
    packet: Packet


def generate_frames(cfg: dict, frames_per_vl: int) -> Iterator[FrameRecord]:
    """Frames per VL at BAG spacing. A and B copies share time and sequence number."""
    assert_valid(cfg)
    records: list[FrameRecord] = []
    for vl in cfg["virtual_links"]:
        counter = SequenceCounter(vl["initial_sn"])
        for i in range(frames_per_vl):
            sn = counter.next()
            pkt = build_frame(vl, sn)
            for net in vl["networks"]:
                # independent copy per network, so altering one copy never changes the other
                records.append(FrameRecord(net, i * vl["bag_ms"] / 1000.0, vl["vl_id"], sn, Ether(bytes(pkt))))
    yield from sorted(records, key=lambda r: (r.t_s, r.vl_id, r.network))


def parse_frame(frame: bytes | Packet) -> dict:
    """Independent of the builder's inputs: reads VL ID and SN back from the wire bytes."""
    raw = bytes(frame)
    if len(raw) < MIN_FRAME_NO_FCS:
        raise ValueError(f"frame is {len(raw)} bytes, below the {MIN_FRAME_NO_FCS}-byte minimum (without FCS)")
    if raw[:4] != DST_MAC_PREFIX:
        raise ValueError("destination MAC lacks the AFDX constant field 03:00:00:00")
    if raw[12:14] != b"\x08\x00":
        raise ValueError("EtherType is not IPv4 (0x0800)")
    if raw[14] != 0x45:
        raise ValueError("expected IPv4 with a 20-byte header (version 4, IHL 5)")
    if raw[23] != 17:
        raise ValueError("IP protocol is not UDP (17)")
    if int.from_bytes(raw[16:18], "big") != len(raw) - ETH_HDR:
        raise ValueError("IP total length does not match the frame length")
    if int.from_bytes(raw[38:40], "big") != len(raw) - ETH_HDR - IP_HDR:
        raise ValueError("UDP length does not match the frame length")
    return {
        "vl_id": int.from_bytes(raw[4:6], "big"),
        "sn": raw[-1],
        "dst_mac": raw[0:6].hex(":"),
        "src_mac": raw[6:12].hex(":"),
        "dst_ip": ".".join(str(b) for b in raw[30:34]),
        "udp_dport": int.from_bytes(raw[36:38], "big"),
        "frame_len_no_fcs": len(raw),
    }
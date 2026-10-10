"""Acceptance tests for item 4.1 (ARINC 664 AFDX synthetic schemas).

AC1: When a scapy script builds frames from the schema, each has a valid virtual
     link ID and sequence number field.
AC2: When a schema value violates the spec (e.g. BAG not a power of 2 ms),
     schema validation rejects it.
"""
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from scapy.layers.l2 import Ether

from bench.afdx import afdx_synth as afdx

CASES = json.loads((Path(__file__).parents[1] / "bench" / "afdx" / "cases" / "schema_cases.json").read_text())["cases"]


def example_config() -> dict:
    """Three VLs, mixed BAGs, one starting near the SN wrap, one single-network."""
    def vl(vl_id, bag, sn, nets, payload):
        return {
            "vl_id": vl_id, "bag_ms": bag, "lmax_bytes": 1518, "skew_max_ms": 2 * bag,
            "src_ip": f"10.1.{vl_id >> 8}.{vl_id & 0xFF}", "src_mac_user_id": (0x0100 + vl_id) & 0xFFFF,
            "interface_id": 0, "udp_src_port": 1 + (4000 + vl_id) % 60000, "udp_dst_port": 1 + (5000 + vl_id) % 60000,
            "app_payload_bytes": payload, "networks": nets, "initial_sn": sn,
        }
    return {
        "schema_version": "1.0",
        "redundancy": {"integrity_check": True, "resync_rule": "drop_with_resync"},
        "virtual_links": [
            vl(100, 16, 1, ["A", "B"], 64),
            vl(256, 4, 254, ["A", "B"], 0),      # tiny payload -> padded; SN wraps 255 -> 1
            vl(65535, 128, 250, ["A"], 1471),    # max VL ID, max payload
        ],
    }


# schema file itself
def test_schema_file_exists_and_is_a_valid_json_schema():
    assert afdx.SCHEMA_PATH.is_file()
    Draft202012Validator.check_schema(afdx.load_schema())


# AC1: frames from schema
@pytest.fixture(scope="module")
def generated():
    cfg = example_config()
    return cfg, list(afdx.generate_frames(cfg, frames_per_vl=300))


def test_example_config_is_valid():
    assert afdx.validate_config(example_config()) == []


def test_every_frame_has_valid_vl_id_and_sequence_number(generated):
    cfg, records = generated
    by_id = {v["vl_id"]: v for v in cfg["virtual_links"]}
    assert records
    for rec in records:
        wire = afdx.parse_frame(bytes(rec.packet))       # read back from raw bytes
        assert wire["vl_id"] in by_id                    # VL ID is one the config defines
        assert 1 <= wire["vl_id"] <= 65535
        assert wire["vl_id"] == rec.vl_id
        assert 1 <= wire["sn"] <= 255  # never 0 (reset) in normal traffic
        assert wire["sn"] == rec.sn


def test_vl_id_is_encoded_in_dst_mac_and_dst_ip(generated):
    _, records = generated
    for rec in records:
        eth = Ether(bytes(rec.packet))
        hi, lo = rec.vl_id >> 8, rec.vl_id & 0xFF          # expectation derived here, not via afdx.*
        assert eth.dst == f"03:00:00:00:{hi:02x}:{lo:02x}"
        assert eth["IP"].dst == f"224.224.{hi}.{lo}"
        assert eth.type == 0x0800


def test_sequence_numbers_increment_by_one_and_wrap_255_to_1(generated):
    _, records = generated
    per_stream = defaultdict(list)
    for rec in records:
        per_stream[(rec.vl_id, rec.network)].append(rec.sn)
    for (vl_id, net), sns in per_stream.items():
        for prev, cur in zip(sns, sns[1:]):
            expected = 1 if prev == 255 else prev + 1
            assert cur == expected, f"VL {vl_id} net {net}: {prev} -> {cur}"
    assert per_stream[(256, "A")][:4] == [254, 255, 1, 2]   # the wrap, explicitly


def test_networks_a_and_b_carry_identical_sn_and_bytes(generated):
    cfg, records = generated
    seen = defaultdict(dict)
    for rec in records:
        seen[(rec.vl_id, rec.t_s)][rec.network] = rec
    for (vl_id, _), nets in seen.items():
        expected = next(v for v in cfg["virtual_links"] if v["vl_id"] == vl_id)["networks"]
        assert sorted(nets) == sorted(expected)
        if len(nets) == 2:
            assert nets["A"].sn == nets["B"].sn
            assert bytes(nets["A"].packet) == bytes(nets["B"].packet)


def test_frame_sizes_respect_ethernet_minimum_and_lmax(generated):
    cfg, records = generated
    lmax = {v["vl_id"]: v["lmax_bytes"] for v in cfg["virtual_links"]}
    for rec in records:
        n = len(bytes(rec.packet))
        assert n >= 60                      # 64-byte Ethernet minimum minus 4-byte FCS
        assert n + 4 <= lmax[rec.vl_id]
    # SN stays the last byte even when padding is inserted before it
    padded = [r for r in records if r.vl_id == 256][0]
    assert bytes(padded.packet)[-1] == padded.sn


def test_frames_are_spaced_at_bag(generated):
    cfg, records = generated
    for vl in cfg["virtual_links"]:
        times = sorted({r.t_s for r in records if r.vl_id == vl["vl_id"]})
        gaps = {round(b - a, 9) for a, b in zip(times, times[1:])}
        assert gaps == {vl["bag_ms"] / 1000.0}


def test_generation_is_deterministic():
    a = [bytes(r.packet) for r in afdx.generate_frames(example_config(), 20)]
    b = [bytes(r.packet) for r in afdx.generate_frames(example_config(), 20)]
    assert a == b


def test_sequence_counter_rejects_reserved_zero():
    with pytest.raises(ValueError):
        afdx.SequenceCounter(0)


def test_builder_refuses_invalid_config():
    bad = example_config()
    bad["virtual_links"][0]["bag_ms"] = 3
    with pytest.raises(afdx.SchemaValidationError):
        list(afdx.generate_frames(bad, 1))




# golden frames (hand-computed)
def golden_vl(**kw):
    vl = {"vl_id": 258, "bag_ms": 16, "lmax_bytes": 512, "skew_max_ms": 2, "src_ip": "10.1.2.3",
          "src_mac_user_id": 0x1234, "interface_id": 1, "udp_src_port": 4660, "udp_dst_port": 5001,
          "app_payload_bytes": 64, "networks": ["A", "B"], "initial_sn": 1}
    vl.update(kw)
    return vl


def test_golden_frame_header_bytes():
    """VL 258 = 0x0102 (asymmetric on purpose, so a byte swap changes the answer)."""
    raw = bytes(afdx.build_frame(golden_vl(), 7))
    assert raw[0:6].hex(":") == "03:00:00:00:01:02"       # constant 03:00:00:00 + VL ID big-endian
    assert raw[6:12].hex(":") == "02:00:00:12:34:20"      # 02:00:00 + user id 0x1234 + (interface 1 << 5)
    assert raw[12:14] == b"\x08\x00"                      # EtherType IPv4
    assert raw[26:30] == bytes([10, 1, 2, 3])             # IP source
    assert raw[30:34] == bytes([224, 224, 1, 2])          # IP destination 224.224.<VL hi>.<VL lo>
    assert raw[23] == 17                                  # IP protocol UDP
    assert raw[34:36] == (4660).to_bytes(2, "big") and raw[36:38] == (5001).to_bytes(2, "big")
    assert len(raw) == 14 + 20 + 8 + 64 + 1               # no padding needed at 64-byte payload
    assert raw[-1] == 7                                   # sequence number is the last byte


def test_golden_frame_padding_keeps_sn_last_and_hits_minimum_size():
    raw = bytes(afdx.build_frame(golden_vl(app_payload_bytes=0), 255))
    assert len(raw) == 60                                 # padded up to the 64-byte minimum minus FCS
    assert raw[-1] == 255
    assert raw[42:-1] == b"\x00" * 17                     # 17 zero pad bytes before the SN


def test_golden_frame_at_largest_legal_size():
    raw = bytes(afdx.build_frame(golden_vl(lmax_bytes=1518, app_payload_bytes=1471), 1))
    assert len(raw) + 4 == 1518
    assert raw[-1] == 1


def test_golden_vl_id_extremes():
    low = bytes(afdx.build_frame(golden_vl(vl_id=1), 1))
    high = bytes(afdx.build_frame(golden_vl(vl_id=65535), 1))
    assert low[0:6].hex(":") == "03:00:00:00:00:01" and low[30:34] == bytes([224, 224, 0, 1])
    assert high[0:6].hex(":") == "03:00:00:00:ff:ff" and high[30:34] == bytes([224, 224, 255, 255])


# structure and independence
def test_ip_and_udp_length_fields_match_frame_length(generated):
    _, records = generated
    for rec in records:
        raw = bytes(rec.packet)
        eth = Ether(raw)
        assert eth["IP"].len == len(raw) - 14
        assert eth["UDP"].len == len(raw) - 34
        assert eth["IP"].ihl == 5 and eth["IP"].proto == 17


def test_parse_frame_rejects_structurally_broken_frames():
    good = bytes(afdx.build_frame(golden_vl(), 3))
    assert afdx.parse_frame(good)["sn"] == 3

    def patched(offset, value):
        b = bytearray(good)
        b[offset:offset + len(value)] = value
        return bytes(b)

    # each case breaks ONE thing and must be rejected for that specific reason
    broken = {
        "EtherType": (patched(12, b"\x08\x06"), "EtherType"),
        "IP options (IHL 6)": (patched(14, b"\x46"), "IHL"),
        "not UDP": (patched(23, b"\x06"), "UDP \\(17\\)"),
        "IP total length field only": (patched(16, (len(good)).to_bytes(2, "big")), "IP total length"),
        "UDP length field only": (patched(38, (len(good)).to_bytes(2, "big")), "UDP length"),
        "dst MAC constant": (patched(0, b"\x01"), "constant field"),
        "runt frame": (good[:30], "minimum"),
    }
    for name, (frame, reason) in broken.items():
        with pytest.raises(ValueError, match=reason):
            afdx.parse_frame(frame)


def test_ab_copies_are_independent_packet_objects():
    recs = list(afdx.generate_frames(example_config(), 3))
    a = next(r for r in recs if r.vl_id == 100 and r.network == "A" and r.sn == 1)
    b = next(r for r in recs if r.vl_id == 100 and r.network == "B" and r.sn == 1)
    assert a.packet is not b.packet
    assert bytes(a.packet) == bytes(b.packet)
    before = bytes(b.packet)
    a.packet["Raw"].load = b"\x00" * len(a.packet["Raw"].load)     # simulate an anomaly on A only
    assert bytes(b.packet) == before                                # B untouched


# AC2: validation rejects
@pytest.mark.parametrize("bag", [0, 3, 5, 6, 7, 9, 12, 24, 48, 100, 129, 256, 1.5, -2, "16", None, True])
def test_bag_not_a_power_of_two_ms_is_rejected(bag):
    cfg = example_config()
    cfg["virtual_links"][0]["bag_ms"] = bag
    issues = afdx.validate_config(cfg)
    assert any(i.path == "virtual_links/0/bag_ms" for i in issues), issues


@pytest.mark.parametrize("bag", [1, 2, 4, 8, 16, 32, 64, 128])   # literal on purpose, not afdx.VALID_BAG_MS
def test_every_permitted_bag_is_accepted(bag):
    cfg = example_config()
    cfg["virtual_links"][0]["bag_ms"] = bag
    cfg["virtual_links"][0]["skew_max_ms"] = 0
    assert afdx.validate_config(cfg) == []


def test_skew_max_boundary():
    cfg = example_config()
    vl = cfg["virtual_links"][0]
    vl["skew_max_ms"] = 253 * vl["bag_ms"]
    assert afdx.validate_config(cfg) == []
    vl["skew_max_ms"] += 0.001
    assert any(i.path.endswith("skew_max_ms") for i in afdx.validate_config(cfg))


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_synthetic_schema_cases(case):
    issues = afdx.validate_config(deepcopy(case["config"]))
    if case["expect_valid"]:
        assert issues == [], f"expected valid, got: {issues}"
    else:
        assert issues, "expected rejection, config was accepted"
        assert any(i.path == case["expect_path"] for i in issues), (
            f"expected issue at {case['expect_path']!r}, got {[str(i) for i in issues]}")


def test_case_file_covers_both_outcomes():
    assert sum(c["expect_valid"] for c in CASES) >= 5
    assert sum(not c["expect_valid"] for c in CASES) >= 30
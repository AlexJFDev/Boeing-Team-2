"""Writes cases/schema_cases.json: synthetic valid and invalid AFDX configs.

Each invalid case mutates exactly one thing from a known-good base so the expected
rejection path is unambiguous. Re-run after changing the schema:
    python generate_schema_cases.py
"""
import json
from copy import deepcopy
from pathlib import Path

BASE_VL = {
    "vl_id": 100, "name": "vl_100", "bag_ms": 16, "lmax_bytes": 512, "skew_max_ms": 2,
    "src_ip": "10.1.2.3", "src_mac_user_id": 0x1234, "interface_id": 1,
    "udp_src_port": 5000, "udp_dst_port": 5001, "app_payload_bytes": 64,
    "networks": ["A", "B"], "initial_sn": 1,
}


def vl(**kw):
    d = deepcopy(BASE_VL)
    d.update(kw)
    return d


def vl_without(key):
    d = deepcopy(BASE_VL)
    del d[key]
    return d


def cfg(*vls, redundancy=None, **top):
    c = {
        "schema_version": "1.0",
        "redundancy": redundancy or {"integrity_check": True, "resync_rule": "drop_with_resync"},
        "virtual_links": list(vls) or [vl()],
    }
    c.update(top)
    return c


cases = []


def ok(name, config, note=""):
    cases.append({"name": name, "expect_valid": True, "expect_path": None, "note": note, "config": config})


def bad(name, config, path, note=""):
    cases.append({"name": name, "expect_valid": False, "expect_path": path, "note": note, "config": config})


P = "virtual_links/0/"

# --valid--
ok("minimal_valid", cfg())
ok("every_permitted_bag",
   cfg(*[vl(vl_id=200 + i, bag_ms=b, skew_max_ms=0) for i, b in enumerate([1, 2, 4, 8, 16, 32, 64, 128])]),
   "BAG 1..128 ms, powers of two")
ok("boundary_values",
   cfg(vl(vl_id=65535, bag_ms=128, skew_max_ms=253 * 128, lmax_bytes=1518, app_payload_bytes=1471,
          networks=["A"], initial_sn=255),
       vl(vl_id=1, bag_ms=1, skew_max_ms=253, lmax_bytes=64, app_payload_bytes=17)),
   "SkewMax exactly 253 x BAG; payload exactly fills Lmax; SN starts at 255")
ok("sn_wrap_start", cfg(vl(initial_sn=254)), "255 -> 1 wrap inside the first few frames")
ok("resync_without", cfg(redundancy={"integrity_check": True, "resync_rule": "drop_without_resync"}))
ok("single_network_b", cfg(vl(networks=["B"])))

# -- invalid: BAG (the stated acceptance example) --
bad("bag_not_power_of_two_3", cfg(vl(bag_ms=3)), P + "bag_ms")
bad("bag_not_power_of_two_6", cfg(vl(bag_ms=6)), P + "bag_ms", "looks plausible, still wrong")
bad("bag_not_power_of_two_100", cfg(vl(bag_ms=100)), P + "bag_ms")
bad("bag_zero", cfg(vl(bag_ms=0)), P + "bag_ms")
bad("bag_above_128", cfg(vl(bag_ms=256)), P + "bag_ms")
bad("bag_fractional", cfg(vl(bag_ms=1.5)), P + "bag_ms")
bad("bag_string", cfg(vl(bag_ms="16")), P + "bag_ms")

# -- invalid: SkewMax --
bad("skew_over_253_bag", cfg(vl(bag_ms=1, skew_max_ms=254)), P + "skew_max_ms")
bad("skew_negative", cfg(vl(skew_max_ms=-1)), P + "skew_max_ms")

# -- invalid: VL ID --
bad("vl_id_zero", cfg(vl(vl_id=0)), P + "vl_id")
bad("vl_id_16bit_overflow", cfg(vl(vl_id=65536)), P + "vl_id")
bad("vl_id_negative", cfg(vl(vl_id=-5)), P + "vl_id")
bad("vl_id_duplicate", cfg(vl(vl_id=7), vl(vl_id=7)), "virtual_links/1/vl_id")

# -- invalid: sequence number --
bad("initial_sn_zero_reserved", cfg(vl(initial_sn=0)), P + "initial_sn", "0 is reserved for reset")
bad("initial_sn_256", cfg(vl(initial_sn=256)), P + "initial_sn")

# -- invalid: frame size --
bad("lmax_below_min", cfg(vl(lmax_bytes=63)), P + "lmax_bytes")
bad("lmax_above_max", cfg(vl(lmax_bytes=1519)), P + "lmax_bytes")
bad("payload_exceeds_lmax", cfg(vl(lmax_bytes=64, app_payload_bytes=18)), P + "app_payload_bytes")
bad("payload_negative", cfg(vl(app_payload_bytes=-1)), P + "app_payload_bytes")

# -- invalid: networks / ports / address --
bad("networks_empty", cfg(vl(networks=[])), P + "networks")
bad("networks_unknown_C", cfg(vl(networks=["C"])), P + "networks/0")
bad("networks_duplicate", cfg(vl(networks=["A", "A"])), P + "networks")
bad("udp_port_zero", cfg(vl(udp_dst_port=0)), P + "udp_dst_port")
bad("udp_port_overflow", cfg(vl(udp_src_port=70000)), P + "udp_src_port")
bad("interface_id_4bit", cfg(vl(interface_id=8)), P + "interface_id")
bad("src_ip_not_10_net", cfg(vl(src_ip="192.168.1.1")), P + "src_ip")
bad("src_ip_bad_octet", cfg(vl(src_ip="10.1.2.300")), P + "src_ip")

# -- invalid: structure / premises --
bad("missing_bag", cfg(vl_without("bag_ms")), "virtual_links/0")
bad("unknown_field", cfg(vl(priority="high")), "virtual_links/0")
bad("no_virtual_links", cfg(virtual_links=[]) | {"virtual_links": []}, "virtual_links")
bad("bad_resync_rule", cfg(redundancy={"integrity_check": True, "resync_rule": "ignore"}), "redundancy/resync_rule")
bad("missing_resync_rule", cfg(redundancy={"integrity_check": True}), "redundancy",
    "tasks must state the resync rule (catalogue C1)")
bad("wrong_schema_version", cfg(schema_version="2.0"), "schema_version")

out = Path(__file__).parent / "cases" / "schema_cases.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps({"schema": "afdx_schema.json", "cases": cases}, indent=2) + "\n")
print(f"wrote {len(cases)} cases ({sum(c['expect_valid'] for c in cases)} valid) -> {out}")

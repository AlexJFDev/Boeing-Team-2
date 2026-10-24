"""Generate labeled AFDX scenarios: one pcap per network + a ground-truth JSON each.

    python -m bench.afdx.generate_afdx_traces [out_dir]        (default: <repo>/data/afdx)

Layout:  out_dir/pcap/<id>_netA.pcap, <id>_netB.pcap   (captured per network, as in real life)
         out_dir/truth/<id>.json                       (config, preamble, labels, decisions)
         out_dir/manifest.json                         (counts)

Anomalies are injected by construction, so their labels are exact. Each is paired with a
benign oddity that looks similar but is conformant (a late frame, an SN skip of one, a B
network dropout, a wrap 255 -> 1, a constant B offset inside SkewMax). Deterministic: seeded.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

from scapy.layers.l2 import Ether
from scapy.utils import wrpcap

from bench.afdx import afdx_oracle as oracle
from bench.afdx import afdx_synth as afdx

T0 = 1000.0
N = 40                      # logical frames per VL (each sent on A and B)
BAGS = (2, 8, 32, 128)
SEEDS_PER_KIND = 2
N_MIXED = 40
N_MIXED_BENIGN = 16         # multi-VL traces with NO anomalies requested (zero violations expected)
PAYLOAD = 64
UDP_PAYLOAD_OFF = 42


# building blocks
def make_vl(vl_id: int, bag: int, rng: random.Random) -> dict:
    return {
        "vl_id": vl_id, "name": f"vl_{vl_id}", "bag_ms": bag, "lmax_bytes": 128,
        "skew_max_ms": bag * 0.25, "src_ip": f"10.1.{vl_id >> 8}.{vl_id & 0xFF}",
        "src_mac_user_id": 0x0100 + vl_id, "interface_id": 0,
        "udp_src_port": 4000 + vl_id, "udp_dst_port": 5000 + vl_id,
        "app_payload_bytes": PAYLOAD, "networks": ["A", "B"],
        "initial_sn": rng.randint(1, 200),   # <=200 so a 40-frame run never wraps by accident
    }


def new_frame(net, t, vl, sn):
    return {"net": net, "t": t, "vl_cfg": vl, "vl_id": vl["vl_id"], "sn": sn,
            "src_user": None, "mod": None, "labels": [], "benign": []}


def make_stream(vl: dict) -> list[dict]:
    counter = afdx.SequenceCounter(vl["initial_sn"])
    rows = []
    for k in range(N):
        sn = counter.next()
        rows.append({net: new_frame(net, T0 + k * vl["bag_ms"] / 1000, vl, sn) for net in vl["networks"]})
    return rows


def frames_of(row):
    return [f for f in row.values() if f is not None]


# events
# Each event mutates one VL's stream `st` and may return extra frames.
def ev_clean(st, vl, rng):
    return []


def ev_bag_violation(st, vl, rng):
    k = rng.randint(8, N - 10)
    for f in frames_of(st[k]):
        f["t"] = st[k - 1]["A"]["t"] + 0.5 * vl["bag_ms"] / 1000
        f["labels"].append("bag_violation")
    return []


def ev_late_frame_gap(st, vl, rng):               # benign: BAG is a MINIMUM gap
    k = rng.randint(8, N - 10)
    for row in st[k:]:
        for f in frames_of(row):
            f["t"] += 0.5 * vl["bag_ms"] / 1000
    for f in frames_of(st[k]):
        f["benign"].append("late_frame_gap")
    return []


def ev_skew_violation(st, vl, rng):               # B delayed beyond SkewMax for the last frames
    for row in st[N - 6:]:
        row["B"]["t"] += 0.5 * vl["bag_ms"] / 1000
        row["B"]["labels"].append("skew_violation")
    return []


def ev_skew_offset_within_limit(st, vl, rng):     # benign: constant B offset < SkewMax
    for row in st:
        row["B"]["t"] += 0.2 * vl["bag_ms"] / 1000
        row["B"]["benign"].append("skew_offset_within_limit")
    return []


def ev_sn_gap(st, vl, rng):
    k, jump = rng.randint(8, N - 10), rng.randint(3, 12)
    for row in st[k:]:
        for f in frames_of(row):
            f["sn"] = oracle.nxt(f["sn"], jump - 1)
    for f in frames_of(st[k]):
        f["labels"].append("sn_gap")
    return []


def ev_sn_replay(st, vl, rng):
    k = rng.randint(8, N - 10)
    for net in vl["networks"]:
        st[k][net]["sn"] = st[k - 3][net]["sn"]
        st[k][net]["labels"].append("sn_replay")
    return []


def ev_single_frame_loss(st, vl, rng):            # benign: skip of exactly one SN is in-window
    k = rng.randint(8, N - 10)
    for row in st[k:]:
        for f in frames_of(row):
            f["sn"] = oracle.nxt(f["sn"], 1)
    for f in frames_of(st[k]):
        f["benign"].append("single_frame_loss")
    return []


def pre_sn_wrap(vl, rng):
    vl["initial_sn"] = 255 - rng.randint(3, 12)


def ev_sn_wrap(st, vl, rng):                      # benign: 255 -> 1 is not a gap
    for j, row in enumerate(st):
        if j > 0 and row["A"]["sn"] == 1:
            for f in frames_of(row):
                f["benign"].append("sn_wrap_255_to_1")
    return []


def ev_network_dropout(st, vl, rng):              # benign: B goes quiet for 5 frames then resumes
    k = rng.randint(8, N - 12)
    for j in range(k, k + 5):
        st[j]["B"] = None
    st[k + 5]["B"]["benign"].append("network_b_resume_after_dropout")
    return []


def ev_unknown_vl(st, vl, rng):
    k = rng.randint(8, N - 10)
    ghost = {**vl, "vl_id": 40000 + vl["vl_id"]}
    f = new_frame("A", st[k]["A"]["t"] + 0.5 * vl["bag_ms"] / 1000, ghost, rng.randint(1, 255))
    f["labels"].append("unknown_vl")
    return [f]


def ev_lmax_exceeded(st, vl, rng):
    k = rng.randint(8, N - 10)
    big = {**vl, "lmax_bytes": 1518, "app_payload_bytes": 200}
    for f in frames_of(st[k]):
        f["vl_cfg"] = big
        f["labels"].append("lmax_exceeded")
    return []


def ev_spoofed_source(st, vl, rng):
    k = rng.randint(8, N - 10)
    st[k]["A"]["src_user"] = (vl["src_mac_user_id"] + 1) & 0xFFFF
    st[k]["A"]["labels"].append("spoofed_source")
    return []


def ev_bus_fault(st, vl, rng):                    # bit flips, UDP checksum left broken
    f = st[rng.randint(8, N - 10)]["A"]
    f["mod"] = "fault"
    f["labels"].append("bus_fault_corruption")
    return []


def ev_injection(st, vl, rng):                    # payload rewritten, checksum recomputed
    f = st[rng.randint(8, N - 10)]["A"]
    f["mod"] = "inject"
    f["labels"].append("injected_payload")
    return []


EVENTS = {
    # anomalies
    "bag_violation": ev_bag_violation, "skew_violation": ev_skew_violation,
    "sn_gap": ev_sn_gap, "sn_replay": ev_sn_replay, "unknown_vl": ev_unknown_vl,
    "lmax_exceeded": ev_lmax_exceeded, "spoofed_source": ev_spoofed_source,
    "bus_fault_corruption": ev_bus_fault, "injected_payload": ev_injection,
    # benign look-alikes
    "late_frame_gap": ev_late_frame_gap, "skew_offset_within_limit": ev_skew_offset_within_limit,
    "single_frame_loss": ev_single_frame_loss, "sn_wrap": ev_sn_wrap,
    "network_dropout": ev_network_dropout,
    # control
    "clean": ev_clean,
}
PRE = {"sn_wrap": pre_sn_wrap}
ANOMALY_KINDS = ["bag_violation", "skew_violation", "sn_gap", "sn_replay", "unknown_vl",
                 "lmax_exceeded", "spoofed_source", "bus_fault_corruption", "injected_payload"]
BENIGN_KINDS = ["late_frame_gap", "skew_offset_within_limit", "single_frame_loss", "sn_wrap",
                "network_dropout"]


# rendering
def render(f: dict, rng: random.Random) -> bytes:
    vl = dict(f["vl_cfg"])
    if f["src_user"] is not None:
        vl["src_mac_user_id"] = f["src_user"]
    raw = bytearray(bytes(afdx.build_frame(vl, f["sn"])))
    n_app = vl["app_payload_bytes"]
    if f["mod"] == "fault":                       # flip 1-3 bits in the app region; checksum stale
        while True:
            trial = bytearray(raw)
            for pos in rng.sample(range(n_app), rng.randint(1, 3)):
                trial[UDP_PAYLOAD_OFF + pos] ^= 1 << rng.randrange(8)
            if not oracle.udp_checksum_ok(bytes(trial)):
                raw = trial
                break
    elif f["mod"] == "inject":                    # rewrite bytes, then recompute UDP checksum
        for pos in rng.sample(range(n_app), rng.randint(4, 8)):
            raw[UDP_PAYLOAD_OFF + pos] = (raw[UDP_PAYLOAD_OFF + pos] + rng.randint(1, 255)) % 256
        pkt = Ether(bytes(raw))
        del pkt["UDP"].chksum
        raw = bytearray(bytes(Ether(bytes(pkt))))
        assert oracle.udp_checksum_ok(bytes(raw))
    return bytes(raw)


def preamble(vls: list[dict], rule: str) -> str:
    lines = ["ARINC 664 Part 7 capture. Networks A and B were captured separately (netA / netB pcaps).",
             "Per virtual link:"]
    for v in vls:
        mac = afdx.src_mac(v["src_mac_user_id"], v["interface_id"])
        lines.append(
            f"  VL {v['vl_id']}: BAG {v['bag_ms']} ms (MINIMUM gap between consecutive frames on one "
            f"network; longer gaps are allowed); SkewMax {v['skew_max_ms']} ms (max arrival difference "
            f"between the A and B copy of the same frame); Lmax {v['lmax_bytes']} bytes including the "
            f"4-byte FCS; registered source MAC {mac}.")
    lines += [
        "Integrity check, per VL and per network: the first frame is accepted; a later frame is accepted "
        "only if its sequence number is the previous plus 1 or plus 2 (255 wraps to 1; 0 is reserved).",
        "Resync rule: " + ("after a dropped frame, the 'previous' number becomes the received number."
                           if rule == "drop_with_resync" else
                           "after a dropped frame, the 'previous' number is left unchanged."),
        "Redundancy management: of the integrity-accepted copies the first is delivered; the later copy "
        "with the same sequence number is discarded. This is normal and is not an anomaly.",
    ]
    return "\n".join(lines)


def write_scenario(sid, kind, category, vls, rows_by_vl, extras, rule, rng, out: Path, seed=None):
    frames = [f for rows in rows_by_vl for row in rows for f in frames_of(row)] + extras
    for o, f in enumerate(frames):
        f["order"] = o
        f["raw"] = render(f, rng)
    paths = {}
    for net in ("A", "B"):
        netf = sorted((f for f in frames if f["net"] == net), key=lambda f: (f["t"], f["vl_id"], f["order"]))
        pkts = []
        for pos, f in enumerate(netf):
            f["pos"], f["id"] = pos, f"{net}{pos}"
            p = Ether(f["raw"])
            p.time = f["t"]
            pkts.append(p)
        paths[net] = out / "pcap" / f"{sid}_net{net}.pcap"
        wrpcap(str(paths[net]), pkts)

    vl_map = {v["vl_id"]: v for v in vls}
    loaded = oracle.load_frames(paths["A"], paths["B"])
    dec = {r: oracle.decisions(loaded, vl_map, r) for r in oracle.RESYNC_RULES}
    by_id = {f["id"]: f for f in frames}

    frame_rows = []
    for f in sorted(frames, key=lambda f: (f["t"], f["net"], f["pos"])):
        frame_rows.append({
            "id": f["id"], "network": f["net"], "t_s": round(f["t"] - T0, 6), "vl_id": f["vl_id"],
            "sn": f["sn"], "labels": f["labels"], "benign": f["benign"],
            "decisions": {r: dec[r][f["id"]] for r in oracle.RESYNC_RULES},
        })

    def group(key):
        g: dict = {}
        for f in frames:
            for lab in f[key]:
                g.setdefault((lab, f["vl_id"] if f["vl_id"] in vl_map else f["vl_cfg"]["vl_id"] - 40000), []).append(f["id"])
        return [{"kind": k, "vl_id": v, "frame_ids": sorted(ids)} for (k, v), ids in sorted(g.items())]

    # Flat, time-ordered list: one entry per injected violation, each with its exact timestamp.
    # t_s is seconds from trace start; t_us is the same instant as an integer microsecond count
    # (pcap's resolution), so a consumer can compare without float rounding.
    violations = []
    for f in frames:
        for lab in f["labels"]:
            violations.append({
                "kind": lab, "frame_id": f["id"], "network": f["net"], "vl_id": f["vl_id"],
                "sn": f["sn"], "t_s": round(f["t"] - T0, 6), "t_us": round((f["t"] - T0) * 1e6),
            })
    violations.sort(key=lambda v: (v["t_us"], v["network"], v["frame_id"], v["kind"]))
    requested = [k for k in kind.removeprefix("mixed:").split("+") if k in ANOMALY_KINDS]

    truth = {
        "scenario_id": sid, "kind": kind, "category": category, "resync_rule": rule, "seed": seed,
        "anomalies_requested": requested,
        "violations": violations,
        "config": {"schema_version": "1.0",
                   "redundancy": {"integrity_check": True, "resync_rule": rule}, "virtual_links": vls},
        "pcap": {n: str(paths[n].relative_to(out)) for n in paths},
        "preamble": preamble(vls, rule),
        "anomalies": group("labels"), "benign_oddities": group("benign"),
        "frames": frame_rows,
    }
    (out / "truth" / f"{sid}.json").write_text(json.dumps(truth, indent=1) + "\n")
    return {
        "id": sid, "kind": kind, "category": category, "seed": seed, "resync_rule": rule,
        "vl_ids": [v["vl_id"] for v in vls], "bags_ms": [v["bag_ms"] for v in vls],
        "frames": len(frames),
        "anomalous_frames": sum(bool(f["labels"]) for f in frames),   # labeled frames, not events
        "benign_oddity_frames": sum(bool(f["benign"]) for f in frames),
        "violations": len(violations),                                 # entries in truth["violations"]
        "violations_by_kind": dict(Counter(v["kind"] for v in violations)),
    }


# scenario sets
def single_scenarios(out: Path, manifest: list):
    n = 0
    for kind in EVENTS:
        category = "anomaly" if kind in ANOMALY_KINDS else "benign" if kind in BENIGN_KINDS else "clean"
        for bi, bag in enumerate(BAGS):
            for seed in range(SEEDS_PER_KIND):
                n += 1
                seed_str = f"{kind}-{bag}-{seed}"
                rng = random.Random(seed_str)
                vl1, vl2 = make_vl(100, bag, rng), make_vl(101, BAGS[(bi + 1) % len(BAGS)], rng)
                if kind in PRE:
                    PRE[kind](vl1, rng)
                st1, st2 = make_stream(vl1), make_stream(vl2)
                extras = EVENTS[kind](st1, vl1, rng)
                rule = oracle.RESYNC_RULES[(bi + seed) % 2]
                sid = f"S{n:03d}_{kind}_bag{bag}_s{seed}"
                manifest.append(write_scenario(sid, kind, category, [vl1, vl2], [st1, st2], extras, rule, rng, out,
                                               seed=seed_str))


def build_mixed(out: Path, manifest: list, rng, seed: str, kinds: list, category: str, sid: str, rule: str):
    """One multi-VL trace: a VL per event kind in `kinds`."""
    vls = [make_vl(200 + i, rng.choice(BAGS), rng) for i in range(len(kinds))]
    for vl, kind in zip(vls, kinds):
        if kind in PRE:
            PRE[kind](vl, rng)
    streams = [make_stream(vl) for vl in vls]
    extras = []
    for vl, st, kind in zip(vls, streams, kinds):
        extras += EVENTS[kind](st, vl, rng)
    manifest.append(write_scenario(sid, "mixed:" + "+".join(kinds), category, vls, streams, extras, rule, rng, out,
                                   seed=seed))


def mixed_scenarios(out: Path, manifest: list):
    """A4-style triage traces: 4 VLs, one event each (2 anomalies, 1 benign look-alike, 1 clean/benign).
    The four kinds are always distinct."""
    for m in range(N_MIXED):
        seed = f"mixed-{m}"
        rng = random.Random(seed)
        anomalies = rng.sample(ANOMALY_KINDS, 2)
        first = rng.choice(BENIGN_KINDS)
        second = rng.choice([k for k in BENIGN_KINDS + ["clean"] if k != first])
        kinds = anomalies + [first, second]
        rng.shuffle(kinds)
        build_mixed(out, manifest, rng, seed, kinds, "mixed", f"M{m + 1:02d}_mixed", oracle.RESYNC_RULES[m % 2])


def mixed_benign_scenarios(out: Path, manifest: list):
    """Multi-VL traces with NO anomalies requested: 4 VLs, four distinct conformant events (or clean).
    Zero violations are expected in every one of them."""
    pool = BENIGN_KINDS + ["clean"]
    for m in range(N_MIXED_BENIGN):
        seed = f"mixed-benign-{m}"
        rng = random.Random(seed)
        kinds = rng.sample(pool, 4)
        build_mixed(out, manifest, rng, seed, kinds, "mixed_benign", f"MB{m + 1:02d}_mixed_benign",
                    oracle.RESYNC_RULES[m % 2])


def main(out_dir: str | None = None):
    out = Path(out_dir) if out_dir else Path(__file__).resolve().parents[2] / "data" / "afdx"
    (out / "pcap").mkdir(parents=True, exist_ok=True)
    (out / "truth").mkdir(parents=True, exist_ok=True)
    manifest: list = []
    single_scenarios(out, manifest)
    mixed_scenarios(out, manifest)
    mixed_benign_scenarios(out, manifest)
    by_violation_kind: Counter = Counter()
    for m in manifest:
        by_violation_kind.update(m["violations_by_kind"])
    summary = {
        "scenarios": len(manifest),
        "frames_total": sum(m["frames"] for m in manifest),
        "anomalous_frames": sum(m["anomalous_frames"] for m in manifest),
        "benign_oddity_frames": sum(m["benign_oddity_frames"] for m in manifest),
        "violations_total": sum(m["violations"] for m in manifest),
        "by_violation_kind": dict(by_violation_kind),
        "by_category": dict(Counter(m["category"] for m in manifest)),
        "by_kind": dict(Counter(m["kind"] for m in manifest if not m["category"].startswith("mixed"))),
    }
    generator = {
        "schema_version": "1.0", "t0_epoch_s": T0, "frames_per_vl": N, "bags_ms": list(BAGS),
        "seeds_per_kind": SEEDS_PER_KIND, "n_mixed": N_MIXED, "n_mixed_benign": N_MIXED_BENIGN,
        "app_payload_bytes": PAYLOAD,
        "anomaly_kinds": ANOMALY_KINDS, "benign_kinds": BENIGN_KINDS,
        "seeding": "each scenario uses random.Random(<seed string>); the seed is recorded per scenario",
        "timestamps": "violation t_s / t_us are relative to trace start; pcap time = t0_epoch_s + t_s",
    }
    doc = {"generator": generator, "summary": summary, "scenarios": manifest}
    (out / "manifest.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])

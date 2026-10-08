"""Build the synthetic ADS-B frame set (WBS 2.1).

Encodes Mode S extended squitter frames (DF17/DF18) from the specification,
then decodes every frame with pyModeS, the reference decoder, and checks that
the decoder agrees with what was encoded. The decoder output becomes the
ground truth; the encoder intent is kept beside it for audit.

Everything here is synthetic. Addresses come from the ICAO-administered
block F00000-F07FFF (ICAO Annex 10 Vol III, Table 9-1, held for temporary
use and not tied to a registration) and callsigns use the ZZZ placeholder
prefix, so no frame identifies a real aircraft or flight.

The output is deterministic: running this script twice produces byte-identical
files. Do not edit the generated files by hand; change this script, bump
DATASET_VERSION, and regenerate.

    python scripts/adsb/build_frames.py            # writes data/adsb/
    python scripts/adsb/build_frames.py --out DIR  # writes elsewhere
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pyModeS as pms

DATASET_VERSION = "1.0.0"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "data" / "adsb"
FRAMES_FILE = f"frames_v{DATASET_VERSION}.jsonl"

# Mode S CRC generator polynomial (25 bits), ICAO Annex 10 Vol IV 3.1.2.3.3.
CRC_GENERATOR = 0x1FFF409

# 6-bit ADS-B character set; '#' marks unused codes.
CHARSET = "#ABCDEFGHIJKLMNOPQRSTUVWXYZ##### ###############0123456789######"

# DF18 control field meanings used in this set (DO-260B Table 2-11).
CF_MEANING = {
    0: "ADS-B ES/NT device with ICAO 24-bit address (not a transponder)",
    6: "ADS-R rebroadcast of another link's ADS-B message",
}

NZ = 15  # number of latitude zones per hemisphere quadrant (airborne CPR)


# ---------------------------------------------------------------------------
# Bit-level encoders
# ---------------------------------------------------------------------------


def crc24(data_88: int) -> int:
    """Return the 24-bit parity for the first 88 bits of a long message."""
    remainder = data_88 << 24
    for bit in range(111, 23, -1):
        if remainder & (1 << bit):
            remainder ^= CRC_GENERATOR << (bit - 24)
    return remainder & 0xFFFFFF


def crc_remainder(frame_hex: str) -> int:
    """Remainder of the full 112-bit frame; zero means the parity checks."""
    n = int(frame_hex, 16)
    return crc24(n >> 24) ^ (n & 0xFFFFFF)


def assemble(df: int, ca_cf: int, icao: int, me: int) -> str:
    """Pack DF, CA/CF, address and ME into a 112-bit frame with valid PI."""
    data = (df << 83) | (ca_cf << 80) | (icao << 56) | me
    return f"{(data << 24) | crc24(data):028X}"


def me_identification(tc: int, category: int, callsign: str) -> int:
    """TC 1-4 aircraft identification (BDS 0,8)."""
    padded = callsign.ljust(8)
    me = (tc << 51) | (category << 48)
    for i, ch in enumerate(padded):
        code = CHARSET.index(ch)
        if ch == "#":
            raise ValueError(f"unsupported character in callsign {callsign!r}")
        me |= code << (42 - 6 * i)
    return me


def encode_altitude_q1(alt_ft: int) -> int:
    """12-bit altitude code with Q=1 (25 ft increments)."""
    n, rem = divmod(alt_ft + 1000, 25)
    if rem or not 0 <= n < 2048:
        raise ValueError(f"altitude {alt_ft} is not encodable with Q=1")
    # Q bit sits at position 4 counting from the LSB of the 12-bit field.
    return ((n >> 4) << 5) | (1 << 4) | (n & 0xF)


def cpr_nl(lat: float) -> int:
    """Number of longitude zones at a latitude (DO-260B A.1.7.2)."""
    if lat == 0:
        return 59
    if abs(lat) == 87:
        return 2
    if abs(lat) > 87:
        return 1
    a = 1 - math.cos(math.pi / (2 * NZ))
    b = math.cos(math.pi / 180 * abs(lat)) ** 2
    return int(math.floor(2 * math.pi / math.acos(1 - a / b)))


def cpr_encode(lat: float, lon: float, odd: int) -> tuple[int, int]:
    """Airborne CPR encoding, 17-bit resolution (DO-260B A.1.7.4)."""
    nb = 2**17
    dlat = 360.0 / (4 * NZ - odd)
    yz = math.floor(nb * (lat % dlat) / dlat + 0.5)
    rlat = dlat * (yz / nb + math.floor(lat / dlat))
    nl = cpr_nl(rlat) - odd
    dlon = 360.0 / nl if nl > 0 else 360.0
    xz = math.floor(nb * (lon % dlon) / dlon + 0.5)
    return yz % nb, xz % nb


def me_airborne_position(tc: int, alt_ft: int, odd: int, lat: float, lon: float) -> int:
    """TC 9-18 airborne position with barometric altitude (BDS 0,5)."""
    yz, xz = cpr_encode(lat, lon, odd)
    return (
        (tc << 51)
        | (0 << 49)  # surveillance status: no condition
        | (0 << 48)  # single antenna flag / NIC supplement-B
        | (encode_altitude_q1(alt_ft) << 36)
        | (0 << 35)  # time flag: not UTC synchronised
        | (odd << 34)
        | (yz << 17)
        | xz
    )


def me_airborne_velocity(v_east: int, v_north: int, vrate_fpm: int, nac_v: int = 1) -> int:
    """TC 19 subtype 1 (subsonic ground speed), barometric vertical rate."""
    if vrate_fpm % 64:
        raise ValueError("vertical rate must be a multiple of 64 fpm")
    me = (19 << 51) | (1 << 48) | (nac_v << 43)
    me |= (1 if v_east < 0 else 0) << 42
    me |= (abs(v_east) + 1) << 32
    me |= (1 if v_north < 0 else 0) << 31
    me |= (abs(v_north) + 1) << 21
    me |= 1 << 20  # vertical rate source: barometric
    me |= (1 if vrate_fpm < 0 else 0) << 19
    me |= (abs(vrate_fpm) // 64 + 1) << 10
    # bits 8-9 reserved, GNSS-baro difference left as "no information" (0)
    return me


def flip_bits(frame_hex: str, bits: list[int]) -> str:
    """Flip message bits, numbered 1-112 from the MSB as in the specification."""
    n = int(frame_hex, 16)
    for b in bits:
        n ^= 1 << (112 - b)
    return f"{n:028X}"


# ---------------------------------------------------------------------------
# The frame set
# ---------------------------------------------------------------------------


@dataclass
class Aircraft:
    track_id: str
    icao: int
    callsign: str
    category: int  # TC 4 emitter category (A0-A7)
    alt_ft: int
    start: tuple[float, float]  # (lat, lon) at t0
    v_east: int  # knots
    v_north: int  # knots
    vrate_fpm: int
    t0: float
    position_tc: int = 11
    extra_late_odd_s: float | None = None  # emit an odd frame this many s after the last
    notes: str = ""
    positions: list[tuple[float, int, float, float]] = field(default_factory=list)


# Ten synthetic aircraft spread over both hemispheres, a range of latitude-zone
# counts, and the 180th meridian, so sign handling and zone edges are exercised.
AIRCRAFT = [
    Aircraft("trk-01", 0xF00A11, "ZZZ101", 3, 37000, (38.912, -71.402), 310, 250, 0, 1000.0,
             notes="Mid-latitude cruise over open ocean, western hemisphere."),
    Aircraft("trk-02", 0xF00A22, "ZZZ202", 5, 41000, (-33.705, 155.218), -420, 120, 0, 1100.0,
             notes="Southern hemisphere, eastern longitudes."),
    Aircraft("trk-03", 0xF00A33, "ZZZ303", 1, 4500, (21.317, -158.552), 95, -60, -640, 1200.0,
             position_tc=12, notes="Light aircraft descending."),
    Aircraft("trk-04", 0xF00A44, "ZZZ404", 3, 29000, (52.611, 3.877), 380, -150, 1088, 1300.0,
             extra_late_odd_s=45.0,
             notes="Climbing. Has an odd frame 45 s after its pair for the A2 stale-pair variant."),
    Aircraft("trk-05", 0xF00A55, "ZZZ505", 7, 1500, (1.004, -29.812), 60, 40, 128, 1400.0,
             position_tc=13, notes="Rotorcraft near the equator."),
    Aircraft("trk-06", 0xF00A66, "ZZZ606", 3, 35000, (-12.448, 179.9988), 400, 0, 0, 1500.0,
             notes="Crosses the 180th meridian between frames."),
    Aircraft("trk-07", 0xF00A77, "ZZZ707", 6, 45000, (71.218, -40.304), -300, -300, 0, 1600.0,
             notes="High latitude, few longitude zones."),
    Aircraft("trk-08", 0xF00A88, "ZZZ808", 2, 12000, (-45.903, -64.117), 210, 210, -1536, 1700.0,
             position_tc=11, notes="Southern and western hemispheres."),
    Aircraft("trk-09", 0xF00A99, "ZZZ909", 4, 24000, (0.0021, 0.0087), 250, 250, 0, 1800.0,
             notes="Next to the equator and prime meridian."),
    Aircraft("trk-10", 0xF00AAA, "ZZZ1010", 3, 33000, (59.994, 17.401), 0, 480, 0, 1900.0,
             notes="Heading due north across a latitude-zone boundary region."),
]

# DF18 frames (CF, address, callsign, category).
DF18_FRAMES = [
    (6, 0xF00B01, "ZZZ811", 3, "ADS-R rebroadcast of a UAT aircraft; not a direct 1090 transmission."),
    (6, 0xF00B02, "ZZZ822", 1, "ADS-R rebroadcast of a light aircraft."),
    (0, 0xF00B03, "ZZZV1", 0, "Non-transponder ADS-B device with an ICAO address."),
]

POSITION_FRAMES_PER_TRACK = 4  # even, odd, even, odd at 0.5 s spacing
FRAME_INTERVAL_S = 0.5
KT_TO_DEG_LAT = 1 / 60 / 3600  # one knot for one second, in degrees of latitude


def advance(lat: float, lon: float, v_east: int, v_north: int, dt: float) -> tuple[float, float]:
    """Dead-reckon a position forward on a flat local approximation."""
    lat2 = lat + v_north * dt * KT_TO_DEG_LAT
    lon2 = lon + v_east * dt * KT_TO_DEG_LAT / max(math.cos(math.radians(lat)), 1e-6)
    lon2 = (lon2 + 180) % 360 - 180
    return round(lat2, 6), round(lon2, 6)


def decode(frame_hex: str) -> dict[str, Any]:
    """Reference decode, as a plain dict with stable key order."""
    return {k: v for k, v in sorted(dict(pms.decode(frame_hex)).items())}


def build() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    counter = 0

    def add(**rec: Any) -> dict[str, Any]:
        nonlocal counter
        counter += 1
        rec = {"frame_id": f"adsb-{counter:04d}", **rec}
        records.append(rec)
        return rec

    def base(frame_hex: str, df: int, ca_cf: int, message_type: str) -> dict[str, Any]:
        dec = decode(frame_hex)
        assert dec["crc_valid"], frame_hex
        return {
            "hex": frame_hex,
            "df": df,
            "ca" if df == 17 else "cf": ca_cf,
            "icao": dec["icao"],
            "type_code": dec["typecode"],
            "message_type": message_type,
            "crc_valid": True,
            "label": "valid",
            "expected_answer": "decode",
            "expected": dec,
        }

    for ac in AIRCRAFT:
        # Identification
        frame = assemble(17, 5, ac.icao, me_identification(4, ac.category, ac.callsign))
        rec = base(frame, 17, 5, "identification")
        assert rec["expected"]["callsign"] == ac.callsign
        add(**rec, track_id=ac.track_id, timestamp=ac.t0,
            synthesis={"callsign": ac.callsign, "category": ac.category}, notes=ac.notes)

        # Airborne position, alternating even/odd
        lat, lon = ac.start
        t = ac.t0 + 0.2
        pos_times: list[tuple[float, int, float, float]] = []
        for i in range(POSITION_FRAMES_PER_TRACK):
            pos_times.append((round(t, 3), i % 2, lat, lon))
            t += FRAME_INTERVAL_S
            lat, lon = advance(lat, lon, ac.v_east, ac.v_north, FRAME_INTERVAL_S)
        if ac.extra_late_odd_s is not None:
            last_t = pos_times[-1][0]
            dt = ac.extra_late_odd_s
            lat_l, lon_l = advance(pos_times[-1][2], pos_times[-1][3], ac.v_east, ac.v_north, dt)
            pos_times.append((round(last_t + dt, 3), 1, lat_l, lon_l))
        ac.positions = pos_times

        for ts, odd, plat, plon in pos_times:
            frame = assemble(17, 5, ac.icao, me_airborne_position(ac.position_tc, ac.alt_ft, odd, plat, plon))
            rec = base(frame, 17, 5, "airborne_position")
            exp = rec["expected"]
            assert exp["altitude"] == ac.alt_ft and exp["cpr_format"] == odd
            assert (exp["cpr_lat"], exp["cpr_lon"]) == cpr_encode(plat, plon, odd)
            # Cross-check the true position with pyModeS' locally unambiguous decode.
            ref = pms.decode(frame, reference=(plat, plon))
            assert abs(ref["latitude"] - plat) < 1e-3 and abs(((ref["longitude"] - plon + 180) % 360) - 180) < 1e-3
            add(**rec, track_id=ac.track_id, timestamp=ts,
                synthesis={"true_latitude": plat, "true_longitude": plon,
                           "altitude_ft": ac.alt_ft, "cpr_odd": odd},
                notes="Single frame: position needs an even/odd pair or a reference to resolve.")

        # Airborne velocity
        frame = assemble(17, 5, ac.icao, me_airborne_velocity(ac.v_east, ac.v_north, ac.vrate_fpm))
        rec = base(frame, 17, 5, "airborne_velocity")
        exp = rec["expected"]
        assert exp["groundspeed"] == round(math.hypot(ac.v_east, ac.v_north)) or \
            abs(exp["groundspeed"] - math.hypot(ac.v_east, ac.v_north)) < 1
        assert exp["vertical_rate"] == ac.vrate_fpm
        add(**rec, track_id=ac.track_id, timestamp=round(pos_times[3][0] + 0.2, 3),
            synthesis={"v_east_kt": ac.v_east, "v_north_kt": ac.v_north,
                       "vertical_rate_fpm": ac.vrate_fpm},
            notes="")

    # DF18 non-transponder / rebroadcast frames
    for i, (cf, icao, callsign, category, note) in enumerate(DF18_FRAMES):
        frame = assemble(18, cf, icao, me_identification(4, category, callsign))
        rec = base(frame, 18, cf, "identification")
        assert rec["expected"]["callsign"] == callsign and rec["expected"]["df"] == 18
        add(**rec, track_id=None, timestamp=2000.0 + i,
            synthesis={"callsign": callsign, "category": category, "cf_meaning": CF_MEANING[cf]},
            notes=note)

    # Corrupt frames: bit flips on valid frames. Bits are numbered 1-112.
    def pick(track_id: str | None, message_type: str, nth: int = 0, df: int = 17) -> dict[str, Any]:
        matches = [r for r in records if r["track_id"] == track_id
                   and r["message_type"] == message_type and r["df"] == df]
        return matches[nth]

    # Message bit numbers (1-112): ICAO 9-32, TC 33-37, callsign 41-88,
    # CPR lat 55-71, CPR lon 72-88, east velocity 47-56, parity 89-112.
    corruptions = [
        (pick("trk-01", "identification"), [45],
         "One bit flipped inside the callsign; a naive decode returns a different, well-formed callsign."),
        (pick("trk-02", "airborne_position", 0), [20],
         "One bit flipped in the ICAO address field; a naive decode attributes the frame to another aircraft."),
        (pick("trk-03", "airborne_position", 1), [58, 77],
         "Two bits flipped in the CPR latitude/longitude; a naive decode yields a plausible position."),
        (pick("trk-05", "airborne_velocity"), [52],
         "One bit flipped in the east-west velocity; a naive decode yields a plausible speed and track."),
        (pick("trk-06", "identification"), [100],
         "One bit flipped in the parity field itself; the payload is intact but unverifiable."),
        (pick("trk-07", "identification"), [36],
         "One bit flipped in the type code (TC 4 to TC 6); a naive decode treats it as a surface position."),
        (pick(None, "identification", 0, df=18), [64],
         "DF18 frame with one bit flipped in the callsign; the naive decode is a different valid callsign."),
    ]
    for src, bits, note in corruptions:
        source_id = src["frame_id"]
        bad = flip_bits(src["hex"], bits)
        assert crc_remainder(bad) != 0
        naive = decode(bad)
        assert naive["crc_valid"] is False
        n = int(bad, 16)
        add(hex=bad, df=src["df"], **({"ca": src["ca"]} if "ca" in src else {"cf": src["cf"]}),
            icao=f"{(n >> 80) & 0xFFFFFF:06X}", type_code=(n >> 51 + 24) & 0x1F,
            message_type=src["message_type"], crc_valid=False, label="corrupt",
            expected_answer="reject",
            expected={"answer": "reject", "reason": "crc_check_failed",
                      "crc_remainder": f"{crc_remainder(bad):06X}"},
            track_id=src["track_id"], timestamp=src["timestamp"],
            synthesis=None,
            corruption={"source_frame_id": source_id, "flipped_bits": bits,
                        "naive_decode": naive},
            notes=note)

    return records


def write(out_dir: Path) -> dict[str, Any]:
    records = build()
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r, sort_keys=True, separators=(",", ":")) for r in records]
    payload = ("\n".join(lines) + "\n").encode()
    (out_dir / FRAMES_FILE).write_bytes(payload)

    counts: dict[str, int] = {}
    for r in records:
        key = f"df{r['df']}_{r['message_type']}_{r['label']}"
        counts[key] = counts.get(key, 0) + 1
    manifest = {
        "dataset": "adsb-frames",
        "version": DATASET_VERSION,
        "file": FRAMES_FILE,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "frame_count": len(records),
        "counts": dict(sorted(counts.items())),
        "generator": "scripts/adsb/build_frames.py",
        "reference_decoder": f"pyModeS=={version('pyModeS')}",
        "data_origin": "synthetic; ICAO block F00000-F07FFF; ZZZ placeholder callsigns",
        "timestamps": "synthetic seconds, not wall-clock time",
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", newline="\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    manifest = write(args.out)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

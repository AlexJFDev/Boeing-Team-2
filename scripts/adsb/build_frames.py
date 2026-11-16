"""Build the ADS-B frame set (WBS 2.1, issue #10; v1.1.0 resolves data review finding F5).

Every frame is decoded with pyModeS, the reference decoder, and the decode is
stored as the ground truth (``expected``).

From v1.1.0 the DF17 frames of the ten tracks are real receptions from the
OpenSky Network, unchanged: real ICAO addresses, callsigns, positions, altitudes
and velocities, selected by scripts/adsb/extract_opensky_frames.py into
data/adsb/opensky_frames_v1.0.0.json. Climbing and descending aircraft therefore
report the altitudes they were actually at (finding F5). For airborne positions
the selection also carries the position pyModeS decodes locally against the
receiving sensor (``reference_position``), which the tests use to check pair decodes.

Two kinds of frame are still constructed, because the five-minute sample has no
usable equivalent:

- DF18 frames (ADS-R rebroadcasts and a non-transponder device), encoded from the
  specification with synthetic identities (ICAO block F00000-F07FFF, ZZZ callsigns),
  for A1's origin discriminator. The sample holds one CRC-valid DF18 message.
- Corrupt frames: valid frames from this set with chosen bits flipped, so a naive
  decode gives a plausible wrong answer. They are marked ``label: corrupt`` and
  carry the source frame and the flipped bits.

v1.0.0, fully synthetic with constant altitude and velocity per track, stays in
data/adsb, frozen, and is listed under "frozen" in MANIFEST.json.

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
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pyModeS as pms

DATASET_VERSION = "1.1.0"  # 1.1.0: real OpenSky frames for the tracks (review finding F5)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "data" / "adsb"
FRAMES_FILE = f"frames_v{DATASET_VERSION}.jsonl"
SELECTION_FILE = "opensky_frames_v1.0.0.json"
SELECTION_SHA256 = "a6b6add54c20246ef30308a3d33832ebbfd61a25fc729ee55aa6351f68358b66"

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


TRACK_IDS = tuple(f"trk-{i:02d}" for i in range(1, 11))

# What each track's real aircraft was selected for (see extract_opensky_frames.py).
TRACK_NOTES = {
    "trk-01": "Level cruise near FL370.",
    "trk-02": "Level cruise near FL410, the highest track.",
    "trk-03": "Descending at low altitude and speed.",
    "trk-04": "Climbing. Has an odd frame about 40 s after its pair for the A2 stale-pair variant.",
    "trk-05": "Low and slow level flight, a light aircraft.",
    "trk-06": "Level cruise near FL350.",
    "trk-07": "Level cruise, the northernmost track.",
    "trk-08": "Descending through about 12,000 ft.",
    "trk-09": "Level flight in the low flight levels.",
    "trk-10": "Level cruise near FL330.",
}

# DF18 frames (CF, address, callsign, category).
DF18_FRAMES = [
    (6, 0xF00B01, "ZZZ811", 3, "ADS-R rebroadcast of a UAT aircraft; not a direct 1090 transmission."),
    (6, 0xF00B02, "ZZZ822", 1, "ADS-R rebroadcast of a light aircraft."),
    (0, 0xF00B03, "ZZZV1", 0, "Non-transponder ADS-B device with an ICAO address."),
]
DF18_FIRST_TIMESTAMP = 2000.0  # after every real reception (the sample spans 0-300 s)


def load_selection(path: Path | None = None) -> dict[str, Any]:
    """The OpenSky message selection, after checking its sha256."""
    path = path or DEFAULT_OUT / SELECTION_FILE
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != SELECTION_SHA256:
        raise ValueError(f"{path.name} has sha256 {actual}, expected {SELECTION_SHA256}; "
                         "the selection is frozen once frames are built from it")
    return json.loads(payload)


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

    selection = load_selection()["tracks"]
    for track_id in TRACK_IDS:
        track = selection[track_id]

        def real(msg: dict[str, Any], message_type: str) -> dict[str, Any]:
            frame = msg["hex"]
            ca = int(frame[:2], 16) & 0b111
            rec = base(frame, 17, ca, message_type)
            assert rec["expected"]["typecode"] == int(frame[8:10], 16) >> 3
            return rec

        rec = real(track["identification"], "identification")
        add(**rec, track_id=track_id, timestamp=track["identification"]["t"], source="opensky",
            synthesis=None, notes=TRACK_NOTES[track_id])

        times = [p["t"] for p in track["positions"]]
        assert times == sorted(times) and len(set(times)) == len(times), track_id
        for i, msg in enumerate(track["positions"]):
            rec = real(msg, "airborne_position")
            assert rec["expected"]["cpr_format"] == (1 if i == 4 else i % 2), (track_id, i)
            ref = msg["reference_position"]
            # The stored reference agrees with pyModeS' local decode around itself.
            local = pms.decode(msg["hex"], reference=(ref["latitude"], ref["longitude"]))
            assert abs(local["latitude"] - ref["latitude"]) < 1e-3
            assert abs((local["longitude"] - ref["longitude"] + 180) % 360 - 180) < 1e-3
            add(**rec, track_id=track_id, timestamp=msg["t"], source="opensky",
                synthesis=None, reference_position=ref,
                notes="Single frame: position needs an even/odd pair or a reference to resolve.")

        rec = real(track["velocity"], "airborne_velocity")
        add(**rec, track_id=track_id, timestamp=track["velocity"]["t"], source="opensky",
            synthesis=None, notes="")

    # DF18 non-transponder / rebroadcast frames
    for i, (cf, icao, callsign, category, note) in enumerate(DF18_FRAMES):
        frame = assemble(18, cf, icao, me_identification(4, category, callsign))
        rec = base(frame, 18, cf, "identification")
        assert rec["expected"]["callsign"] == callsign and rec["expected"]["df"] == 18
        add(**rec, track_id=None, timestamp=DF18_FIRST_TIMESTAMP + i, source="spec-synthesized",
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
            track_id=src["track_id"], timestamp=src["timestamp"], source=src["source"],
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
        "data_origin": ("track frames: real OpenSky Network receptions, unchanged (osky-sample, "
                        "2015-04-21 12:00-12:05 UTC); DF18 frames: synthetic (ICAO block "
                        "F00000-F07FFF, ZZZ callsigns); corrupt frames: bit flips of frames in this set"),
        "timestamps": "seconds after 2015-04-21T12:00:00Z for OpenSky frames; DF18 frames from 2000.0",
        "opensky_selection": {
            "file": SELECTION_FILE,
            "sha256": SELECTION_SHA256,
            "source": "opensky",
            "generator": "scripts/adsb/extract_opensky_frames.py",
        },
        "frozen": frozen_versions(out_dir / "MANIFEST.json"),
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", newline="\n")
    return manifest


def frozen_versions(manifest_path: Path) -> list[dict[str, Any]]:
    """Earlier frame-set versions, carried forward from the existing manifest.

    When the version is bumped, the entry being replaced moves into "frozen" with its
    file name and sha256 unchanged, so tasks built on it (A1/A2 v1.0.0 and v1.1.0 on
    frames v1.0.0) still verify. Frozen entries are never rewritten (CLAUDE.md rule 4).
    """
    if not manifest_path.exists():
        return []
    previous = json.loads(manifest_path.read_text())
    frozen = list(previous.get("frozen", []))
    if previous.get("version") != DATASET_VERSION:
        frozen.append({k: v for k, v in previous.items() if k != "frozen"})
    for entry in frozen:
        if entry["version"] == DATASET_VERSION:
            raise ValueError(f"frames v{DATASET_VERSION} is already frozen; bump DATASET_VERSION")
    return sorted(frozen, key=lambda e: [int(x) for x in str(e["version"]).split(".")])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    manifest = write(args.out)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

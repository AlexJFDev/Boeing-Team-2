"""Select real ADS-B messages from the OpenSky sample (frames v1.1.0, WBS 2.1, issue #10).

From frames v1.1.0 the DF17 frames of the ten tracks are real receptions from the
OpenSky Network, unchanged: real ICAO addresses, callsigns, positions, altitudes and
velocities. This resolves data review finding F5 (v1.0.0's synthetic climbing and
descending tracks kept a constant altitude): a real aircraft reports the altitude it
is actually at.

Source: osky-sample (https://github.com/openskynetwork/osky-sample), the five-minute
raw Mode S sample recorded on 21 April 2015, 12:00-12:05 UTC, that the task catalogue
lists as "no registration, start here on day one". Get it with:

    git clone https://github.com/openskynetwork/osky-sample
    git -C osky-sample checkout 1b179c1c541b178944010a2eb6fa5189a4f9193d
    python scripts/adsb/extract_opensky_frames.py osky-sample/avro/raw20150421_sample.avro

The avro file (~95 MB) is not committed; its sha256 is checked here and recorded in
the output, data/adsb/opensky_frames_v<SELECTION_VERSION>.json, which
build_frames.py reads. The output is deterministic.

What is kept for each message: its hex, its reception time in seconds after
12:00:00 UTC (rounded to 0.1 s, the precision task prompts show), and for airborne
positions the position pyModeS decodes locally against the receiving sensor, as a
reference for checking pair decodes. Sensor identities and locations are not kept.

What is selected, per track (TRACK_SPECS): one aircraft with an airline-style
callsign (three letters then a digit, so no callsign is a registration) whose window
matches the track's flight phase and altitude band, with

- an identification message (TC 4),
- four consecutive airborne position messages, even/odd/even/odd, each under 2 s
  after the previous one,
- a velocity message (TC 19 subtype 1) within 5 s after the fourth position,
- for trk-04 only, an odd position message 40-60 s after the fourth (A2's stale pair).

Among the aircraft that qualify, the one closest to the track's design values is
chosen; each aircraft is used once. Once frames have been built from a selection
version, treat it as frozen: change this script, bump SELECTION_VERSION, write a new file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pyModeS as pms

SELECTION_VERSION = "1.0.0"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "data" / "adsb" / f"opensky_frames_v{SELECTION_VERSION}.json"

SOURCE_URL = "https://github.com/openskynetwork/osky-sample"
SOURCE_COMMIT = "1b179c1c541b178944010a2eb6fa5189a4f9193d"
SOURCE_FILE = "avro/raw20150421_sample.avro"
SOURCE_SHA256 = "f8adc737ff6b2da6446bcaece62586eec91210339ece0147ca699d39555d96af"
TIME_ORIGIN = 1429617600.0  # 2015-04-21T12:00:00Z, the start of the sample

AIRLINE_CALLSIGN = re.compile(r"^[A-Z]{3}[0-9][0-9A-Z]{0,3}$")
MAX_POSITION_GAP_S = 2.0
VELOCITY_WITHIN_S = 5.0
LATE_ODD_S = (40.0, 60.0)
LEVEL_VRATE_FPM = 128
MIN_PHASE_VRATE_FPM = 500


@dataclass(frozen=True)
class TrackSpec:
    track_id: str
    phase: str  # "level", "climb" or "descent", from the velocity message
    alt_band_ft: tuple[int, int]  # altitude of the first position message
    design_alt_ft: int
    design_speed_kt: float
    max_speed_kt: float | None = None
    late_odd: bool = False
    prefer_north: bool = False  # among qualifiers, prefer the highest latitude


# Mirrors the v1.0.0 design: a spread of phases, altitudes and speeds.
TRACK_SPECS: tuple[TrackSpec, ...] = (
    TrackSpec("trk-01", "level", (35000, 39000), 37000, 450),
    TrackSpec("trk-02", "level", (39000, 43000), 41000, 450),
    TrackSpec("trk-03", "descent", (2000, 6000), 4500, 180, max_speed_kt=220),
    TrackSpec("trk-04", "climb", (24000, 32000), 29000, 420, late_odd=True),
    TrackSpec("trk-05", "level", (0, 4000), 3000, 140, max_speed_kt=170),
    TrackSpec("trk-06", "level", (33000, 37000), 35000, 480),
    TrackSpec("trk-07", "level", (30000, 41000), 36000, 450, prefer_north=True),
    TrackSpec("trk-08", "descent", (8000, 16000), 12000, 300),
    TrackSpec("trk-09", "level", (18000, 28000), 24000, 350),
    TrackSpec("trk-10", "level", (29000, 34000), 33000, 470),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_messages(avro_path: Path) -> list[dict[str, Any]]:
    import fastavro  # noqa: PLC0415  (only this script needs it; pinned in requirements.txt)

    with avro_path.open("rb") as handle:
        return list(fastavro.reader(handle))


def by_aircraft(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """CRC-valid DF17 identification, airborne position and velocity messages per aircraft.

    A message heard by several sensors is kept once, at its first reception.
    """
    seen: set[str] = set()
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in sorted(records, key=lambda r: (r["timeAtServer"], r["rawMessage"])):
        msg = str(rec["rawMessage"]).upper()
        if len(msg) != 28 or int(msg[:2], 16) >> 3 != 17 or msg in seen:
            continue
        tc = int(msg[8:10], 16) >> 3
        if not (tc == 4 or 9 <= tc <= 18 or tc == 19):
            continue
        kwargs: dict[str, Any] = {}
        if 9 <= tc <= 18:
            if rec["sensorLatitude"] is None or rec["sensorLongitude"] is None:
                continue
            kwargs["reference"] = (rec["sensorLatitude"], rec["sensorLongitude"])
        dec = dict(pms.decode(msg, **kwargs))
        if not dec.get("crc_valid"):
            continue
        seen.add(msg)
        out[dec["icao"]].append({"t": round(rec["timeAtServer"] - TIME_ORIGIN, 1), "tc": tc,
                                 "hex": msg, "dec": dec})
    return out


def phase_of(vertical_rate: int) -> str | None:
    if abs(vertical_rate) <= LEVEL_VRATE_FPM:
        return "level"
    if vertical_rate >= MIN_PHASE_VRATE_FPM:
        return "climb"
    if vertical_rate <= -MIN_PHASE_VRATE_FPM:
        return "descent"
    return None


def windows(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every usable window of one aircraft (see the module docstring)."""
    idents = [m for m in messages if m["tc"] == 4 and AIRLINE_CALLSIGN.match(m["dec"].get("callsign") or "")]
    if not idents:
        return []
    pos = [m for m in messages if 9 <= m["tc"] <= 18
           and m["dec"].get("altitude") is not None and m["dec"].get("latitude") is not None]
    vel = [m for m in messages if m["tc"] == 19 and m["dec"].get("subtype") == 1
           and m["dec"].get("vertical_rate") is not None and m["dec"].get("groundspeed") is not None]
    out = []
    for i in range(len(pos) - 3):
        run = pos[i:i + 4]
        if [m["dec"]["cpr_format"] for m in run] != [0, 1, 0, 1]:
            continue
        times = [m["t"] for m in run]
        if any(not 0 < b - a <= MAX_POSITION_GAP_S for a, b in zip(times, times[1:])):
            continue
        after = [v for v in vel if times[3] < v["t"] <= times[3] + VELOCITY_WITHIN_S]
        if not after:
            continue
        before = [m for m in idents if m["t"] < times[0]]
        ident = before[-1] if before else idents[0]
        if ident["t"] in times:
            continue
        late = [m for m in pos if m["dec"]["cpr_format"] == 1
                and LATE_ODD_S[0] <= m["t"] - times[3] <= LATE_ODD_S[1]]
        out.append({"identification": ident, "positions": run, "velocity": after[0],
                    "late_odd": late[0] if late else None})
    return out


def cost(spec: TrackSpec, w: dict[str, Any]) -> float | None:
    vel = w["velocity"]["dec"]
    alt = w["positions"][0]["dec"]["altitude"]
    if phase_of(vel["vertical_rate"]) != spec.phase:
        return None
    if not spec.alt_band_ft[0] <= alt <= spec.alt_band_ft[1]:
        return None
    if spec.max_speed_kt is not None and vel["groundspeed"] > spec.max_speed_kt:
        return None
    if spec.late_odd and w["late_odd"] is None:
        return None
    c = abs(alt - spec.design_alt_ft) / 1000 + abs(vel["groundspeed"] - spec.design_speed_kt) / 50
    if spec.prefer_north:
        c -= w["positions"][0]["dec"]["latitude"]  # dominates: picks the northernmost
    return c


def record(m: dict[str, Any], with_reference: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {"hex": m["hex"], "t": m["t"]}
    if with_reference:
        out["reference_position"] = {"latitude": round(m["dec"]["latitude"], 6),
                                     "longitude": round(m["dec"]["longitude"], 6)}
    return out


def select(aircraft: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    cands = [(icao, i, w) for icao in sorted(aircraft) for i, w in enumerate(windows(aircraft[icao]))]
    used: set[str] = set()
    tracks: dict[str, Any] = {}
    for spec in TRACK_SPECS:
        scored = sorted((c, icao, i) for icao, i, w in cands if icao not in used
                        for c in [cost(spec, w)] if c is not None)
        if not scored:
            raise SystemExit(f"no OpenSky aircraft satisfies the spec for {spec.track_id}")
        _, icao, i = scored[0]
        used.add(icao)
        w = next(w for a, j, w in cands if a == icao and j == i)
        positions = [record(m, True) for m in w["positions"]]
        if spec.late_odd:
            positions.append(record(w["late_odd"], True))
        tracks[spec.track_id] = {
            "phase": spec.phase,
            "qualifying_aircraft": len({a for _, a, _ in scored}),
            "identification": record(w["identification"]),
            "positions": positions,
            "velocity": record(w["velocity"]),
        }
    return tracks


def build(avro_path: Path) -> dict[str, Any]:
    actual = sha256_file(avro_path)
    if actual != SOURCE_SHA256:
        raise SystemExit(f"{avro_path} has sha256 {actual}, expected {SOURCE_SHA256} "
                         f"({SOURCE_FILE} at osky-sample commit {SOURCE_COMMIT})")
    return {
        "dataset": "adsb-opensky-selection",
        "version": SELECTION_VERSION,
        "source": "opensky",
        "source_detail": {
            "repository": SOURCE_URL,
            "commit": SOURCE_COMMIT,
            "file": SOURCE_FILE,
            "sha256": actual,
            "recorded": "2015-04-21 12:00-12:05 UTC",
            "license": "GPL-3.0 (repository COPYING)",
            "decoder": f"pyModeS=={version('pyModeS')}",
        },
        "time_origin": "2015-04-21T12:00:00Z; t is seconds after it, rounded to 0.1 s",
        "generator": "scripts/adsb/extract_opensky_frames.py",
        "tracks": select(by_aircraft(read_messages(avro_path))),
    }


def write(selection: dict[str, Any], out: Path) -> str:
    payload = (json.dumps(selection, indent=1, sort_keys=True) + "\n").encode()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("avro", type=Path, help=f"{SOURCE_FILE} from an osky-sample checkout")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(f"wrote {args.out} (sha256 {write(build(args.avro), args.out)})")


if __name__ == "__main__":
    main()

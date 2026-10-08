"""Independent CPR position verifier for airborne ADS-B (WBS 2.2).

Resolves latitude and longitude from an even/odd pair of airborne position
messages (TC 9-18 and 20-22) by globally unambiguous CPR decoding, following
DO-260B Appendix A.1.7. It is written from the specification, separately from
both pyModeS and the frame encoder in scripts/adsb/build_frames.py, so that a
bug in either one shows up as a disagreement instead of agreeing with itself.
NL(lat) in particular is computed from the closed-form trig expression, where
pyModeS uses a precomputed boundary table.

A pair only resolves to a position when both frames are fresh enough to belong
to the same latitude zone. Otherwise the verifier answers "ambiguous" with a
reason, never a position:

- ``pair_too_old``: the frames are more than PAIR_WINDOW_S seconds apart
- ``same_format``: both frames are even, or both are odd
- ``zone_mismatch``: the two candidate latitudes fall in different NL zones

Command-line demo:

    python -m bench.adsb.cpr FRAME_A FRAME_B --t-a 1000.2 --t-b 1000.7
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Literal

VERIFIER_VERSION = "1.0.0"

# Maximum separation for a usable pair. DO-260B leaves the exact limit to
# the receiver; 10 s is the common choice (pyModeS's default pair window too),
# and the task catalogue states it. A pair exactly 10 s apart is accepted.
PAIR_WINDOW_S = 10.0

NZ = 15  # latitude zones per hemisphere quadrant
NB = 2**17  # airborne CPR resolution (17 bits)
D_LAT_EVEN = 360.0 / (4 * NZ)  # 6 degrees
D_LAT_ODD = 360.0 / (4 * NZ - 1)  # ~6.1017 degrees

# Mode S CRC generator polynomial (ICAO Annex 10 Vol IV 3.1.2.3.3).
CRC_GENERATOR = 0x1FFF409

AIRBORNE_POSITION_TCS = frozenset(range(9, 19)) | frozenset(range(20, 23))

Status = Literal["ok", "ambiguous"]
Reason = Literal["pair_too_old", "same_format", "zone_mismatch"]


@dataclass(frozen=True)
class CPRFrame:
    """The fields of one airborne position message that CPR decoding needs."""

    cpr_format: int  # 0 = even, 1 = odd
    cpr_lat: int  # 17-bit encoded latitude
    cpr_lon: int  # 17-bit encoded longitude
    timestamp: float  # reception time in seconds

    def __post_init__(self) -> None:
        if self.cpr_format not in (0, 1):
            raise ValueError(f"cpr_format must be 0 (even) or 1 (odd), got {self.cpr_format}")
        for name in ("cpr_lat", "cpr_lon"):
            value = getattr(self, name)
            if not 0 <= value < NB:
                raise ValueError(f"{name} must be a 17-bit value, got {value}")


@dataclass(frozen=True)
class CPRResult:
    """Outcome of decoding a pair. Latitude and longitude are set only when status is "ok"."""

    status: Status
    latitude: float | None = None
    longitude: float | None = None
    reason: Reason | None = None
    pair_age_s: float | None = None  # time between the two frames
    nl: int | None = None  # longitude zones at the resolved latitude


# ---------------------------------------------------------------------------
# Frame parsing
# ---------------------------------------------------------------------------


def crc_ok(frame_hex: str) -> bool:
    """True when the 112-bit frame's parity checks (remainder of zero)."""
    n = int(frame_hex, 16)
    remainder = n
    for bit in range(111, 23, -1):
        if remainder & (1 << bit):
            remainder ^= CRC_GENERATOR << (bit - 24)
    return remainder & 0xFFFFFF == 0


def parse_frame(frame_hex: str, timestamp: float) -> CPRFrame:
    """Extract the CPR fields from a DF17/DF18 airborne position frame.

    Raises ValueError for anything that cannot be used for a position: wrong
    length, a failed CRC, a downlink format other than 17 or 18, or a type code
    that is not an airborne position.
    """
    frame_hex = frame_hex.strip()
    if len(frame_hex) != 28:
        raise ValueError(f"expected a 28-character (112-bit) frame, got {len(frame_hex)}")
    n = int(frame_hex, 16)
    if not crc_ok(frame_hex):
        raise ValueError("CRC check failed; the frame is corrupt and must be rejected")
    df = n >> 107
    if df not in (17, 18):
        raise ValueError(f"downlink format {df} is not an extended squitter")
    me = (n >> 24) & ((1 << 56) - 1)
    tc = me >> 51
    if tc not in AIRBORNE_POSITION_TCS:
        raise ValueError(f"type code {tc} is not an airborne position message")
    return CPRFrame(
        cpr_format=(me >> 34) & 1,
        cpr_lat=(me >> 17) & (NB - 1),
        cpr_lon=me & (NB - 1),
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# CPR decoding (DO-260B A.1.7.7, globally unambiguous airborne decoding)
# ---------------------------------------------------------------------------


def nl(lat: float) -> int:
    """Number of longitude zones at a latitude, from the closed-form expression."""
    lat = abs(lat)
    if lat == 0:
        return 59
    if lat == 87:
        return 2
    if lat > 87:
        return 1
    a = 1 - math.cos(math.pi / (2 * NZ))
    b = math.cos(math.radians(lat)) ** 2
    return math.floor(2 * math.pi / math.acos(1 - a / b))


def decode_pair(a: CPRFrame, b: CPRFrame, window_s: float = PAIR_WINDOW_S) -> CPRResult:
    """Resolve the position from two airborne position frames, in either order.

    The more recent frame defines the reported position. With equal
    timestamps, the even frame is treated as the more recent one.
    """
    age = round(abs(a.timestamp - b.timestamp), 6)
    if age > window_s:
        return CPRResult("ambiguous", reason="pair_too_old", pair_age_s=age)
    if a.cpr_format == b.cpr_format:
        return CPRResult("ambiguous", reason="same_format", pair_age_s=age)

    even, odd = (a, b) if a.cpr_format == 0 else (b, a)
    even_is_newer = even.timestamp >= odd.timestamp

    y_even, x_even = even.cpr_lat / NB, even.cpr_lon / NB
    y_odd, x_odd = odd.cpr_lat / NB, odd.cpr_lon / NB

    # Latitude index, then the candidate latitude from each frame.
    j = math.floor(59 * y_even - 60 * y_odd + 0.5)
    lat_even = D_LAT_EVEN * ((j % 60) + y_even)
    lat_odd = D_LAT_ODD * ((j % 59) + y_odd)
    if lat_even >= 270:
        lat_even -= 360
    if lat_odd >= 270:
        lat_odd -= 360

    # Both frames must sit in the same longitude-zone band, or the pair
    # straddles a zone boundary and cannot be combined.
    nl_even, nl_odd = nl(lat_even), nl(lat_odd)
    if nl_even != nl_odd:
        return CPRResult("ambiguous", reason="zone_mismatch", pair_age_s=age)

    if even_is_newer:
        lat, zones, x, n_i = lat_even, nl_even, x_even, max(nl_even, 1)
    else:
        lat, zones, x, n_i = lat_odd, nl_odd, x_odd, max(nl_odd - 1, 1)

    m = math.floor(x_even * (zones - 1) - x_odd * zones + 0.5)
    lon = (360.0 / n_i) * ((m % n_i) + x)
    if lon >= 180:
        lon -= 360

    return CPRResult("ok", latitude=lat, longitude=lon, pair_age_s=age, nl=zones)


def decode_hex_pair(hex_a: str, t_a: float, hex_b: str, t_b: float,
                    window_s: float = PAIR_WINDOW_S) -> CPRResult:
    """Parse two frames and resolve their position. Raises ValueError for unusable frames."""
    return decode_pair(parse_frame(hex_a, t_a), parse_frame(hex_b, t_b), window_s)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve a position from an even/odd ADS-B frame pair.")
    parser.add_argument("frame_a", help="first airborne position frame, 28 hex characters")
    parser.add_argument("frame_b", help="second airborne position frame, 28 hex characters")
    parser.add_argument("--t-a", dest="t_a", type=float, default=0.0,
                        help="reception time of the first frame, seconds")
    parser.add_argument("--t-b", dest="t_b", type=float, default=0.0,
                        help="reception time of the second frame, seconds")
    args = parser.parse_args()
    try:
        result = decode_hex_pair(args.frame_a, args.t_a, args.frame_b, args.t_b)
    except ValueError as exc:
        raise SystemExit(f"rejected: {exc}") from exc
    print(json.dumps({"verifier_version": VERIFIER_VERSION, **asdict(result)}, indent=2))


if __name__ == "__main__":
    main()

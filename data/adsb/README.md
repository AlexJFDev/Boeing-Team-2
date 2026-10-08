# ADS-B frame set

Synthetic Mode S extended squitter frames with ground truth, for the Family A
(ADS-B / Mode S) tasks. Built for WBS 2.1 (issue #10). Tasks A1 (decode) and A2 (CPR) are authored from this set in WBS 2.5 and 2.6.

| File | Contents |
| --- | --- |
| `frames_v1.0.0.jsonl` | One frame per line (71 frames) |
| `MANIFEST.json` | Version, SHA-256 of the frames file, counts, pyModeS version |

Do not edit these files by hand. Change `scripts/adsb/build_frames.py`, bump `DATASET_VERSION`, and regenerate:

```bash
python scripts/adsb/build_frames.py
pytest tests/test_adsb_frames.py
```

Once a version is used by a task, treat it as frozen: a new version gets a new file name rather than replacing the old one.

## Where the ground truth comes from

Frames are encoded from the specification (DO-260B message layout, Mode S CRC, airborne CPR encoding), then decoded with **pyModeS 3.6.0**. The pyModeS output is stored as `expected`; what the encoder intended is stored as `synthesis`, and the build fails if the two disagree. The tests decode every frame again with pyModeS, so the labels never rest on our encoder alone.

## Data boundary

Everything is synthetic, per the project rule against real identifiers.

- **ICAO addresses** come from `F00000`–`F07FFF`, the block ICAO Annex 10 Vol III
  Table 9-1 reserves for ICAO-administered temporary addresses. None maps to a registration.
- **Callsigns** use the `ZZZ` placeholder prefix.
- **Timestamps** are synthetic seconds (1000.0 onward), not wall-clock time.

The catalogue's example frame (`8D4840D6...`, KLM1023) is a real aircraft and is deliberately not in this set.

## Record fields

| Field | Meaning |
| --- | --- |
| `frame_id` | Stable ID, `adsb-0001` onward |
| `hex` | The 112-bit frame, 28 uppercase hex characters |
| `df` | Downlink format, 17 or 18 |
| `ca` / `cf` | Capability (DF17) or control field (DF18) |
| `icao` | 24-bit address as read from the frame |
| `type_code` | ADS-B type code as read from the frame |
| `message_type` | `identification`, `airborne_position` or `airborne_velocity` |
| `crc_valid` | Whether the parity checks |
| `label` | `valid` or `corrupt` |
| `expected_answer` | `decode` for valid frames, `reject` for corrupt ones |
| `expected` | Valid: the pyModeS decode. Corrupt: `{"answer": "reject", ...}` |
| `track_id` | Groups frames from one synthetic aircraft (`null` for DF18) |
| `timestamp` | Synthetic reception time in seconds, used for CPR pairing |
| `synthesis` | Encoder intent: true position, velocity components, callsign |
| `corruption` | Corrupt frames only: source frame, flipped bits (1–112), and the `naive_decode` a careless decoder would report |
| `notes` | What the frame is for |

## What is in the set

| | Valid | Corrupt |
| --- | --- | --- |
| DF17 identification (TC 4) | 10 | 3 |
| DF17 airborne position (TC 11–13) | 41 | 2 |
| DF17 airborne velocity (TC 19 subtype 1) | 10 | 1 |
| DF18 identification (CF 0 and CF 6) | 3 | 1 |

Ten synthetic aircraft (`trk-01` to `trk-10`) each send one identification frame,
four airborne position frames alternating even/odd at 0.5 s spacing, and one
velocity frame. The tracks cover both hemispheres, the equator and prime meridian,
a high latitude with few longitude zones, and a crossing of the 180th meridian
(`trk-06`).

Cases the catalogue asks for:

- **DF18 rebroadcasts (A1).** CF 6 frames are ADS-R rebroadcasts and the CF 0 frame
  is a non-transponder device, so none is a direct transponder transmission.
- **Corrupt frames (A1).** Each corrupt frame is a valid frame with one or two bits
  flipped, chosen so a naive decode still gives a well-formed, plausible answer: a
  different callsign, another aircraft's address, a believable position or speed, a
  type code that turns identification into surface position, or a broken parity
  field over an intact payload.
- **CPR negative variants (A2).** Any two even frames from a track give the
  "two even frames" case. `trk-04` has an extra odd frame 45 s after its last
  position, for the "pair too far apart" case. In both, the correct answer is that
  the position cannot be determined.

## Loading in Inspect AI

The file is JSON Lines, so Inspect's `json_dataset` reads it directly. How each
record becomes a `Sample` (prompt wording, target format) belongs to the task in
WBS 2.5; `tests/test_adsb_frames.py` shows a minimal `record_to_sample`.

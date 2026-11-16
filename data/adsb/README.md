# ADS-B frame set

Mode S extended squitter frames with ground truth, for the Family A
(ADS-B / Mode S) tasks. Built for WBS 2.1 (issue #10). Tasks A1 (decode) and A2 (CPR) are authored from this set in WBS 2.5 and 2.6.

| File | Contents |
| --- | --- |
| `frames_v1.1.0.jsonl` | Current frame set (71 frames). The DF17 track frames are real OpenSky Network receptions, unchanged; DF18 and corrupt frames are constructed. Used by A1/A2 v1.2.0 |
| `frames_v1.0.0.jsonl` | **Frozen.** Fully synthetic, constant altitude and velocity per track. Used by A1/A2 v1.0.0 and v1.1.0 and the regression baseline |
| `opensky_frames_v1.0.0.json` | The OpenSky messages selected for v1.1.0, one window per track (see below) |
| `MANIFEST.json` | Current version, SHA-256, counts, pyModeS version; the selection file under `opensky_selection`; earlier versions under `frozen` |

Do not edit these files by hand. Change `scripts/adsb/build_frames.py`, bump `DATASET_VERSION`, and regenerate:

```bash
python scripts/adsb/build_frames.py
pytest tests/test_adsb_frames.py
```

Once a version is used by a task, treat it as frozen: a new version gets a new file name rather than replacing the old one. When `DATASET_VERSION` is bumped, the generator moves the previous manifest entry into `frozen` unchanged, and `bench/adsb/frames.py` loads any listed version by number, so frozen task files keep verifying against the frames they were built from.

## Version 1.1.0: real OpenSky frames (review finding F5)

The October 2026 data review found that climbing and descending tracks in v1.0.0
kept a constant altitude (trk-04 reported 29,000 ft at 1301.7 s and 1346.7 s while
climbing at +1,088 ft/min). In v1.1.0 the ten tracks are real aircraft: every
identification, airborne position and velocity frame of a track is a message the
OpenSky Network received, unchanged, from the public sample
[osky-sample](https://github.com/openskynetwork/osky-sample) (21 April 2015,
12:00-12:05 UTC; the task catalogue's "start here on day one" source). A climbing
aircraft therefore reports the altitudes it was actually at: trk-04 (RYR39XF)
goes from 29,050 ft to 29,800 ft between its pair and its late frame.

`scripts/adsb/extract_opensky_frames.py` picks, for each track, the real aircraft
closest to the v1.0.0 design (flight phase, altitude band, speed) that has an
identification message, an even/odd/even/odd run of position messages each under
2 s apart, a velocity message right after, and, for trk-04, an odd position about
40 s later for A2's stale pair. Only aircraft with airline-style callsigns (three
letters then a digit) are used, so no callsign is a registration. It needs the avro
file from osky-sample (not committed, ~95 MB):

```bash
git clone https://github.com/openskynetwork/osky-sample
git -C osky-sample checkout 1b179c1c541b178944010a2eb6fa5189a4f9193d
python scripts/adsb/extract_opensky_frames.py osky-sample/avro/raw20150421_sample.avro
```

The output is deterministic and its SHA-256 is pinned in `build_frames.py`, so the
frame set can be rebuilt from the committed selection without the avro file.

Still constructed, because the sample has no usable equivalent:

- **DF18 frames** (two ADS-R rebroadcasts, one non-transponder device), with
  synthetic identities (`F00000`–`F07FFF` addresses, `ZZZ` callsigns), byte-identical
  to v1.0.0. The sample holds a single CRC-valid DF18 message.
- **Corrupt frames**: real frames from this set with chosen bits flipped, as in
  v1.0.0, so a naive decode still gives a plausible answer (for example callsign
  AFL2548 read as CFL2548).

What changes for the tasks: the traffic is all over western Europe, so v1.0.0's
southern-hemisphere, equator and 180th-meridian cases have no v1.1.0 counterpart
(they stay in the frozen v1.0.0 set and the tasks built on it). Frame IDs and the
per-track layout are unchanged, so A2's case structure carries over with new
receivers (see `bench/adsb/a2.py`).

Reception times are OpenSky server times (`timeAtServer`), so they include network
jitter; speeds estimated from a two-second run of positions are unreliable, and the
tests compare direction of travel with the velocity message rather than speed.

## Where the ground truth comes from

Every frame is decoded with **pyModeS 3.6.0** and the decode is stored as `expected`. For the constructed frames (v1.0.0, and DF18 in v1.1.0) what the encoder intended is stored as `synthesis`, and the build fails if the two disagree. For real airborne positions, `reference_position` is the position pyModeS decodes locally against the receiving sensor, recorded at extraction; the tests check every even/odd pair decode against it. The tests decode every frame again with pyModeS, so the labels never rest on our own code alone.

## Data boundary

CLAUDE.md rule 1: real identities only as received from an approved open source.

- **v1.1.0 track frames** carry real ICAO addresses and callsigns from the OpenSky
  Network (an approved source in `bench/boundary.py`). Callsigns are airline-style;
  none is a registration. No sensor identity or location is kept.
- **Constructed frames** use synthetic identities: ICAO addresses from `F00000`–`F07FFF`,
  the block ICAO Annex 10 Vol III Table 9-1 reserves for ICAO-administered temporary
  addresses, and `ZZZ` callsigns.
- **Timestamps** are seconds after 2015-04-21 12:00:00 UTC for OpenSky frames,
  rounded to 0.1 s; DF18 frames use 2000.0 onward. v1.0.0 uses synthetic seconds
  from 1000.0.

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
| `track_id` | Groups frames from one aircraft (`null` for DF18) |
| `source` | v1.1.0: `opensky` for real receptions (and corrupt frames made from them), `spec-synthesized` for DF18 frames |
| `reference_position` | v1.1.0 real airborne positions: pyModeS' local decode against the receiving sensor |
| `timestamp` | Synthetic reception time in seconds, used for CPR pairing |
| `synthesis` | Constructed frames: encoder intent (true position, velocity components, callsign). `null` for real receptions |
| `corruption` | Corrupt frames only: source frame, flipped bits (1–112), and the `naive_decode` a careless decoder would report |
| `notes` | What the frame is for |

## What is in the set

| | Valid | Corrupt |
| --- | --- | --- |
| DF17 identification (TC 4) | 10 | 3 |
| DF17 airborne position (TC 11–18; 11–13 in v1.0.0) | 41 | 2 |
| DF17 airborne velocity (TC 19 subtype 1) | 10 | 1 |
| DF18 identification (CF 0 and CF 6) | 3 | 1 |

Ten aircraft (`trk-01` to `trk-10`) each send one identification frame, four
airborne position frames alternating even/odd under 2 s apart, and one velocity
frame. In v1.1.0 they are real airliners and one light aircraft over western Europe:
level cruise from FL210 to FL410, a climb, two descents and a slow aircraft at
3,000 ft, on both sides of the prime meridian and across several longitude-zone
bands. In v1.0.0 they are synthetic, covering both hemispheres, the equator, and a
crossing of the 180th meridian (`trk-06`).

Cases the catalogue asks for:

- **DF18 rebroadcasts (A1).** CF 6 frames are ADS-R rebroadcasts and the CF 0 frame
  is a non-transponder device, so none is a direct transponder transmission.
- **Corrupt frames (A1).** Each corrupt frame is a valid frame with one or two bits
  flipped, chosen so a naive decode still gives a well-formed, plausible answer: a
  different callsign, another aircraft's address, a believable position or speed, a
  type code that turns identification into surface position, or a broken parity
  field over an intact payload.
- **CPR negative variants (A2).** Any two even frames from a track give the
  "two even frames" case. `trk-04` has an extra odd frame after its last position
  (about 40 s in v1.1.0, 45 s in v1.0.0), for the "pair too far apart" case. In both, the correct answer is that
  the position cannot be determined.

## Loading in Inspect AI

The file is JSON Lines, so Inspect's `json_dataset` reads it directly. How each
record becomes a `Sample` (prompt wording, target format) belongs to the task in
WBS 2.5; `tests/test_adsb_frames.py` shows a minimal `record_to_sample`.

# AFDX synthetic schema (ARINC 664 Part 7)

This README explains the schema used to describe synthetic AFDX traffic, plus the code that checks configs against it and builds frames from it. It covers Sprint 01 item 4.1.

The ARINC 664 spec is paywalled, so none of this comes from the spec itself. It is built from the free tutorials (Abaco's AFDX tutorial and UEI's ARINC 664 manual), which are secondary sources. 

## What's in here

| File | What it does |
|---|---|
| `afdx_schema.json` | The rules. A JSON Schema describing a valid config: virtual links, BAG, Lmax, SkewMax, sequence numbers, networks A/B, and the resync rule. |
| `afdx_synth.py` | The engine. Validates configs, hands out sequence numbers, builds scapy frames, and parses frames back. |
| `generate_schema_cases.py` | Writes the test cases. Each one takes a good config and breaks exactly one thing. |
| `cases/schema_cases.json` | The 39 cases it writes (6 valid, 33 invalid). Plain data. |

The tests live in `tests/test_afdx_schema.py`.

## How the pieces fit together

```
generate_schema_cases.py ──writes──▶ cases/schema_cases.json ──┐
                                                               ├──▶ tests/test_afdx_schema.py
afdx_schema.json ──loaded by──▶ afdx_synth.py ─────────────────┘
```

The schema says what's allowed. `afdx_synth.py` enforces it and builds frames. The cases are the test inputs, and the test file checks all of it.

## Running it

From the repo root:

```
pip install -r requirements.txt
python -m pytest tests/test_afdx_schema.py -v
```

You should see 84 tests pass. Use `python -m pytest`, not plain `pytest`, so `bench` can be imported.

To change or add cases, edit `generate_schema_cases.py` and rerun it:

```
python bench/afdx/generate_schema_cases.py
```

Then run the tests again.

## What the rules are

The schema checks one field at a time:

- **VL ID:** 1 to 65535. We reject 0 on purpose.
- **BAG:** 1, 2, 4, 8, 16, 32, 64 or 128 ms. Nothing else.
- **Lmax:** 64 to 1518 bytes, including the 4-byte FCS.
- **Sequence number start:** 1 to 255. 0 is reserved for reset.
- **Networks:** A, B, or both, no repeats.
- **Resync rule:** required, either `drop_with_resync` or `drop_without_resync`.

`afdx_synth.py` adds the rules that need more than one field:

- SkewMax can't be more than 253 times the BAG.
- No two virtual links can share a VL ID.
- The payload has to fit inside Lmax.

## What a built frame looks like

- Destination MAC is `03:00:00:00` followed by the VL ID, so VL 258 becomes `03:00:00:00:01:02`.
- Destination IP is `224.224.<VL ID high byte>.<VL ID low byte>`.
- The sequence number is the very last byte. If the frame needs padding to reach the Ethernet minimum, the padding goes before it.
- Sequence numbers count 1 to 255, then wrap back to 1.
- Frames on network A and B have identical bytes and are spaced one BAG apart. Each network gets its own packet object, so changing one copy never changes the other.

## What the tests check

1. The schema file exists and is a valid JSON Schema.
2. **Frames from the schema have a valid VL ID and sequence number.** We build frames, read the fields back from the raw bytes, and check them, including the 255 to 1 wrap.
3. **Hand-computed golden frames.** Expected header bytes are written out by hand in the tests, so the builder isn't only being checked against itself.
4. **Frame structure.** The IP and UDP length fields match the real frame length, `parse_frame()` rejects broken frames (wrong EtherType, IP options, not UDP, bad length fields, runt frames), and the A and B copies are separate objects.
5. **Bad values get rejected.** A BAG that isn't a power of 2 (and 16 other bad values) fails, every legal BAG passes, and each of the 39 cases gets the right verdict at the right field.

The tests have also been checked by breaking the code on purpose (wrong byte order, wrong wrap, removed checks) and making sure the tests failed each time. 

## Other things to know and limitations
The spec is paywalled, so the frame layout (MAC and IP encoding, where the sequence number sits, padding, the source MAC fields) comes from free tutorials. 

The independent check for that is a Wireshark/tshark cross-check of the frames, which is a separate item. Until it's done, treat "valid frame" as "valid according to our reading of the tutorials."



The schema describes configs, not packets. Frames get built from a config that passes validation.

Anomalies here are config-level. The invalid cases are configs that break the spec. Traffic anomalies (frames arriving too fast, sequence gaps, spoofed sources) belong to the pcap trace work, which is a separate task.

No pcap writing here. `generate_frames()` returns packet objects. Writing them to pcap files comes with the trace work.

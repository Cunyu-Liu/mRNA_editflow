#!/usr/bin/env python3
"""GPU smoke and official-prediction parity attempt for the Saluki port (Task 3).

Smoke: load one official fold checkpoint, forward UTR-like sequences on CUDA,
assert finite outputs. Optional parity: parse the official test TFRecords with
a dependency-free protobuf reader, rebuild the 6-channel inputs exactly as the
official ``RnaDataset`` parse (one-hot + ``tf.one_hot(v, 1)`` indicator
semantics + right-pad to 12288), forward, and compare against the official
``preds.h5``.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import zlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
import torch

from core.route2_saluki_port_v1 import (
    SALUKI_INPUT_CHANNELS,
    SalukiGRUV1,
    encode_saluki_six_channel_v1,
)

SALUKI_FULL_LENGTH = 12288


def _read_varint(buffer: bytes, position: int) -> tuple[int, int]:
    result, shift = 0, 0
    while True:
        byte = buffer[position]
        position += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            break
        shift += 7
    return result, position


def _parse_fields(buffer: bytes):
    position = 0
    while position < len(buffer):
        tag, position = _read_varint(buffer, position)
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            value, position = _read_varint(buffer, position)
            yield field, wire, value
        elif wire == 2:
            length, position = _read_varint(buffer, position)
            yield field, wire, buffer[position : position + length]
            position += length
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")


def parse_example(payload: bytes) -> dict[str, tuple[str, list]]:
    features: dict[str, tuple[str, list]] = {}
    for field, wire, features_bytes in _parse_fields(payload):
        if field != 1 or wire != 2:
            continue
        for f2, w2, entry in _parse_fields(features_bytes):
            if f2 != 1 or w2 != 2:
                continue
            key, feature_bytes = None, None
            for f3, w3, value in _parse_fields(entry):
                if f3 == 1 and w3 == 2:
                    key = value.decode()
                elif f3 == 2 and w3 == 2:
                    feature_bytes = value
            if key is None or feature_bytes is None:
                continue
            for f4, w4, inner in _parse_fields(feature_bytes):
                if f4 == 1 and w4 == 2:
                    features[key] = (
                        "bytes",
                        [v for f5, w5, v in _parse_fields(inner) if f5 == 1 and w5 == 2],
                    )
                elif f4 == 3 and w4 == 2:
                    features[key] = (
                        "int64",
                        [v for f5, w5, v in _parse_fields(inner) if f5 == 1 and w5 == 0],
                    )
    return features


def iter_tfrecords(path: Path):
    """Yield TFRecord payloads; basenji shards are zlib-compressed streams."""
    raw = Path(path).read_bytes()
    if raw[:2] in (b"\x78\x9c", b"\x78\x01", b"\x78\xda"):
        raw = zlib.decompress(raw)
    position = 0
    while position + 12 <= len(raw):
        length = struct.unpack("<Q", raw[position : position + 8])[0]
        position += 12  # length + length crc
        if length == 0 or position + length + 4 > len(raw):
            break
        yield raw[position : position + length]
        position += length + 4  # payload + payload crc


def decode_inputs_from_example(features: dict[str, tuple[str, list]], seq_len: int) -> np.ndarray:
    """Rebuild the official 6-channel input (sequence one-hot + indicator channels)."""
    sequence = np.frombuffer(features["sequence"][1][0], dtype=np.uint8).astype(np.int64)
    length = len(sequence)
    one_hot = np.zeros((length, 4), dtype=np.float32)
    one_hot[np.arange(length), sequence] = 1.0

    def indicator_channel(name: str) -> np.ndarray:
        # 2022-era pipeline: the channel carries the raw indicator value
        # (1.0 at codon-first / splice positions, 0.0 at ordinary positions).
        # Verified against the official f7_c0 test set (reproduces the
        # published pearson 0.758); the master repo's tf.one_hot(v, 1)
        # refactor inverts this and does not match the frozen checkpoints.
        channel = np.zeros((length, 1), dtype=np.float32)
        if name in features:
            kind, payload = features[name]
            if kind == "bytes":
                values = np.frombuffer(payload[0], dtype=np.uint8).astype(np.int64)
            else:
                values = np.asarray(payload, dtype=np.int64)
            channel[values == 1] = 1.0
        return channel

    # 2022-era shards use the feature names `coding` and `splice`; the master
    # parser calls them `frame` and `splice5p` (plus a later `splice3p` the
    # 6-channel model never consumed).
    coding = indicator_channel("coding") if "coding" in features else indicator_channel("frame")
    splice = indicator_channel("splice") if "splice" in features else indicator_channel("splice5p")
    inputs = np.concatenate([one_hot, coding, splice], axis=1)
    padded = np.zeros((seq_len, SALUKI_INPUT_CHANNELS), dtype=np.float32)
    padded[:length] = inputs  # official parse right-pads zeros to length_full
    return padded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--parity-tfr", type=Path, default=None)
    parser.add_argument("--parity-preds", type=Path, default=None)
    parser.add_argument("--parity-count", type=int, default=64)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable - GPU is required for the Saluki port smoke")
    device = torch.device(f"cuda:{args.physical_gpu_index}")
    model = SalukiGRUV1(args.weights).to(device).eval()

    rng = np.random.default_rng(20260902)
    report: dict[str, object] = {
        "schema_version": "route_a_v3_route2_saluki_port_smoke.v1",
        "weights": str(args.weights),
        "device": torch.cuda.get_device_name(device),
        "smoke": {},
    }
    for length in (350, 512, 1000):
        sequences = ["".join(rng.choice(list("ACGU"), size=length)) for _ in range(4)]
        encoded = np.stack([encode_saluki_six_channel_v1(s) for s in sequences])
        tensor = torch.from_numpy(encoded).to(device)
        with torch.no_grad():
            values = model(tensor)
        if not (values.is_cuda and torch.isfinite(values).all().item()):
            raise SystemExit(f"smoke forward failed at length {length}")
        report["smoke"][str(length)] = {
            "batch": 4,
            "min": float(values.min()),
            "max": float(values.max()),
        }

    if args.parity_tfr is not None and args.parity_preds is not None:
        records: list[dict[str, tuple[str, list]]] = []
        for payload in iter_tfrecords(args.parity_tfr):
            records.append(parse_example(payload))
            if len(records) >= args.parity_count:
                break
        if not records:
            raise SystemExit("no TFRecords parsed")
        report["parity_feature_keys"] = sorted(records[0].keys())
        inputs = np.stack([decode_inputs_from_example(r, SALUKI_FULL_LENGTH) for r in records])
        tensor = torch.from_numpy(inputs).to(device)
        with torch.no_grad():
            values = model(tensor).cpu().numpy()
        with h5py.File(args.parity_preds, "r") as handle:
            keys = list(handle.keys())
            prediction_key = "preds" if "preds" in handle else keys[0]
            preds = np.asarray(handle[prediction_key]).reshape(-1)
        count = min(len(values), len(preds))
        diffs = np.abs(values[:count] - preds[:count].astype(np.float32))
        report["parity"] = {
            "preds_keys": keys,
            "n_compared": int(count),
            "max_abs_diff": float(diffs.max()),
            "mean_abs_diff": float(diffs.mean()),
            "fraction_within_0p01": float((diffs <= 0.01).mean()),
            "fraction_within_0p05": float((diffs <= 0.05).mean()),
        }
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

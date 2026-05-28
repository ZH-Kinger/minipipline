from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    root = Path("data/raw")
    streams = {
        "cam_left": (".jpg", b"demo-rgb"),
        "depth": (".npy", b"demo-depth"),
        "imu": (".json", None),
        "pose": (".json", None),
    }
    timestamps = [
        1716810000000000000,
        1716810000033000000,
        1716810000066000000,
    ]

    for timestamp_ns in timestamps:
        for stream, (suffix, payload) in streams.items():
            stream_dir = root / stream
            stream_dir.mkdir(parents=True, exist_ok=True)
            path = stream_dir / f"{timestamp_ns}{suffix}"
            if payload is not None:
                path.write_bytes(payload)
            else:
                body = {
                    "timestamp_ns": timestamp_ns,
                    "label": "pick",
                    "confidence": 0.96,
                    "stream": stream,
                }
                path.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote demo data under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

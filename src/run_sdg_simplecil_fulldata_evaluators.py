#!/usr/bin/env python3
"""Run the full-data CIL evaluator producer and independent verifier.

`--source-root` is the output root of sdg_simplecil_matched.py (it holds
canonical_cls_features.pt). The protocol file pins the hashes of that cache;
see configs/simplecil_fulldata_evaluators_protocol.json and the README.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--commit", default=None)
    args = parser.parse_args()
    source_cache = args.source_root / "canonical_cls_features.pt"
    producer = [
        sys.executable,
        "-u",
        str(Path(__file__).with_name("sdg_simplecil_fulldata_evaluators.py")),
        "--source-root",
        str(args.source_root),
        "--protocol",
        str(args.protocol),
        "--protocol-sha256",
        args.protocol_sha256,
        "--output-root",
        str(args.output_root),
    ]
    if args.commit:
        producer += ["--commit", args.commit]
    subprocess.run(producer, check=True)
    validation = args.output_root / "independent_validation.json"
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(
                Path(__file__).with_name(
                    "verify_sdg_simplecil_fulldata_evaluators.py"
                )
            ),
            "--root",
            str(args.output_root),
            "--protocol",
            str(args.protocol),
            "--protocol-sha256",
            args.protocol_sha256,
            "--source-cache",
            str(source_cache),
            "--output",
            str(validation),
        ],
        check=True,
    )
    completion = args.output_root / "completion.json"
    if completion.exists():
        raise FileExistsError(completion)
    completion.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "accepted": True,
                "protocol_sha256": args.protocol_sha256,
                "result_sha256": sha256(args.output_root / "result.json"),
                "validation_sha256": sha256(validation),
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

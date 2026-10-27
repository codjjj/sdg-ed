#!/usr/bin/env python3
"""Run the SimpleCIL/FSCIL producer and its independent verifier in one job.

The producer is sdg_simplecil_matched.py; the run is accepted only if
verify_sdg_simplecil_matched.py passes, after which completion.json is
written next to result.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sdg_paths import MODEL_ROOT, RELEASE_ROOT  # noqa: E402

MODEL_REPO = "facebook/dinov2-base"
MODEL_REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cub-root", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, default=RELEASE_ROOT / "splits" / "fscil_cub200")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT / "dinov2-base")
    parser.add_argument("--model-repo", default=MODEL_REPO)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--commit", default=None)
    args = parser.parse_args()
    producer = [
        sys.executable,
        str(Path(__file__).with_name("sdg_simplecil_matched.py")),
        "--cub-root", str(args.cub_root),
        "--index-root", str(args.index_root),
        "--output-root", str(args.output_root),
        "--model-root", str(args.model_root),
        "--model-repo", args.model_repo,
        "--model-revision", args.model_revision,
        "--batch-size", str(args.batch_size),
        "--workers", str(args.workers),
    ]
    if args.commit:
        producer += ["--commit", args.commit]
    subprocess.run(producer, check=True)
    validation = args.output_root / "independent_validation.json"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("verify_sdg_simplecil_matched.py")),
            "--root", str(args.output_root),
            "--output", str(validation),
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

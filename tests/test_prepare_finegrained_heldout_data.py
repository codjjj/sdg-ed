import io
import json
import tarfile
from pathlib import Path

import pytest

from prepare_finegrained_heldout_data import (
    FOODX_OFFICIAL_SPLIT_COUNTS,
    SOURCES,
    download,
    parse_foodx_classes,
    parse_foodx_csv,
    parse_key_value_lines,
    source_id_sort_key,
    suffix_lookup,
    stage_project_source,
    verify_downloaded_source,
)


def test_nabirds_key_value_parser_preserves_names_with_spaces():
    parsed = parse_key_value_lines("1 path/to/a.jpg\n2 Northern Cardinal adult male\n")
    assert parsed == {"1": "path/to/a.jpg", "2": "Northern Cardinal adult male"}


def test_nabirds_source_ids_preserve_numeric_order_and_accept_uuid_ids():
    source_ids = ["10", "2", "0000139e-21dc-4d0c-bfe1-4cae3c85c829", "abc"]
    assert sorted(source_ids, key=source_id_sort_key) == [
        "2",
        "10",
        "0000139e-21dc-4d0c-bfe1-4cae3c85c829",
        "abc",
    ]


def test_suffix_lookup_is_unique_and_fail_closed():
    members = ["root/images/a.jpg", "root/images/b.jpg"]
    assert suffix_lookup(members, ["images/a.jpg"])["images/a.jpg"] == "root/images/a.jpg"
    with pytest.raises(ValueError, match="resolution failed"):
        suffix_lookup(["x/a.jpg", "y/a.jpg"], ["a.jpg"])


def test_foodx_annotation_parser_skips_header_only():
    assert parse_foodx_csv("image_name,label\na.jpg,3\nb.jpg,10\n") == [
        ("a.jpg", "3"),
        ("b.jpg", "10"),
    ]


def test_foodx_class_parser_requires_all_251_classes():
    text = "\n".join(f"class {index}" for index in range(251))
    parsed = parse_foodx_classes(text)
    assert parsed["0"] == "class 0"
    assert parsed["250"] == "class 250"
    with pytest.raises(ValueError, match="251"):
        parse_foodx_classes("only one")


def test_foodx_official_class_format_and_split_counts_are_frozen():
    text = "\n".join(f"{index} class {index}" for index in range(251))
    parsed = parse_foodx_classes(text)
    assert parsed["0"] == "class 0"
    assert parsed["250"] == "class 250"
    assert FOODX_OFFICIAL_SPLIT_COUNTS == {"train": 118_475, "test": 11_994}


def test_nabirds_archive_is_local_only_and_size_and_sha_pinned(tmp_path):
    source = SOURCES["nabirds"][0]
    assert source["url"] is None
    with pytest.raises(FileNotFoundError, match="no public download URL"):
        download(source, tmp_path / "nabirds.tar.gz")
    assert source["bytes_expected"] == 9_884_285_075
    assert source["sha256_expected"] == (
        "cd20ef17a4ae110edcacaf964c4ad4794fc762ad760f237f826dbf30040f713d"
    )

    payload = tmp_path / "archive.tar.gz"
    payload.write_bytes(b"official bytes")
    fixture = {
        "name": payload.name,
        "url": "https://example.invalid/archive.tar.gz",
        "md5": None,
        "bytes_expected": payload.stat().st_size,
        "sha256_expected": __import__("hashlib").sha256(payload.read_bytes()).hexdigest(),
    }
    receipt = verify_downloaded_source(fixture, payload)
    assert receipt["bytes"] == payload.stat().st_size
    assert receipt["sha256"] == fixture["sha256_expected"]

    with pytest.raises(ValueError, match="size mismatch"):
        verify_downloaded_source({**fixture, "bytes_expected": 1}, payload)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_downloaded_source({**fixture, "sha256_expected": "0" * 64}, payload)


def test_all_official_sources_are_size_pinned():
    assert all(
        isinstance(source.get("bytes_expected"), int) and source["bytes_expected"] > 0
        for sources in SOURCES.values()
        for source in sources
    )


def test_local_source_archive_is_hash_bound(tmp_path):
    source_root = tmp_path / "source_archives"
    source_path = source_root / "nabirds" / "archive.tar.gz"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"official bytes")
    destination = tmp_path / "staged.tar.gz"
    fixture = {
        "name": "archive.tar.gz",
        "url": None,
        "md5": None,
        "bytes_expected": source_path.stat().st_size,
        "sha256_expected": __import__("hashlib").sha256(source_path.read_bytes()).hexdigest(),
    }
    receipt = stage_project_source(
        fixture,
        destination,
        source_root=source_root,
        dataset="nabirds",
    )
    assert receipt["transport"] == "local_archive"
    assert receipt["sha256"] == fixture["sha256_expected"]
    with pytest.raises(FileNotFoundError, match="local source archive is missing"):
        stage_project_source(
            fixture,
            tmp_path / "missing-copy.tar.gz",
            source_root=tmp_path / "outside",
            dataset="nabirds",
        )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        stage_project_source(
            {**fixture, "sha256_expected": "0" * 64},
            tmp_path / "bad-copy.tar.gz",
            source_root=source_root,
            dataset="nabirds",
        )

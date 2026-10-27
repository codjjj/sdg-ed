import hashlib
import json

import pytest

import seal_finegrained_data_inputs as module


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_root(tmp_path, dataset, contract):
    root = tmp_path / "safe" / dataset
    manifest_path = root / "manifest" / "dataset_manifest.json"
    archive = root / "raw" / "archive.tar"
    archive.parent.mkdir(parents=True)
    archive.write_bytes((dataset + " bytes").encode())
    manifest = {
        "dataset": dataset,
        "dataset_identity_sha256": hashlib.sha256(dataset.encode()).hexdigest(),
        "n_classes": contract["n_classes"],
        "splits": contract["splits"],
        "split_definition": {
            "n_species": 10_000,
            "class_subset": None,
        } if dataset == "inat2021_mini" else {},
        "archive_sha256": {"raw/archive.tar": _sha(archive)},
    }
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    ready = {
        "schema_version": 1,
        "kind": "sdg_caev_finegrained_dataset_ready",
        "dataset": dataset,
        "fallback_used": False,
        "n_classes": contract["n_classes"],
        "splits": contract["splits"],
        "dataset_manifest_sha256": _sha(manifest_path),
        "dataset_identity_sha256": manifest["dataset_identity_sha256"],
    }
    (root / "READY.json").write_text(json.dumps(ready), encoding="utf-8")
    return root


def test_data_input_bundle_is_exact_and_rejects_inat_subset(tmp_path, monkeypatch):
    safe = tmp_path / "safe"
    monkeypatch.setattr(
        module,
        "load_and_verify_manifest",
        lambda path: json.loads(path.read_text(encoding="utf-8")),
    )
    contracts = {
        dataset: {"n_classes": index + 1, "splits": {"train": 20, "test": 1}}
        for index, dataset in enumerate(module.DATASETS)
    }
    monkeypatch.setattr(module, "EXPECTED_CONTRACTS", contracts)
    bindings = {
        dataset: _make_root(tmp_path, dataset, contracts[dataset])
        for dataset in module.DATASETS
    }
    bundle = module.build_data_input_bundle(bindings, "abc123")
    assert [entry["dataset"] for entry in bundle["datasets"]] == list(module.DATASETS)
    path = safe / "bundle.json"
    path.write_bytes(module.canonical_json_bytes(bundle))
    assert module.verify_dataset_binding(path, bindings["inat2021_mini"])["n_classes"] == 3

    manifest_path = bindings["inat2021_mini"] / "manifest/dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["split_definition"]["class_subset"] = [0]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    ready_path = bindings["inat2021_mini"] / "READY.json"
    ready = json.loads(ready_path.read_text())
    ready["dataset_manifest_sha256"] = _sha(manifest_path)
    ready_path.write_text(json.dumps(ready), encoding="utf-8")
    with pytest.raises(ValueError, match="subset"):
        module.build_data_input_bundle(bindings, "abc123")


def test_bundle_identity_and_root_binding_fail_closed(tmp_path, monkeypatch):
    safe = tmp_path / "safe"
    payload = {
        "schema_version": 1,
        "kind": "sdg_caev_finegrained_data_input_bundle",
        "commit": "abc123",
        "datasets": [
            {
                "dataset": dataset,
                "root": str(safe / dataset),
                "ready_sha256": "0" * 64,
                "dataset_manifest_sha256": "1" * 64,
            }
            for dataset in module.DATASETS
        ],
        "dataset_order": list(module.DATASETS),
        "inat2021_contract": {
            "all_10000_species": True,
            "class_subset": None,
            "train_per_class": 50,
            "validation_per_class": 10,
        },
    }
    payload["bundle_identity_sha256"] = module.sha256_bytes(module.canonical_json_bytes(payload))
    path = safe / "bundle.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(module.canonical_json_bytes(payload))
    loaded = module.load_and_verify_data_input_bundle(path)
    assert loaded["commit"] == "abc123"
    payload["commit"] = "tampered"
    path.write_bytes(module.canonical_json_bytes(payload))
    with pytest.raises(ValueError, match="identity"):
        module.load_and_verify_data_input_bundle(path)


def test_dataset_scoped_bundle_binds_one_ready_dataset_without_weakening_joint_scope(
    tmp_path, monkeypatch
):
    safe = tmp_path / "safe"
    monkeypatch.setattr(
        module,
        "load_and_verify_manifest",
        lambda path: json.loads(path.read_text(encoding="utf-8")),
    )
    contract = {"n_classes": 120, "splits": {"train": 12_000, "test": 8_580}}
    monkeypatch.setitem(module.EXPECTED_CONTRACTS, "stanford_dogs", contract)
    root = _make_root(tmp_path, "stanford_dogs", contract)

    bindings = module.parse_dataset_binding([f"stanford_dogs={root}"])
    bundle = module.build_dataset_input_bundle(bindings, "abc123")
    path = safe / "dogs.json"
    path.write_bytes(module.canonical_json_bytes(bundle))
    loaded = module.load_and_verify_data_input_bundle(path)

    assert loaded["kind"] == "sdg_caev_finegrained_dataset_input_bundle"
    assert loaded["dataset_order"] == ["stanford_dogs"]
    assert module.verify_dataset_binding(path, root)["dataset"] == "stanford_dogs"
    with pytest.raises(ValueError, match="exact four datasets"):
        module.build_data_input_bundle(bindings, "abc123")
    with pytest.raises(ValueError, match="exactly one"):
        module.parse_dataset_binding(
            [f"stanford_dogs={root}", f"nabirds={safe / 'nabirds'}"]
        )


def test_foodx_official_split_contract_accepts_current_counts_and_rejects_legacy(
    tmp_path, monkeypatch
):
    safe = tmp_path / "safe"
    monkeypatch.setattr(
        module,
        "load_and_verify_manifest",
        lambda path: json.loads(path.read_text(encoding="utf-8")),
    )
    contract = module.EXPECTED_CONTRACTS["foodx251"]
    assert contract == {
        "n_classes": 251,
        "splits": {"train": 118_475, "test": 11_994},
    }
    root = _make_root(tmp_path, "foodx251", contract)

    bundle = module.build_dataset_input_bundle({"foodx251": root}, "abc123")
    assert bundle["datasets"][0]["splits"] == contract["splits"]

    manifest_path = root / "manifest/dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["splits"] = {"train": 120_216, "test": 12_170}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    ready_path = root / "READY.json"
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    ready["splits"] = manifest["splits"]
    ready["dataset_manifest_sha256"] = _sha(manifest_path)
    ready_path.write_text(json.dumps(ready), encoding="utf-8")

    with pytest.raises(ValueError, match="split-count contract mismatch"):
        module.build_dataset_input_bundle({"foodx251": root}, "abc123")

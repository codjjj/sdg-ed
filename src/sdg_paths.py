"""Default locations used by the release scripts.

Every script also accepts explicit command-line paths; these defaults only
decide where a script looks when no flag is given. Each default can be moved
with an environment variable:

    SDG_RESULTS_ROOT  released result JSONs        (default: artifacts/results)
    SDG_OUTPUT_ROOT   regenerated outputs          (default: outputs)
    SDG_FEATURE_ROOT  cached frozen features       (default: features)
    SDG_DATA_ROOT     prepared image datasets      (default: data)
    SDG_MODEL_ROOT    local model snapshots        (default: models)

Cached-feature layout (see DATA_AND_FEATURES.md):

    $SDG_FEATURE_ROOT/<dataset>/{train,test}.pt                       DINOv2-B core caches
    $SDG_FEATURE_ROOT/backbones/<backbone>/<dataset>/{train,test}.pt  other frozen backbones

Each cache directory may also keep the two files under a `features/`
subdirectory.
"""

from __future__ import annotations

import os
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


RESULTS_ROOT = _env_path("SDG_RESULTS_ROOT", RELEASE_ROOT / "artifacts" / "results")
OUTPUT_ROOT = _env_path("SDG_OUTPUT_ROOT", RELEASE_ROOT / "outputs")
FEATURE_ROOT = _env_path("SDG_FEATURE_ROOT", RELEASE_ROOT / "features")
DATA_ROOT = _env_path("SDG_DATA_ROOT", RELEASE_ROOT / "data")
MODEL_ROOT = _env_path("SDG_MODEL_ROOT", RELEASE_ROOT / "models")


def _cache_dir(root: Path) -> Path:
    """Return `root` or `root/features`, whichever holds train.pt and test.pt."""

    for candidate in (root, root / "features"):
        if (candidate / "train.pt").exists() and (candidate / "test.pt").exists():
            return candidate
    return root


def core_feature_dir(dataset: str, feature_root: Path | str | None = None) -> Path:
    """Directory of the DINOv2-B core cache for one dataset."""

    return _cache_dir(Path(feature_root or FEATURE_ROOT) / dataset)


def backbone_feature_dir(
    backbone: str, dataset: str, feature_root: Path | str | None = None
) -> Path:
    """Directory of a non-DINOv2 backbone cache for one dataset."""

    if backbone in {"dinov2", "dinov2_b"}:
        return core_feature_dir(dataset, feature_root)
    return _cache_dir(Path(feature_root or FEATURE_ROOT) / "backbones" / backbone / dataset)

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def load_tags_whitelist(path: Path | None = None) -> dict[str, Any]:
    cfg = path or project_root() / "configs" / "tags_whitelist.yaml"
    return _read_yaml(cfg)


def load_constraints(path: Path | None = None) -> dict[str, Any]:
    cfg = path or project_root() / "configs" / "constraints.yaml"
    return _read_yaml(cfg)

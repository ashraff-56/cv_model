"""YAML config loading. Several files merge in order (later wins); `key.sub=value` overrides last."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(*paths: str | Path, overrides: list[str] | None = None) -> dict:
    cfg: dict = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key.sub=value, got {item!r}")
        node = cfg
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg


def resolve_path(path: str | Path, base: Path = REPO_ROOT) -> Path:
    p = Path(path)
    return p if p.is_absolute() else base / p


def load_classes(path: str | Path = REPO_ROOT / "configs" / "classes.yaml") -> list[dict]:
    """Class definitions in their canonical order (this order defines the label index)."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["classes"]

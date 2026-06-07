"""全局配置加载器。读取项目根目录的 config.yaml。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """加载并返回全局配置 dict。"""
    p = Path(path) if path else CONFIG_PATH
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_path(key: str, cfg: dict[str, Any] | None = None) -> Path:
    """把 config.paths 下的相对路径解析为绝对路径，并确保目录存在。"""
    cfg = cfg or load_config()
    rel = cfg["paths"][key]
    p = ROOT / rel
    p.mkdir(parents=True, exist_ok=True)
    return p

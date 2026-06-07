"""全局配置加载器。读取项目根目录的 config.yaml。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"


def user_db_path() -> Path:
    """用户数据 SQLite 路径（规则/校准/事件/检验账本）。

    优先环境变量 QUANTLAB_DB_PATH（部署到带持久卷的主机时指向挂载盘，重启不丢）；
    否则用项目内 data/quantlab.db（本地持久；Streamlit Cloud 上为临时，建议用导出/恢复备份）。"""
    env = os.environ.get("QUANTLAB_DB_PATH")
    p = Path(env) if env else (ROOT / "data" / "quantlab.db")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


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

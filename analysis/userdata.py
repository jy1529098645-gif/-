"""用户数据 备份 / 恢复（零外部依赖的持久化方案）。

Streamlit Community Cloud 文件系统是临时的——重启后 SQLite 里你手填的事件、校准战绩、
FDR 账本、保存的规则都会丢。两条免费出路：
1. 部署到带持久卷的主机，设环境变量 QUANTLAB_DB_PATH 指向挂载盘（config.user_db_path 自动用）；
2. 用本模块**导出为 JSON 下载备份 / 上传恢复**——任何部署都能用，零依赖。

导出表：rules(保存的规则) / signals(校准信号) / event_watch(手填事件) / mt_tests(检验账本)。
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import config

_TABLES = ["rules", "signals", "event_watch", "mt_tests"]


def _conn(db_path: str | Path | None = None):
    p = Path(db_path) if db_path else config.user_db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(p)


def _table_exists(c, name: str) -> bool:
    return c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def export_userdata(db_path: str | Path | None = None) -> dict:
    """导出全部用户表为 {table: {columns:[...], rows:[[...]]}} 的可 JSON 化 dict。"""
    out: dict = {"_schema": "quantlab_userdata_v1", "tables": {}}
    with _conn(db_path) as c:
        for t in _TABLES:
            if not _table_exists(c, t):
                continue
            cur = c.execute(f"SELECT * FROM {t}")
            cols = [d[0] for d in cur.description]
            rows = [list(r) for r in cur.fetchall()]
            out["tables"][t] = {"columns": cols, "rows": rows}
    out["_counts"] = {t: len(v["rows"]) for t, v in out["tables"].items()}
    return out


def export_json(db_path: str | Path | None = None) -> str:
    return json.dumps(export_userdata(db_path), ensure_ascii=False, indent=2, default=str)


def import_userdata(payload: dict | str, mode: str = "merge",
                    db_path: str | Path | None = None) -> dict:
    """从导出的 dict/JSON 恢复。mode='merge'(追加,默认) 或 'replace'(先清空再写)。

    依赖各模块的建表逻辑先建好表，故这里先触发一次建表。返回 {table: 写入行数}。"""
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict) or "tables" not in payload:
        raise ValueError("非法备份格式（缺 tables）。")

    # 确保表结构存在（复用各模块建表）
    from analysis import journal as _j, mt_ledger as _m, event_radar as _e
    from frontend import store as _s
    _j._conn(db_path).close(); _m._conn(db_path).close()
    _e._conn(db_path).close()
    if db_path is None:  # store 用全局路径
        _s._conn().close()

    written: dict = {}
    with _conn(db_path) as c:
        for t, blk in payload["tables"].items():
            if t not in _TABLES or not _table_exists(c, t):
                continue
            cols, rows = blk.get("columns", []), blk.get("rows", [])
            # 只保留目标表真实存在的列（防 schema 漂移）
            real_cols = [d[1] for d in c.execute(f"PRAGMA table_info({t})").fetchall()]
            keep = [(i, col) for i, col in enumerate(cols) if col in real_cols and col != "id"]
            if not keep:
                continue
            colnames = [col for _, col in keep]
            placeholders = ",".join("?" * len(colnames))
            if mode == "replace":
                c.execute(f"DELETE FROM {t}")
            n = 0
            for r in rows:
                vals = [r[i] for i, _ in keep]
                try:
                    c.execute(f"INSERT INTO {t}({','.join(colnames)}) VALUES({placeholders})", vals)
                    n += 1
                except sqlite3.IntegrityError:
                    pass  # 去重键冲突(merge 时重复)跳过
            written[t] = n
    return written

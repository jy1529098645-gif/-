"""规则持久化（SQLite）。保存/复用命名的进出场规则。"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import config

_DB = Path(config.ROOT) / "data" / "quantlab.db"


def _conn():
    _DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(_DB)
    c.execute(
        "CREATE TABLE IF NOT EXISTS rules ("
        "name TEXT PRIMARY KEY, spec TEXT NOT NULL, created TEXT)"
    )
    return c


def save_rule(name: str, spec: dict) -> None:
    """保存（或覆盖）一条命名规则。spec 为可 JSON 序列化的 dict。"""
    payload = json.dumps(_jsonable(spec), ensure_ascii=False)
    with _conn() as c:
        c.execute(
            "INSERT INTO rules(name, spec, created) VALUES(?,?,datetime('now')) "
            "ON CONFLICT(name) DO UPDATE SET spec=excluded.spec, created=datetime('now')",
            (name, payload),
        )


def get_rule(name: str) -> dict | None:
    with _conn() as c:
        row = c.execute("SELECT spec FROM rules WHERE name=?", (name,)).fetchone()
    return json.loads(row[0]) if row else None


def list_rules() -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT name, spec, created FROM rules ORDER BY created DESC").fetchall()
    out = []
    for name, spec, created in rows:
        s = json.loads(spec)
        sigs = "+".join(x[0] for x in s.get("specs", [])) or "?"
        out.append({"名称": name, "信号": sigs, "组合": s.get("op", ""),
                    "止损/止盈": f"{s.get('trailing','')}/{s.get('tp','')}",
                    "条件": s.get("cond_kind", "none"), "保存于": created})
    return out


def delete_rule(name: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM rules WHERE name=?", (name,))


def _jsonable(spec: dict) -> dict:
    """把 tuple（specs）转成 list 以便 JSON 序列化。"""
    out = dict(spec)
    if "specs" in out:
        out["specs"] = [list(s) for s in out["specs"]]
    return out

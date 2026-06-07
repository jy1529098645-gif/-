"""事件雷达（Event Radar）—— 把"已知的未来大事"作为 display-only 风险提醒。

铁律（关键）：这些事件**绝不进回测/量化结果**——
- 它们是前视信息（未来才发生），且多为一次性、无干净历史可统计（如某巨型 IPO 抽走流动性）；
- 强行量化=前视偏差 + 过拟合独特事件。所以本模块**只出提醒报告、不改任何引擎数字/证据等级**。

两类来源：
1. 自动可算事件（无前视、规则确定）：FOMC 利率决议、月度非农(首个周五)、期权到期/四巫日(每月/季三周五)、
   标普季度再平衡(季三周五)、个股财报(来自数据)。
2. 用户手填特大事件（自动源免费拿不到）：如 SpaceX 上市、并购、诉讼判决、大型解禁/再融资。
   存 SQLite，可在前端增删。

每个事件给：日期 / 距今天数 / 范围(全市场或某票) / 类别 / 严重度 / "为什么重要·盯什么"。
"""
from __future__ import annotations

import datetime as _dt
import sqlite3
from pathlib import Path

import pandas as pd

import config

SEVERITY_ORDER = {"高": 3, "中": 2, "低": 1}


# ---------------------------------------------------------------------------
# 自动可算事件（规则确定、无前视）
# ---------------------------------------------------------------------------
# FOMC 利率决议日（决议日=会议第二天，美联储提前公布全年日程；2025–2026）
_FOMC = [
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
    "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29",
    "2026-09-16", "2026-10-28", "2026-12-09",
]


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> _dt.date:
    """某月第 n 个 weekday(0=周一…4=周五)。"""
    d = _dt.date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    return d + _dt.timedelta(days=offset + 7 * (n - 1))


def _first_friday(year: int, month: int) -> _dt.date:
    return _nth_weekday(year, month, 4, 1)


def _third_friday(year: int, month: int) -> _dt.date:
    return _nth_weekday(year, month, 4, 3)


def auto_market_events(today: _dt.date, horizon_days: int = 45) -> list[dict]:
    """生成未来 horizon_days 内的全市场日历事件（规则确定）。"""
    end = today + _dt.timedelta(days=horizon_days)
    ev: list[dict] = []

    for s in _FOMC:
        d = _dt.date.fromisoformat(s)
        if today <= d <= end:
            ev.append({"date": d, "scope": "全市场", "category": "FOMC 利率决议",
                       "severity": "高", "source": "auto",
                       "watch": "利率/点阵图/鲍威尔措辞 → 全市场波动放大；别在决议隔夜满仓押方向。"})

    # 遍历当月与后续月份，取首个周五(非农)、第三个周五(期权到期/四巫)
    y, m = today.year, today.month
    for _ in range(3):  # 覆盖约 3 个月
        nfp = _first_friday(y, m)
        if today <= nfp <= end:
            ev.append({"date": nfp, "scope": "全市场", "category": "非农就业(NFP)",
                       "severity": "中", "source": "auto",
                       "watch": "8:30ET 公布 → 利率预期与开盘波动；数据爆冷/爆热当日易跳空。"})
        tf = _third_friday(y, m)
        if today <= tf <= end:
            is_quad = m in (3, 6, 9, 12)
            ev.append({"date": tf, "scope": "全市场",
                       "category": "四巫日/标普再平衡" if is_quad else "月度期权到期(OpEx)",
                       "severity": "高" if is_quad else "中", "source": "auto",
                       "watch": ("季度四巫(股指/个股期权期货同时到期)+标普再平衡 → 尾盘成交暴增、价格易被钉/异动。"
                                 if is_quad else "月度期权到期 → 临近到期 gamma 效应、关键行权价附近易被磁吸。")})
        m += 1
        if m > 12:
            m = 1; y += 1
    return ev


# ---------------------------------------------------------------------------
# 用户手填特大事件（SQLite）
# ---------------------------------------------------------------------------
def _conn(db_path: str | Path | None = None):
    p = Path(db_path) if db_path else config.user_db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(p)
    c.execute(
        "CREATE TABLE IF NOT EXISTS event_watch ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, scope TEXT, category TEXT, "
        "title TEXT, impact TEXT, severity TEXT, added TEXT)"
    )
    return c


def add_event(date: str, title: str, scope: str = "全市场", category: str = "特大事件",
              impact: str = "", severity: str = "高", db_path: str | Path | None = None) -> None:
    """手填一条特大事件（如 'SpaceX 上市'）。scope='全市场' 或某 ticker。"""
    with _conn(db_path) as c:
        c.execute(
            "INSERT INTO event_watch(date,scope,category,title,impact,severity,added) "
            "VALUES(?,?,?,?,?,?,datetime('now'))",
            (date, scope, category, title, impact, severity),
        )


def delete_event(event_id: int, db_path: str | Path | None = None) -> None:
    with _conn(db_path) as c:
        c.execute("DELETE FROM event_watch WHERE id=?", (int(event_id),))


def manual_events(db_path: str | Path | None = None) -> list[dict]:
    with _conn(db_path) as c:
        rows = c.execute("SELECT id,date,scope,category,title,impact,severity FROM event_watch").fetchall()
    out = []
    for _id, d, scope, cat, title, impact, sev in rows:
        try:
            dd = _dt.date.fromisoformat(str(d)[:10])
        except ValueError:
            continue
        out.append({"id": _id, "date": dd, "scope": scope or "全市场", "category": cat or "特大事件",
                    "title": title, "watch": impact or "", "severity": sev or "高", "source": "manual"})
    return out


# ---------------------------------------------------------------------------
# 汇总：未来事件雷达
# ---------------------------------------------------------------------------
def upcoming(today: _dt.date, ticker: str | None = None, horizon_days: int = 45,
             earnings: list[dict] | None = None, db_path: str | Path | None = None) -> dict:
    """合并 自动 + 手填 事件，过滤到未来 horizon_days，按日期排序。

    ticker 非空时：保留 全市场 事件 + scope==ticker 的事件 + 该票财报(若传入 earnings)。
    返回 {events:[...], summary:str}。events 每项含 days_ahead。"""
    end = today + _dt.timedelta(days=horizon_days)
    ev = auto_market_events(today, horizon_days)
    for m in manual_events(db_path):
        if today <= m["date"] <= end:
            ev.append(m)
    if earnings:
        for e in earnings:
            d = e.get("date")
            if isinstance(d, str):
                try:
                    d = _dt.date.fromisoformat(d[:10])
                except ValueError:
                    continue
            if d and today <= d <= end:
                ev.append({"date": d, "scope": e.get("ticker", ticker or "?"), "category": "财报",
                           "severity": "高", "source": "auto",
                           "watch": "财报隔夜 gap 风险不对称 → 别满仓押方向；财报后按超预期/不及的历史漂移调仓。"})

    if ticker:
        ev = [e for e in ev if e["scope"] == "全市场" or e["scope"] == ticker]

    for e in ev:
        e["days_ahead"] = (e["date"] - today).days
    ev = sorted(ev, key=lambda e: (e["date"], -SEVERITY_ORDER.get(e["severity"], 0)))

    n_high = sum(1 for e in ev if e["severity"] == "高")
    if not ev:
        summary = f"未来 {horizon_days} 天内无登记的重大事件。"
    else:
        nearest = ev[0]
        summary = (f"未来 {horizon_days} 天内 {len(ev)} 项事件（{n_high} 项高风险）。"
                   f"最近：{nearest['date']}（{nearest['days_ahead']}天后）· {nearest['category']}"
                   f"{('·' + nearest['title']) if nearest.get('title') else ''}。"
                   "⚠️ 仅提醒、不入量化——这些是前视/独特事件，模型吃不进，请人工纳入仓位与风险判断。")
    return {"events": ev, "summary": summary, "n_high": n_high}

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
# 全网自动抓取的即将事件（IPO 日历 + 经济日历 + 新闻线索）—— 全部 display-only
# ---------------------------------------------------------------------------
def _parse_money(s: str) -> float:
    """'$86,249,999,880' → 86249999880.0；失败返回 0。"""
    try:
        return float(str(s).replace("$", "").replace(",", "").strip())
    except (ValueError, AttributeError):
        return 0.0


def fetch_ipo_calendar(today: _dt.date, horizon_days: int = 45) -> list[dict]:
    """NASDAQ IPO 日历：未来 horizon_days 内的 IPO，按募资额自动判严重度。

    巨型 IPO(募资额大)会从二级市场抽走资金 → 压制流动性/风险偏好，自动标高风险。免费、失败返回 []。"""
    from data.news import _http_get
    end = today + _dt.timedelta(days=horizon_days)
    months = {today.strftime("%Y-%m"), end.strftime("%Y-%m")}
    seen, out = set(), []
    for mo in months:
        try:
            r = _http_get(f"https://api.nasdaq.com/api/ipo/calendar?date={mo}", timeout=18)
            rows = ((r.json().get("data", {}) or {}).get("upcoming", {}) or {}).get("upcomingTable", {}) or {}
            rows = rows.get("rows") or []
        except Exception:  # noqa: BLE001
            continue
        for row in rows:
            dstr = row.get("expectedPriceDate") or ""
            try:
                d = _dt.datetime.strptime(dstr, "%m/%d/%Y").date()
            except ValueError:
                continue
            if not (today <= d <= end):
                continue
            usd = _parse_money(row.get("dollarValueOfSharesOffered"))
            sym = (row.get("proposedTickerSymbol") or "").strip()
            name = (row.get("companyName") or "").strip()
            key = (sym, dstr)
            if key in seen or usd < 1e9:   # 只报募资额 ≥ $1B 的（够大才算"大事"）
                continue
            seen.add(key)
            mega = usd >= 5e9
            bil = usd / 1e9
            out.append({
                "date": d, "scope": "全市场", "category": "大型IPO" + ("(巨型)" if mega else ""),
                "title": f"{name}({sym}) 上市 ${bil:.0f}B", "severity": "高" if mega else "中",
                "source": "web",
                "watch": (f"募资约 ${bil:.0f}B 的巨型 IPO → 从二级市场抽走巨量资金、短期压制流动性与风险偏好；"
                          "留意成交缩量/波动抬升，别在事件窗口满仓押方向。" if mega
                          else f"募资约 ${bil:.0f}B 的大型 IPO → 短期分流部分资金，留意板块资金面。")})
    return out


def fetch_econ_calendar(today: _dt.date, horizon_days: int = 14) -> list[dict]:
    """ForexFactory 经济日历(本周)：美国高影响宏观(CPI/PPI/零售/就业等)。免费、失败返回 []。"""
    import xml.etree.ElementTree as ET
    from data.news import _http_get
    end = today + _dt.timedelta(days=horizon_days)
    import re as _re
    root = None
    try:
        r = _http_get("https://nfs.faireconomy.media/ff_calendar_thisweek.xml", timeout=18)
        try:
            root = ET.fromstring(r.content)            # bytes：尊重编码声明
        except ET.ParseError:
            # 实时源偶发截断 → 去声明、恢复到最后一个完整 </event>、补根标签
            txt = _re.sub(r"<\?xml[^>]*\?>", "", r.content.decode("windows-1252", errors="ignore"))
            cut = txt.rfind("</event>")
            if cut > 0:
                root = ET.fromstring("<weeklyevents>" + txt[txt.find("<event>"):cut + len("</event>")] + "</weeklyevents>")
    except Exception:  # noqa: BLE001
        return []
    if root is None:
        return []
    out, seen = [], set()
    for e in root.findall(".//event"):
        if (e.findtext("impact") or "") != "High":
            continue
        if (e.findtext("country") or "") != "USD":   # 聚焦美股最相关的美国高影响数据
            continue
        try:
            d = _dt.datetime.strptime((e.findtext("date") or "").strip(), "%m-%d-%Y").date()
        except ValueError:
            continue
        if not (today <= d <= end):
            continue
        title = (e.findtext("title") or "").strip()
        key = (title, d)
        if key in seen:
            continue
        seen.add(key)
        out.append({"date": d, "scope": "全市场", "category": "宏观数据", "title": title,
                    "severity": "高", "source": "web",
                    "watch": f"美国高影响数据({title}) → 数据爆冷/爆热当日易跳空、放大全市场波动。"})
    return out


def fetch_event_news(ticker: str | None, limit: int = 6) -> list[dict]:
    """新闻里的**前瞻性事件线索**(并购/判决/发布/解禁/分拆等)。未核实、无精确日期 → 仅作线索，不进时间线。"""
    from data import news as _n
    if ticker:
        q = f'{_n.COMPANY_NAMES.get(ticker.upper(), ticker)} ({"merger OR acquisition OR ruling OR verdict OR launch OR lockup OR split OR IPO OR guidance"})'
    else:
        q = '(stock market) (IPO OR merger OR "rate decision" OR shutdown OR tariff OR ruling)'
    try:
        items = _n._google_news(q, limit=limit * 2)
    except Exception:  # noqa: BLE001
        return []
    _FWD = ("will", "to acquire", "to merge", "set to", "plans", "expected", "upcoming", "ruling",
            "verdict", "launch", "to report", "lockup", "split", "ipo", "debut", "guidance", "deadline")
    out = []
    for it in items:
        t = (it.get("title") or "")
        if any(w in t.lower() for w in _FWD):
            out.append({"date": it.get("date"), "title": t, "provider": it.get("provider"), "url": it.get("url")})
        if len(out) >= limit:
            break
    return out


def web_events(today: _dt.date, ticker: str | None = None, horizon_days: int = 45) -> list[dict]:
    """聚合全网自动抓取的结构化事件(IPO + 经济日历)。失败的源静默跳过。"""
    ev = fetch_ipo_calendar(today, horizon_days) + fetch_econ_calendar(today, min(horizon_days, 14))
    return ev


# ---------------------------------------------------------------------------
# 用户手填特大事件（SQLite）—— 作为自动抓取的补充/覆盖
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
             earnings: list[dict] | None = None, include_web: bool = True,
             db_path: str | Path | None = None) -> dict:
    """合并 规则日历 + 全网自动抓取(IPO/经济日历) + 手填 事件，过滤到未来 horizon_days，按日期排序。

    ticker 非空时：保留 全市场 事件 + scope==ticker 的事件 + 该票财报(若传入 earnings)。
    include_web=True 时联网抓 IPO/经济日历(失败静默跳过)。返回 {events, summary, news_leads}。"""
    end = today + _dt.timedelta(days=horizon_days)
    ev = auto_market_events(today, horizon_days)
    if include_web:
        ev += web_events(today, ticker, horizon_days)
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

    # 去重：同日期 + 标题/类别前缀相同（手填覆盖自动）
    dedup, seen = [], set()
    for e in sorted(ev, key=lambda x: 0 if x["source"] == "manual" else 1):
        k = (e["date"], (e.get("title") or e["category"])[:8])
        if k in seen:
            continue
        seen.add(k); dedup.append(e)
    ev = dedup

    for e in ev:
        e["days_ahead"] = (e["date"] - today).days
    ev = sorted(ev, key=lambda e: (e["date"], -SEVERITY_ORDER.get(e["severity"], 0)))

    n_high = sum(1 for e in ev if e["severity"] == "高")
    n_web = sum(1 for e in ev if e["source"] == "web")
    if not ev:
        summary = f"未来 {horizon_days} 天内未发现重大事件（已查 IPO/经济日历 + 规则日历）。"
    else:
        nearest = ev[0]
        summary = (f"未来 {horizon_days} 天内 {len(ev)} 项事件（{n_high} 项高风险，{n_web} 项全网自动抓取）。"
                   f"最近：{nearest['date']}（{nearest['days_ahead']}天后）· {nearest['category']}"
                   f"{('·' + nearest['title']) if nearest.get('title') else ''}。"
                   "⚠️ 仅提醒、不入量化——这些是前视/独特事件，模型吃不进，请人工纳入仓位与风险判断。")
    return {"events": ev, "summary": summary, "n_high": n_high, "n_web": n_web}

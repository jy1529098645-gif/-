"""事件雷达测试：自动日历事件 + 手填特大事件 + 汇总(display-only)。"""
from __future__ import annotations

import datetime as dt

import pytest

from analysis import event_radar as er


def test_weekday_helpers():
    # 2026-06：第一个周五=6/5，第三个周五=6/19
    assert er._first_friday(2026, 6) == dt.date(2026, 6, 5)
    assert er._third_friday(2026, 6) == dt.date(2026, 6, 19)


def test_auto_market_events_contains_fomc_and_opex():
    today = dt.date(2026, 6, 7)
    ev = er.auto_market_events(today, horizon_days=45)
    cats = {e["category"] for e in ev}
    # 6/17 FOMC、6/19 四巫日 都在 45 天内
    assert any("FOMC" in c for c in cats)
    assert any("四巫" in c or "OpEx" in c for c in cats)
    assert all("scope" in e and "watch" in e and "severity" in e for e in ev)


def test_add_and_query_manual_event(tmp_path):
    db = tmp_path / "t.db"
    er.add_event("2026-06-12", "SpaceX 上市", scope="全市场", category="大型IPO",
                 impact="抽走二级市场流动性、短期压制风险偏好", severity="高", db_path=db)
    m = er.manual_events(db)
    assert len(m) == 1 and m[0]["title"] == "SpaceX 上市"
    assert m[0]["date"] == dt.date(2026, 6, 12)


def test_upcoming_merges_and_sorts(tmp_path):
    db = tmp_path / "t.db"
    er.add_event("2026-06-12", "SpaceX 上市", scope="全市场", category="大型IPO",
                 impact="抽流动性", severity="高", db_path=db)
    today = dt.date(2026, 6, 7)
    res = er.upcoming(today, ticker="NVDA", horizon_days=45, include_web=False, db_path=db)
    evs = res["events"]
    assert len(evs) >= 2
    # 排序：日期升序
    dates = [e["date"] for e in evs]
    assert dates == sorted(dates)
    # days_ahead 正确
    assert all(e["days_ahead"] == (e["date"] - today).days for e in evs)
    # 手填事件在内
    assert any(e.get("title") == "SpaceX 上市" for e in evs)
    assert "不入量化" in res["summary"]


def test_upcoming_ticker_filter(tmp_path):
    db = tmp_path / "t.db"
    er.add_event("2026-06-10", "AAPL 反垄断判决", scope="AAPL", category="诉讼",
                 impact="单票事件", severity="高", db_path=db)
    today = dt.date(2026, 6, 7)
    # NVDA 视角不应看到 AAPL 专属事件
    nvda = er.upcoming(today, ticker="NVDA", horizon_days=30, include_web=False, db_path=db)
    assert not any(e.get("title") == "AAPL 反垄断判决" for e in nvda["events"])
    # AAPL 视角应看到
    aapl = er.upcoming(today, ticker="AAPL", horizon_days=30, include_web=False, db_path=db)
    assert any(e.get("title") == "AAPL 反垄断判决" for e in aapl["events"])


def test_parse_money():
    assert er._parse_money("$86,249,999,880") == 86249999880.0
    assert er._parse_money("$1,000,000,000") == 1e9
    assert er._parse_money("bad") == 0.0


def test_web_events_graceful_offline(monkeypatch):
    # 模拟网络失败：_http_get 抛错 → web_events 静默返回 []
    import data.news as _n
    def _boom(*a, **k):
        raise RuntimeError("offline")
    monkeypatch.setattr(_n, "_http_get", _boom)
    monkeypatch.setattr("data.news._google_news", lambda *a, **k: [])
    today = dt.date(2026, 6, 7)
    assert er.web_events(today, "NVDA", 45) == []


def test_delete_event(tmp_path):
    db = tmp_path / "t.db"
    er.add_event("2026-07-01", "测试", db_path=db)
    eid = er.manual_events(db)[0]["id"]
    er.delete_event(eid, db_path=db)
    assert er.manual_events(db) == []

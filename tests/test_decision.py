"""统一决策卡测试（离线·确定性状态机）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import decision as dc


def _px(level_from_high=0.0, n=400, trend_up=True):
    """构造一条价格，使当前距前高 = level_from_high(≤0)，并控制是否在200线上方。"""
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    base = np.linspace(100, 200, n) if trend_up else np.linspace(200, 120, n)
    s = pd.Series(base, index=idx)
    # 制造一个前高，再回落到目标回撤
    s.iloc[-60:] = s.iloc[-61] * (1 + np.linspace(0, level_from_high, 60))
    return s


def test_classify():
    assert dc.classify("SMH") == "semi_etf"
    assert dc.classify("NVDA") == "semi_single"
    assert dc.classify("SPY") == "index_etf"
    assert dc.classify("TQQQ") == "leveraged_etf"
    assert dc.classify("CRM") == "single"


def test_near_high_says_chase():
    c = dc.decision_card("AAPL", _px(-0.02), {"has_zone": False}, fragile_now=False)
    assert "追" in c["state"] or "持有" in c["state"]


def test_fragile_forces_defense():
    c = dc.decision_card("AAPL", _px(-0.03), {"has_zone": False}, fragile_now=True)
    assert "防守" in c["state"]
    assert any("降仓" in r or "宽度" in r for r in c["exit_rules"])


def test_index_etf_deep_is_high_conviction_buy():
    c = dc.decision_card("SPY", _px(-0.25), {"has_zone": False}, fragile_now=False)
    assert "建仓" in c["state"]
    assert "高置信" in c["state"] or "ETF" in c["action"]


def test_semi_mid_dip_warns_not_to_catch():
    c = dc.decision_card("SMH", _px(-0.13), {"has_zone": False}, fragile_now=False)
    assert "别" in c["action"] or "无edge" in c["action"] or "别急" in c["state"]


def test_single_deep_needs_conviction():
    c = dc.decision_card("CRM", _px(-0.25), {"has_zone": False}, fragile_now=False)
    assert "把握" in c["state"] or "价值陷阱" in c["action"]


def test_format_card_with_entry():
    be = {"has_zone": True, "anchor_price": 123.4, "zone_label": "距前高10%-15%",
          "horizon": 21, "tier": "稳健最佳入场区", "confident": True,
          "excess_median": 0.02, "dsr": 0.96, "price_band": [118.0, 128.0]}
    c = dc.decision_card("GOOGL", _px(-0.12), be, fragile_now=False)
    s = dc.format_card(c)
    assert "入场参考" in s and "123.4" in s


def test_leveraged_etf_decay_warning():
    c = dc.decision_card("TQQQ", _px(-0.05), {"has_zone": False}, fragile_now=False)
    assert any("衰减" in r or "杠杆" in r for r in c["exit_rules"])

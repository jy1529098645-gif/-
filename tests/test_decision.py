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


# 一个"有稳健入场区"的引擎结果：用于隔离测试纯回撤状态机，
# 避免触发"引擎未找到稳健入场区→转防守"的纪律覆盖（该覆盖另有专测）。
_ZONE_OK = {"has_zone": True, "anchor_price": 150.0, "confident": True,
            "tier": "稳健最佳入场区", "zone_label": "距前高-25%档", "horizon": 63,
            "price_band": [140.0, 160.0], "excess_median": 0.04}


def test_index_etf_deep_is_high_conviction_buy():
    c = dc.decision_card("SPY", _px(-0.25), _ZONE_OK, fragile_now=False)
    assert "建仓" in c["state"]
    assert "高置信" in c["state"] or "ETF" in c["action"]


def test_semi_mid_dip_warns_not_to_catch():
    c = dc.decision_card("SMH", _px(-0.13), _ZONE_OK, fragile_now=False)
    assert "别" in c["action"] or "无edge" in c["action"] or "别急" in c["state"]


def test_single_deep_needs_conviction():
    c = dc.decision_card("CRM", _px(-0.25), _ZONE_OK, fragile_now=False)
    assert "把握" in c["state"] or "价值陷阱" in c["action"]


def test_engine_override_momentum_trap_blocks_build():
    # 动量陷阱：即便在回撤建仓区，决策卡也必须转防守口径，与操作预案一致
    c = dc.decision_card("CRM", _px(-0.25), _ZONE_OK, fragile_now=False, momentum_trap=True)
    assert c["engine_override"] == "momentum_trap"
    assert "越跌越补" in c["action"] and c["posture"] == "caution"


def test_engine_override_grade_f_no_build():
    c = dc.decision_card("CRM", _px(-0.13), _ZONE_OK, fragile_now=False, grade={"grade": "F"})
    assert c["engine_override"] == "grade_F" and c["posture"] == "defend"


def test_engine_override_no_robust_zone_defers_to_defense():
    # best_entry 无稳健入场区(has_zone=False) → 决策卡不再硬给"建仓"
    c = dc.decision_card("SPY", _px(-0.25), {"has_zone": False}, fragile_now=False)
    assert c["engine_override"] == "no_robust_zone"
    assert "建仓" not in c["state"]


def test_fragile_threshold_aligned_with_wait_or_chase():
    # 脆弱 + dd 在 -10%~-15% 之间：旧阈值(-0.10)会落到"分批"，与市场环境横幅"降仓减半"打架；
    # 新阈值(-0.15)统一转防守。
    c = dc.decision_card("AAPL", _px(-0.12), _ZONE_OK, fragile_now=True)
    assert "防守" in c["state"] and c["posture"] == "defend"


def test_holding_advice_consistent_with_entry_posture():
    # 已建仓视角与建仓视角同源：动量陷阱时两者都不主张越跌越补
    c = dc.decision_card("CRM", _px(-0.12), _ZONE_OK, fragile_now=False, momentum_trap=True)
    h = dc.holding_advice(c, {"momentum_trap": True, "tranches": [
        {"tier": "浅", "price": 150, "target": 200, "stop": 140}]}, _ZONE_OK)
    assert "别补" in h["stance"] or "防守" in h["stance"]
    assert h["triggers"]  # 总给触发式止盈止损


def test_format_card_with_entry():
    be = {"has_zone": True, "anchor_price": 123.4, "zone_label": "距前高10%-15%",
          "horizon": 21, "tier": "稳健最佳入场区", "confident": True,
          "excess_median": 0.02, "dsr": 0.96, "price_band": [118.0, 128.0]}
    c = dc.decision_card("GOOGL", _px(-0.12), be, fragile_now=False)
    s = dc.format_card(c)
    assert "入场参考" in s and "123.4" in s


def test_etf_trend_break_says_half_position():
    # ETF 跌破200线 → 应给"减到半仓"(经验证的规则)，而非清仓
    px = pd.Series(np.linspace(200, 130, 400),
                   index=pd.date_range("2022-01-01", periods=400, freq="B"))  # 持续下行→破200线
    c = dc.decision_card("SPY", px, {"has_zone": False}, fragile_now=False)
    assert c["trend_broken"] is True
    assert any("半仓" in r for r in c["exit_rules"])


def test_single_stock_trend_break_not_forced_half():
    px = pd.Series(np.linspace(200, 130, 400),
                   index=pd.date_range("2022-01-01", periods=400, freq="B"))
    c = dc.decision_card("CRM", px, {"has_zone": False}, fragile_now=False)
    # 个股口径：诚实说减半仓只降回撤不升夏普
    assert any("个股" in r or "不升夏普" in r for r in c["exit_rules"])


def test_leveraged_etf_decay_warning():
    c = dc.decision_card("TQQQ", _px(-0.05), {"has_zone": False}, fragile_now=False)
    assert any("衰减" in r or "杠杆" in r for r in c["exit_rules"])


def test_exit_warning_levels():
    """撤离预警分级：跌破200线→红；高位拉伸→黄(防过热)；健康→绿。"""
    import numpy as np, pandas as pd
    from analysis import decision as dc
    idx = pd.date_range("2015-01-01", periods=900, freq="B")
    # 健康上行(现价稳在200线上方、不极端) → 绿
    up = pd.Series(np.linspace(100, 200, 900), index=idx)
    ew = dc.exit_warning(up, fragile_now=False, breadth_pctile=0.5)
    assert ew["red"] is False and ew["dist_ma200"] > 0
    # 跌破200线 → 红
    down = pd.Series(np.r_[np.linspace(100, 200, 700), np.linspace(200, 150, 200)], index=idx)
    ewd = dc.exit_warning(down, fragile_now=False, breadth_pctile=0.5)
    assert ewd["red"] is True and "撤离预警" in ewd["level"]
    # 宽度脆弱(市场) → 红，即使个股趋势健康
    ewf = dc.exit_warning(up, fragile_now=True, breadth_pctile=0.10)
    assert ewf["red"] is True
    # 宽度转弱(分位<25%但未脆弱) → 至少黄
    ewa = dc.exit_warning(up, fragile_now=False, breadth_pctile=0.20)
    assert ewa["amber"] is True or ewa["red"] is True
    # 字段完整
    for k in ("level", "color", "action", "signals", "dist_ma200", "vol_pctile", "overext_pctile"):
        assert k in ew

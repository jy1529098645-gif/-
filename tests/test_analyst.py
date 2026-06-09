"""专业分析师视角测试（离线·mock 基本面 + 合成价格）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import analyst as an


def _px(n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.018, n))), index=idx)


_INFO = {
    "trailingPE": 32.0, "forwardPE": 26.0, "pegRatio": 1.5, "priceToBook": 12.0,
    "revenueGrowth": 0.24, "earningsGrowth": 0.30, "grossMargins": 0.62,
    "operatingMargins": 0.35, "profitMargins": 0.28, "returnOnEquity": 0.45,
    "debtToEquity": 40.0, "currentRatio": 2.1, "freeCashflow": 8.5e9,
    "totalCash": 30e9, "totalDebt": 10e9, "beta": 1.25, "dividendYield": 0.005,
    "targetMeanPrice": 130.0, "targetHighPrice": 160.0, "targetLowPrice": 95.0,
    "recommendationKey": "buy", "numberOfAnalystOpinions": 42,
}


def test_report_structure():
    r = an.analyst_report("NVDA", _px(), _INFO, brief={}, horizon=63)
    for k in ("thesis", "valuation", "growth", "profitability", "health", "scenarios",
              "catalysts", "risks", "invalidations", "disclaimer"):
        assert k in r
    assert r["valuation"]["grade"] in ("贵", "偏贵", "合理", "便宜", "—")
    assert r["growth"]["grade"] == "高增长"          # revGrowth 24% > 20%
    assert r["profitability"]["grade"] == "高盈利"   # profitMargin 28% > 20%


def test_scenarios_ordered():
    r = an.analyst_report("X", _px(), _INFO, horizon=63)
    sc = r["scenarios"]
    assert sc["bear"] <= sc["base"] <= sc["bull"]
    assert sc["n"] > 0 and 0 <= sc["win_rate"] <= 1


def test_format_markdown_has_sections():
    r = an.analyst_report("AAPL", _px(), _INFO, brief={"next_earnings": "2026-07-25", "days_to_earnings": 30}, horizon=63)
    md = an.format_report(r)
    assert "估值" in md and "量化情景" in md and "失效条件" in md
    assert "不纳入量化" in md          # 必须有诚实标注
    assert "牛" in md and "熊" in md


def test_momentum_trap_defensive_thesis():
    r = an.analyst_report("INTC", _px(seed=3), _INFO, brief={"momentum_trap": True}, horizon=63)
    assert "动量陷阱" in r["thesis"] or any("动量陷阱" in x for x in r["risks"])


def test_target_vs_engine_flags_optimism():
    # 目标价远高于现价(+60%)，大概率超过引擎牛市情景 → 应标"偏乐观"
    info = dict(_INFO); info["targetMeanPrice"] = 200.0
    px = _px()
    px.iloc[-1] = 100.0
    r = an.analyst_report("X", px, info, horizon=63)
    tve = r.get("target_vs_engine")
    assert tve is not None and "implied_return" in tve
    md = an.format_report(r)
    assert "卖方 vs 引擎" in md


def test_etf_no_data_insufficient_spam():
    # ETF 只有 PE/PB/股息 → 应给 ETF 口径,不刷"数据不足"
    etf_info = {"trailingPE": 25.0, "priceToBook": 4.5, "dividendYield": 0.013,
                "targetMeanPrice": None}
    r = an.analyst_report("SPY", _px(), etf_info, brief={}, horizon=63)
    assert r["etf_like"] is True
    md = an.format_report(r)
    assert md.count("数据不足") == 0          # 不再刷数据不足
    assert "ETF" in md                        # 给 ETF 口径
    assert "量化情景" in md                    # 情景仍在(ETF的核心)


def test_empty_info_graceful():
    r = an.analyst_report("ZZZ", _px(), {}, brief={}, horizon=63)
    md = an.format_report(r)
    assert isinstance(md, str) and "分析师视角" in md

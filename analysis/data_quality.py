"""数据质量 / 新鲜度监控（Tier-3 运维）。

免费数据(yfinance)不保证可靠：可能停更、有缺口、未除权的异常跳空、整段缺失。
本模块体检价格数据，出"健康报告"，让你知道分析建立在什么质量的数据上。

铁律：只体检与提示，不改数据/不入量化。
"""
from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd


def _series_health(ticker: str, s: pd.Series, today: _dt.date, jump_thr: float = 0.35) -> dict:
    """单票价格序列体检：新鲜度、缺口、异常跳空、样本量。"""
    s = s.dropna()
    if s.empty:
        return {"标的": ticker, "状态": "❌ 无数据", "最新日期": "—", "滞后(天)": None,
                "样本天数": 0, "缺口(>5日)": None, "异常跳空": None, "最新价": None}
    last = s.index[-1].date()
    lag = (today - last).days
    n = len(s)
    # 交易日缺口：相邻自然日差 > 5（约一周无数据）的次数
    gaps = (s.index.to_series().diff().dt.days > 5).sum()
    # 异常跳空：单日涨跌幅 > jump_thr（多为未除权/数据错误，真实暴动也可能，故只提示）
    ret = s.pct_change().abs()
    jumps = int((ret > jump_thr).sum())
    # 状态判定
    if lag <= 4:
        status = "✅ 新鲜"
    elif lag <= 10:
        status = "🟡 略滞后"
    else:
        status = "🔴 陈旧"
    if jumps > 0 or gaps > 3:
        status += " ⚠️"
    return {"标的": ticker, "状态": status, "最新日期": last.isoformat(), "滞后(天)": int(lag),
            "样本天数": int(n), "缺口(>5日)": int(gaps), "异常跳空": jumps,
            "最新价": round(float(s.iloc[-1]), 2)}


def data_health(prices: pd.DataFrame, today: _dt.date | None = None) -> dict:
    """对价格面板逐票体检。返回 {table: DataFrame, summary: str}。"""
    today = today or _dt.date.today()
    rows = [_series_health(c, prices[c], today) for c in prices.columns]
    df = pd.DataFrame(rows)
    n = len(df)
    fresh = int((df["状态"].str.contains("✅")).sum()) if n else 0
    stale = int((df["状态"].str.contains("🔴")).sum()) if n else 0
    flagged = int((df["状态"].str.contains("⚠️")).sum()) if n else 0
    summary = (f"体检 {n} 只：{fresh} 新鲜 / {stale} 陈旧 / {flagged} 有异常标记(缺口或跳空)。"
               + ("⚠️ 陈旧或异常的数据会让分析失真，留意上方标记。" if (stale or flagged) else "数据整体健康。"))
    return {"table": df, "summary": summary, "n_fresh": fresh, "n_stale": stale, "n_flagged": flagged}

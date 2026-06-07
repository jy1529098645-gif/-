"""基本面因子（补充规格 B）。

⚠️ 总则：免费 yfinance 的**财务报表快照**带前视/重述偏差，不可直接用于历史回测；
   SUE / 估值分位 / 质量趋势（F2/F3）需 point-in-time 付费数据源（Sharadar/FMP/Tiingo）。

**本文件当前只实现 F1：财报日历因子**——这条**完全无前视**（财报日期提前排期、Surprise 公布后才可知），
免费数据即可干净计算，是 B 轨的低成本起点。

工作流铁律（B）：假设先行（下方每个因子写明可证伪假设）→ PIT 构造 → 只测对**未来**收益的领先预测力
（见 evaluation/earnings_eval.py）→ 多重检验校正。禁止全指标相关性扫描。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _reported(edates: pd.DataFrame) -> pd.DataFrame:
    """只保留已公布（有 Surprise(%)）的财报行。"""
    return edates.dropna(subset=["Surprise(%)"]).sort_index()


def days_to_next_earnings(index: pd.DatetimeIndex, edates: pd.DataFrame) -> pd.Series:
    """距下次财报的自然日数（含未来已排期）。

    假设（可证伪）：临近财报，不确定性上升、知情交易者调仓 → 财报前收益/波动有结构性差异。
    无前视：财报日期提前数周公布。
    """
    ed = np.sort(edates.index.values.astype("datetime64[ns]"))
    idx = index.values.astype("datetime64[ns]")
    pos = np.searchsorted(ed, idx, side="left")  # 第一个 >= t 的财报
    out = np.full(len(idx), np.nan)
    valid = pos < len(ed)
    out[valid] = (ed[pos[valid]] - idx[valid]) / np.timedelta64(1, "D")
    return pd.Series(out, index=index, name="days_to_next_earnings")


def days_since_last_earnings(index: pd.DatetimeIndex, edates: pd.DataFrame) -> pd.Series:
    """距上次财报的自然日数。

    假设：财报后漂移（PEAD）窗口内，刚公布的标的收益结构不同于平静期。
    """
    ed = np.sort(edates.index.values.astype("datetime64[ns]"))
    idx = index.values.astype("datetime64[ns]")
    pos = np.searchsorted(ed, idx, side="right") - 1  # 最后一个 <= t 的财报
    out = np.full(len(idx), np.nan)
    valid = pos >= 0
    out[valid] = (idx[valid] - ed[pos[valid]]) / np.timedelta64(1, "D")
    return pd.Series(out, index=index, name="days_since_last_earnings")


def last_surprise(index: pd.DatetimeIndex, edates: pd.DataFrame) -> pd.Series:
    """最近一次**已公布**财报的 Surprise(%)，PIT（财报盘后公布，故只在公布日之后可知）。

    假设（PEAD）：盈利超预期后，股价在公布后一段时间继续同向漂移 → 正 surprise 领先正远期收益。
    """
    rep = _reported(edates)
    ed = rep.index.values.astype("datetime64[ns]")
    sv = rep["Surprise(%)"].to_numpy(dtype=float)
    idx = index.values.astype("datetime64[ns]")
    pos = np.searchsorted(ed, idx, side="left") - 1  # 严格早于 t 的最近财报（盘后公布→次日才知）
    out = np.full(len(idx), np.nan)
    valid = pos >= 0
    out[valid] = sv[pos[valid]]
    return pd.Series(out, index=index, name="last_surprise")


def last_beat(index: pd.DatetimeIndex, edates: pd.DataFrame) -> pd.Series:
    """上次是否超预期（last_surprise > 0），布尔，PIT。"""
    s = last_surprise(index, edates)
    return (s > 0).where(s.notna()).astype("boolean")


def pre_earnings_window(index: pd.DatetimeIndex, edates: pd.DataFrame, window: int = 10) -> pd.Series:
    """是否处于下次财报前 window 自然日内（布尔，无前视）。"""
    d = days_to_next_earnings(index, edates)
    return ((d > 0) & (d <= window)).where(d.notna()).astype("boolean")


def post_earnings_window(index: pd.DatetimeIndex, edates: pd.DataFrame, window: int = 10) -> pd.Series:
    """是否处于上次财报后 window 自然日内（布尔，PEAD 窗口）。"""
    d = days_since_last_earnings(index, edates)
    return ((d >= 0) & (d < window)).where(d.notna()).astype("boolean")


REGISTRY = {
    "days_to_next_earnings": days_to_next_earnings,
    "days_since_last_earnings": days_since_last_earnings,
    "last_surprise": last_surprise,
    "last_beat": last_beat,
}

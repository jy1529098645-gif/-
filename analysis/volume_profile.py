"""Volume Profile（按价位的成交量，日线近似）。

用日线 high-low 把当日成交量**均匀摊**到价格区间，堆叠出成交密集区与 POC。
免费日线只能粗近似（够看大级别筹码密集区），**不作主信号**，仅可视化 + regime 辅助。
日内精细版需付费分钟数据（已砍）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def volume_profile(ohlcv: pd.DataFrame, bins: int = 50, lookback: int | None = None) -> dict:
    """返回 {centers, volumes, poc, value_area}：价位中心、各价位成交量、POC、70% 价值区间。"""
    df = ohlcv.dropna(subset=["high", "low", "volume"])
    if lookback:
        df = df.tail(lookback)
    if df.empty:
        raise ValueError("OHLCV 为空")

    lo = float(df["low"].min())
    hi = float(df["high"].max())
    if hi <= lo:
        hi = lo * 1.001 + 1e-9
    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    vol = np.zeros(bins)

    low = df["low"].to_numpy(float)
    high = df["high"].to_numpy(float)
    v = df["volume"].to_numpy(float)
    e_lo, e_hi = edges[:-1], edges[1:]
    for l, h, vv in zip(low, high, v):
        if h <= l:
            j = min(bins - 1, max(0, int(np.searchsorted(edges, l) - 1)))
            vol[j] += vv
            continue
        overlap = np.clip(np.minimum(e_hi, h) - np.maximum(e_lo, l), 0, None)
        span = overlap.sum()
        if span > 0:
            vol += vv * overlap / span

    poc = float(centers[int(np.argmax(vol))])

    # 价值区间（70% 成交量，从 POC 向两侧扩展）
    order = np.argsort(vol)[::-1]
    cum = np.cumsum(vol[order])
    total = vol.sum()
    keep = order[: max(1, int(np.searchsorted(cum, 0.7 * total) + 1))] if total > 0 else order[:1]
    va_low, va_high = float(centers[keep].min()), float(centers[keep].max())

    return {"centers": centers, "volumes": vol, "poc": poc, "value_area": (va_low, va_high)}

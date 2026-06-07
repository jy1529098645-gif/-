"""入场信号（补充规格 A，纯价格，可组合）。

每个信号是纯函数：输入单只标的价格 Series → 输出**布尔** Series（True=该日满足入场条件）。
可用 combine_and / combine_or 组合；组合数应计入多重检验计数（见 rule_eval / deflated Sharpe）。

定位红线：这些不是"最佳买点"，只是把入场从感觉变成可回测、可统计的规则。
"""
from __future__ import annotations

import pandas as pd

import config

_CFG = config.load_config()
_E = _CFG["single_name"]["entry"]


def dip_from_high(price: pd.Series, lookback: int | None = None, pct: float | None = None) -> pd.Series:
    """距 lookback 日内高点回撤 ≥ pct（状态型：处于回撤中即 True）。"""
    lookback = lookback or _E["dip_from_high"]["lookback"]
    pct = pct if pct is not None else _E["dip_from_high"]["pct"]
    high = price.rolling(lookback, min_periods=lookback // 2).max()
    return (price / high - 1.0) <= -pct


def ma_cross(price: pd.Series, fast: int | None = None, slow: int | None = None) -> pd.Series:
    """金叉事件：快线上穿慢线那一天 True（事件型，仅穿越当日）。"""
    fast = fast or _E["ma_cross"]["fast"]
    slow = slow or _E["ma_cross"]["slow"]
    ma_f = price.rolling(fast, min_periods=fast // 2).mean()
    ma_s = price.rolling(slow, min_periods=slow // 2).mean()
    above = ma_f > ma_s
    return above & ~above.shift(1, fill_value=False)


def rsi(price: pd.Series, window: int = 14) -> pd.Series:
    """Wilder RSI。"""
    delta = price.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    import numpy as np

    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(avg_loss != 0, 100.0)              # 无下跌 → RSI=100
    out = out.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)  # 完全走平 → 50
    return out.astype(float)


def rsi_oversold(price: pd.Series, window: int | None = None, level: float | None = None) -> pd.Series:
    """RSI 超卖（状态型：RSI < level 即 True）。"""
    window = window or _E["rsi_oversold"]["window"]
    level = level if level is not None else _E["rsi_oversold"]["level"]
    return rsi(price, window) < level


def vol_regime(price: pd.Series, window: int = 20, low: bool = True, ref: int = 252) -> pd.Series:
    """波动率状态过滤：近 window 日已实现波动率相对其 ref 日中位的高/低。

    low=True → 低波动状态（vol < 滚动中位）；low=False → 高波动状态。
    """
    rv = price.pct_change().rolling(window, min_periods=window // 2).std()
    med = rv.rolling(ref, min_periods=ref // 2).median()
    return (rv < med) if low else (rv > med)


def combine_and(*signals: pd.Series) -> pd.Series:
    out = signals[0].astype("boolean")
    for s in signals[1:]:
        out = out & s.astype("boolean")
    return out.fillna(False).astype(bool)


def combine_or(*signals: pd.Series) -> pd.Series:
    out = signals[0].astype("boolean")
    for s in signals[1:]:
        out = out | s.astype("boolean")
    return out.fillna(False).astype(bool)


# 入场信号注册表（组合时用于计数/命名）
REGISTRY = {
    "dip_from_high": dip_from_high,
    "ma_cross": ma_cross,
    "rsi_oversold": rsi_oversold,
    "vol_regime": vol_regime,
}


def make_single(name: str, p1: float, p2: float = 0.0):
    """按 (名称, 参数1, 参数2) 构造单个入场信号函数 price->bool。"""
    if name == "dip_from_high":
        return lambda px: dip_from_high(px, lookback=252, pct=p1)
    if name == "rsi_oversold":
        return lambda px: rsi_oversold(px, window=int(p1), level=p2)
    if name == "ma_cross":
        return lambda px: ma_cross(px, fast=int(p1), slow=int(p2))
    if name == "vol_regime":
        return lambda px: vol_regime(px, window=int(p1), low=(p2 >= 0.5))
    raise ValueError(f"未知信号 {name}")


def build_entry(specs: list[tuple], op: str = "and"):
    """把多个 (名称,p1,p2) 信号按 AND/OR 组合成一个入场函数。

    specs：[(name, p1, p2), ...]；op："and" 或 "or"。组合数（len(specs)）应计入多重检验计数。
    """
    funcs = [make_single(n, a, b) for (n, a, b) in specs]

    def entry(price):
        sigs = [f(price) for f in funcs]
        if len(sigs) == 1:
            return sigs[0].reindex(price.index).fillna(False).astype(bool)
        return combine_and(*sigs) if op == "and" else combine_or(*sigs)

    return entry

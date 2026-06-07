"""免费可观测的市场状态（补充产品总纲 U2）。

全部用价格即可算、无前视：实现波动率分位、趋势位置、回撤状态、七姐妹相互相关。
用途：(a) 作个股规则的「条件门」做分条件池化评估；(b) 今日状态面板。

铁律：只作条件/展示，不作单一信号；任何条件结果仍要对比无条件基准。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def realized_vol(price: pd.Series, window: int = 21) -> pd.Series:
    """年化已实现波动率（近 window 日）。"""
    return price.pct_change().rolling(window, min_periods=window // 2).std() * np.sqrt(252)


def realized_vol_percentile(price: pd.Series, window: int = 21, ref: int = 252) -> pd.Series:
    """已实现波动率在过去 ref 日的滚动百分位（0–1），无前视。"""
    rv = realized_vol(price, window)
    return rv.rolling(ref, min_periods=ref // 2).apply(
        lambda x: (x[-1] >= x).mean(), raw=True
    )


def trend_position(price: pd.Series, ma: int = 200) -> pd.Series:
    """价格相对 ma 日均线的偏离（>0 在均线上方=上升趋势）。"""
    m = price.rolling(ma, min_periods=ma // 2).mean()
    return price / m - 1.0


def drawdown(price: pd.Series) -> pd.Series:
    """距历史前高的回撤（≤0）。"""
    return price / price.cummax() - 1.0


def mutual_correlation(prices: pd.DataFrame, window: int = 63) -> pd.Series:
    """一篮子标的的滚动平均两两相关（高相关=系统性 beta 主导，分散无效）。"""
    rets = prices.pct_change()
    out = {}
    idx = rets.index
    cols = rets.columns
    k = len(cols)
    arr = rets.to_numpy()
    for i in range(window, len(idx)):
        win = arr[i - window:i]
        c = np.corrcoef(win, rowvar=False)
        off = c[~np.eye(k, dtype=bool)]
        out[idx[i]] = float(np.nanmean(off))
    return pd.Series(out, name="avg_corr")


# ---------------------------------------------------------------------------
# 状态分桶（供条件门）
# ---------------------------------------------------------------------------
def vol_state(price: pd.Series, window: int = 21, ref: int = 252) -> pd.Series:
    """低/中/高波动三分位（按滚动百分位）。"""
    pct = realized_vol_percentile(price, window, ref)
    return pd.Series(
        np.where(pct <= 1 / 3, "low_vol", np.where(pct >= 2 / 3, "high_vol", "mid_vol")),
        index=price.index,
    ).where(pct.notna())


def trend_state(price: pd.Series, ma: int = 200) -> pd.Series:
    """趋势上/下（相对均线）。"""
    tp = trend_position(price, ma)
    return pd.Series(np.where(tp > 0, "up_trend", "down_trend"), index=price.index).where(tp.notna())


def drawdown_state(price: pd.Series, threshold: float = 0.10) -> pd.Series:
    """是否处于 >threshold 回撤。"""
    dd = drawdown(price)
    return pd.Series(np.where(dd < -threshold, "in_drawdown", "near_high"), index=price.index)


def today_panel(price: pd.Series) -> dict:
    """今日状态快照：当前值 + 历史百分位（供人对照，不给结论）。"""
    price = price.dropna()
    rv = realized_vol(price, 21)
    tp = trend_position(price, 200)
    dd = drawdown(price)
    rv_pct = realized_vol_percentile(price, 21, 252)
    return {
        "date": price.index[-1],
        "realized_vol": float(rv.iloc[-1]),
        "vol_percentile": float(rv_pct.iloc[-1]) if rv_pct.notna().iloc[-1] else float("nan"),
        "vol_state": str(vol_state(price).iloc[-1]),
        "trend_position": float(tp.iloc[-1]),
        "trend_state": str(trend_state(price).iloc[-1]),
        "drawdown": float(dd.iloc[-1]),
        "drawdown_state": str(drawdown_state(price).iloc[-1]),
    }

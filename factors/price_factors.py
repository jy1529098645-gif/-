"""价格类因子（只用价格，无前视偏差）。

每个因子是纯函数：输入价格面板（date × ticker）→ 输出同形状的因子值面板（截面可比）。
所有因子只用截至 t 时刻的价格，不引入未来信息。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def momentum_12_1(prices: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """经典动量：过去 lookback 日收益，剔除最近 skip 日。

    factor_t = P_{t-skip} / P_{t-lookback} - 1
    """
    return prices.shift(skip) / prices.shift(lookback) - 1.0


def short_reversal(prices: pd.DataFrame, lookback: int = 21) -> pd.DataFrame:
    """短期反转：过去 lookback 日收益的负值。

    factor_t = -(P_t / P_{t-lookback} - 1)
    """
    return -(prices / prices.shift(lookback) - 1.0)


def low_volatility(prices: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """低波动异象：过去 window 日收益波动率的负值（波动越低因子越高）。

    factor_t = -std(daily_returns over last `window` days)
    """
    rets = prices.pct_change()
    vol = rets.rolling(window, min_periods=max(2, window // 2)).std()
    return -vol


def trend(prices: pd.DataFrame, ma_window: int = 200) -> pd.DataFrame:
    """趋势/regime：价格相对 ma_window 日均线的位置。

    factor_t = P_t / MA_{ma_window}(P) - 1   （>0 在均线上方）
    """
    ma = prices.rolling(ma_window, min_periods=max(2, ma_window // 2)).mean()
    return prices / ma - 1.0


def random_factor(prices: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """随机因子（健全性检查用）：与价格同形状的纯噪音，IC 应≈0。"""
    rng = np.random.default_rng(seed)
    vals = rng.standard_normal(size=prices.shape)
    out = pd.DataFrame(vals, index=prices.index, columns=prices.columns)
    return out.where(prices.notna())  # 价格缺失处也置 NaN，保持可比口径


def blend(panels: dict[str, pd.DataFrame], weights: dict[str, float] | None = None) -> pd.DataFrame:
    """多因子组合（Phase 6 免费版）：各因子按日截面 z-score 后加权求和。

    所有因子约定"越高越好"（动量正、反转/低波已取负、趋势正），故等权 z-score 求和合理。
    panels：{因子名: 因子值面板}；weights：{因子名: 权重}（缺省等权）。

    校验（加固）：panels 不能为空；权重不能为负（负权=偷偷反转因子，会让"越高越好"约定失效）；
    截面标准差为 0（当日所有票因子值相同）→ 置 NaN 而非 0 除，避免 inf 污染。
    """
    if not panels:
        raise ValueError("blend: panels 为空，至少需要一个因子面板")
    weights = weights or {}
    bad = {k: v for k, v in weights.items() if v is not None and v < 0}
    if bad:
        raise ValueError(f"blend: 权重不可为负（因子已统一为'越高越好'，负权请改在因子侧取负）：{bad}")
    if weights and sum(v for v in weights.values() if v is not None) <= 0:
        raise ValueError("blend: 权重之和必须为正")

    total = None
    for name, panel in panels.items():
        mu = panel.mean(axis=1)
        sd = panel.std(axis=1).replace(0.0, np.nan)
        z = panel.sub(mu, axis=0).div(sd, axis=0)
        w = weights.get(name, 1.0)
        total = z * w if total is None else total.add(z * w, fill_value=0.0)
    return total


# 因子注册表：名称 → 函数
REGISTRY = {
    "momentum_12_1": momentum_12_1,
    "short_reversal": short_reversal,
    "low_volatility": low_volatility,
    "trend": trend,
    "random_factor": random_factor,
}

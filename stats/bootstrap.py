"""块自助法（circular block bootstrap）置信区间。

为什么要块：收益序列有时间相关性（波动聚集、动量），逐点重采样会破坏相关结构、
虚增有效样本量、从而虚增显著性。按「块」重采样保留局部时间结构，给出诚实的 CI。
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def block_bootstrap_ci(
    returns: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    block_size: int,
    n: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """对时间序列统计量给出 circular block bootstrap 置信区间。

    参数
    ----
    returns : 1D 收益序列
    stat_fn : 作用于 1D 序列、返回标量的统计量函数（如 np.mean、夏普）
    block_size : 块长（应覆盖主要自相关尺度，如 21 ≈ 1 个月）
    n : 重采样次数
    ci : 置信水平（0.95 → 返回 2.5/97.5 分位）
    seed : 随机种子，保证可复现

    返回
    ----
    (point_estimate, lower, upper)
    """
    x = np.asarray(returns, dtype=float)
    x = x[~np.isnan(x)]
    L = len(x)
    if L == 0:
        raise ValueError("returns 为空")
    if block_size < 1 or block_size > L:
        raise ValueError(f"block_size 需在 [1, {L}]，收到 {block_size}")

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(L / block_size))

    # 预生成所有块起点：(n, n_blocks)
    starts = rng.integers(0, L, size=(n, n_blocks))
    offsets = np.arange(block_size)
    stats = np.empty(n, dtype=float)
    for i in range(n):
        idx = (starts[i][:, None] + offsets[None, :]).ravel() % L  # 循环块
        sample = x[idx[:L]]  # 拼接后裁到原长
        stats[i] = stat_fn(sample)

    point = float(stat_fn(x))
    alpha = (1.0 - ci) / 2.0
    lower = float(np.nanpercentile(stats, 100 * alpha))
    upper = float(np.nanpercentile(stats, 100 * (1 - alpha)))
    return point, lower, upper


def sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """年化夏普（无风险利率视为 0），供 bootstrap 的 stat_fn 使用。"""
    x = np.asarray(returns, dtype=float)
    x = x[~np.isnan(x)]
    sd = x.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(x.mean() / sd * np.sqrt(periods_per_year))

"""探索性信号研究：哪些状态下远期收益/回撤明显偏离基准（建仓 vs 止盈）。

诚实声明：这是多信号探索（有多重检验风险）。每个信号都对比该股**无条件基准**，
报独立事件数 N（连续在状态内算一次），小 N 一律不可信。结论仅供后续假设，不是择时圣杯。
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import loader  # noqa: E402
from factors import signals as sg  # noqa: E402
from regime.conditional_returns import _path_min_return, _count_independent_events  # noqa: E402

T = ["GOOGL", "MSFT", "NVDA"]
START, END = "2014-01-01", "2026-06-07"
H = 63          # 远期收益窗口（约一季）
DD_H = 21       # 止盈风险：未来 21 日途中最大浮亏


def states(price: pd.Series) -> pd.DataFrame:
    p = price.dropna()
    ma50 = p.rolling(50, min_periods=25).mean()
    ma200 = p.rolling(200, min_periods=100).mean()
    dist200 = p / ma200 - 1.0
    high20 = p.rolling(20, min_periods=10).max()
    dip20 = p / high20 - 1.0
    rsi = sg.rsi(p, 14)
    high252 = p.rolling(252, min_periods=120).max()
    rv = p.pct_change().rolling(21, min_periods=10).std()
    rv_pct = rv.rolling(252, min_periods=120).apply(lambda x: (x[-1] >= x).mean(), raw=True)
    # 估值代理：价格相对 200 日线偏离的历史分位（越高越"拉伸/贵"）
    stretch_pct = dist200.rolling(504, min_periods=200).apply(lambda x: (x[-1] >= x).mean(), raw=True)

    return pd.DataFrame({
        "above200": dist200 > 0,
        "below200": dist200 <= 0,
        "dip_in_uptrend": (dist200 > 0) & (dip20 <= -0.08),
        "rsi_oversold": rsi < 35,
        "rsi_overbought": rsi > 70,
        "breakout_52w": p >= high252.shift(1) * 0.999,
        "stretched_top20": stretch_pct >= 0.80,    # 极度拉伸（止盈候选）
        "high_vol": rv_pct >= 0.70,
    }, index=p.index)


def run():
    prices = {t: loader.load_prices([t], START, END)[t].dropna() for t in T}
    # 预先算每股 fwd 收益 / 远期浮亏 / 基准
    fwd, ddown, base_med = {}, {}, {}
    for t, p in prices.items():
        fwd[t] = p.shift(-H) / p - 1.0
        ddown[t] = _path_min_return(p, DD_H)
        base_med[t] = float(fwd[t].dropna().median())

    signals = ["above200", "below200", "dip_in_uptrend", "rsi_oversold", "rsi_overbought",
               "breakout_52w", "stretched_top20", "high_vol"]
    st = {t: states(prices[t]) for t in T}

    print(f"研究区间 {START}~{END}，远期收益={H}日，风险浮亏窗口={DD_H}日")
    print("各股无条件基准（{}日中位收益）: ".format(H) +
          " ".join(f"{t} {base_med[t]:+.1%}" for t in T))
    print("=" * 96)
    print(f"{'信号':<16}{'池N(事件)':>9}{'池中位':>8}{'胜率':>7}{'超额*':>8}{'未来21d浮亏中位':>14}   单股中位(G/M/N)")
    print("-" * 96)

    pooled_rows = []
    for signame in signals:
        all_fwd, all_dd, n_events = [], [], 0
        per_stock_med = []
        excess_list = []
        for t in T:
            mask = st[t][signame].reindex(prices[t].index).fillna(False)
            f = fwd[t][mask].dropna()
            d = ddown[t][mask].dropna()
            all_fwd.append(f); all_dd.append(d)
            n_events += _count_independent_events(mask)
            m = float(f.median()) if len(f) else float("nan")
            per_stock_med.append(m)
            if len(f):
                excess_list.append(m - base_med[t])
        F = pd.concat(all_fwd) if all_fwd else pd.Series(dtype=float)
        D = pd.concat(all_dd) if all_dd else pd.Series(dtype=float)
        if len(F) < 20:
            continue
        pool_med = float(F.median())
        winrate = float((F > 0).mean())
        excess = float(np.nanmean(excess_list))   # 平均（各股相对自身基准）超额
        dd_med = float(D.median())
        ps = "/".join(f"{m:+.0%}" for m in per_stock_med)
        print(f"{signame:<16}{n_events:>9}{pool_med:>8.1%}{winrate:>7.0%}{excess:>8.1%}{dd_med:>14.1%}   {ps}")
        pooled_rows.append((signame, n_events, pool_med, winrate, excess, dd_med))

    print("-" * 96)
    print("*超额 = 该信号下中位远期收益 − 该股无条件基准中位（各股平均）。正=看涨倾斜，负=看跌/止盈倾斜。")
    print("⚠️ 探索性、多信号：小 N、跨0即不可信；真正采用前须 walk-forward + deflated Sharpe。")


if __name__ == "__main__":
    run()

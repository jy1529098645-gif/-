"""风险管理叠加测试（离线）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import overlay as ov


def _trending(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-01-01", periods=n, freq="B")
    # 牛市+两段回撤
    r = rng.normal(0.0005, 0.012, n)
    r[400:460] = rng.normal(-0.01, 0.03, 60)   # 急跌段(高波动)
    r[800:840] = rng.normal(-0.012, 0.035, 40)
    return pd.Series(100 * np.exp(np.cumsum(r)), index=idx)


def test_position_in_bounds():
    px = _trending()
    pos = ov.risk_managed_position(px)
    assert ((pos.dropna() >= 0) & (pos.dropna() <= 1)).all()


def test_position_drops_in_high_vol():
    px = _trending()
    pos = ov.risk_managed_position(px)
    # 急跌高波动段(400-460)的平均仓位应低于平静段
    calm = pos.iloc[200:380].mean()
    crash = pos.iloc[410:460].mean()
    assert crash < calm


def test_backtest_structure_and_drawdown_reduced():
    px = _trending()
    bt = ov.backtest_overlay(px)
    assert set(["equity", "strategy", "hold", "current_position"]).issubset(bt)
    # 叠加应降低回撤(更小的负数=更浅)
    assert bt["strategy"]["maxdd"] >= bt["hold"]["maxdd"]
    assert 0 <= bt["current_position"] <= 1


def test_verdict_string():
    bt = ov.backtest_overlay(_trending())
    v = ov.verdict(bt)
    assert "风险管理叠加" in v and "夏普" in v


def test_stats_has_sortino_calmar():
    bt = ov.backtest_overlay(_trending())
    for k in ("cagr", "vol", "sharpe", "sortino", "maxdd", "calmar"):
        assert k in bt["strategy"]


def test_backtest_portfolio():
    prices = {f"A{i}": _trending(seed=i) for i in range(4)}
    bm = _trending(seed=99)
    bt = ov.backtest_portfolio(prices, benchmark=bm)
    assert bt["available"] and bt["n_assets"] == 4
    assert "overlay" in bt and "hold" in bt and "benchmark" in bt
    assert not bt["equity"].empty
    # 组合叠加回撤应不深于持有(更小的负数)
    assert bt["overlay"]["maxdd"] >= bt["hold"]["maxdd"]


def test_sector_effectiveness():
    assert "能源" in ov.sector_effectiveness("XOM")
    assert "防御" in ov.sector_effectiveness("KO")
    assert "升夏普" in ov.sector_effectiveness("NVDA")
    assert "升夏普" in ov.sector_effectiveness("XLK")


def test_param_insensitive_runs():
    px = _trending()
    for tv in (0.12, 0.15, 0.20):
        for w in (0.4, 0.5, 0.6):
            bt = ov.backtest_overlay(px, target_vol=tv, blend=w)
            assert bt["strategy"]["sharpe"] == bt["strategy"]["sharpe"]  # not nan

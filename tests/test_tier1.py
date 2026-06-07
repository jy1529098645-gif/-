"""Tier-1 专业级升级测试：LW收缩 / EWMA波动 / 多因子归因 / 横截面中性化+NW / FDR账本 / sp500。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis import quant_edge as qe
from analysis import mt_ledger as mt


def _px(seed, n=600, vol=0.01, drift=0.0003):
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    return pd.Series(np.cumprod(1 + rng.randn(n) * vol + drift) * 100, index=idx)


# ---------------------------------------------------------------------------
# Ledoit-Wolf 收缩
# ---------------------------------------------------------------------------
def test_portfolio_uses_ledoit_wolf():
    prices = pd.DataFrame({f"T{i}": _px(i, vol=0.006 + 0.004 * i) for i in range(5)})
    res = qe.portfolio_weights(prices, method="min_var", single_cap=1.0)
    assert res["shrinkage"] == "Ledoit-Wolf"
    # 收缩后最小方差仍应 ≤ 等权
    assert res["compare"]["最小方差"] <= res["compare"]["等权"] + 1e-9
    assert abs(sum(res["weights"].values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# EWMA 波动（无前视 + 形状）
# ---------------------------------------------------------------------------
def test_ewma_vol_no_lookahead():
    r = _px(1).pct_change()
    ev = qe.ewma_vol(r)
    # 用了 shift(1)：首值 NaN（无前视）；起点 r²=0 那点为 0，之后应全为正
    assert ev.notna().sum() > 100
    assert pd.isna(ev.iloc[0])           # 无前视：第一天无预测
    assert (ev.dropna().iloc[5:] > 0).all()


def test_vol_target_uses_ewma_runs():
    px = _px(2)
    res = qe.vol_target_backtest(px, target_vol=0.15)
    assert res["avg_exposure"] <= 1.0
    assert "cagr" in res["overlay"]


# ---------------------------------------------------------------------------
# 多因子归因
# ---------------------------------------------------------------------------
def test_factor_attribution_pure_market():
    spy = _px(3)
    spy_r = spy.pct_change()
    strat = spy_r * 1.0  # 纯市场 beta=1，无风格、无 alpha
    fp = {"SPY": spy, "MTUM": _px(4), "IWD": _px(5), "IWM": _px(6), "USMV": _px(7)}
    res = qe.factor_attribution(strat, fp, n_boot=300)
    assert abs(res["betas"]["市场"] - 1.0) < 0.1
    assert not res["alpha_significant"]


def test_factor_attribution_detects_alpha():
    spy = _px(3)
    spy_r = spy.pct_change().dropna()
    strat = spy_r + 0.002  # 真 alpha
    fp = {"SPY": spy, "MTUM": _px(4), "IWD": _px(5)}
    res = qe.factor_attribution(strat, fp, n_boot=400)
    assert res["alpha_ann"] > 0
    assert res["alpha_significant"]


def test_factor_attribution_insufficient():
    s = pd.Series([0.01, -0.01], index=pd.date_range("2020-01-01", periods=2))
    res = qe.factor_attribution(s, {"SPY": _px(3)})
    assert res["n"] < 120


# ---------------------------------------------------------------------------
# 横截面中性化 + Newey-West
# ---------------------------------------------------------------------------
def test_cross_section_neutralized_fields():
    prices = pd.DataFrame({f"T{i}": _px(10 + i, n=700) for i in range(8)})
    res = qe.cross_section_edge(prices, n_boot=150, neutralize_beta=True)
    assert res["neutralized"] is True
    assert "ic_t_newey_west" in res and "ic_mean" in res
    assert res["ic_n_periods"] > 0


def test_newey_west_t_basic():
    rng = np.random.RandomState(0)
    x = rng.randn(200) * 0.1 + 0.05  # 正均值
    t = qe._newey_west_t(x, lags=5)
    assert t > 2  # 明显正且样本大 → 显著


# ---------------------------------------------------------------------------
# BH-FDR 账本
# ---------------------------------------------------------------------------
def test_benjamini_hochberg_known():
    # 经典例子：大量纯噪声 p ~ U(0,1) + 几个强信号
    rng = np.random.RandomState(0)
    noise = rng.uniform(0, 1, 95)
    signal = np.array([0.0001, 0.0002, 0.0005, 0.001, 0.002])
    p = np.concatenate([signal, noise])
    reject, pstar = qe_bh(p)
    assert reject.sum() >= 3  # 强信号应存活
    assert reject[:5].sum() >= 3


def qe_bh(p):
    return mt.benjamini_hochberg(p, alpha=0.10)


def test_fdr_report_empty(tmp_path):
    rep = mt.fdr_report(db_path=tmp_path / "t.db")
    assert rep["n_tests"] == 0


def test_ledger_log_and_report(tmp_path):
    db = tmp_path / "t.db"
    for i in range(20):
        mt.log_test("noise", f"h{i}", p_value=0.5, db_path=db)        # 噪声
    mt.log_test("real", "pead_h5", p_value=0.0001, db_path=db)        # 真信号
    rep = mt.fdr_report(alpha=0.10, db_path=db)
    assert rep["n_tests"] == 21
    assert rep["n_sig_bh"] >= 1
    # 去重：同 family+name 重复 log 不增加条数
    mt.log_test("real", "pead_h5", p_value=0.0002, db_path=db)
    assert mt.fdr_report(db_path=db)["n_tests"] == 21


# ---------------------------------------------------------------------------
# S&P500 universe
# ---------------------------------------------------------------------------
def test_load_sp500():
    from data import loader
    syms = loader.load_sp500()
    assert len(syms) > 400
    assert "AAPL" in syms and "MSFT" in syms
    assert all(s.isascii() for s in syms)

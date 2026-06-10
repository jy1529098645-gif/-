"""quant_edge 层测试：α/β 分解、regime 暴露、波动目标、横截面 edge。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import quant_edge as qe


def _series(seed, n=900, drift=0.0003, vol=0.012):
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    r = rng.randn(n) * vol + drift
    return pd.Series(np.cumprod(1 + r) * 100, index=idx)


# ---------------------------------------------------------------------------
# alpha_beta
# ---------------------------------------------------------------------------
def test_alpha_beta_pure_beta_has_no_alpha():
    mkt = _series(1).pct_change()
    strat = (1.0 * mkt).copy()  # 纯 beta=1，无 α
    res = qe.alpha_beta(strat, mkt, n_boot=300)
    assert abs(res["beta"] - 1.0) < 0.05
    assert not res["alpha_significant"]  # 无真 α
    assert res["r2"] > 0.95


def test_alpha_beta_detects_real_alpha():
    mkt = _series(2).pct_change().dropna()
    strat = mkt + 0.002  # 每日 +0.2% 真 α
    res = qe.alpha_beta(strat, mkt, n_boot=400)
    assert res["alpha_ann"] > 0
    assert res["alpha_significant"]


def test_alpha_beta_small_sample():
    s = pd.Series([0.01, -0.01, 0.02], index=pd.date_range("2020-01-01", periods=3))
    res = qe.alpha_beta(s, s)
    assert res["n"] == 3
    assert "样本不足" in res["verdict"]


# ---------------------------------------------------------------------------
# alpha_beta_profile（β漂移 / 上下行β / 诚实α裁决）
# ---------------------------------------------------------------------------
def test_ab_profile_amplifier_high_beta():
    mkt = _series(5)
    # 个股 = 1.5x 市场收益 + 噪声 → β≈1.5、无真α
    mr = mkt.pct_change()
    rng = np.random.RandomState(9)
    sr = 1.5 * mr + pd.Series(rng.randn(len(mr)) * 0.004, index=mr.index)
    stock = (1 + sr.fillna(0)).cumprod() * 100
    r = qe.alpha_beta_profile(stock, mkt, n_boot=300)
    assert r["available"]
    assert 1.3 <= r["beta"] <= 1.7
    assert "放大市场" in r["beta_note"]
    assert not r["alpha_significant"]  # 纯放大无真α
    assert r["beta_down"] == r["beta_down"] and r["beta_up"] == r["beta_up"]


def test_ab_profile_downside_beta_asymmetry():
    # 构造跌时β更高的不对称：市场跌的日子放大2x、涨的日子1x
    mkt = _series(6)
    mr = mkt.pct_change().dropna()
    sr = mr.where(mr >= 0, mr * 2.0)  # 负收益日翻倍
    stock = (1 + sr).cumprod() * 100
    r = qe.alpha_beta_profile(stock, mkt.reindex(stock.index), n_boot=200)
    assert r["beta_down"] > r["beta_up"]
    assert "下行β" in r["risk_note"]


def test_ab_profile_small_sample():
    s = _series(7, n=40)
    r = qe.alpha_beta_profile(s, s)
    assert not r["available"]


# ---------------------------------------------------------------------------
# regime_exposure / vol_target
# ---------------------------------------------------------------------------
def test_regime_exposure_bounds():
    px = _series(3)
    r = qe.regime_exposure(px, None)
    assert 0.2 <= r["exposure"] <= 1.0
    assert r["factors"]  # 至少有波动/趋势因子


def test_regime_high_vol_lowers_exposure():
    # 构造近端高波动序列 → 暴露应明显<1
    rng = np.random.RandomState(7)
    idx = pd.date_range("2018-01-01", periods=600, freq="B")
    calm = rng.randn(400) * 0.005
    storm = rng.randn(200) * 0.05
    px = pd.Series(np.cumprod(1 + np.concatenate([calm, storm])) * 100, index=idx)
    r = qe.regime_exposure(px, None)
    assert r["exposure"] < 1.0


def test_vol_target_backtest_no_leverage():
    px = _series(4)
    res = qe.vol_target_backtest(px, target_vol=0.15, max_lev=1.0)
    assert res["avg_exposure"] <= 1.0
    assert set(res["equity"].columns) == {"波动目标", "持有"}
    assert "cagr" in res["overlay"]


# ---------------------------------------------------------------------------
# cross_section_edge
# ---------------------------------------------------------------------------
def test_cross_section_edge_runs():
    prices = pd.DataFrame({f"T{i}": _series(10 + i, n=700) for i in range(6)})
    res = qe.cross_section_edge(prices, n_boot=200)
    assert "sharpe" in res and "deflated_sharpe_prob" in res
    assert 0.0 <= res["deflated_sharpe_prob"] <= 1.0
    assert "robust" in res

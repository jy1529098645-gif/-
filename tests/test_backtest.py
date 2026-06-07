"""Phase 5 验收：回测整合。

投入计划用合成数据单测（不联网）；策略对比/因子回测用真实数据（缓存/联网，失败 skip）。
核心：结果带置信区间（不报点估计）；三策略同预算公平对比。
"""
import numpy as np
import pandas as pd
import pytest

from backtest import strategies as bt


@pytest.fixture
def close():
    idx = pd.bdate_range("2010-01-01", periods=600)
    rng = np.random.default_rng(0)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 600))), index=idx)


def test_schedules_sum_to_budget(close):
    """三策略的总投入计划都应等于预算（公平对比的前提）。"""
    budget = 10000.0
    for k in bt.STRATEGIES:
        s = bt._BUILDERS[k](close, budget=budget, deploy=252, n_dca=12, dip=0.05)
        assert abs(np.nansum(s.to_numpy()) - budget) < 1e-6, f"{k} 总投入 != budget"


def test_lump_sum_invests_day0(close):
    s = bt._schedule_lump_sum(close, budget=5000.0)
    assert s.iloc[0] == 5000.0
    assert s.iloc[1:].isna().all()


def test_dca_has_n_tranches(close):
    s = bt._schedule_dca(close, budget=1200.0, deploy=252, n_dca=12)
    nz = s.dropna()
    assert len(nz) == 12
    assert abs(nz.sum() - 1200.0) < 1e-6
    assert np.allclose(nz.to_numpy(), 100.0)


def test_average_down_deploys_on_dips():
    """构造一段持续下跌：average_down 应触发多次补仓（早于 deploy 末）。"""
    idx = pd.bdate_range("2010-01-01", periods=300)
    close = pd.Series(np.linspace(100, 60, 300), index=idx)  # 单调下跌 40%
    s = bt._schedule_average_down(close, budget=1000.0, deploy=252, n_dca=10, dip=0.05)
    buys = s.dropna()
    assert len(buys) > 2                      # 多次补仓
    assert abs(buys.sum() - 1000.0) < 1e-6    # 总投入仍 = budget


# ---------------------------------------------------------------------------
# 集成（真实数据）
# ---------------------------------------------------------------------------
def _load(tickers, start="1995-01-01", end="2024-01-01"):
    from data import loader

    try:
        px = loader.load_prices(tickers, start, end)
        if px.dropna(how="all").shape[0] < 2000:
            pytest.skip("数据不足")
        return px
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"数据不可用：{type(e).__name__}: {e}")


def test_compare_entry_strategies_reports_ci():
    px = _load(["SPY"])
    res = bt.compare_entry_strategies(px, asset="SPY", n_boot=200, start_step=126)

    assert set(res["per_strategy"]) == set(bt.STRATEGIES)
    for k, v in res["per_strategy"].items():
        # 必须带 CI（不报点估计）
        assert "median_ci_low" in v and "median_ci_high" in v
        assert v["median_ci_low"] <= v["median"] <= v["median_ci_high"]
        assert v["n_windows"] > 10

    # 配对差值带 CI + 显著性判断
    for k, v in res["vs_lump_sum"].items():
        assert "ci_low" in v and "ci_high" in v and "significant" in v


def test_factor_quantile_backtest_reports_ci():
    from data import loader
    from factors import price_factors as pf

    tickers = loader.load_universe()
    px = _load(tickers, "2005-01-01", "2020-01-01")
    mom = pf.momentum_12_1(px)
    res = bt.factor_quantile_backtest(mom, px, quantiles=5, long_short=True, n_boot=200)
    assert "sharpe" in res and "sharpe_ci_low" in res and "sharpe_ci_high" in res
    assert res["sharpe_ci_low"] <= res["sharpe"] <= res["sharpe_ci_high"]

"""Phase S3 验收：walk-forward + deflated Sharpe 包裹规则选择。

per_trade_sharpe / 网格用合成或纯逻辑单测；deflated + walk-forward 用真实数据（缓存/联网，失败 skip）。
核心：每个"最优规则"带出样本表现 + 折扣后夏普；IS vs OOS 缺口可量化。
"""
import numpy as np
import pytest

from evaluation import rule_select as rs


def test_per_trade_sharpe():
    assert np.isnan(rs.per_trade_sharpe([0.1, 0.2]))          # 样本不足
    s = rs.per_trade_sharpe([0.05, 0.05, 0.05, 0.05])         # 零方差
    assert s == 0.0
    s2 = rs.per_trade_sharpe([0.1, -0.05, 0.2, 0.0, 0.15])
    assert np.isfinite(s2)


def test_candidate_grid_from_current_rule():
    specs = [("dip_from_high", 0.15, 0.0), ("rsi_oversold", 14, 40)]
    grid = rs.candidate_grid_from(specs, "and", {"trailing_stop": 0.20, "take_profit": 0.25, "time_stop": 63})
    assert len(grid) == 9                      # 3×3 缩放
    for c in grid:
        assert callable(c["entry_fn"]) and "trailing_stop" in c["exit_spec"]
    # 含 base(×1.0) 组合
    trails = {round(c["exit_spec"]["trailing_stop"], 3) for c in grid}
    assert 0.2 in trails


def test_candidate_grid_structure():
    grid = rs.candidate_grid(dip_pcts=(0.10, 0.15), trail_stops=(0.15, 0.20))
    assert len(grid) == 4
    for c in grid:
        assert {"name", "entry_fn", "exit_spec"} <= set(c)
        assert callable(c["entry_fn"])
        assert "trailing_stop" in c["exit_spec"]


def _check_data():
    from data import loader

    try:
        px = loader.load_prices(["AAPL", "MSFT"], "2012-01-01", "2024-01-01")
        if px.dropna().shape[0] < 1000:
            pytest.skip("数据不足")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"数据不可用：{type(e).__name__}: {e}")


def test_deflated_rule_sharpe():
    _check_data()
    grid = rs.candidate_grid(dip_pcts=(0.10, 0.15), trail_stops=(0.15, 0.20))
    d = rs.deflated_rule_sharpe(candidates=grid, start="2012-01-01", end="2024-01-01", min_trades=10)
    assert d["n_trials"] == len(grid)
    assert 0.0 <= d["deflated_sharpe_prob"] <= 1.0
    assert d["best_name"] in set(d["table"]["name"])
    # 表按夏普降序
    sh = d["table"]["sharpe"].dropna().to_numpy()
    assert (np.diff(sh) <= 1e-9).all()


def test_walk_forward_reports_oos_and_gap():
    _check_data()
    grid = rs.candidate_grid(dip_pcts=(0.10, 0.15), trail_stops=(0.15, 0.20))
    wf = rs.walk_forward_rule(candidates=grid, start="2012-01-01", end="2024-01-01",
                              train_years=4, test_years=1, min_trades=5)
    tab, summ = wf["table"], wf["summary"]
    assert {"is_sharpe", "oos_sharpe", "selected", "oos_start"} <= set(tab.columns)
    assert summ["n_splits"] >= 3
    # 每段选出的规则必须来自候选
    assert set(tab["selected"]).issubset({c["name"] for c in grid})
    # overfit_gap = IS均 - OOS均
    assert abs(summ["overfit_gap"] - (summ["mean_is_sharpe"] - summ["mean_oos_sharpe"])) < 1e-9


def test_s3_plots_produced():
    _check_data()
    from data import loader
    from reports import plots

    grid = rs.candidate_grid(dip_pcts=(0.10, 0.15), trail_stops=(0.15, 0.20))
    tickers = ["AAPL", "MSFT", "GOOGL"]
    prices = {t: loader.load_prices([t], "2012-01-01", "2024-01-01")[t].dropna() for t in tickers}
    tr = rs._candidate_trades(grid[1], prices)
    wf = rs.walk_forward_rule(candidates=grid, tickers=tickers, start="2012-01-01", end="2024-01-01")

    import os
    p1 = plots.underwater(tr, "test_s3")
    p2 = plots.walk_forward_plot(wf["table"].dropna(subset=["oos_sharpe"]), "test_s3")
    assert os.path.getsize(p1) > 5000 and os.path.getsize(p2) > 5000

"""U6 验收：前端图表冒烟测试（不联网，合成数据，确认每个图能生成 plotly Figure）。"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from frontend import charts as ch


def _trades(n=30):
    idx = pd.bdate_range("2018-01-01", periods=n)
    rng = np.random.default_rng(0)
    r = rng.normal(0.05, 0.15, n)
    return pd.DataFrame({
        "entry_date": idx, "exit_date": idx + pd.Timedelta(days=20),
        "entry_price": 100 + np.arange(n), "exit_price": 100 + np.arange(n) * (1 + r),
        "return": r, "mae": -np.abs(rng.normal(0.05, 0.03, n)), "duration_bars": 20,
    })


def test_return_and_mae_hist():
    t = _trades()
    assert isinstance(ch.return_hist(t["return"], median=0.05, baseline_median=0.02, n=30, n_eff=12), go.Figure)
    assert isinstance(ch.mae_hist(t["mae"], n=30), go.Figure)
    assert isinstance(ch.equity_underwater(t), go.Figure)


def test_corr_and_cone():
    df = pd.DataFrame(np.random.default_rng(0).normal(0, 1, (100, 3)), columns=list("ABC"))
    assert isinstance(ch.corr_heatmap(df.corr()), go.Figure)
    h = [21, 63, 252]
    assert isinstance(ch.forward_cone(h, [-.1]*3, [-.05]*3, [.02]*3, [.08]*3, [.15]*3, [.01]*3), go.Figure)


def test_candles_and_vp():
    idx = pd.bdate_range("2021-01-01", periods=120)
    c = 100 + np.cumsum(np.random.default_rng(1).normal(0, 1, 120))
    ohlcv = pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1e6}, index=idx)
    assert isinstance(ch.trade_map_candles(ohlcv, _trades(10)), go.Figure)
    vp = {"centers": np.linspace(90, 120, 40), "volumes": np.random.default_rng(2).random(40),
          "poc": 105.0, "value_area": (98.0, 112.0)}
    assert isinstance(ch.volume_profile_bars(vp), go.Figure)
    assert isinstance(ch.candle_with_levels(ohlcv, vp), go.Figure)


def test_ic_and_strategy_charts():
    ic = pd.DataFrame({"IC_mean": [0.02, 0.04], "IC_std": [0.4, 0.4], "n_days": [1000, 1000]}, index=["1D", "5D"])
    assert isinstance(ch.factor_ic_bars(ic), go.Figure)
    per = {"lump_sum": {"median": 0.2, "median_ci_low": 0.1, "median_ci_high": 0.3},
           "dca": {"median": 0.15, "median_ci_low": 0.08, "median_ci_high": 0.22}}
    assert isinstance(ch.strategy_compare(per), go.Figure)


def test_glossary_term_html():
    from frontend import glossary as gl
    html = gl.term("IC")
    assert "qterm" in html and "data-tip" in html
    assert gl.help_for("N_eff")

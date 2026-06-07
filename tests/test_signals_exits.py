"""Phase S1 验收：入场信号 + 出场规则 + 单票逐笔交易。

信号/出场映射用合成数据单测；逐笔交易 + 出图用真实数据（缓存/联网，失败 skip）。
"""
import numpy as np
import pandas as pd
import pytest

from backtest import exits as ex
from factors import signals as sg


@pytest.fixture
def rise_then_drop():
    idx = pd.bdate_range("2018-01-01", periods=400)
    up = np.linspace(100, 200, 250)              # 先涨到 200
    down = np.linspace(200, 150, 150)            # 再跌 25%
    return pd.Series(np.concatenate([up, down]), index=idx)


# --------------------------- signals ---------------------------
def test_dip_from_high(rise_then_drop):
    sig = sg.dip_from_high(rise_then_drop, lookback=200, pct=0.15)
    assert sig.dtype == bool
    assert not sig.iloc[:250].any()      # 上涨段不触发
    assert sig.iloc[-1]                   # 跌 25% > 15% 阈值 → 触发


def test_ma_cross_is_event():
    idx = pd.bdate_range("2018-01-01", periods=300)
    price = pd.Series(np.concatenate([np.linspace(100, 80, 150), np.linspace(80, 140, 150)]), index=idx)
    sig = sg.ma_cross(price, fast=10, slow=50)
    assert sig.sum() >= 1
    assert sig.sum() <= 3                 # 事件型：只在穿越当日，不是一直 True


def test_rsi_bounds_and_oversold():
    idx = pd.bdate_range("2018-01-01", periods=200)
    up = pd.Series(np.linspace(100, 300, 200), index=idx)   # 单调上涨
    r = sg.rsi(up, 14).dropna()
    assert (r >= 0).all() and (r <= 100).all()
    assert r.iloc[-1] > 70                                  # 持续上涨 RSI 偏高
    assert not sg.rsi_oversold(up, 14, 30).iloc[-1]         # 不超卖


def test_combine_and_or():
    idx = pd.bdate_range("2020-01-01", periods=5)
    a = pd.Series([True, True, False, False, True], index=idx)
    b = pd.Series([True, False, True, False, True], index=idx)
    assert list(sg.combine_and(a, b)) == [True, False, False, False, True]
    assert list(sg.combine_or(a, b)) == [True, True, True, False, True]


# --------------------------- exits ---------------------------
def test_build_exit_kwargs_mapping():
    idx = pd.bdate_range("2020-01-01", periods=20)
    price = pd.Series(np.arange(100, 120), index=idx, dtype=float)
    entries = pd.Series(False, index=idx)
    entries.iloc[0] = True

    kw = ex.build_exit_kwargs(price, entries, {"trailing_stop": 0.2})
    assert kw["sl_stop"] == 0.2 and kw["sl_trail"] is True

    kw = ex.build_exit_kwargs(price, entries, {"stop_loss": 0.1})
    assert kw["sl_stop"] == 0.1 and "sl_trail" not in kw

    kw = ex.build_exit_kwargs(price, entries, {"take_profit": 0.25})
    assert kw["tp_stop"] == 0.25

    kw = ex.build_exit_kwargs(price, entries, {"time_stop": 5})
    assert "exits" in kw and kw["exits"].iloc[5]            # 入场后第5日标记出场


def test_run_and_extract_trades_synthetic(rise_then_drop):
    entries = pd.Series(False, index=rise_then_drop.index)
    entries.iloc[0] = True                                  # day0 入场
    pf = ex.run_trades(rise_then_drop, entries, {"take_profit": 0.25}, fees=0.0, slippage=0.0)
    trades = ex.extract_trades(pf, rise_then_drop)
    assert len(trades) >= 1
    cols = {"entry_date", "exit_date", "return", "duration_days", "mae"}
    assert cols <= set(trades.columns)
    # 价格从 100 涨到 200，+25% 止盈应被触发 → 首笔收益≈+25%
    assert trades.iloc[0]["return"] > 0.2
    # MAE 恒 ≤ 0（途中浮亏不为正）
    assert (trades["mae"].dropna() <= 1e-9).all()


# --------------------------- integration（真实数据）---------------------------
def test_nvda_trades_and_plots(tmp_path):
    from data import loader
    from reports import plots

    try:
        px = loader.load_prices(["NVDA"], "2010-01-01", "2024-01-01")["NVDA"]
        if px.dropna().shape[0] < 1000:
            pytest.skip("数据不足")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"数据不可用：{type(e).__name__}: {e}")

    entries = sg.dip_from_high(px, lookback=252, pct=0.15)
    pf = ex.run_trades(px, entries, {"trailing_stop": 0.20, "take_profit": 0.25, "time_stop": 63})
    trades = ex.extract_trades(pf, px)
    assert len(trades) >= 5

    import os
    rule = "test_nvda_dip15"
    p1 = plots.trade_map(px, trades, rule)
    p2 = plots.return_hist(trades, rule)
    p3 = plots.mae_hist(trades, rule)
    for p in (p1, p2, p3):
        assert os.path.exists(p) and os.path.getsize(p) > 5000

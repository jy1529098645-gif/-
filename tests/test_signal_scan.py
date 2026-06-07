"""Phase G 验收：信号扫描 + FDR + 趋势跌破出场（ma_exit）。"""
import numpy as np
import pandas as pd
import pytest

from analysis import signal_scan as ss


def test_bh_fdr_logic():
    # 全是小 p → 多数通过；全是大 p → 都不通过
    assert all(ss._bh_fdr([0.001, 0.002, 0.003], q=0.1))
    assert not any(ss._bh_fdr([0.6, 0.7, 0.8], q=0.1))
    # 混合：极小的那个应通过
    res = ss._bh_fdr([0.0001, 0.5, 0.9], q=0.1)
    assert res[0] is True and res[1] is False


def test_block_boot_shape():
    arr = np.random.default_rng(0).normal(0.01, 0.05, 500)
    boot = ss._block_boot(arr, block=21, n=200, seed=1)
    assert boot.shape == (200,)
    assert np.isfinite(boot).all()


def test_features_and_states_causal():
    idx = pd.bdate_range("2015-01-01", periods=600)
    rng = np.random.default_rng(0)
    price = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, 600))), index=idx)
    f = ss._features(price)
    es = ss.entry_states(f)
    rs = ss.risk_states(f)
    assert "上升趋势中的回调" in es and "高波动" in rs
    for s in list(es.values()) + list(rs.values()):
        assert s.dtype == bool or s.dropna().isin([True, False]).all()


def test_ma_exit_builds_exits():
    from backtest import exits as ex
    idx = pd.bdate_range("2020-01-01", periods=300)
    price = pd.Series(np.r_[np.linspace(100, 150, 150), np.linspace(150, 110, 150)], index=idx)
    entries = pd.Series(False, index=idx); entries.iloc[0] = True
    kw = ex.build_exit_kwargs(price, entries, {"ma_exit": 50})
    assert "exits" in kw and kw["exits"].dtype == bool
    assert kw["exits"].iloc[160:].any()      # 下跌段应触发跌破均线出场


def test_humanize_scan_readable():
    df = pd.DataFrame([
        {"kind": "entry", "signal": "上升趋势中的回调", "n_events": 98, "n_days": 300,
         "median_fwd": 0.158, "win_rate": 0.75, "excess": 0.048, "ci_low": -0.06, "ci_high": 0.14,
         "pval": 0.21, "sig_raw": False, "sig_fdr": False, "fwd_drawdown": -0.042},
        {"kind": "risk", "signal": "高波动", "n_events": 71, "n_days": 200,
         "median_fwd": 0.11, "win_rate": 0.72, "excess": 0.0, "ci_low": -0.05, "ci_high": 0.07,
         "pval": 0.9, "sig_raw": False, "sig_fdr": False, "fwd_drawdown": -0.046},
    ])
    disp, summary = ss.humanize_scan(df)
    assert "比闭眼买多/少" in disp.columns and "结论" in disp.columns
    assert "sig_raw" not in disp.columns and "ci_low" not in disp.columns
    assert "建仓倾斜最强" in summary
    assert disp.iloc[0]["类别"].startswith("📈")


def test_scan_columns_optional():
    """scan 联网（缓存）；失败 skip。验证关键列齐全。"""
    try:
        df = ss.scan(["AAPL", "MSFT"], start="2014-01-01", end="2024-01-01", horizon=63, n_boot=120)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"数据不可用：{type(e).__name__}: {e}")
    need = {"kind", "signal", "n_events", "excess", "ci_low", "ci_high", "pval", "sig_raw", "sig_fdr", "fwd_drawdown"}
    assert need <= set(df.columns)
    assert (df["ci_low"] <= df["excess"]).all() and (df["excess"] <= df["ci_high"]).all()

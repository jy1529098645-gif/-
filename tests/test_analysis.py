"""U3 验收：Volume Profile（合成 OHLCV，不联网）+ 期权快照（联网/skip）。"""
import numpy as np
import pandas as pd
import pytest

from analysis import volume_profile as vpm


def _synth_ohlcv():
    idx = pd.bdate_range("2021-01-01", periods=300)
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, 300))
    high = close + np.abs(rng.normal(0, 1, 300))
    low = close - np.abs(rng.normal(0, 1, 300))
    vol = rng.integers(1e6, 5e6, 300).astype(float)
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


def test_volume_profile_conserves_volume():
    df = _synth_ohlcv()
    vp = vpm.volume_profile(df, bins=40)
    assert abs(vp["volumes"].sum() - df["volume"].sum()) / df["volume"].sum() < 0.02  # 摊量守恒
    assert len(vp["centers"]) == 40
    lo, hi = df["low"].min(), df["high"].max()
    assert lo <= vp["poc"] <= hi
    va_lo, va_hi = vp["value_area"]
    assert va_lo <= vp["poc"] <= va_hi


def test_volume_profile_poc_at_dense_price():
    """构造价格长期在某区间盘整 → POC 应落在该密集区。"""
    idx = pd.bdate_range("2021-01-01", periods=200)
    close = np.r_[np.full(150, 100.0), np.linspace(100, 130, 50)]  # 150天在100附近
    df = pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5,
                       "close": close, "volume": np.full(200, 1e6)}, index=idx)
    vp = vpm.volume_profile(df, bins=50)
    assert abs(vp["poc"] - 100) < 5      # POC 接近 100


def test_options_snapshot_optional():
    try:
        from data import options
        snap = options.options_snapshot("AAPL")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"期权数据不可用：{type(e).__name__}: {e}")
    assert snap["spot"] > 0
    assert {"atm_iv_call", "put_call_oi_ratio", "max_oi_call_strike"} <= set(snap)


def test_regime_path_distribution():
    """趋势全程分布：分布字段完整、内部一致(median↔再跌过-10%频率)、非预测口径。"""
    import numpy as np, pandas as pd
    from analysis import analogs as an
    idx = pd.date_range("2008-01-01", periods=2000, freq="B")
    rng = np.random.RandomState(0)
    px = pd.Series(100*np.exp(np.cumsum(rng.normal(0.0003, 0.015, 2000))), index=idx)
    rd = an.regime_path_distribution(px, window=504)
    assert rd["available"] and rd["n"] >= 30
    f = rd["further_dd"]; assert f["p10"] <= f["p50"] <= f["p90"]
    r = rd["runup"]; assert r["p10"] <= r["p50"] <= r["p90"]
    # 一致性：再跌过-10%频率 与 中位 further_dd 同向
    assert (rd["further_dd"]["p50"] <= -0.10) == (rd["further_le10"] >= 0.5)
    assert 0 <= rd["further_le10"] <= 1 and 0 <= rd["recovery_rate"] <= 1
    fmt = an.format_regime_path(rd)
    assert fmt and "headline" in fmt and len(fmt["lines"]) >= 3 and "非预测" in fmt["caveat"]
    # 样本不足→优雅降级
    short = px.iloc[:100]
    assert an.regime_path_distribution(short)["available"] is False

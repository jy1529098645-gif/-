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

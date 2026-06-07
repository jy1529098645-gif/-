"""Phase 3 新功能离线测试：波动自适应止损 + 滚动IC/因子衰减。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.exits import vol_scaled_trail
from evaluation import factor_eval as fe


# ---------------------------------------------------------------------------
# 波动自适应移动止损
# ---------------------------------------------------------------------------
def _series(vol_daily, n=300, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, vol_daily, n)
    px = 100 * np.exp(np.cumsum(rets))
    return pd.Series(px, index=pd.date_range("2020-01-01", periods=n, freq="B"))


def test_vol_trail_in_bounds_and_scales():
    quiet = _series(0.008)
    loud = _series(0.05)
    sq = vol_scaled_trail(quiet, k=3.0)
    sl = vol_scaled_trail(loud, k=3.0)
    assert 0.05 <= sq <= 0.50 and 0.05 <= sl <= 0.50
    assert sl > sq                       # 波动大的票 → 更宽的止损
    assert vol_scaled_trail(quiet, k=6.0) >= vol_scaled_trail(quiet, k=3.0)  # k 越大越宽


def test_vol_trail_handles_degenerate():
    flat = pd.Series([100.0] * 50, index=pd.date_range("2020-01-01", periods=50, freq="B"))
    assert 0.05 <= vol_scaled_trail(flat) <= 0.50  # 零波动不崩，回退默认


# ---------------------------------------------------------------------------
# 滚动 IC / 因子衰减
# ---------------------------------------------------------------------------
def _panel(n_assets=8, n=500, signal=True, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    cols = [f"A{i}" for i in range(n_assets)]
    rets = rng.normal(0.0004, 0.02, (n, n_assets))
    px = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=cols)
    h = 21
    fwd = px.shift(-h) / px - 1.0
    if signal:                       # 因子 = 未来收益 + 噪声 → 应有正 IC
        fac = fwd + rng.normal(0, 0.03, (n, n_assets))
    else:                            # 随机因子 → IC≈0
        fac = pd.DataFrame(rng.normal(0, 1, (n, n_assets)), index=idx, columns=cols)
    return fac, px


def test_cross_sectional_ic_positive_for_signal():
    fac, px = _panel(signal=True)
    ic = fe.cross_sectional_ic(fac, px, horizon=21)
    assert len(ic) > 50
    assert ic.mean() > 0.05           # 构造的信号因子应有明显正 IC


def test_cross_sectional_ic_near_zero_for_random():
    fac, px = _panel(signal=False)
    ic = fe.cross_sectional_ic(fac, px, horizon=21)
    assert abs(ic.mean()) < 0.05      # 随机因子 IC≈0（健全性检查）


def test_rolling_ic_shape_and_decay():
    fac, px = _panel(signal=True)
    rdf = fe.rolling_ic(fac, px, horizon=21, window=120)
    assert set(["ic", "roll_ic", "roll_ir"]).issubset(rdf.columns)
    decay = fe.ic_decay(fac, px, horizons=(1, 21, 63, 252))
    assert decay.notna().any()
    v = fe.decay_verdict(decay)
    assert isinstance(v, str) and "因子" in v

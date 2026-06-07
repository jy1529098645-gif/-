"""Phase 2 验收：价格因子 + 评估。

因子公式用合成数据单测（不联网）；评估部分联网（或用缓存），失败则 skip。
核心健全性检查：随机因子 IC≈0；momentum IC 明显高于随机。
"""
import numpy as np
import pandas as pd
import pytest

from factors import price_factors as pf


@pytest.fixture
def synth_prices():
    """3 只标的、稳定上行 + 不同波动的合成价格，用于因子公式单测。"""
    idx = pd.bdate_range("2015-01-01", periods=300)
    t = np.arange(300)
    a = 100 * (1.0005) ** t                         # 平稳上行、低波动
    b = 100 * (1.0005) ** t * (1 + 0.05 * np.sin(t / 5))  # 同趋势、高波动
    c = 100 * (0.9995) ** t                         # 下行
    return pd.DataFrame({"A": a, "B": b, "C": c}, index=idx)


def test_factor_shapes(synth_prices):
    for fn in (pf.momentum_12_1, pf.short_reversal, pf.low_volatility, pf.trend):
        out = fn(synth_prices)
        assert out.shape == synth_prices.shape
        assert list(out.columns) == list(synth_prices.columns)


def test_momentum_definition(synth_prices):
    """momentum = P_{t-skip}/P_{t-lookback} - 1，且上行标的为正、下行为负。"""
    out = pf.momentum_12_1(synth_prices, lookback=60, skip=10)
    expected = synth_prices.shift(10) / synth_prices.shift(60) - 1.0
    pd.testing.assert_frame_equal(out, expected)
    last = out.iloc[-1]
    assert last["A"] > 0 and last["C"] < 0


def test_short_reversal_sign(synth_prices):
    """上行标的过去收益为正 → 反转因子为负。"""
    out = pf.short_reversal(synth_prices, lookback=21)
    assert out["A"].iloc[-1] < 0
    assert out["C"].iloc[-1] > 0


def test_low_volatility_nonpositive(synth_prices):
    """低波动因子是波动率的负值，恒 ≤ 0；低波动标的应高于高波动标的。"""
    out = pf.low_volatility(synth_prices, window=30).iloc[-1]
    assert (out.dropna() <= 0).all()
    assert out["A"] > out["B"]  # A 低波动 → 因子更高（更接近 0）


def test_trend_sign(synth_prices):
    out = pf.trend(synth_prices, ma_window=50).iloc[-1]
    assert out["A"] > 0 and out["C"] < 0


def test_blend_zscore_combination(synth_prices):
    """多因子组合：z-score 等权混合，形状一致、按日截面近似零均值。"""
    panels = {"mom": pf.momentum_12_1(synth_prices, lookback=60, skip=10),
              "lowvol": pf.low_volatility(synth_prices, window=30)}
    comp = pf.blend(panels)
    assert comp.shape == synth_prices.shape
    # 某有效日的截面（z-score 加权和）均值应接近 0
    row = comp.dropna(how="all").iloc[-1].dropna()
    if len(row) >= 3:
        assert abs(row.mean()) < 1.0


def test_no_lookahead(synth_prices):
    """因子值在 t 只能用 ≤t 的价格：前 lookback 行必为 NaN。"""
    out = pf.momentum_12_1(synth_prices, lookback=60, skip=10)
    assert out.iloc[:60].isna().all().all()


# ---------------------------------------------------------------------------
# 评估（联网/缓存）：随机因子健全性检查
# ---------------------------------------------------------------------------
def _load_demo():
    from data import loader

    try:
        tickers = loader.load_universe()
        px = loader.load_prices(tickers, "2005-01-01", "2020-01-01")
        if px.dropna(how="all").shape[0] < 1000:
            pytest.skip("价格数据不足")
        return px
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"数据不可用：{type(e).__name__}: {e}")


def test_random_factor_ic_near_zero_and_momentum_higher():
    from evaluation import factor_eval as fe

    px = _load_demo()
    rnd = pf.random_factor(px, seed=42)
    mom = pf.momentum_12_1(px)

    res_rnd = fe.evaluate_factor(rnd, px, quantiles=5, periods=(1, 5, 21, 63))
    res_mom = fe.evaluate_factor(mom, px, quantiles=5, periods=(1, 5, 21, 63))

    rnd_abs = res_rnd["ic"]["IC_mean"].abs().max()
    mom_abs = res_mom["ic"]["IC_mean"].abs().max()

    assert rnd_abs < 0.02, f"随机因子 IC 应≈0，实测 {rnd_abs:.4f}"
    assert mom_abs > rnd_abs, "momentum IC 应明显高于随机因子"
    # 结果带样本期信息
    assert res_mom["sample_years"] > 5
    assert res_mom["n_obs"] > 1000

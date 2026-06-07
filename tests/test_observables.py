"""U2 验收：免费可观测状态 + regime 条件门（合成数据，不联网）。"""
import numpy as np
import pandas as pd

from regime import observables as ob


def _synth():
    idx = pd.bdate_range("2015-01-01", periods=800)
    rng = np.random.default_rng(0)
    price = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, 800))), index=idx)
    return price


def test_realized_vol_percentile_range():
    p = _synth()
    pct = ob.realized_vol_percentile(p, 21, 252).dropna()
    assert (pct >= 0).all() and (pct <= 1).all()


def test_states_categories():
    p = _synth()
    assert set(ob.vol_state(p).dropna().unique()) <= {"low_vol", "mid_vol", "high_vol"}
    assert set(ob.trend_state(p).dropna().unique()) <= {"up_trend", "down_trend"}
    assert set(ob.drawdown_state(p).dropna().unique()) <= {"in_drawdown", "near_high"}


def test_today_panel_keys():
    p = _synth()
    panel = ob.today_panel(p)
    assert {"vol_state", "trend_state", "drawdown_state", "realized_vol", "drawdown"} <= set(panel)


def test_mutual_correlation():
    idx = pd.bdate_range("2018-01-01", periods=400)
    rng = np.random.default_rng(1)
    common = rng.normal(0, 0.01, 400)
    df = pd.DataFrame({f"x{i}": 100 * np.exp(np.cumsum(common + rng.normal(0, 0.003, 400))) for i in range(4)}, index=idx)
    mc = ob.mutual_correlation(df, 63).dropna()
    assert (mc > 0.5).mean() > 0.5      # 共同因子主导 → 高相关


def test_regime_condition_gates():
    from evaluation import rule_eval as re
    p = _synth()
    cond = re.regime_condition("up_trend")
    s = cond("X", p)
    assert s.dtype == bool and s.shape[0] == p.shape[0]
    # 与另一个条件 AND
    comb = re.combine_conditions(re.regime_condition("up_trend"), re.regime_condition("low_vol"))
    s2 = comb("X", p)
    assert (s2 <= s).all()              # AND 后不多于单条件

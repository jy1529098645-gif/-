"""完整版新增模块测试：信号日志/校准、历史案例、风险偏好、组合优化、信号衰减。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis import journal as jn
from analysis import engine_discipline as ed
from analysis import quant_edge as qe


# ---------------------------------------------------------------------------
# 信号日志 + 校准
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "test.db"


def _rec(ticker, date, h=21, win=0.6, exc=0.03, base=0.05, price=100.0, grade="B"):
    return {"ticker": ticker, "signal_date": date, "horizon": h, "price": price,
            "grade": grade, "bucket": "回撤桶", "pred_win_rate": win, "pred_excess": exc,
            "baseline_median": base, "momentum_trap": False}


def test_log_and_dedup(tmp_db):
    assert jn.log_signal(_rec("AAA", "2024-01-02"), tmp_db) is True
    assert jn.log_signal(_rec("AAA", "2024-01-02"), tmp_db) is False  # 同键去重
    df = jn.load_signals(tmp_db)
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "AAA"


def test_evaluate_and_calibration(tmp_db):
    idx = pd.date_range("2024-01-01", periods=120, freq="B")
    # 造一只稳涨的票：21日后必为正
    price = pd.Series(np.linspace(100, 130, 120), index=idx)
    for d in ("2024-01-02", "2024-01-15", "2024-02-01"):
        jn.log_signal(_rec("AAA", d, h=21, win=0.6, base=0.0), tmp_db)
    df = jn.load_signals(tmp_db)
    ev = jn.evaluate(df, prices={"AAA": price})
    matured = ev[ev["matured"]]
    assert len(matured) >= 1
    assert (matured["realized_pos"] == 1.0).all()  # 稳涨 → 全部命中
    cal = jn.calibration_summary(ev)
    assert cal["n_matured"] >= 1
    assert cal["realized_hit"] == 1.0
    assert 0 <= cal["brier"] <= 1


def test_calibration_empty():
    cal = jn.calibration_summary(pd.DataFrame())
    assert cal["n_matured"] == 0


# ---------------------------------------------------------------------------
# 风险偏好
# ---------------------------------------------------------------------------
def test_risk_profile_scaling():
    g = {"max_position_fraction": 0.6, "grade": "B"}
    cons = ed.apply_risk_profile(g, "保守", single_name_cap=0.5)
    aggr = ed.apply_risk_profile(g, "激进", single_name_cap=0.5)
    assert cons["max_position_fraction"] == pytest.approx(0.3)  # 0.6*0.5
    # 激进 0.6*1.25=0.75 但被单票上限 0.5 截断
    assert aggr["max_position_fraction"] == pytest.approx(0.5)
    # 默认单票上限 0.25 会进一步截断保守档(0.3→0.25)
    assert ed.apply_risk_profile(g, "保守")["max_position_fraction"] == pytest.approx(0.25)
    assert cons["require_trigger"] is True
    assert aggr["require_trigger"] is False


# ---------------------------------------------------------------------------
# 组合优化
# ---------------------------------------------------------------------------
def _px(seed, n=400, vol=0.01, drift=0.0003):
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(np.cumprod(1 + rng.randn(n) * vol + drift) * 100, index=idx)


def test_portfolio_weights_sum_and_cap():
    prices = pd.DataFrame({f"T{i}": _px(i, vol=0.008 + 0.004 * i) for i in range(4)})
    res = qe.portfolio_weights(prices, method="min_var", single_cap=0.5)
    w = res["weights"]
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert all(v <= 0.5 + 1e-9 for v in w.values())
    assert res["port_vol"] == res["port_vol"]  # not nan


def test_min_var_lower_vol_than_equal():
    prices = pd.DataFrame({f"T{i}": _px(i, vol=0.006 + 0.006 * i) for i in range(5)})
    res = qe.portfolio_weights(prices, method="min_var", single_cap=1.0)
    # 最小方差的组合波动应 ≤ 等权
    assert res["compare"]["最小方差"] <= res["compare"]["等权"] + 1e-9


def test_risk_parity_runs():
    prices = pd.DataFrame({f"T{i}": _px(i) for i in range(4)})
    res = qe.portfolio_weights(prices, method="risk_parity")
    assert abs(sum(res["weights"].values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 信号衰减
# ---------------------------------------------------------------------------
def test_signal_decay_runs():
    prices = pd.DataFrame({f"T{i}": _px(10 + i, n=800) for i in range(6)})
    res = qe.signal_decay(prices, horizon=5)
    assert "recent_ic" in res and "early_ic" in res
    assert isinstance(res["decayed"], bool)

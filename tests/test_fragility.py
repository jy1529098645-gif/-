"""市场脆弱性 + 等/追指南 测试（离线）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import fragility as fr


def _panel(n=900, k=10, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    rets = rng.normal(0.0004, 0.015, (n, k))
    px = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(px, index=idx, columns=[f"S{i}" for i in range(k)])


def test_breadth_in_range():
    p = _panel()
    br = fr.breadth_above_ma(p, ma=200)
    b = br.dropna()
    assert ((b >= 0) & (b <= 1)).all()


def test_fragility_frame_columns():
    p = _panel()
    f = fr.fragility_frame(p)
    assert set(["breadth", "pctile", "fragile"]).issubset(f.columns)
    assert f["fragile"].dtype == bool


def test_current_fragility_structure():
    p = _panel()
    cur = fr.current_fragility(p)
    assert cur["available"] in (True, False)
    if cur["available"]:
        assert 0 <= cur["pctile"] <= 1
        assert "light" in cur


def test_evaluate_breadth_warning_keys():
    p = _panel()
    idx = p.mean(axis=1)  # 等权指数当代理
    out = fr.evaluate_breadth_warning(p, idx, horizon=63)
    assert "base" in out
    # 触发样本够时应有 lift
    if "lift" in out:
        assert out["lift"] == out["lift"]  # not nan structurally


# —— 等/追 决策逻辑（确定性，逐情形验证）——
def test_wait_or_chase_near_high_says_chase():
    g = fr.wait_or_chase(current_dd=-0.02, fragile_now=False)
    assert "追" in g["action"]
    assert "别等" in g["detail"]


def test_wait_or_chase_index_deep_is_accumulate():
    g = fr.wait_or_chase(current_dd=-0.25, fragile_now=False, is_index=True)
    assert "建仓区" in g["action"]


def test_wait_or_chase_single_deep_no_conviction_defensive():
    g = fr.wait_or_chase(current_dd=-0.25, fragile_now=False, is_index=False, conviction=False)
    assert "观望" in g["action"] or "轻仓" in g["action"]


def test_wait_or_chase_fragile_forces_defense_near_high():
    g = fr.wait_or_chase(current_dd=-0.03, fragile_now=True)
    assert "降仓" in g["action"] or "减半" in g["action"]
    assert g["fragile"] is True

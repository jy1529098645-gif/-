"""最佳入场区 + 锚点价（regime.entry_cockpit.best_entry_zone）离线测试。

铁律：给"最佳入场区+锚点价"，但锚点必须落在区间内、CI 跨 0 时降级标注、
无正超额档时转防守，绝不硬凑买点。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from regime import entry_cockpit as ec


def _mean_reverting_series(n=2200, period=120, amp=0.15, drift=0.0003):
    """温和上行 + 周期回调的合成价：回调后必回升 → 回撤档历史远期为正、超额为正。"""
    t = np.arange(n)
    base = 100.0 * (1 + drift * t)
    osc = 1 + amp * np.sin(2 * np.pi * t / period)
    idx = pd.date_range("2012-01-01", periods=n, freq="B")
    return pd.Series(base * osc, index=idx, name="X")


def test_best_entry_zone_found_and_anchor_in_band():
    px = _mean_reverting_series()
    bez = ec.best_entry_zone(px, asset="X", horizon=21, n_boot=200, single_name=True)
    assert bez["has_zone"] is True
    lo, hi = bez["price_band"]
    assert lo <= bez["anchor_price"] <= hi          # 锚点必须在区间内
    assert bez["n_events"] >= 1
    assert bez["excess_median"] > 0                  # 最佳档必须正超额
    # 个股口径必须带幸存者偏差提醒
    assert any("幸存者偏差" in c for c in bez["caveats"])
    # 一定带"区间非点位"的诚实提醒
    assert any("不是预测" in c or "若到达" in c for c in bez["caveats"])


def test_format_best_entry_string():
    px = _mean_reverting_series()
    bez = ec.best_entry_zone(px, asset="X", horizon=21, n_boot=200)
    s = ec.format_best_entry(bez)
    assert "最佳入场区" in s
    assert "锚点价" in s
    assert "CI" in s


def test_index_not_single_name_drops_survivorship_caveat():
    px = _mean_reverting_series()
    bez = ec.best_entry_zone(px, asset="X", horizon=21, n_boot=200, single_name=False)
    assert bez["has_zone"] is True
    assert not any("幸存者偏差" in c for c in bez["caveats"])


def test_methodology_fields_present():
    px = _mean_reverting_series()
    bez = ec.best_entry_zone(px, asset="X", horizon=21, n_boot=150)
    for k in ("dsr", "dsr_ok", "n_trials", "n_independent", "regime_clustered", "open_ended"):
        assert k in bez, f"缺字段 {k}"
    assert bez["n_trials"] >= 1


def test_confident_implies_all_gates_pass():
    # 「稳健」是强承诺：必须 DSR 达标 + 有效独立窗口≥5 + 非 regime 聚集 + 非个股深档
    px = _mean_reverting_series()
    for sn in (True, False):
        bez = ec.best_entry_zone(px, asset="X", horizon=21, n_boot=200, single_name=sn)
        if bez.get("has_zone") and bez.get("confident"):
            assert bez["dsr_ok"] is True
            assert bez["n_independent"] >= 5
            assert bez["regime_clustered"] is False
            assert not (sn and bez["zones"].loc[bez["zones"]["zone"] == bez["zone_label"], "depth_hi"].iloc[0] > 0.30)


def test_anchor_is_median_dd_projection_in_band():
    # 锚点=历史命中回撤深度中位投影到当前前高 → 必落在 [price_low, price_high]
    px = _mean_reverting_series()
    bez = ec.best_entry_zone(px, asset="X", horizon=21, n_boot=150)
    if bez.get("has_zone") and not bez.get("open_ended"):
        lo, hi = bez["price_band"]
        assert lo <= bez["anchor_price"] <= hi


def test_low_effective_n_not_confident():
    # 长持有期 + 短样本 → 有效独立窗口很少 → 不得"稳健"
    px = _mean_reverting_series(n=900, period=120)
    bez = ec.best_entry_zone(px, asset="X", horizon=252, n_boot=120, single_name=True)
    if bez.get("has_zone"):
        # n_independent = n_days//252，短样本下任一档都很小
        if bez["n_independent"] < 5:
            assert bez["confident"] is False


def test_defensive_when_no_positive_excess():
    # 加速下跌(动量陷阱型)：越跌远期越差 → 不应硬给最佳入场点
    n = 1500
    t = np.arange(n)
    px = pd.Series(100.0 - 2e-5 * t * t, index=pd.date_range("2014-01-01", periods=n, freq="B"), name="D")
    px = px.clip(lower=1.0)
    bez = ec.best_entry_zone(px, asset="D", horizon=21, n_boot=150, single_name=True)
    if not bez["has_zone"]:
        assert bez["tier"] == "防守"
        assert "观望" in bez["verdict"] or "轻仓" in bez["verdict"]
    else:
        # 若仍找到正超额档，至少不能是无样本的深档
        assert bez["excess_median"] > 0 and bez["n_days"] >= ec._MIN_N

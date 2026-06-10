"""最佳入场区 + 历史常驻价（regime.entry_cockpit.best_entry_zone）离线测试。

铁律：给"最佳入场区+历史常驻价"，但常驻价必须落在区间内、CI 跨 0 时降级标注、
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
    assert lo <= bez["anchor_price"] <= hi          # 历史常驻价必须落在区间内
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
    assert "历史常驻价" in s
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
    # 历史常驻价=历史命中回撤深度中位投影到当前前高 → 必落在 [price_low, price_high]
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


def test_across_horizons_picks_most_confident():
    px = _mean_reverting_series(n=2200)
    out = ec.best_entry_across_horizons(px, asset="X", horizons=(21, 63, 126, 252),
                                        single_name=True, n_boot=150)
    assert "horizon_scan" in out and len(out["horizon_scan"]) >= 1
    # 胜出者应是各候选里 (confident, dsr, ci_low) 字典序最大的
    scan = out["horizon_scan"]
    if out.get("has_zone"):
        best_conf = bool(out.get("confident"))
        # 不应存在"比胜出者更可信"的候选被漏选
        for s in scan:
            if s.get("confident") and not best_conf:
                raise AssertionError("有 confident 候选却没被选中")
        # 胜出者的 horizon 必在候选周期内
        assert out["horizon"] in (21, 63, 126, 252)


def test_across_horizons_all_defensive_returns_defensive():
    # 加速下跌：各周期都无正超额 → 跨周期也应返回防守
    import numpy as np
    n = 1400
    t = np.arange(n)
    px = pd.Series(100.0 - 2e-5 * t * t, index=pd.date_range("2014-01-01", periods=n, freq="B")).clip(lower=1.0)
    out = ec.best_entry_across_horizons(px, asset="D", horizons=(21, 63), single_name=True, n_boot=120)
    assert "horizon_scan" in out
    if not out.get("has_zone"):
        assert out["tier"] == "防守"


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


def test_entry_miss_risk_and_format():
    """踏空风险：构造一段持续上行价格，深档历史常驻价几乎不会被触及→应判'踏空风险高'。"""
    import numpy as np, pandas as pd
    from regime import entry_cockpit as ec
    idx = pd.date_range("2012-01-01", periods=1500, freq="B")
    px = pd.Series(np.linspace(100, 400, 1500) * (1 + 0.02*np.sin(np.arange(1500)/30)), index=idx)
    cur = float(px.iloc[-1])
    bez = {"has_zone": True, "anchor_price": cur*0.80, "price_band": [cur*0.78, cur*0.82],
           "horizon": 63, "tier": "稳健最佳入场区", "zone_label": "距前高-20%档"}
    mr = ec.entry_miss_risk(px, bez)
    assert mr["available"] and not mr.get("already_in")
    assert 0.0 <= mr["reach_prob"] <= 1.0 and abs(mr["reach_prob"]+mr["miss_prob"]-1.0) < 1e-9
    assert mr["miss_prob"] > 0.5  # 单边上行，深档难触及
    line = ec.format_miss_risk(mr)
    assert "踏空风险" in line and "Plan" not in line  # 句子里有裁决
    # already-in：历史常驻价在现价之上→无需等
    bez2 = dict(bez); bez2["price_band"] = [cur*1.0, cur*1.05]; bez2["anchor_price"] = cur*1.02
    mr2 = ec.entry_miss_risk(px, bez2)
    assert mr2.get("already_in") and "无踏空风险" in ec.format_miss_risk(mr2)


def test_best_entry_prefers_adequate_samples_and_ci_low():
    """选档应优先样本充足档(独立窗口≥5)并按 ci_low 排名，不再选深档少样本/烂CI的噪声。"""
    from data import loader as ld
    from regime import entry_cockpit as ec
    px = ld.load_prices(["SPY"], "1995-01-01", "2026-06-09")["SPY"].dropna()
    z = ec.entry_zones(px, asset="SPY", horizon=63, bands=ec.DEEP_BANDS)
    base = z.attrs["baseline_median"]
    bez = ec.best_entry_zone(px, asset="SPY", horizon=63, single_name=False)
    if not bez.get("has_zone"):
        return  # 防守也可接受
    row = z[z["zone"] == bez["zone_label"]].iloc[0]
    elig = z[(z["enough"]) & (z["excess_median"] > 0)]
    adequate = elig[elig["n_independent"] >= 5]
    if not adequate.empty:
        # 选中档必须来自样本充足池
        assert int(row["n_independent"]) >= 5, f"选中档 N独立={row['n_independent']}<5 但存在充足档"
        # 且是该池(按 robust/降级后)里 ci_low 最高者之一
        pool = adequate[adequate["ci_low"] > base]
        pool = pool if not pool.empty else adequate
        pool_b = pool[pool["depth_hi"] < 1.0]
        pool_b = pool_b if not pool_b.empty else pool
        assert abs(row["ci_low"] - pool_b["ci_low"].max()) < 1e-9, "未按 ci_low 择优"


def test_cross_horizon_avoids_tiny_sample_bands():
    """跨周期择优不得选到 N独立<5 的少样本档（如长周期开口深档 N=1）——只要有充足样本档可选。"""
    from data import loader as ld
    from regime import entry_cockpit as ec
    for a in ["SPY", "XLF", "TSM"]:
        zs = "1995-01-01" if a == "SPY" else "2008-01-01"
        px = ld.load_prices([a], zs, "2026-06-09")[a].dropna()
        bez = ec.best_entry_across_horizons(px, asset=a, single_name=False)
        if not bez.get("has_zone"):
            continue
        # 若任一周期存在 N独立≥5 的有区档，选中档必须 N独立≥5
        scan = bez.get("horizon_scan", [])
        has_adequate = any(s.get("has_zone") and (s.get("n_independent") or 0) >= 5 for s in scan)
        if has_adequate:
            assert (bez.get("n_independent") or 0) >= 5, f"{a}: 选中 N独立={bez.get('n_independent')}<5 但存在充足档"

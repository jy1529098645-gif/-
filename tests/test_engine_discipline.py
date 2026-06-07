"""引擎纪律层测试：体检/证据等级/多周期对账/组合预算/一致性。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import engine_discipline as ed


# ---------------------------------------------------------------------------
# 1. 基本面体检
# ---------------------------------------------------------------------------
def test_dividend_yield_unit_normalization():
    # yfinance 返回 0.87（其实是 0.87%），不应被读成 87%
    out = ed.sanity_check_fundamentals({"dividendYield": 0.87})
    assert "dividendYield" in out["clean"]
    assert abs(out["clean"]["dividendYield"] - 0.0087) < 1e-9
    assert not out["suspicious"]


def test_dividend_yield_truly_absurd_flagged():
    # 即便按百分数还原(/100)仍 >15% 才标可疑；这里给一个归一后仍越界的极端值
    out = ed.sanity_check_fundamentals({"dividendYield": 50.0})  # /100 -> 0.5 = 50% 仍越界
    assert "dividendYield" in out["suspicious"]
    assert out["warnings"]


def test_pe_and_margin_bounds():
    # PE=999 现在判为"极端但真实"(亏损/微利公司可能如此)：保留进 clean + 登记 extreme，不再剔除。
    out = ed.sanity_check_fundamentals({"trailingPE": 999.0, "grossMargins": 0.6})
    assert "trailingPE" in out["clean"]
    assert "trailingPE" in out["extreme"]
    assert "grossMargins" in out["clean"]
    # 真正不可能的值(毛利率 200% / PE 十万)才进 suspicious。
    bad = ed.sanity_check_fundamentals({"trailingPE": 99999.0, "grossMargins": 2.0})
    assert "trailingPE" in bad["suspicious"]
    assert "grossMargins" in bad["suspicious"]


# ---------------------------------------------------------------------------
# 2. 证据等级硬规则
# ---------------------------------------------------------------------------
def _bucket(excess, ci, sig=True, n=200, neff=40, low_power=False, rr=1.5):
    return {"excess": excess, "ci_low": ci[0], "ci_high": ci[1], "significant": sig,
            "n_events": n, "n_independent": neff, "low_power": low_power, "reward_risk": rr}


def test_negative_excess_never_AB():
    g = ed.evidence_grade(_bucket(-0.05, (-0.10, 0.02), sig=False))
    assert g["grade"] not in ("A", "B")
    assert g["max_position_fraction"] <= 0.25


def test_ci_crosses_zero_not_high_conf():
    g = ed.evidence_grade(_bucket(0.04, (-0.01, 0.09), sig=False))
    assert g["grade"] == "C"
    assert g["confidence"] == "低"
    assert g["ci_crosses_zero"]


def test_strong_signal_can_reach_A():
    g = ed.evidence_grade(_bucket(0.05, (0.02, 0.09), sig=True, rr=1.6))
    assert g["grade"] in ("A", "B")
    assert g["max_position_fraction"] >= 0.6


def test_significant_negative_reaches_F():
    # 显著为负 + 盈亏比<1 → F(0% 不建仓)
    g = ed.evidence_grade(_bucket(-0.05, (-0.10, -0.02), sig=True, rr=0.6))
    assert g["grade"] == "F"
    assert g["max_position_fraction"] == 0.0


def test_high_vol_negative_excess_forces_defensive():
    g = ed.evidence_grade(_bucket(-0.02, (-0.08, 0.03), sig=False), vol_percentile=0.97)
    assert g["grade"] in ("D", "F")
    assert g["max_position_fraction"] <= 0.20


def test_weak_sample_caps_grade():
    g = ed.evidence_grade(_bucket(0.05, (0.02, 0.09), sig=True, n=10, neff=3, low_power=True))
    assert g["grade"] == "C"
    assert g["sample_weak"]


# ---------------------------------------------------------------------------
# 3. 多周期对账
# ---------------------------------------------------------------------------
def test_reconcile_conflict_flag():
    per = {21: _bucket(0.05, (0.02, 0.08), sig=True),   # 短期正
           504: _bucket(-0.05, (-0.09, -0.01), sig=True)}  # 长期负
    r = ed.reconcile_horizons(per)
    assert r["conflict"]


def test_reconcile_aligned():
    per = {21: _bucket(0.05, (0.02, 0.08), sig=True),
           504: _bucket(0.06, (0.02, 0.10), sig=True)}
    r = ed.reconcile_horizons(per)
    assert not r["conflict"]
    assert "回踩可分批" in r["action"]


# ---------------------------------------------------------------------------
# 4. 组合风险预算
# ---------------------------------------------------------------------------
def test_portfolio_budget_correlation_cluster():
    idx = pd.date_range("2022-01-01", periods=400, freq="B")
    base = pd.Series(np.cumsum(np.random.RandomState(0).randn(400)) + 100, index=idx)
    prices = pd.DataFrame({
        "AAA": base,
        "BBB": base * 1.01 + 0.5,                       # 与 AAA 高度相关
        "ZZZ": pd.Series(np.cumsum(np.random.RandomState(9).randn(400)) + 100, index=idx),
    })
    briefs = [{"ticker": t, "vol_percentile": 0.5} for t in ("AAA", "BBB", "ZZZ")]
    budg = ed.portfolio_budget(briefs, prices)
    clusters = [set(c) for c in budg["clusters"]]
    assert {"AAA", "BBB"} in clusters
    assert budg["per_name"]["AAA"]["cap"] <= 0.25


def test_portfolio_budget_high_vol_capped():
    briefs = [{"ticker": "AAA", "vol_percentile": 0.96}, {"ticker": "BBB", "vol_percentile": 0.5}]
    budg = ed.portfolio_budget(briefs, None)
    assert budg["per_name"]["AAA"]["cap"] <= 0.15
    assert budg["per_name"]["AAA"]["vol_capped"]


# ---------------------------------------------------------------------------
# 5. 一致性校验
# ---------------------------------------------------------------------------
def test_consistency_take_profit_out_of_order():
    brief = {"engine_headline": _bucket(0.05, (0.02, 0.09)), "momentum_trap": False,
             "vol_percentile": 0.5}
    pb = {"if_up": ["第一减仓区 ≈ 600.0 减1/3", "下一减仓区 ≈ 前高 500"], "if_down": []}
    issues = ed.validate_consistency(brief, pb)
    assert any(i["code"] == "take_profit_out_of_order" for i in issues)


def test_consistency_trap_allows_average_down():
    brief = {"engine_headline": _bucket(-0.02, (-0.05, 0.02), sig=False), "momentum_trap": True,
             "vol_percentile": 0.5}
    pb = {"if_up": [], "if_down": ["跌到下一档可补仓"]}
    issues = ed.validate_consistency(brief, pb)
    assert any(i["code"] == "trap_allows_average_down" for i in issues)


def test_consistency_clean_when_ok():
    brief = {"engine_headline": _bucket(0.05, (0.02, 0.09)), "momentum_trap": False,
             "vol_percentile": 0.5}
    g = ed.evidence_grade(brief["engine_headline"], vol_percentile=0.5)
    pb = {"if_up": ["第一减仓区 ≈ 500.0", "下一减仓区 ≈ 前高 600"], "if_down": ["硬止损 400"]}
    issues = ed.validate_consistency(brief, pb, g)
    assert not [i for i in issues if i["severity"] == "high"]

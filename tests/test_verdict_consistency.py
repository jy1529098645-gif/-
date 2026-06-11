"""裁决一致性回归测试：防"前端裁决自相矛盾"（入场/离场/优质回踩 三者口径必须自洽）。

来历：scripts/contradiction_scan.py 跑 3 万样本×日 全历史扫描，逮到一个静态看代码看不出的隐藏矛盾——
市场宽度恶化触发离场红灯、但价仍>200线时，fear_pullback 仍为真 → 前端同时显示"🔴暂不建仓"+"✨优质回踩可加码"。
本测试把那次扫描的核心不变量固化：以后任何人改裁决逻辑，pytest 自动检出同类矛盾。

不变量（前端实际依赖）：
  · 离场红灯 / 飞刀 ⟹ 入场评级必须 🔴（不能一边撤离一边可建仓）。
  · fear_pullback(会弹"优质回踩·可加码"高亮) ⟹ 评级必须 🟢 且 warn_red=False（高亮不能和🔴评级并存）。
  · 历史小样本扫描：无 [红灯but可买] / [fear但🔴] / [在支撑but红灯] 矛盾。
"""
import warnings
import pytest

warnings.filterwarnings("ignore")

from data import loader
from analysis import decision as dec
from regime import entry_cockpit as ec
import factors.signals as sg

_DUMMY = {"has_zone": False}   # 跳过 best_entry_zone 的 bootstrap，加速；冲突检测不依赖统计锚定价


@pytest.fixture(scope="module")
def aapl():
    return loader.load_ohlcv("AAPL", "2010-01-01", "2024-12-31").dropna()


@pytest.fixture(scope="module")
def spy():
    return loader.load_ohlcv("SPY", "2010-01-01", "2024-12-31").dropna()


def test_red_suppresses_fear_pullback(aapl, spy):
    """**核心回归**：在"本会触发优质回踩"(趋势内深回踩+高VIX)的日子，强制离场红灯 →
    fear_pullback 必须被压成 False、评级必须🔴。否则前端会同时显示"🔴暂不建仓"+"✨优质回踩"绿条。
    这是 scripts/contradiction_scan.py 全历史扫描逮到的隐藏 bug 的精确回归（撤掉 fix 此测试必红）。"""
    checked = 0
    for o, nm in [(aapl, "AAPL"), (spy, "SPY")]:
        px = o["close"]; ma200 = px.rolling(200, min_periods=100).mean(); r = sg.rsi(px, 14)
        dip_days = px[(px > ma200) & (r < 38)].index            # 趋势内深回踩 → 高VIX下会触发 fear_pullback
        for d in list(dip_days)[-40:]:
            base = ec.entry_confluence(o.loc[:d], asset=nm, best_entry=_DUMMY, vix_pctile=0.9)
            if not base["fear_pullback"]:
                continue                                          # 这天没触发优质回踩，测不到该 bug，跳过
            red = ec.entry_confluence(o.loc[:d], asset=nm, best_entry=_DUMMY,
                                      warn_red=True, warn_label="测试红灯", vix_pctile=0.9)
            assert red["fear_pullback"] is False, f"红灯下 fear_pullback 仍为真(会弹优质回踩绿条) @ {nm} {d.date()}"
            assert red["grade_tag"] == "🔴", f"红灯未强制入场🔴 @ {nm} {d.date()}: {red['grade']}"
            checked += 1
    assert checked >= 1, "没采到'会触发优质回踩'的场景——回归测试无效，需调整切片条件"


def test_fear_pullback_implies_green_no_red(aapl, spy):
    """fear_pullback==True ⟹ 评级🟢(优质回踩) 且 warn_red=False —— 高亮不能和🔴/红灯并存。"""
    seen_fear = False
    for o, nm in [(aapl, "AAPL"), (spy, "SPY")]:
        px = o["close"]; ma200 = px.rolling(200, min_periods=100).mean(); r = sg.rsi(px, 14)
        dip_days = px[(px > ma200) & (r < 38)].index           # 趋势内深回踩 → 易触发 fear_pullback
        for d in list(dip_days)[-25:]:
            ef = ec.entry_confluence(o.loc[:d], asset=nm, best_entry=_DUMMY, vix_pctile=0.9)
            if ef["fear_pullback"]:
                seen_fear = True
                assert ef["grade_tag"] == "🟢" and "优质回踩" in ef["grade"], f"fear但非🟢优质回踩 @ {nm} {d.date()}"
                assert ef["warn_red"] is False, f"fear_pullback 与 warn_red 并存 @ {nm} {d.date()}"
    assert seen_fear, "构造不出 fear_pullback 场景——测试无效，需调整切片条件"


@pytest.mark.parametrize("wr,wa", [(False, False), (True, False), (False, True), (True, True)])
def test_grade_gating_matrix(aapl, wr, wa):
    """红/黄灯门控矩阵：红灯⟹🔴；任何组合下 fear_pullback⟹评级≠🔴（覆盖4种预警组合）。"""
    px = aapl["close"]
    for frac in (0.6, 0.8, 1.0):
        cut = px.index[int(len(px) * frac) - 1]
        ef = ec.entry_confluence(aapl.loc[:cut], asset="AAPL", best_entry=_DUMMY,
                                 warn_red=wr, warn_amber=wa, vix_pctile=0.85)
        if wr:
            assert ef["grade_tag"] == "🔴"
        if ef["fear_pullback"]:
            assert ef["grade_tag"] != "🔴"


def test_no_contradictions_historical_sample(aapl, spy):
    """小样本历史扫描(本地缓存)：无 [红灯but可买] / [fear但🔴] / [在支撑🟢but红灯] 矛盾。"""
    bad = []
    for o, nm in [(aapl, "AAPL"), (spy, "SPY")]:
        idx = o.index[210::90]                                  # 每~90交易日采样一次
        for d in idx:
            sub = o.loc[:d]
            ew = dec.exit_warning(sub["close"], False, None)    # fragile=False：红灯来自价<200线
            ef = ec.entry_confluence(sub, asset=nm, best_entry=_DUMMY,
                                     warn_red=ew["red"], warn_amber=ew["amber"],
                                     warn_label=ew["level"], vix_pctile=0.5)
            g = ef["grade_tag"]
            if ew["red"] and g != "🔴":
                bad.append(("A 红灯but可买", nm, d.date(), g))
            if ef["fear_pullback"] and g == "🔴":
                bad.append(("B fear但🔴", nm, d.date(), g))
            if g == "🟢" and "支撑" in ef["grade"] and ew["red"]:
                bad.append(("D 在支撑but红灯", nm, d.date(), g))
    assert not bad, f"发现裁决矛盾 {len(bad)} 处：{bad[:8]}"

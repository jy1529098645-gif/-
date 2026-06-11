"""建仓作战室模块验收（块1 价位带 / 块2 事件 / 块3 阶梯回测）。

用本地缓存数据(AAPL/SPY/财报/宏观)，小 horizon + 小 n_boot 保证快。
铁律检查：价位带给的是区间+分布(非点)、对比基准、阶梯回测带 CI。
"""
import numpy as np
import pandas as pd
import pytest

from data import loader
from regime import entry_cockpit as ec


@pytest.fixture(scope="module")
def aapl():
    return loader.load_prices(["AAPL"], "2010-01-01", "2024-12-31")


@pytest.fixture(scope="module")
def aapl_edates():
    return loader.load_earnings_dates("AAPL", limit=80)


@pytest.fixture(scope="module")
def aapl_ohlcv():
    return loader.load_ohlcv("AAPL", "2010-01-01", "2024-12-31").dropna()


def test_entry_confluence_structure(aapl_ohlcv):
    c = ec.entry_confluence(aapl_ohlcv, asset="AAPL")
    # 必含关键字段 + 类型合理
    for k in ("current_price", "falling_knife", "confluence", "confirms", "supports",
              "grade", "grade_tag", "note", "at_support_now"):
        assert k in c
    assert c["confluence"] == len(c["confirms"]) >= 0
    assert isinstance(c["supports"], list) and len(c["supports"]) >= 2     # 至少均线两条
    # 每个技术支撑都有 label/price/dist_pct
    for s in c["supports"]:
        assert s["price"] > 0 and "label" in s and "dist_pct" in s
    # 共振确认必是支撑的子集、且确实落在锚定价 ±tol 内
    if c.get("anchor"):
        for cf in c["confirms"]:
            assert abs(cf["price"] / c["anchor"] - 1.0) <= 0.025 + 1e-9


def test_entry_confluence_falling_knife_guard(aapl_ohlcv):
    # 构造'飞刀'：跌破200线 + 均线下行的深跌段 → 必标 falling_knife + 观望
    px = aapl_ohlcv["close"]
    ma200 = px.rolling(200, min_periods=100).mean()
    # 找一个 价<ma200 且 ma200 下行 的日期切片
    cond = (px < ma200) & (ma200 < ma200.shift(21))
    knife_days = cond[cond].index
    if len(knife_days):
        cut = knife_days[-1]
        c = ec.entry_confluence(aapl_ohlcv.loc[:cut], asset="AAPL")
        assert c["falling_knife"] is True
        assert "飞刀" in c["grade"] and c["grade_tag"] == "🔴"


def test_third_friday():
    # 2024-06 第三个周五 = 6/21
    assert ec._third_friday(2024, 6) == pd.Timestamp("2024-06-21")
    # 2024-12 第三个周五 = 12/20
    assert ec._third_friday(2024, 12) == pd.Timestamp("2024-12-20")
    assert ec._third_friday(2024, 6).dayofweek == 4  # 周五


def test_ladder_schedule_total_equals_budget(aapl):
    close = aapl["AAPL"].dropna().iloc[:520]
    s = ec._ladder_schedule(close, budget=10000.0, bands=(0.05, 0.10, 0.15))
    # 总投入必须等于预算（窗口末补齐剩余档）
    assert abs(np.nansum(s.to_numpy()) - 10000.0) < 1e-6
    assert pd.notna(s.iloc[0])  # day0 必投一档


def test_entry_zones_shape_and_rules(aapl):
    z = ec.entry_zones(aapl, asset="AAPL", horizon=126, n_boot=120)
    # 每个带都有价位区间（区间而非点）
    assert {"zone", "price_high", "price_low", "n_events", "is_current", "enough"} <= set(z.columns)
    assert (z["price_high"] >= z["price_low"]).all()
    # 恰有一个"今日所处带"
    assert z["is_current"].sum() == 1
    # 基准存在；够样本的带必须有 excess(对比基准)、期望值、盈亏比
    assert np.isfinite(z.attrs["baseline_median"])
    eok = z[z["enough"]]
    assert len(eok) >= 2
    assert eok["excess_median"].notna().all()
    assert eok["expectancy"].notna().all()
    # 价位随回撤加深递减
    assert z["price_high"].is_monotonic_decreasing


def test_zone_verdict_text(aapl):
    z = ec.entry_zones(aapl, asset="AAPL", horizon=126, n_boot=120)
    row = z[z["enough"]].iloc[0]
    txt = ec.format_zone_verdict(row, horizon=126)
    assert "价位带" in txt and "基准" in txt and "CI" in txt
    # 不得出现"目标价/最佳买点"等点预测措辞
    assert "目标价" not in txt and "最佳买" not in txt


def test_earnings_reaction_stats(aapl, aapl_edates):
    st = ec.earnings_reaction_stats(aapl["AAPL"], aapl_edates, pre=10, post=20)
    assert st["n_events"] > 0
    assert np.isfinite(st["day_abs_move"]["median"])
    # 财报前 drift 与财报后(分超预期)都应有分布
    assert st["pre_drift"]["n"] > 0
    assert st["post_beat"]["n"] + st["post_miss"]["n"] > 0


def test_upcoming_events(aapl, aapl_edates):
    ev = ec.upcoming_events(aapl["AAPL"], aapl_edates, n_opex=3)
    # 至少有 3 个期权到期日，且都在未来、按 days_ahead 升序
    assert (ev["event"] == "月度期权到期").sum() == 3
    assert (ev["days_ahead"] > 0).all()
    assert ev["days_ahead"].is_monotonic_increasing


def test_ladder_backtest_has_ci(aapl):
    res = ec.ladder_plan_backtest(aapl, asset="AAPL", hold=378, deploy=189,
                                  start_step=63, n_boot=120)
    assert set(res["per_strategy"]) == {"lump_sum", "dca", "ladder"}
    for k, v in res["per_strategy"].items():
        assert np.isfinite(v["median_ci_low"]) and np.isfinite(v["median_ci_high"])
        assert v["median_ci_low"] <= v["median"] <= v["median_ci_high"] + 1e-9
    assert "ladder" in res["vs_lump_sum"]

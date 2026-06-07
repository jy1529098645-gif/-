"""验收闸门 + 长周期 CI 低功效标记 验收。"""
import numpy as np
import pandas as pd

from data import loader
from evaluation import acceptance as acc
from evaluation import rule_eval as re_
from factors import signals as sg
from regime import conditional_returns as cr

EXIT = {"trailing_stop": 0.20, "take_profit": 0.25, "time_stop": 63}
MAG7 = ["AAPL", "MSFT", "GOOGL", "NVDA"]  # 取子集够快


def test_acceptance_gate_structure():
    entry = sg.build_entry([("dip_from_high", 0.15, 0.0)], "and")
    res = re_.evaluate_rule(entry, EXIT, tickers=MAG7, start="2013-01-01", n_boot=120)
    gate = acc.acceptance_gate(res, entry, EXIT, MAG7, start="2013-01-01",
                               oos_sharpe_min=1.0, max_dd_tol=0.35, train=600, test=200)
    assert set(gate["criteria"]) == {"beat_baseline", "oos_sharpe", "drawdown"}
    assert isinstance(gate["overall"], bool)
    # 每条都有 pass 布尔
    for c in gate["criteria"].values():
        assert isinstance(c["pass"], bool)
    # overall = 三条与
    crits = gate["criteria"]
    assert gate["overall"] == (crits["beat_baseline"]["pass"] and crits["oos_sharpe"]["pass"]
                               and crits["drawdown"]["pass"])
    # 回撤是负数、OOS 天数 > 0
    assert crits["drawdown"]["max_drawdown"] <= 0
    txt = acc.format_gate(gate)
    assert ("PASS" in txt or "FAIL" in txt) and "跑赢基准" in txt


def test_random_entry_fails_beat_baseline():
    """随机进场不应通过『跑赢基准』这条。"""
    rng = np.random.default_rng(1)
    rand = lambda price: pd.Series(rng.random(len(price)) < 0.03, index=price.index)  # noqa: E731
    res = re_.evaluate_rule(rand, EXIT, tickers=MAG7, start="2013-01-01",
                            rule_name="rand", n_boot=120)
    gate = acc.acceptance_gate(res, rand, EXIT, MAG7, start="2013-01-01", train=600, test=200)
    assert gate["criteria"]["beat_baseline"]["pass"] is False


def test_long_horizon_low_power_flag():
    """h=252 重叠窗口 → 有效独立窗口少 → low_power 标记应触发。"""
    macro = loader.load_macro("1990-01-01")
    price = loader.load_prices(["AAPL"], "2013-01-01")["AAPL"].dropna()
    tab = cr.conditional_forward_returns(price, macro, asset="AAPL", horizons=(252,),
                                         groupings=[["in_drawdown"]], n_boot=120)
    dd = tab[(tab.bucket == "in_drawdown=True") & (tab.horizon == 252)].iloc[0]
    assert "n_independent" in tab.columns and "low_power" in tab.columns
    # 12 年数据、252 日窗口 → 独立窗口数应远小于名义天数，且触发 low_power
    assert dd["n_independent"] < dd["n_days"]
    assert bool(dd["low_power"]) is True


def test_short_horizon_not_low_power():
    """h=21 短周期 → 独立窗口多 → 不应 low_power。"""
    macro = loader.load_macro("1990-01-01")
    price = loader.load_prices(["AAPL"], "2013-01-01")["AAPL"].dropna()
    tab = cr.conditional_forward_returns(price, macro, asset="AAPL", horizons=(21,),
                                         groupings=[["in_drawdown"]], n_boot=120)
    dd = tab[(tab.bucket == "in_drawdown=True") & (tab.horizon == 21)].iloc[0]
    assert bool(dd["low_power"]) is False

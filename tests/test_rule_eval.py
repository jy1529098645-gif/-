"""Phase S2 验收：池化评估 + N_eff + 基准对比。

N_eff 折算用合成数据单测；池化评估 + 随机入场超额≈0 健全性检查用真实数据（缓存/联网，失败 skip）。
"""
import numpy as np
import pandas as pd
import pytest

from evaluation import rule_eval as re
from factors import signals as sg


# --------------------------- N_eff ---------------------------
def test_effective_n_correlated_vs_independent():
    idx = pd.bdate_range("2015-01-01", periods=500)
    rng = np.random.default_rng(0)

    # 完全相关：N_eff ≈ N/k
    base = rng.normal(0, 0.01, 500)
    corr_df = pd.DataFrame({f"x{i}": base for i in range(4)}, index=idx)
    e = re.effective_n(corr_df, n_trades=100)
    assert e["rho_bar"] > 0.99
    assert abs(e["n_eff"] - 100 / 4) < 5

    # 相互独立：N_eff ≈ N
    ind_df = pd.DataFrame({f"x{i}": rng.normal(0, 0.01, 500) for i in range(4)}, index=idx)
    e2 = re.effective_n(ind_df, n_trades=100)
    assert abs(e2["rho_bar"]) < 0.1
    assert e2["n_eff"] > 85


# --------------------------- 集成（真实数据）---------------------------
def _check_data():
    from data import loader

    try:
        px = loader.load_prices(["AAPL", "MSFT"], "2010-01-01", "2024-01-01")
        if px.dropna().shape[0] < 1000:
            pytest.skip("数据不足")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"数据不可用：{type(exc).__name__}: {exc}")


def test_evaluate_rule_pooled_has_neff_and_ci():
    _check_data()
    res = re.evaluate_rule(
        entry_fn=lambda p: sg.dip_from_high(p, lookback=252, pct=0.15),
        exit_spec={"trailing_stop": 0.20, "take_profit": 0.25, "time_stop": 63},
        start="2010-01-01", end="2024-01-01", rule_name="t_dip15", n_boot=300,
    )
    p = res["pooled"]
    need = {"n_trades", "n_eff", "win_rate", "median_return", "median_mae",
            "excess_median", "excess_ci_low", "excess_ci_high", "excess_significant",
            "longest_losing_streak", "baseline_median"}
    assert need <= set(p)
    assert p["n_eff"] <= p["n_trades"]            # 折算后不大于名义 N
    assert p["n_eff"] < p["n_trades"]             # 七姐妹相关 → 严格变小
    assert p["excess_ci_low"] <= p["excess_median"] <= p["excess_ci_high"]
    # 单票仅展示，各自带 N
    assert all("n_trades" in v for v in res["per_ticker"].values())


def test_random_entry_excess_near_zero():
    """健全性检查：随机入场规则相对随机基准的超额应≈0、且不显著。"""
    _check_data()

    def random_entry(price: pd.Series) -> pd.Series:
        # 用价格长度做确定性种子，约 8% 的日子触发入场
        rng = np.random.default_rng(len(price))
        return pd.Series(rng.random(len(price)) < 0.08, index=price.index)

    res = re.evaluate_rule(
        entry_fn=random_entry,
        exit_spec={"time_stop": 63},
        start="2010-01-01", end="2024-01-01", rule_name="t_random", n_boot=300,
    )
    p = res["pooled"]
    assert abs(p["excess_median"]) < 0.05          # 超额≈0
    assert not p["excess_significant"]             # 不显著


def test_conditional_rule_eval_gates_entries():
    """分条件池化评估：财报条件门控应减少交易数，且结果结构完整。"""
    _check_data()
    entry = lambda p: sg.dip_from_high(p, lookback=252, pct=0.10)
    spec = {"trailing_stop": 0.20, "time_stop": 63}
    base = re.evaluate_rule(entry, spec, start="2012-01-01", end="2024-01-01", rule_name="t_base", n_boot=200)
    cond = re.evaluate_rule(
        entry, spec, start="2012-01-01", end="2024-01-01", rule_name="t_cond", n_boot=200,
        condition_fn=re.earnings_condition("post_beat", window=20),
    )
    assert cond["pooled"]["n_trades"] < base["pooled"]["n_trades"]   # 条件更严 → 更少
    assert "excess_ci_low" in cond["pooled"]


def test_format_rule_verdict_mentions_neff_ci():
    _check_data()
    res = re.evaluate_rule(
        entry_fn=lambda p: sg.dip_from_high(p, lookback=252, pct=0.15),
        exit_spec={"trailing_stop": 0.20, "time_stop": 63},
        start="2010-01-01", end="2024-01-01", rule_name="t_dip15b", n_boot=200,
    )
    text = re.format_rule_verdict(res)
    assert "N_eff" in text and "CI" in text and "基准" in text

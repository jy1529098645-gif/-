"""Phase 4 验收：建仓概率引擎。

独立事件计数、状态构造用合成数据单测；条件收益表/指纹面板用真实数据（缓存/联网，失败 skip）。
核心：每桶有 N(独立事件) + CI + 基准差值；禁止光秃秃概率（措辞模板含分布与 CI）。
"""
import numpy as np
import pandas as pd
import pytest

from regime import conditional_returns as cr


def test_count_independent_events():
    """连续 True 段算一次事件，对重叠去重。"""
    mask = pd.Series([False, True, True, False, True, False, True, True, True])
    assert cr._count_independent_events(mask) == 3
    assert cr._count_independent_events(pd.Series([False] * 5)) == 0
    assert cr._count_independent_events(pd.Series([True] * 5)) == 1


def test_forward_and_path_min():
    price = pd.Series([100, 110, 90, 120], index=pd.bdate_range("2020-01-01", periods=4))
    fwd = cr._forward_return(price, 2)
    assert abs(fwd.iloc[0] - (90 / 100 - 1)) < 1e-9     # P_{t+2}/P_t - 1
    pmin = cr._path_min_return(price, 2)
    # 从第 0 天起未来 2 天最差： min(110,90)/100 - 1 = -0.10
    assert abs(pmin.iloc[0] - (-0.10)) < 1e-9


def test_build_state_columns():
    idx = pd.bdate_range("2000-01-01", periods=1500)
    rng = np.random.default_rng(0)
    price = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 1500))), index=idx)
    macro = pd.DataFrame(
        {
            "credit_spread": pd.Series(2 + rng.normal(0, 0.3, 1500).cumsum() * 0.01, index=idx),
            "yield_curve": pd.Series(rng.normal(1, 0.2, 1500), index=idx),
        }
    )
    state = cr.build_state(price, macro)
    assert {"in_drawdown", "valuation_tercile", "credit_trend"} <= set(state.columns)
    assert state["in_drawdown"].dtype == bool
    assert set(state["valuation_tercile"].dropna().unique()) <= {"low", "mid", "high"}
    assert set(state["credit_trend"].dropna().unique()) <= {"widening", "narrowing"}


# ---------------------------------------------------------------------------
# 集成测试（真实数据）
# ---------------------------------------------------------------------------
def _load():
    from data import loader

    try:
        px = loader.load_prices(["SPY"], "1995-01-01", "2024-01-01")
        macro = loader.load_macro("1990-01-01", "2024-01-01")
        if px["SPY"].dropna().shape[0] < 2000:
            pytest.skip("价格数据不足")
        return px, macro
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"数据不可用：{type(e).__name__}: {e}")


def test_conditional_returns_has_N_CI_baseline():
    px, macro = _load()
    tab = cr.conditional_forward_returns(px, macro, asset="SPY", horizons=(63, 252), n_boot=300)

    required = {
        "grouping", "bucket", "horizon", "n_events", "n_days", "win_rate",
        "median", "p10", "p90", "median_path_drawdown",
        "baseline_median", "excess_median", "ci_low", "ci_high",
    }
    assert required <= set(tab.columns)

    # 必须有无条件基准行，且其 excess_median == 0
    base = tab[tab.grouping == "__baseline__"]
    assert not base.empty
    assert (base["excess_median"].abs() < 1e-12).all()

    # N 按独立事件计：条件桶的 n_events 应远小于 n_days（去重叠生效）
    cond = tab[tab.grouping != "__baseline__"]
    assert (cond["n_events"] <= cond["n_days"]).all()
    assert (cond["n_events"] < cond["n_days"]).any()

    # excess_median = median - baseline_median
    chk = (cond["median"] - cond["baseline_median"] - cond["excess_median"]).abs()
    assert (chk < 1e-9).all()

    # CI 合理：下界 ≤ 中位 ≤ 上界（至少多数桶成立）
    ok = ((cond["ci_low"] <= cond["median"]) & (cond["median"] <= cond["ci_high"])).mean()
    assert ok > 0.8


def test_verdict_template_mentions_N_and_CI():
    px, macro = _load()
    tab = cr.conditional_forward_returns(px, macro, asset="SPY", horizons=(252,), n_boot=300)
    row = tab[tab.grouping == "in_drawdown"].iloc[0]
    text = cr.format_bucket_verdict(row)
    assert "独立事件" in text and "CI" in text and "基准" in text


def test_current_fingerprint_panel():
    px, macro = _load()
    fp = cr.current_fingerprint(px, macro, asset="SPY", top_k=5)
    assert "TODAY" in fp.index
    assert {"drawdown", "credit_spread", "credit_trend", "yield_curve", "valuation_tercile"} <= set(fp.columns)
    assert "disclaimer" in fp.attrs
    # 历史谷底回撤应明显为负
    troughs = fp.drop(index="TODAY")
    assert (troughs["drawdown"] < -0.05).all()

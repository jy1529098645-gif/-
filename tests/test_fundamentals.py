"""Phase F1 验收：财报日历因子（免费 yfinance 数据）。

PIT 因子逻辑用合成数据单测（无前视严格验证）；PEAD 领先 IC + 假财报日对照用真实数据（缓存/联网，失败 skip）。
核心健全性检查：随机假财报日对照平均 IC≈0。
"""
import numpy as np
import pandas as pd
import pytest

from factors import fundamentals as fu


@pytest.fixture
def synth():
    idx = pd.date_range("2020-01-01", periods=70, freq="D")
    edates = pd.DataFrame(
        {"EPS Estimate": [1.0, 1.0], "Reported EPS": [1.1, 0.9], "Surprise(%)": [5.0, -3.0]},
        index=pd.to_datetime(["2020-01-15", "2020-02-15"]),
    )
    edates.index.name = "earnings_date"
    return idx, edates


def test_days_to_next_and_since(synth):
    idx, ed = synth
    dtn = fu.days_to_next_earnings(idx, ed)
    assert dtn.loc["2020-01-10"] == 5      # 距 01-15 五天
    assert dtn.loc["2020-01-15"] == 0      # 当天
    dsl = fu.days_since_last_earnings(idx, ed)
    assert dsl.loc["2020-01-20"] == 5
    assert (dtn.dropna() >= 0).all() and (dsl.dropna() >= 0).all()


def test_last_surprise_is_pit(synth):
    """无前视：财报盘后公布，公布日当天及之前不得"看见"该次 surprise。"""
    idx, ed = synth
    ls = fu.last_surprise(idx, ed)
    assert pd.isna(ls.loc["2020-01-14"])   # 首份财报前 → 未知
    assert pd.isna(ls.loc["2020-01-15"])   # 公布当天（盘后）→ 仍未知
    assert ls.loc["2020-01-16"] == 5.0     # 次日才可知
    assert ls.loc["2020-02-16"] == -3.0    # 第二份公布后
    assert bool(fu.last_beat(idx, ed).loc["2020-01-16"]) is True
    assert bool(fu.last_beat(idx, ed).loc["2020-02-16"]) is False


def test_windows(synth):
    idx, ed = synth
    pre = fu.pre_earnings_window(idx, ed, window=10)
    assert bool(pre.loc["2020-02-10"]) is True     # 距 02-15 五天 ≤10
    assert bool(pre.loc["2020-01-25"]) is False    # 距下次 >10 天
    post = fu.post_earnings_window(idx, ed, window=10)
    assert bool(post.loc["2020-01-18"]) is True    # 上次 01-15 后 3 天


# --------------------------- 集成（真实数据）---------------------------
def _load():
    from data import loader

    try:
        tickers = loader.load_universe("mag7")
        prices = {t: loader.load_prices([t], "2010-01-01", "2024-01-01")[t].dropna() for t in tickers}
        edates = {t: loader.load_earnings_dates(t, limit=80) for t in tickers}
        if any(e.dropna(subset=["Surprise(%)"]).shape[0] < 10 for e in edates.values()):
            pytest.skip("财报数据不足")
        return prices, edates
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"数据不可用：{type(e).__name__}: {e}")


def test_load_earnings_dates_columns():
    from data import loader

    try:
        ed = loader.load_earnings_dates("AAPL", limit=40)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"数据不可用：{type(e).__name__}: {e}")
    assert {"EPS Estimate", "Reported EPS", "Surprise(%)"} <= set(ed.columns)
    assert ed.index.is_monotonic_increasing
    assert ed.dropna(subset=["Surprise(%)"]).shape[0] > 5


def test_pead_ic_fake_control_near_zero():
    """健全性检查：随机假财报日对照的平均 IC 应≈0。"""
    from evaluation import earnings_eval as ee

    prices, edates = _load()
    ic = ee.earnings_drift_ic(prices, edates, horizons=(1, 5, 21, 63), n_control=60)
    tab = ic["ic_table"]
    assert ic["n_events"] > 100
    assert tab["ic_fake_mean"].abs().max() < 0.03      # 假对照≈0
    # 短周期 PEAD 真实 IC 应为正且显著（文献现象）
    short = tab[tab["horizon"] == 5].iloc[0]
    assert short["ic_real"] > 0.03
    assert short["significant"]


def test_event_study_structure():
    from evaluation import earnings_eval as ee

    prices, edates = _load()
    study = ee.earnings_event_study(prices, edates, pre=10, post=20, by_beat=True)
    assert study["n_events"] > 100
    assert len(study["mean_car"]) == 31
    # 公布后超预期组应高于不及预期组（反应日之后）
    post0 = np.where(study["offsets"] == 5)[0][0]
    assert study["mean_car_beat"][post0] > study["mean_car_miss"][post0]

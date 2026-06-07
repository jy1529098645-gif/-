"""P0/P1/P2 加固改动的离线回归测试（不联网）。

覆盖：
- P0  时间止损方向（entry@t → exit@t+n，非 shift(-n)）
- P0  acceptance 读 config 的 walk-forward 参数
- P1  blend 权重校验（负权/空 panels 抛错）
- P1  deflated_rule_sharpe 的 extra_trials 计入多重检验（DSR 随尝试数增多而下降）
- P2  价格/OHLCV 缓存文件名带版本号
- P2  基本面阈值分级（极端但真实 vs 不可能）
- P2  新闻进程内 TTL 缓存命中
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# P0：时间止损方向
# ---------------------------------------------------------------------------
def test_time_stop_direction():
    from backtest.exits import _time_stop_exits

    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    entries = pd.Series(False, index=idx)
    entries.iloc[2] = True  # t=2 入场
    exits = _time_stop_exits(entries, 3)
    # 出场必须在 t=2+3=5（向未来搬），且 t<5 全 False（不得在入场前/当日出现）
    assert bool(exits.iloc[5]) is True
    assert not exits.iloc[:5].any()
    assert exits.sum() == 1


def test_time_stop_disabled_when_n_lt_1():
    from backtest.exits import _time_stop_exits

    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    entries = pd.Series([True, False, False, True, False], index=idx)
    assert not _time_stop_exits(entries, 0).any()


# ---------------------------------------------------------------------------
# P0：acceptance 读 config walk-forward
# ---------------------------------------------------------------------------
def test_acceptance_reads_config_wf():
    import config
    from evaluation import acceptance as ac

    wf = config.load_config().get("stats", {}).get("walk_forward", {})
    assert ac._WF_TRAIN == int(wf.get("train", 1260))
    assert ac._WF_TEST == int(wf.get("test", 252))
    assert ac._WF_STEP == int(wf.get("step", ac._WF_TEST))


# ---------------------------------------------------------------------------
# P1：blend 权重校验
# ---------------------------------------------------------------------------
def _toy_panel(seed):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=20, freq="D")
    return pd.DataFrame(rng.standard_normal((20, 3)), index=idx, columns=["A", "B", "C"])


def test_blend_rejects_negative_weight():
    from factors.price_factors import blend

    with pytest.raises(ValueError):
        blend({"mom": _toy_panel(1)}, weights={"mom": -1.0})


def test_blend_rejects_empty_panels():
    from factors.price_factors import blend

    with pytest.raises(ValueError):
        blend({})


def test_blend_zero_weight_sum_rejected():
    from factors.price_factors import blend

    with pytest.raises(ValueError):
        blend({"mom": _toy_panel(1)}, weights={"mom": 0.0})


def test_blend_normal_path_ok():
    from factors.price_factors import blend

    out = blend({"a": _toy_panel(1), "b": _toy_panel(2)})
    assert out.shape == (20, 3)
    assert np.isfinite(out.to_numpy()).any()


# ---------------------------------------------------------------------------
# P1：多重检验计数（extra_trials 让 DSR 下降）—— 直接用底层公式验证单调性
# ---------------------------------------------------------------------------
def test_more_trials_lowers_deflated_sharpe():
    from stats.deflated_sharpe import deflated_sharpe_ratio

    common = dict(sr=0.15, sr_trials_std=0.05, n_obs=200)
    dsr_few = deflated_sharpe_ratio(n_trials=5, **common)
    dsr_many = deflated_sharpe_ratio(n_trials=50, **common)
    # 试验越多，"运气撞出来"门槛越高 → 同一夏普的 DSR 概率越低
    assert dsr_many < dsr_few


# ---------------------------------------------------------------------------
# P2：缓存版本化
# ---------------------------------------------------------------------------
def test_cache_paths_carry_version():
    from data import loader

    assert loader.CACHE_VERSION in loader._price_cache_path("AAPL").name
    assert loader.CACHE_VERSION in loader._ohlcv_cache_path("AAPL").name


# ---------------------------------------------------------------------------
# P2：基本面阈值分级
# ---------------------------------------------------------------------------
def test_fundamentals_three_tier():
    from analysis import engine_discipline as ed

    out = ed.sanity_check_fundamentals({
        "trailingPE": 25.0,      # 正常
        "forwardPE": 800.0,      # 极端但可能(亏损股)
        "grossMargins": 3.0,     # 不可能(>100%)
    })
    assert out["clean"].get("trailingPE") == 25.0
    assert "forwardPE" in out["clean"] and "forwardPE" in out["extreme"]
    assert "grossMargins" in out["suspicious"]


# ---------------------------------------------------------------------------
# P2：新闻 TTL 缓存
# ---------------------------------------------------------------------------
def test_news_ttl_cache(monkeypatch):
    from data import news

    news.clear_news_cache()
    calls = {"n": 0}

    def fake_google(query, limit=20):
        calls["n"] += 1
        return [{"date": pd.Timestamp("2024-01-01"), "title": f"headline {calls['n']}",
                 "provider": "X", "url": "http://x", "summary": ""}]

    monkeypatch.setattr(news, "_google_news", fake_google)
    df1 = news.stock_news("AAPL", limit=5, sources=("google",))
    df2 = news.stock_news("AAPL", limit=5, sources=("google",))  # 应命中缓存，不再调用
    assert calls["n"] == 1
    assert df1.equals(df2)
    # use_cache=False 强制刷新
    news.stock_news("AAPL", limit=5, sources=("google",), use_cache=False)
    assert calls["n"] == 2
    news.clear_news_cache()

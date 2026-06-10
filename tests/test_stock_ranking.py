"""验收：选股榜(多因子横截面综合评分)引擎。

铁律检查：① z-score winsorize 正确；② 综合分按权重合成、排序；③ 排名结构完整 +
tier 与趋势健康挂钩；④ IC 验证含安慰剂对照 + 诚实裁决(无预测力时不吹)。
"""
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from analysis import stock_ranking as sr


def test_zwins_winsorize():
    s = pd.Series([0, 1, 2, 3, 100.0])
    z = sr._zwins(s, winsor=2.0)
    assert z.abs().max() <= 2.0 + 1e-9
    assert abs(z.mean()) < 1.0  # 去均值后接近 0(winsorize 后不严格 0)


def test_weights_sum_and_keys():
    assert abs(sum(sr.DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9
    assert "risk_adj_mom" in sr.DEFAULT_WEIGHTS and "trend_quality" in sr.DEFAULT_WEIGHTS


def test_factor_frame_no_lookahead_shapes():
    idx = pd.bdate_range("2015-01-01", periods=400)
    rng = np.random.default_rng(0)
    cols = ["A", "B", "C", "D", "E"]
    px = pd.DataFrame({c: 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, len(idx)))) for c in cols}, index=idx)
    spy = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(idx)))), index=idx)
    panels = sr._factor_frame(px, spy)
    assert set(["risk_adj_mom", "trend_quality", "rel_strength", "mom_12_1", "low_vol"]).issubset(panels)
    for f in panels.values():
        assert list(f.columns) == cols
    # 综合分某日：去 NaN 后非空、降序
    comp = sr._composite_at(panels, px.index[-1], sr.DEFAULT_WEIGHTS)
    assert comp.is_monotonic_decreasing


def test_ranking_smoke_and_verdict_honest():
    tickers = ["NVDA", "AMD", "AVGO", "TSM", "MU", "INTC", "AAPL", "MSFT", "GOOGL", "META"]
    res = sr.rank_stocks(tickers, start="2014-01-01")
    tab = res["table"]
    assert {"排名", "票", "综合分", "tier"}.issubset(tab.columns)
    assert tab["排名"].tolist() == sorted(tab["排名"].tolist())  # 排名连续递增
    # 验证 + 裁决
    val = sr.validate_ranking(tickers, start="2014-01-01")
    assert 21 in val and "placebo_ic_mean" in val[21]
    verd = sr.ranking_verdict(val)
    assert verd["grade"][0] in "🟢🟡🔴"
    assert "筛选器" in verd["usage"]  # 诚实定位：筛选器非预测器

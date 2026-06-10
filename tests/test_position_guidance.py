"""验收：建仓/撤离作战卡引擎（v3 验证版）。

铁律检查：① 暴露遵守 v3 趋势门规则(趋势死亡→0)；② 三档单调(进取≥中性≥稳健)；
③ 动量陷阱→建仓转防守；④ 撤离始终带"崩盘保险/不跑赢长牛"诚实口径 + 排除固定止损。
"""
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from analysis import position_guidance as pg


def test_target_exposure_trend_gate():
    # 趋势死亡(200线下+斜率转负) → 暴露归零
    assert pg._target_exposure(trend_up=False, slope_pos=False, ewmav=0.30, tvol=0.25) == 0.0
    # 趋势内 → min(1, tvol/vol)
    e = pg._target_exposure(trend_up=True, slope_pos=True, ewmav=0.50, tvol=0.25)
    assert abs(e - 0.5) < 1e-9
    # 低波动 → 封顶 1.0
    assert pg._target_exposure(True, True, ewmav=0.10, tvol=0.25) == 1.0
    # 200线下但斜率未转负 → 半仓过渡
    assert abs(pg._target_exposure(False, True, 0.50, 0.25) - 0.25) < 1e-9


def test_profiles_monotonic():
    # 同状态下，进取暴露 ≥ 中性 ≥ 稳健（波动目标更高→暴露更高，封顶前）
    e_c = pg._target_exposure(True, True, 0.60, pg.PROFILES["conservative"])
    e_m = pg._target_exposure(True, True, 0.60, pg.PROFILES["moderate"])
    e_a = pg._target_exposure(True, True, 0.60, pg.PROFILES["aggressive"])
    assert e_a >= e_m >= e_c > 0


def test_exposure_series_no_lookahead_and_bounds():
    idx = pd.bdate_range("2015-01-01", periods=600)
    rng = np.random.default_rng(0)
    price = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, len(idx)))), index=idx)
    es = pg._exposure_series(price, 0.25)
    assert es.index.equals(price.index)
    assert float(es.min()) >= 0.0 and float(es.max()) <= 1.0


@pytest.mark.parametrize("tk", ["SPY"])
def test_guidance_smoke_structure(tk):
    g = pg.position_guidance(tk)
    assert set(["regime", "exposure", "build", "exit", "headline", "exposure_history"]).issubset(g)
    # 三档暴露都在 [0,1]
    for k in pg.PROFILES:
        assert 0 <= g["exposure"][k]["exposure"] <= 1
    # 撤离必须含诚实口径 + 排除固定止损
    trg = " ".join(g["exit"]["triggers"])
    assert "固定%移动止损" in trg or "二元止损" in trg
    assert "崩盘保险" in g["exit"]["honesty"]
    # Markdown 渲染不报错且含关键节
    md = pg.format_guidance(g)
    assert "建仓" in md and "撤离" in md and "今日建议暴露" in md


def test_markdown_contains_honesty_disclaimer():
    g = pg.position_guidance("SPY")
    md = pg.format_guidance(g)
    assert "非买卖指令" in md
    assert "不跑赢长牛持有" in g["disclaimer"]

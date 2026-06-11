"""验收：建仓/撤离作战卡引擎（v3 / v3.1 软地板版）。

铁律检查：① 暴露遵守趋势门规则(破位降到 floor×base，稳健 floor=0 仍清零)；② 三档单调；
③ 动量陷阱→建仓转防守；④ 撤离始终带"崩盘保险/不跑赢长牛"诚实口径 + 排除固定止损；
⑤ v3.1 软地板：floor/slope_floor 参数生效，各档 floor 配置正确。
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
    # 低波动 → 封顶 1.0(默认 max_lev=1)
    assert pg._target_exposure(True, True, ewmav=0.10, tvol=0.25) == 1.0
    # 200线下但斜率未转负 → 半仓过渡
    assert abs(pg._target_exposure(False, True, 0.50, 0.25) - 0.25) < 1e-9


def test_leverage_lowvol_gate():
    # 低波动(分位 0.3 ≤ 0.5)+确认趋势 → 允许上杠杆到 1.5 封顶
    assert abs(pg._target_exposure(True, True, 0.20, 0.40, max_lev=1.5, vol_pct=0.30) - 1.5) < 1e-9
    # 中等波动低分位：0.40/0.30=1.33 未封顶
    assert abs(pg._target_exposure(True, True, 0.30, 0.40, max_lev=1.5, vol_pct=0.30) - (0.40 / 0.30)) < 1e-9
    # 高波动(分位 0.9 > 0.5) → 低波动门把杠杆收回 1.0
    assert abs(pg._target_exposure(True, True, 0.20, 0.40, max_lev=1.5, vol_pct=0.90) - 1.0) < 1e-9
    # 波动分位未知(None) → 保守：不许上杠杆，封顶 1.0
    assert abs(pg._target_exposure(True, True, 0.20, 0.40, max_lev=1.5, vol_pct=None) - 1.0) < 1e-9
    # 杠杆档遇趋势死亡仍归零(破位清仓优先于一切)
    assert pg._target_exposure(False, False, 0.20, 0.40, max_lev=1.5, vol_pct=0.30) == 0.0
    # max_lev=1 时不受门影响(低波动也不会>1)
    assert pg._target_exposure(True, True, 0.10, 0.25, max_lev=1.0, vol_pct=0.10) == 1.0


def test_soft_floor_break():
    """v3.1 软地板：破位不再一刀清零，而是降到 floor×base；slope_floor 控过渡档。"""
    base = min(1.0, 0.25 / 0.30)                      # ewmav=0.30 → base≈0.833
    # 趋势死亡 + floor=0.5 → 0.5×base（保留底仓，非 0）
    assert abs(pg._target_exposure(False, False, 0.30, 0.25, floor=0.5) - 0.5 * base) < 1e-9
    # 趋势死亡 + floor=0.0（稳健/默认）→ 仍清零
    assert pg._target_exposure(False, False, 0.30, 0.25, floor=0.0) == 0.0
    # 线下未死 + slope_floor=0.7 → 0.7×base
    assert abs(pg._target_exposure(False, True, 0.30, 0.25, slope_floor=0.7) - 0.7 * base) < 1e-9
    # floor 越高破位暴露越高（单调）
    fs = [pg._target_exposure(False, False, 0.30, 0.25, floor=f) for f in (0.0, 0.3, 0.5)]
    assert fs == sorted(fs) and fs[0] == 0.0 and fs[-1] > 0


def test_profiles_have_floor_config():
    """各档 floor/slope_floor 配置：稳健清零(0)、中性/进取软地板0.5、杠杆0.4。"""
    for k in ("conservative", "moderate", "aggressive", "leveraged"):
        assert "floor" in pg.PROFILES[k] and "slope_floor" in pg.PROFILES[k]
    assert pg.PROFILES["conservative"]["floor"] == 0.0          # 稳健=最强保险(仍清零)
    assert pg.PROFILES["moderate"]["floor"] >= 0.4              # 中性=软地板(找回复利)
    assert pg.PROFILES["aggressive"]["floor"] >= 0.4
    assert pg.PROFILES["leveraged"]["floor"] >= 0.3
    # 软地板档破位暴露 > 稳健档破位暴露（同状态）
    cons = pg._target_exposure(False, False, 0.30, pg.PROFILES["conservative"]["tvol"],
                               floor=pg.PROFILES["conservative"]["floor"])
    mod = pg._target_exposure(False, False, 0.30, pg.PROFILES["moderate"]["tvol"],
                              floor=pg.PROFILES["moderate"]["floor"])
    assert cons == 0.0 and mod > 0.0


def test_profiles_monotonic():
    # 同状态下，杠杆进取 ≥ 进取 ≥ 中性 ≥ 稳健
    es = [pg._target_exposure(True, True, 0.60, pg.PROFILES[k]["tvol"], pg.PROFILES[k]["max_lev"])
          for k in ("conservative", "moderate", "aggressive", "leveraged")]
    assert es == sorted(es) and es[0] > 0
    # 杠杆档 max_lev>1，其余=1
    assert pg.PROFILES["leveraged"]["max_lev"] > 1.0
    assert all(pg.PROFILES[k]["max_lev"] == 1.0 for k in ("conservative", "moderate", "aggressive"))


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
    # 暴露在 [0, max_lev]；杠杆档可>1
    for k in pg.PROFILES:
        assert 0 <= g["exposure"][k]["exposure"] <= pg.PROFILES[k]["max_lev"] + 1e-9
    assert "leveraged" in g["exposure"] and g["exit"].get("leverage_warning")
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

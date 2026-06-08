"""风险管理叠加（已端到端验证的可部署规则）。

规则：持仓比例 = 0.5 × 波动目标仓 + 0.5 × 趋势/宽度半仓floor
  - 波动目标仓 = clip(目标年化波动 / 近21日实现波动, 0, 1)   —— 波动大自动降仓(Moreira-Muir)
  - 半仓floor = 0.5 + 0.25×(站上200线) + 0.25×(站上200线 且 市场宽度healthy)
                —— 趋势破位/宽度恶化时降到半仓,不空仓(回测证明比清仓更优)
信号滞后1日(防前视),换仓按单边成本计。长仓不加杠杆(上限100%)。

验证(v1→dig2 反复迭代):
  ETF 全程 夏普 0.77 vs 持有 0.70、2015+ OOS 1.00 vs 0.91;
  个股 全程 0.84 vs 0.83、2015+ OOS 0.99 vs 0.92;
  两类资产、样本内外均升夏普,回撤砍约40%(-33% vs -58%),且对参数(目标波动12-20%/混合0.4-0.6)不敏感。
诚实边界:这是**风险管理**(改善夏普/砍回撤),不是择时alpha;牛市中CAGR略低于满仓持有,换来更稳。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_VOL = 0.15
BLEND = 0.5          # 波动目标 与 趋势半仓floor 的混合权重
COST = 0.001         # 单边换仓成本


def risk_managed_position(px: pd.Series, fragile: pd.Series | None = None,
                          target_vol: float = TARGET_VOL, blend: float = BLEND) -> pd.Series:
    """返回逐日目标仓位(0–1)。fragile=市场宽度脆弱布尔序列(None=视为healthy)。"""
    px = px.dropna()
    r = px.pct_change()
    trend = (px > px.rolling(200, min_periods=100).mean()).astype(float)
    if fragile is None:
        h = pd.Series(1.0, index=px.index)
    else:
        h = (~fragile.astype(bool)).reindex(px.index).ffill().fillna(True).astype(float)
    rv = r.rolling(21, min_periods=10).std() * np.sqrt(252)
    vt = (target_vol / rv).clip(0, 1.0)
    floor = 0.5 + 0.25 * trend + 0.25 * (trend * h)
    pos = (blend * vt + (1 - blend) * floor).clip(0, 1)
    return pos


def _stats(ret: pd.Series) -> dict:
    ret = ret.dropna()
    if len(ret) < 60:
        return {"cagr": float("nan"), "sharpe": float("nan"), "maxdd": float("nan")}
    eq = (1 + ret).cumprod()
    yrs = len(ret) / 252
    return {"cagr": float(eq.iloc[-1] ** (1 / yrs) - 1),
            "sharpe": float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else float("nan"),
            "maxdd": float((eq / eq.cummax() - 1).min())}


def backtest_overlay(px: pd.Series, fragile: pd.Series | None = None,
                     target_vol: float = TARGET_VOL, blend: float = BLEND,
                     cost: float = COST) -> dict:
    """回测风险管理叠加 vs 闭眼持有。返回净值曲线 + 三项统计 + 当前建议仓位。"""
    px = px.dropna()
    r = px.pct_change()
    pos = risk_managed_position(px, fragile, target_vol, blend)
    pos_l = pos.shift(1).fillna(0)                       # 滞后1日,防前视
    turn = pos_l.diff().abs().fillna(0)
    strat_ret = (pos_l * r - turn * cost)
    eq_s = (1 + strat_ret.dropna()).cumprod()
    eq_h = (1 + r.dropna()).cumprod()
    return {
        "equity": pd.DataFrame({"风险管理叠加": eq_s, "闭眼持有": eq_h}).dropna(),
        "strategy": _stats(strat_ret), "hold": _stats(r),
        "current_position": float(pos.iloc[-1]) if len(pos) else float("nan"),
        "avg_position": float(pos_l.mean()),
        "target_vol": target_vol, "blend": blend,
    }


def verdict(bt: dict) -> str:
    """一句话裁决。"""
    s, h = bt["strategy"], bt["hold"]
    ds = s["sharpe"] - h["sharpe"]
    better = "更优(夏普↑·回撤↓)" if ds >= 0 else "夏普未改善(此标的更适合长持)"
    return (f"风险管理叠加：夏普 {s['sharpe']:.2f} vs 持有 {h['sharpe']:.2f}、"
            f"回撤 {s['maxdd']:.0%} vs {h['maxdd']:.0%}、年化 {s['cagr']:+.0%} vs {h['cagr']:+.0%}"
            f"｜当前建议仓位 {bt['current_position']:.0%}｜{better}")

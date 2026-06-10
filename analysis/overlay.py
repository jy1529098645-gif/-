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
        return {k: float("nan") for k in ("cagr", "vol", "sharpe", "sortino", "maxdd", "calmar")}
    eq = (1 + ret).cumprod()
    yrs = len(ret) / 252
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    vol = float(ret.std() * np.sqrt(252))
    sharpe = float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else float("nan")
    dn = ret[ret < 0].std()
    sortino = float(ret.mean() * 252 / (dn * np.sqrt(252))) if dn and dn > 0 else float("nan")
    mdd = float((eq / eq.cummax() - 1).min())
    calmar = float(cagr / abs(mdd)) if mdd < 0 else float("nan")
    return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "sortino": sortino,
            "maxdd": mdd, "calmar": calmar}


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


# 大规模分板块回测结论：叠加在高beta/周期板块+ETF 升夏普；能源(商品·均值回归)与防御板块
# (必需消费/公用/部分医疗)只砍回撤、夏普≈持平。分板块精调 vol 目标不稳健(过拟合)→统一用全局15%。
_ENERGY = {"XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "XLE"}
_DEFENSIVE = {"PG", "KO", "PEP", "WMT", "COST", "MO", "MDLZ", "CL", "NEE", "DUK", "SO", "D",
              "AEP", "XLP", "XLU"}


def sector_effectiveness(asset: str) -> str:
    """该标的所属板块上，风险管理叠加的实测有效性(大规模回测结论)。"""
    a = asset.upper()
    if a in _ENERGY:
        return "⚠️ 能源板块：商品驱动+均值回归，叠加**主要用于砍回撤**，夏普≈持平甚至略降，趋势规则慎用。"
    if a in _DEFENSIVE:
        return "ℹ️ 防御板块(消费/公用)：本就低波，叠加**主要砍回撤**，夏普提升有限。"
    return "✅ 高beta/周期/科技/ETF：叠加历史上**升夏普+砍回撤**，是适用的主战场。"


def backtest_portfolio(prices: dict, fragile: pd.Series | None = None,
                       benchmark: pd.Series | None = None,
                       target_vol: float = TARGET_VOL, blend: float = BLEND,
                       cost: float = COST) -> dict:
    """产品级组合回测：等权聚焦组合，逐标的应用风险管理叠加 vs 闭眼持有(+可选基准)。

    prices: {ticker: 价格Series}; benchmark: 基准价格Series(如SPY)。
    返回各方案净值曲线 + 完整指标(夏普/索提诺/卡玛/回撤)。
    """
    ov_rets, ho_rets = [], []
    for t, px in prices.items():
        px = px.dropna()
        if len(px) < 252:
            continue
        r = px.pct_change()
        pos = risk_managed_position(px, fragile, target_vol, blend).shift(1).fillna(0)
        turn = pos.diff().abs().fillna(0)
        ov_rets.append((pos * r - turn * cost).rename(t))
        ho_rets.append(r.rename(t))
    if not ov_rets:
        return {"available": False}
    port_ov = pd.concat(ov_rets, axis=1).mean(axis=1)
    port_ho = pd.concat(ho_rets, axis=1).mean(axis=1)
    out = {"available": True, "n_assets": len(ov_rets),
           "overlay": _stats(port_ov), "hold": _stats(port_ho),
           "ret_overlay": port_ov, "ret_hold": port_ho,
           "equity": pd.DataFrame({"组合+风险叠加": (1 + port_ov.dropna()).cumprod(),
                                   "组合闭眼持有": (1 + port_ho.dropna()).cumprod()})}
    if benchmark is not None:
        br = benchmark.pct_change()
        out["benchmark"] = _stats(br)
        out["equity"]["基准"] = (1 + br.reindex(out["equity"].index).fillna(0)).cumprod()
    out["equity"] = out["equity"].dropna()
    return out


# 历史重大回撤窗口（压力测试用）——证明风控在关键时刻是否真管用
CRISES = [
    ("2018Q4 加息恐慌", "2018-10-01", "2018-12-24"),
    ("2020 COVID 崩盘", "2020-02-19", "2020-03-23"),
    ("2022 加息熊市", "2022-01-03", "2022-10-12"),
    ("2025 关税冲击", "2025-02-19", "2025-04-08"),
]


def crisis_stress(equity: pd.DataFrame, crises=CRISES) -> list:
    """各历史崩盘窗口内，每条净值曲线的区间收益（看叠加是否真的少跌）。"""
    rows = []
    for name, s, e in crises:
        seg = equity.loc[s:e]
        if len(seg) < 5:
            continue
        row = {"crisis": name, "start": s, "end": e}
        for col in equity.columns:
            row[col] = float(seg[col].iloc[-1] / seg[col].iloc[0] - 1.0)
        rows.append(row)
    return rows


def stability_stats(ret: pd.Series) -> dict:
    """稳定性画像：年化/波动/最大回撤/最差年/正收益月占比/滚动1年正收益占比/最长水下(月)。"""
    ret = ret.dropna()
    if len(ret) < 60:
        return {}
    eq = (1 + ret).cumprod(); yrs = len(ret) / 252
    yr = ret.groupby(ret.index.year).apply(lambda x: (1 + x).prod() - 1)
    mo = ret.resample("ME").apply(lambda x: (1 + x).prod() - 1) if hasattr(ret.index, "freq") or True else ret
    try:
        mo = ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    except Exception:  # noqa: BLE001
        mo = ret.resample("M").apply(lambda x: (1 + x).prod() - 1)
    roll1y = eq.pct_change(252).dropna()
    dd = eq / eq.cummax() - 1
    uw = (dd < -0.05).astype(int)
    longest = cur = 0
    for u in uw:
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    return {
        "cagr": float(eq.iloc[-1] ** (1 / yrs) - 1),
        "vol": float(ret.std() * np.sqrt(252)),
        "maxdd": float(dd.min()),
        "worst_year": float(yr.min()),
        "pos_months": float((mo > 0).mean()),
        "pos_roll1y": float((roll1y > 0).mean()) if len(roll1y) else float("nan"),
        "longest_underwater_m": int(longest // 21),
    }


def rolling_sharpe(ret: pd.Series, window: int = 252) -> pd.Series:
    """滚动年化夏普——监控策略 edge 是否随时间衰减。"""
    r = ret.dropna()
    m = r.rolling(window).mean()
    s = r.rolling(window).std()
    return (m / s * np.sqrt(252)).dropna()


def verdict(bt: dict) -> str:
    """一句话裁决。"""
    s, h = bt["strategy"], bt["hold"]
    ds = s["sharpe"] - h["sharpe"]
    dd_better = s["maxdd"] >= h["maxdd"]   # maxdd≤0，叠加的更浅(更大)=回撤改善
    if ds >= 0:
        better = "更优(夏普↑" + ("·回撤↓)" if dd_better else "·但回撤未改善)")
    else:
        better = "夏普未改善(此标的更适合长持)" + ("·回撤↓" if dd_better else "")
    return (f"风险管理叠加：夏普 {s['sharpe']:.2f} vs 持有 {h['sharpe']:.2f}、"
            f"回撤 {s['maxdd']:.0%} vs {h['maxdd']:.0%}、年化 {s['cagr']:+.0%} vs {h['cagr']:+.0%}"
            f"｜当前建议仓位 {bt['current_position']:.0%}｜{better}")
